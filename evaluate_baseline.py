#!/usr/bin/env python
"""Evaluate visual baseline models (frozen-embedding or fine-tuned PTM + FCN / CNN-maxpool head).

Example
-------
python evaluate_baseline.py --metadata data/meta.csv --config config.json \
    --filepath image_path --target label --out_dir runs_out --device auto \
    --metrics accuracy eer loss f1_macro --monitor loss
"""
import argparse
import hashlib
import copy
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms as T

try:                                   # progress bars are optional (pip install tqdm)
    from tqdm.auto import tqdm
except ImportError:                    # pragma: no cover
    tqdm = None

from models import DOWNSTREAMS, MODEL_REGISTRY, BaselinePipeline, build_backbone, build_head
from models.weights import set_cache_dir

# --------------------------------------------------------------------------------------
# Config handling
# --------------------------------------------------------------------------------------
RESERVED_MODEL_KEYS = {"enabled", "mode", "downstream", "frozen", "finetune", *DOWNSTREAMS.keys()}
DEFAULTS = dict(
    pretrained=True, embedding_layer="final", optimizer="adam", lr=1e-3, backbone_lr=None,
    weight_decay=0.0, lr_scheduler="none", lr_decay=0.95, lr_step=10, epochs=10, batch_size=128,
    patience=0, augment=False, num_workers=16, normalize_embeddings="standardize",
    amp=True, amp_dtype="bfloat16",
    dropout=0.2, activation="relu", norm="batch",
)
ALL_METRICS = ["accuracy", "balanced_accuracy", "loss", "eer", "auc", "f1_macro", "f1_weighted",
               "precision_macro", "recall_macro"]
LOWER_IS_BETTER = {"loss", "eer"}


def resolve_cfg(config, mname, mode, ds):
    """defaults < global < downstream section < model < model[mode] < model[downstream]."""
    c = dict(DEFAULTS)
    c.update(config.get("global", {}))
    d = dict(config.get("downstream", {}).get(ds, {}))
    d.pop("enabled", None)
    c.update(d)
    m = config["models"][mname]
    c.update({k: v for k, v in m.items() if k not in RESERVED_MODEL_KEYS})
    for sub in (mode, ds):
        c.update(m.get(sub, {}))
    return c


def validate_config(config):
    if "models" not in config or not isinstance(config["models"], dict):
        raise ValueError("config must contain a 'models' object")
    for name, m in config["models"].items():
        if name not in MODEL_REGISTRY:
            raise ValueError(f"config: unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
        if m.get("mode", "frozen") not in ("frozen", "finetune", "both"):
            raise ValueError(f"config[{name}].mode must be frozen | finetune | both")
        for ds in m.get("downstream", list(DOWNSTREAMS)):
            if ds not in DOWNSTREAMS:
                raise ValueError(f"config[{name}].downstream: unknown '{ds}'. Available: {sorted(DOWNSTREAMS)}")


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def log(*a):
    """print that does not break active tqdm bars."""
    if tqdm is not None:
        tqdm.write(" ".join(str(x) for x in a))
    else:
        print(*a, flush=True)


class _NullBar:
    def update(self, *a, **k): pass
    def set_postfix(self, *a, **k): pass
    def set_description(self, *a, **k): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


def make_bar(total, desc, enabled=True, leave=False, unit="it"):
    if enabled and tqdm is not None:
        return tqdm(total=total, desc=desc, leave=leave, unit=unit, dynamic_ncols=True)
    return _NullBar()



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("--device mps requested but MPS is not available")
    return torch.device(name)


def amp_dtype_from_config(device, cfg=None):
    """Return a supported CUDA autocast dtype; AMP is disabled on non-CUDA devices."""
    cfg = cfg or {}
    if device.type != "cuda" or not cfg.get("amp", True):
        return None
    requested = str(cfg.get("amp_dtype", "bfloat16")).lower()
    if requested in ("bf16", "bfloat16"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        log("[warn] CUDA BF16 is unsupported; disabling autocast to preserve FP32 training stability")
        return None
    if requested in ("fp16", "float16", "half"):
        return torch.float16
    if requested in ("fp32", "float32", "none"):
        return None
    raise ValueError("amp_dtype must be bfloat16, float16, or float32")


def autocast_context(device, cfg=None):
    dtype = amp_dtype_from_config(device, cfg)
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype is not None))


def clean_json(o):
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(float(o)) or math.isinf(float(o))) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    return o


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(clean_json(obj), f, indent=2)


def count_params(m, trainable_only=True):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = self.transform(im.convert("RGB"))
        return img, int(self.labels[i])



class UInt8ImageDataset(Dataset):
    """Returns resized RGB images as uint8 CHW tensors (no normalization)."""
    def __init__(self, paths, size):
        self.paths = paths
        self.transform = T.Compose([T.Resize((size, size)), T.PILToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            image = self.transform(im.convert("RGB"))
        return image


def _image_cache_fingerprint(paths, size):
    """Invalidate the tensor cache if paths, file sizes, mtimes, or resize size change."""
    h = hashlib.sha256()
    h.update(str(int(size)).encode())
    for path in paths:
        st = os.stat(path)
        h.update(os.path.abspath(path).encode(errors="replace"))
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()[:20]


def load_or_build_gpu_image_cache(ctx, size, cfg):
    """
    Build/load a single uint8 NCHW .pt cache for all images, then keep it on CUDA.
    Images are resized once on CPU; scaling and model-specific normalization happen on GPU.
    """
    if ctx.device.type != "cuda":
        return None

    if not hasattr(ctx, "gpu_image_cache"):
        ctx.gpu_image_cache = {}
    if size in ctx.gpu_image_cache:
        return ctx.gpu_image_cache[size]

    cache_dir = getattr(ctx.args, "image_cache_dir", None) or os.path.join(ctx.args.out_dir, "image_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fingerprint = _image_cache_fingerprint(ctx.paths, size)
    cache_path = os.path.join(cache_dir, f"rgb_uint8_{size}_{fingerprint}.pt")

    if not os.path.isfile(cache_path):
        log(f"  building preprocessed image cache: {cache_path}")
        nw = ctx.args.num_workers if ctx.args.num_workers is not None else cfg["num_workers"]
        ds = UInt8ImageDataset(ctx.paths, size)
        loader = make_loader(ds, int(cfg["batch_size"]), False, nw)
        # Use a memory-mapped temporary array to avoid allocating another full dataset in RAM.
        tmp_npy = cache_path + ".building.npy"
        arr = np.lib.format.open_memmap(
            tmp_npy, mode="w+", dtype=np.uint8,
            shape=(len(ctx.paths), 3, int(size), int(size))
        )
        offset = 0
        with make_bar(len(loader), "  cache images", ctx.args.progress, unit="batch") as bar:
            for images in loader:
                n = int(images.shape[0])
                arr[offset:offset+n] = images.numpy()
                offset += n
                bar.update(1)
        arr.flush()
        if offset != len(ctx.paths):
            raise RuntimeError(f"Image cache incomplete: {offset}/{len(ctx.paths)} images")
        tensor = torch.from_numpy(arr)
        torch.save({"images": tensor, "size": int(size), "fingerprint": fingerprint}, cache_path)
        del tensor, arr
        try:
            os.remove(tmp_npy)
        except OSError:
            pass
        log(f"  image cache saved ({os.path.getsize(cache_path) / 1024**3:.2f} GiB)")

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") != fingerprint or int(payload.get("size", -1)) != int(size):
        raise RuntimeError(f"Image cache metadata mismatch: {cache_path}")
    images_cpu = payload["images"]
    images_gpu = images_cpu.to(ctx.device, non_blocking=True)
    del images_cpu, payload
    ctx.gpu_image_cache[size] = images_gpu
    log(f"  GPU image cache ready: {images_gpu.shape[0]} images, "
        f"{images_gpu.numel() / 1024**3:.2f} GiB uint8 on {ctx.device}")
    return images_gpu


@torch.inference_mode()
def extract_embeddings_from_gpu_cache(ctx, backbone, split, images_gpu, cfg, desc="extract"):
    """Run PTM inference directly from GPU-resident uint8 images."""
    backbone.eval().to(ctx.device)
    size = int(backbone.input_size)
    mean = torch.as_tensor(backbone.mean, device=ctx.device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.as_tensor(backbone.std, device=ctx.device, dtype=torch.float32).view(1, 3, 1, 1)
    indices = torch.as_tensor(ctx.splits[split], device=ctx.device, dtype=torch.long)
    batch_size = int(cfg["batch_size"])
    features_all = []
    with make_bar((len(indices) + batch_size - 1) // batch_size, desc,
                  ctx.args.progress, unit="batch") as bar:
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            x = images_gpu.index_select(0, batch_idx).to(torch.float32).div_(255.0)
            x.sub_(mean).div_(std)
            with autocast_context(ctx.device, cfg):
                features = backbone(x)
            features_all.append(features.float())
            bar.update(1)
    X = torch.cat(features_all, dim=0).cpu().numpy()
    y = ctx.y[ctx.splits[split]]
    return X, y


def make_transform(size, mean, std, augment=False):
    ops = [T.Resize((size, size))]
    if augment:
        ops += [T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(0.2, 0.2, 0.2)]
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def _load_metadata_file(metadata_path, args, required_columns):
    """Load one metadata CSV and resolve image paths relative to that CSV."""
    frame = pd.read_csv(metadata_path)
    for col in required_columns:
        if col not in frame.columns:
            raise ValueError(f"Column '{col}' not in {metadata_path}. Columns: {list(frame.columns)}")
    frame = frame.dropna(subset=required_columns).reset_index(drop=True)
    csv_dir = os.path.dirname(os.path.abspath(metadata_path))

    def resolve(path):
        if args.root_dir:
            return os.path.join(args.root_dir, str(path))
        path = str(path)
        if os.path.isabs(path) or os.path.isfile(path):
            return path
        alt = os.path.join(csv_dir, path)
        return alt if os.path.isfile(alt) else path

    paths = np.asarray([resolve(v) for v in frame[args.filepath].astype(str)])
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image files not found in {metadata_path}, e.g. {missing[:3]} (use --root_dir?)"
        )
    frame = frame.copy()
    frame["_resolved_path"] = paths
    return frame


def load_metadata(args):
    """Load the main pool and, optionally, independent validation/test CSVs."""
    base = _load_metadata_file(args.metadata, args, [args.filepath, args.target])
    if args.split_col and args.split_col not in base.columns:
        raise ValueError(f"--split_col '{args.split_col}' not in metadata columns: {list(base.columns)}")
    has_external = bool(args.val_metadata or args.test_metadata)

    if not has_external:
        df = base
        paths = df["_resolved_path"].to_numpy()
        classes = sorted(df[args.target].astype(str).unique())
        cmap = {c: i for i, c in enumerate(classes)}
        y = df[args.target].astype(str).map(cmap).to_numpy()
        if len(classes) < 2:
            raise ValueError("Need at least 2 classes in the target column")
        return df, paths, y, classes

    if args.split_col:
        raise ValueError("Use either --split_col or separate --val_metadata/--test_metadata, not both")

    # The main metadata is the training pool. Missing held-out partitions are carved
    # from this pool; explicitly supplied CSVs are kept entirely independent.
    base = base.copy()
    if args.val_metadata:
        val = _load_metadata_file(args.val_metadata, args, [args.filepath, args.target])
    else:
        val = None
    if args.test_metadata:
        test = _load_metadata_file(args.test_metadata, args, [args.filepath, args.target])
    else:
        test = None

    base_labels = base[args.target].astype(str)
    classes = sorted(base_labels.unique())
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes in the training metadata")

    rng_idx = np.arange(len(base))
    def holdout(indices, fraction, labels, name):
        if not (0 < fraction < 1):
            raise ValueError(f"The derived {name} split fraction must be in (0,1); got {fraction}")
        try:
            return train_test_split(indices, test_size=fraction, stratify=labels, random_state=args.seed)
        except ValueError:
            log(f"  [warn] stratified {name} split failed (too few samples per class); using random split")
            return train_test_split(indices, test_size=fraction, random_state=args.seed)

    base["_explicit_split"] = "train"
    if val is None and test is None:
        raise AssertionError("external split handling requires at least one external CSV")
    if val is not None and test is None:
        tr_idx, te_idx = holdout(rng_idx, args.test_size, base_labels.to_numpy(), "test")
        base.loc[base.index[te_idx], "_explicit_split"] = "test"
    elif val is None and test is not None:
        tr_idx, va_idx = holdout(rng_idx, args.val_size, base_labels.to_numpy(), "validation")
        base.loc[base.index[va_idx], "_explicit_split"] = "val"

    frames = [base]
    if val is not None:
        val = val.copy()
        val["_explicit_split"] = "val"
        frames.append(val)
    if test is not None:
        test = test.copy()
        test["_explicit_split"] = "test"
        frames.append(test)
    df = pd.concat(frames, ignore_index=True, sort=False)

    cmap = {c: i for i, c in enumerate(classes)}
    labels = df[args.target].astype(str)
    unknown = sorted(set(labels) - set(cmap))
    if unknown:
        raise ValueError(
            f"Validation/test metadata contains labels absent from training metadata: {unknown}"
        )
    y = labels.map(cmap).to_numpy()
    paths = df["_resolved_path"].to_numpy()

    # Guard against accidental train/validation/test leakage when separate files overlap.
    split_values = df["_explicit_split"].to_numpy()
    path_sets = {name: set(paths[split_values == name]) for name in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = path_sets[left] & path_sets[right]
        if overlap:
            raise ValueError(
                f"Image leakage: {len(overlap)} image paths occur in both {left} and {right} sets; "
                f"examples: {list(overlap)[:3]}"
            )
    return df, paths, y, classes


def make_splits(df, y, args):
    n = len(y)
    if "_explicit_split" in df.columns:
        values = df["_explicit_split"].astype(str).to_numpy()
        return {name: np.where(values == name)[0] for name in ("train", "val", "test")}
    if args.split_col:
        norm = {"train": "train", "val": "val", "valid": "val", "validation": "val", "dev": "val", "test": "test"}
        s = df[args.split_col].astype(str).str.lower().map(norm)
        if s.isna().any():
            raise ValueError(f"--split_col values must be train/val/test; got {df[args.split_col].unique()}")
        return {k: np.where(s.to_numpy() == k)[0] for k in ("train", "val", "test")}
    if not (0 < args.val_size < 1 and 0 < args.test_size < 1 and args.val_size + args.test_size < 1):
        raise ValueError("--val_size and --test_size must be in (0,1) and sum to < 1")
    idx = np.arange(n)

    def split(ix, size, strat):
        try:
            return train_test_split(ix, test_size=size, stratify=strat, random_state=args.seed)
        except ValueError:
            log("  [warn] stratified split failed (too few samples per class); using random split")
            return train_test_split(ix, test_size=size, random_state=args.seed)

    trval, te = split(idx, args.test_size, y)
    tr, va = split(trval, args.val_size / (1 - args.test_size), y[trval])
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def _eer_binary(y_true, score):
    fpr, tpr, _ = roc_curve(y_true, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def compute_eer(y, probs):
    """Binary: EER of positive-class score. Multiclass: macro-average of one-vs-rest EERs."""
    C = probs.shape[1]
    if C == 2:
        return _eer_binary(y, probs[:, 1]) if len(np.unique(y)) == 2 else float("nan")
    vals = []
    for c in range(C):
        yt = (y == c).astype(int)
        if 0 < yt.sum() < len(yt):
            vals.append(_eer_binary(yt, probs[:, c]))
    return float(np.mean(vals)) if vals else float("nan")


def compute_metrics(logits, y):
    C = logits.shape[1]
    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    pred = probs.argmax(1)
    labels = list(range(C))
    out = {
        "loss": float(F.cross_entropy(torch.tensor(logits, dtype=torch.float32), torch.tensor(y)).item()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)),
    }
    try:
        out["eer"] = compute_eer(y, probs)
    except Exception:
        out["eer"] = float("nan")
    try:
        if C == 2:
            out["auc"] = float(roc_auc_score(y, probs[:, 1]))
        else:
            out["auc"] = float(roc_auc_score(y, probs, multi_class="ovr", labels=labels))
    except Exception:
        out["auc"] = float("nan")
    return out


# --------------------------------------------------------------------------------------
# Embedding extraction / training
# --------------------------------------------------------------------------------------
def make_loader(ds, bs, shuffle, nw, drop_last=False):
    """Build a loader tuned for accelerator workloads without changing sample order semantics."""
    kwargs = dict(
        dataset=ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
    )
    if nw > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(**kwargs)


@torch.inference_mode()
def extract_embeddings(backbone, loader, device, desc="extract", show=True, cfg=None):
    backbone.eval().to(device)
    # Keep batch outputs on the GPU and transfer once to avoid per-batch synchronization.
    xs, ys = [], []
    with make_bar(len(loader), desc, show, unit="batch") as bar:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with autocast_context(device, cfg):
                features = backbone(x)
            xs.append(features.float())
            ys.append(y)
            bar.update(1)
    return torch.cat(xs, dim=0).cpu().numpy(), torch.cat(ys).numpy()


@torch.inference_mode()
def predict(model, loader, device, cfg=None):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast_context(device, cfg):
            logits = model(x)
        outs.append(logits.float())
        ys.append(y)
    return torch.cat(outs, dim=0).cpu().numpy(), torch.cat(ys).numpy()


def build_optimizer(model, cfg, finetune):
    wd, lr = cfg["weight_decay"], cfg["lr"]
    if finetune:
        blr = cfg["backbone_lr"] if cfg["backbone_lr"] is not None else lr
        groups = [{"params": model.backbone.parameters(), "lr": blr},
                  {"params": model.head.parameters(), "lr": lr}]
    else:
        groups = [{"params": model.parameters(), "lr": lr}]
    opt = cfg["optimizer"].lower()
    if opt == "adam":
        return torch.optim.Adam(groups, weight_decay=wd)
    if opt == "adamw":
        return torch.optim.AdamW(groups, weight_decay=wd)
    if opt == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=wd)
    raise ValueError(f"unknown optimizer '{opt}' (adam | adamw | sgd)")


def build_scheduler(opt, cfg):
    s = cfg["lr_scheduler"].lower()
    if s == "none":
        return None
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=int(cfg["lr_step"]), gamma=cfg["lr_decay"])
    if s == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg["lr_decay"])
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(cfg["epochs"])))
    raise ValueError(f"unknown lr_scheduler '{s}' (none | step | exponential | cosine)")


def train_model(model, train_loader, val_loader, cfg, device, monitor, finetune, show_progress=True):
    model.to(device)
    opt = build_optimizer(model, cfg, finetune)
    sched = build_scheduler(opt, cfg)
    lower = monitor in LOWER_IS_BETTER
    best_score, best_state, best_epoch, bad, history = None, None, 0, 0, []
    use_bar = show_progress and tqdm is not None
    ebar = make_bar(int(cfg["epochs"]), "  epochs", use_bar, leave=True, unit="ep")
    for ep in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        train_correct = torch.zeros((), device=device, dtype=torch.long)
        tn = 0
        bbar = make_bar(len(train_loader), f"  epoch {ep}", use_bar and finetune, unit="batch")
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, cfg):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            batch_n = int(y.shape[0])
            train_loss_sum += loss.detach().float() * batch_n
            train_correct += (logits.detach().argmax(1) == y).sum()
            tn += batch_n
            if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                bbar.set_postfix(loss=f"{(train_loss_sum / tn).item():.4f}")
            bbar.update(1)
        bbar.close()

        train_loss = (train_loss_sum / max(tn, 1)).item()
        train_accuracy = (train_correct.float() / max(tn, 1)).item()
        if sched:
            sched.step()
        v_logits, v_y = predict(model, val_loader, device, cfg)
        vm = compute_metrics(v_logits, v_y)
        row = {"epoch": ep, "train_loss": train_loss, "train_accuracy": train_accuracy,
               **{f"val_{k}": v for k, v in vm.items()}, "lr": opt.param_groups[-1]["lr"]}
        history.append(row)
        score = vm[monitor]
        if math.isnan(score):
            score = float("inf") if lower else float("-inf")
        improved = best_score is None or (score < best_score if lower else score > best_score)
        if improved:
            best_score, best_epoch, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        ebar.update(1)
        ebar.set_postfix(train_loss=f"{row['train_loss']:.3f}", val_loss=f"{vm['loss']:.3f}",
                         val_acc=f"{vm['accuracy']:.3f}", best_ep=best_epoch)
        if not use_bar:
            log(f"    ep {ep:03d} train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.3f} "
                f"val_loss={vm['loss']:.4f} val_acc={vm['accuracy']:.3f} {'*' if improved else ''}")
        if cfg["patience"] and bad >= int(cfg["patience"]):
            log(f"    early stopping at epoch {ep}")
            break
    ebar.close()
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch


def save_confusion(cm, classes, path_prefix, title):
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(path_prefix + ".csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = max(4, 0.6 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path_prefix + ".png", dpi=150)
    plt.close(fig)


def save_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("loss")
    ax[0].legend()
    ax[1].plot(h.epoch, h.train_accuracy, label="train")
    ax[1].plot(h.epoch, h.val_accuracy, label="val")
    ax[1].set_title("accuracy")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_embeddings(out_dir, name, emb, labels, paths, classes):
    d = os.path.join(out_dir, "embeddings", name)
    os.makedirs(d, exist_ok=True)
    for split, (X, y, idx) in emb.items():
        np.savez_compressed(os.path.join(d, f"{split}.npz"), embeddings=X, labels=y,
                            paths=paths[idx], classes=np.array(classes))
    return d


# --------------------------------------------------------------------------------------
# One run (backbone x mode x downstream)
# --------------------------------------------------------------------------------------
class Context:
    """Everything shared by all runs."""

    def __init__(self, args, device, paths, y, classes, splits):
        self.args, self.device, self.paths, self.y, self.classes, self.splits = args, device, paths, y, classes, splits
        self.metrics = args.metrics


def image_loaders(ctx, backbone, cfg, train_aug):
    nw = ctx.args.num_workers if ctx.args.num_workers is not None else cfg["num_workers"]
    ev_tf = make_transform(backbone.input_size, backbone.mean, backbone.std, False)
    tr_tf = make_transform(backbone.input_size, backbone.mean, backbone.std, train_aug)
    bs = int(cfg["batch_size"])
    mk = lambda split, tf: ImageDataset(ctx.paths[ctx.splits[split]], ctx.y[ctx.splits[split]], tf)
    n_tr = len(ctx.splits["train"])
    train = make_loader(mk("train", tr_tf), bs, True, nw, drop_last=(n_tr > 1 and n_tr % bs == 1))
    evals = {s: make_loader(mk(s, ev_tf), bs, False, nw) for s in ("train", "val", "test")}
    return train, evals


def extract_all_splits(ctx, backbone, evals=None):
    out = {}
    cfg = ctx.current_cfg
    if ctx.device.type == "cuda":
        images_gpu = load_or_build_gpu_image_cache(ctx, int(backbone.input_size), cfg)
        for split in ("train", "val", "test"):
            X, y = extract_embeddings_from_gpu_cache(
                ctx, backbone, split, images_gpu, cfg, desc=f"  extract {split}"
            )
            out[split] = (X, y, ctx.splits[split])
    else:
        for split, loader in evals.items():
            X, y = extract_embeddings(
                backbone, loader, ctx.device, desc=f"  extract {split}",
                show=ctx.args.progress, cfg=cfg
            )
            out[split] = (X, y, ctx.splits[split])
    return out


def finalize_run(ctx, run_dir, run_meta, cfg, model, test_loader, val_loader, history, best_epoch,
                 extra_ckpt, t0):
    args = ctx.args
    os.makedirs(run_dir, exist_ok=True)
    res = {"meta": run_meta, "config": cfg, "best_epoch": best_epoch}
    for split, ld in (("val", val_loader), ("test", test_loader)):
        logits, yy = predict(model, ld, ctx.device, cfg)
        m = compute_metrics(logits, yy)
        res[split] = m
        if split == "test":
            pred = logits.argmax(1)
            cm = confusion_matrix(yy, pred, labels=range(len(ctx.classes)))
            res["classification_report"] = classification_report(
                yy, pred, labels=list(range(len(ctx.classes))), target_names=ctx.classes,
                output_dict=True, zero_division=0)
            if args.save_confusion_matrix:
                save_confusion(cm, ctx.classes, os.path.join(run_dir, "confusion_matrix_test"),
                               f"{run_meta['run']} (test)")
            if args.save_predictions:
                probs = F.softmax(torch.tensor(logits), 1).numpy()
                dfp = pd.DataFrame({"path": ctx.paths[ctx.splits["test"]],
                                    "true": [ctx.classes[i] for i in yy],
                                    "pred": [ctx.classes[i] for i in pred]})
                for i, c in enumerate(ctx.classes):
                    dfp[f"prob_{c}"] = probs[:, i]
                dfp.to_csv(os.path.join(run_dir, "predictions_test.csv"), index=False)
    res["history"] = history
    res["train_time_sec"] = time.time() - t0
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    if args.save_plots:
        save_curves(history, os.path.join(run_dir, "training_curves.png"))
    if args.save_checkpoints:
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": cfg, "meta": run_meta, "classes": ctx.classes, "best_epoch": best_epoch,
                    **extra_ckpt}, os.path.join(run_dir, "checkpoints", "best.pt"))
    save_json(res, os.path.join(run_dir, "results.json"))
    return res


def run_one(ctx, mname, mode, ds, cfg, emb_cache):
    args = ctx.args
    ctx.current_cfg = cfg
    layer = cfg["embedding_layer"]
    layer_tag = str(layer).replace(".", "-")
    run = f"{mname}__{mode}__{ds}" + ("" if layer == "final" else f"__{layer_tag}")
    run_dir = os.path.join(args.out_dir, "runs", run)
    if args.skip_existing and os.path.isfile(os.path.join(run_dir, "results.json")):
        log(f"  [skip] {run} (results.json exists)")
        with open(os.path.join(run_dir, "results.json")) as f:
            return json.load(f)
    log(f"\n=== {run} ===")
    set_seed(args.seed)
    t0 = time.time()
    C = len(ctx.classes)
    head_kw = {k: cfg[k] for k in ("dropout", "activation", "norm")}
    extra = {}

    if mode == "frozen":
        key = (mname, layer, cfg["pretrained"])
        if key not in emb_cache:
            bb = build_backbone(mname, cfg["pretrained"], layer)
            log(f"  extracting embeddings (layer='{layer}', dim={bb.embedding_dim})")
            evals = None
            if ctx.device.type != "cuda":
                _, evals = image_loaders(ctx, bb, cfg, False)
            emb_cache.clear()
            emb_cache[key] = extract_all_splits(ctx, bb, evals)
            del bb
            if args.save_embeddings:
                d = save_embeddings(args.out_dir, f"{mname}__frozen__{layer_tag}", emb_cache[key],
                                    ctx.y, ctx.paths, ctx.classes)
                log(f"  embeddings saved -> {d}")
        emb = emb_cache[key]
        Xtr = emb["train"][0]
        if cfg["normalize_embeddings"] == "standardize":
            mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        else:
            mu, sd = np.zeros((1, Xtr.shape[1]), np.float32), np.ones((1, Xtr.shape[1]), np.float32)
        extra = {"emb_mean": torch.tensor(mu), "emb_std": torch.tensor(sd)}
        tds = {s: TensorDataset(torch.tensor((emb[s][0] - mu) / sd, dtype=torch.float32),
                                torch.tensor(emb[s][1])) for s in emb}
        bs, n_tr = int(cfg["batch_size"]), len(tds["train"])
        train_loader = make_loader(tds["train"], bs, True, 0, drop_last=(n_tr > 1 and n_tr % bs == 1))
        val_loader, test_loader = make_loader(tds["val"], bs, False, 0), make_loader(tds["test"], bs, False, 0)
        model = build_head(ds, Xtr.shape[1], C, **head_kw)
        dim = Xtr.shape[1]
        n_train_params = count_params(model)
    else:  # finetune
        bb = build_backbone(mname, cfg["pretrained"], layer)
        dim = bb.embedding_dim
        for p in bb.parameters():
            p.requires_grad = True
        model = BaselinePipeline(bb, build_head(ds, dim, C, **head_kw))
        train_loader, evals = image_loaders(ctx, bb, cfg, bool(cfg["augment"]))
        val_loader, test_loader = evals["val"], evals["test"]
        n_train_params = count_params(model)

    log(f"  mode={mode} head={ds} emb_dim={dim} trainable_params={n_train_params:,} device={ctx.device}")
    history, best_epoch = train_model(model, train_loader, val_loader, cfg, ctx.device, args.monitor,
                                      finetune=(mode == "finetune"), show_progress=args.progress)
    meta = {"run": run, "model": mname, "mode": mode, "downstream": ds, "embedding_layer": layer,
            "embedding_dim": dim, "trainable_params": n_train_params, "pretrained": cfg["pretrained"]}
    res = finalize_run(ctx, run_dir, meta, cfg, model, test_loader, val_loader, history, best_epoch, extra, t0)

    if mode == "finetune" and args.save_embeddings:   # embeddings of the fine-tuned backbone
        emb = extract_all_splits(ctx, model.backbone, evals)
        d = save_embeddings(args.out_dir, run, emb, ctx.y, ctx.paths, ctx.classes)
        log(f"  fine-tuned embeddings saved -> {d}")
    log(f"  TEST: " + " ".join(f"{m}={res['test'][m]:.4f}" for m in ctx.metrics))
    del model
    return res


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", help="metadata CSV file")
    p.add_argument("--config", help="config JSON file (models on/off + hyper-parameters)")
    p.add_argument("--filepath", help="metadata column holding image paths")
    p.add_argument("--target", help="metadata column holding the (categorical) target")
    p.add_argument("--root_dir", default=None, help="optional prefix prepended to image paths")
    p.add_argument("--split_col", default=None, help="optional column with train/val/test; else random stratified split")
    p.add_argument("--val_metadata", default=None,
                   help="optional independent validation CSV; if omitted, validation is split from --metadata")
    p.add_argument("--test_metadata", default=None,
                   help="optional independent test CSV; if omitted, test is split from --metadata")
    p.add_argument("--val_size", type=float, default=0.15,
                   help="validation fraction when validation CSV is not supplied (default: 0.15)")
    p.add_argument("--test_size", type=float, default=0.15,
                   help="test fraction when test CSV is not supplied (default: 0.15)")
    p.add_argument("--out_dir", default="baseline_results")
    p.add_argument("--metrics", nargs="+", default=["accuracy", "eer", "loss"], choices=ALL_METRICS,
                   help="metrics reported in the summary")
    p.add_argument("--monitor", default="loss", choices=ALL_METRICS,
                   help="validation metric used for checkpoint selection / early stopping")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=None, help="only run these models from the config")
    p.add_argument("--epochs", type=int, default=None, help="override epochs of every run")
    p.add_argument("--num_workers", type=int, default=None, help="override dataloader workers")
    p.add_argument("--batch_size", type=int, default=None, help="override batch size for every run")
    p.add_argument("--no_pretrained", action="store_true", help="random init instead of pre-trained weights (offline smoke tests)")
    p.add_argument("--save_checkpoints", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save_embeddings", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_confusion_matrix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_predictions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_plots", action=argparse.BooleanOptionalAction, default=True, help="training curves")
    p.add_argument("--no_progress", dest="progress", action="store_false", help="disable progress bars")
    p.add_argument("--cache_dir", default=None,
                   help="folder with pre-trained weights (same value as used by download_models.py)")
    p.add_argument("--image_cache_dir", default=None,
                   help="persistent preprocessed uint8 image .pt cache directory (default: OUT_DIR/image_cache)")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--fail_fast", action="store_true", help="stop at the first failing run")
    p.add_argument("--list_models", action="store_true")
    p.add_argument("--list_layers", metavar="MODEL", help="print layer names usable as embedding_layer")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_models:
        print("backbones:", ", ".join(sorted(MODEL_REGISTRY)))
        print("downstream:", ", ".join(sorted(DOWNSTREAMS)))
        return 0
    if args.list_layers:
        bb = build_backbone(args.list_layers, pretrained=False)
        print("\n".join(bb.available_layers(max_depth=3)))
        return 0
    for req in ("metadata", "config", "filepath", "target"):
        if not getattr(args, req):
            sys.exit(f"error: --{req} is required")
    if args.monitor not in args.metrics:
        args.metrics = list(args.metrics) + [args.monitor]

    set_cache_dir(args.cache_dir)
    with open(args.config) as f:
        config = json.load(f)
    validate_config(config)
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    df, paths, y, classes = load_metadata(args)
    splits = make_splits(df, y, args)
    for s, ix in splits.items():
        if len(ix) == 0:
            sys.exit(f"error: split '{s}' is empty")
    log(f"device={device} samples={len(y)} classes={classes} "
        f"split sizes={ {k: len(v) for k, v in splits.items()} }")
    sp = np.empty(len(y), dtype=object)
    for k, ix in splits.items():
        sp[ix] = k
    pd.DataFrame({"path": paths, "target": df[args.target].astype(str), "split": sp}).to_csv(
        os.path.join(args.out_dir, "split.csv"), index=False)
    save_json({"args": vars(args), "config": config}, os.path.join(args.out_dir, "config_used.json"))

    ctx = Context(args, device, paths, y, classes, splits)
    plan = []                                   # every (model, mode, head) that will run
    for mname, mcfg in config["models"].items():
        if not mcfg.get("enabled", True) or (args.models and mname not in args.models):
            continue
        modes = ["frozen", "finetune"] if mcfg.get("mode", "frozen") == "both" else [mcfg.get("mode", "frozen")]
        for mode in modes:
            for ds in mcfg.get("downstream", list(DOWNSTREAMS)):
                if config.get("downstream", {}).get(ds, {}).get("enabled", True):
                    plan.append((mname, mode, ds))
    log(f"{len(plan)} runs planned")
    rows, failures, emb_cache, prev_model = [], 0, {}, None
    overall = make_bar(len(plan), "overall", args.progress, leave=True, unit="run")
    for mname, mode, ds in plan:
        if mname != prev_model:                 # embeddings are only shared within one model
            emb_cache, prev_model = {}, mname
            if device.type == "cuda":
                torch.cuda.empty_cache()
        overall.set_description(f"overall [{mname}/{mode}/{ds}]")
        cfg = resolve_cfg(config, mname, mode, ds)
        if args.no_pretrained:
            cfg["pretrained"] = False
        if args.epochs is not None:
            cfg["epochs"] = args.epochs
        if args.num_workers is not None:
            cfg["num_workers"] = args.num_workers
        if args.batch_size is not None:
            cfg["batch_size"] = args.batch_size
        try:
            res = run_one(ctx, mname, mode, ds, cfg, emb_cache)
            row = {**{k: res["meta"][k] for k in ("run", "model", "mode", "downstream",
                                                  "embedding_layer", "embedding_dim", "trainable_params")},
                   "best_epoch": res["best_epoch"], "train_time_sec": round(res["train_time_sec"], 1),
                   "status": "ok"}
            for m in args.metrics:
                row[f"val_{m}"] = res["val"][m]
                row[f"test_{m}"] = res["test"][m]
        except Exception as e:  # keep going so one bad model does not kill the sweep
            failures += 1
            log(traceback.format_exc())
            row = {"run": f"{mname}__{mode}__{ds}", "model": mname, "mode": mode, "downstream": ds,
                   "status": f"FAILED: {type(e).__name__}: {e}"}
            if args.fail_fast:
                rows.append(row)
                pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
                overall.close()
                return 1
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
        save_json(rows, os.path.join(args.out_dir, "summary.json"))
        overall.update(1)
    overall.close()

    log("\n================ SUMMARY ================")
    if rows:
        cols = ["run", "status"] + [f"test_{m}" for m in args.metrics if f"test_{m}" in pd.DataFrame(rows)]
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(pd.DataFrame(rows)[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"\nresults in: {os.path.abspath(args.out_dir)}   failed runs: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
