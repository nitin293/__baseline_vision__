#!/usr/bin/env python
"""Run a checkpoint saved by evaluate_baseline.py (--save_checkpoints) on images and, if a target
column is given, compute performance against the ground-truth labels.

Place this file next to evaluate_baseline.py and the `models/` directory.

Examples
--------
# predictions only
python infer_baseline.py --checkpoint results/runs/resnet18__finetune__fcn \
    --metadata new.csv --filepath image_path --out_dir infer_out

# predictions + performance (target column holds the true labels)
python infer_baseline.py --checkpoint results/runs/resnet18__finetune__fcn/checkpoints/best.pt \
    --metadata meta.csv --filepath image_path --target label \
    --metrics accuracy eer loss f1_macro --out_dir infer_out

# evaluate only the test rows of a metadata file that has a split column
python infer_baseline.py --checkpoint results/runs/vgg16__frozen__fcn --metadata meta.csv \
    --filepath image_path --target label --split_col fold --split test --cache_dir ./pretrained

Note: "frozen" checkpoints store only the downstream head, so the pre-trained backbone is rebuilt
from its pre-trained weights (same --cache_dir as used for download_models.py / evaluation).
Fine-tuned checkpoints contain the full model and need no download.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

from evaluate_baseline import (ALL_METRICS, ImageDataset, autocast_context, compute_metrics,
                               log, make_bar, make_loader, make_transform, resolve_device,
                               save_confusion, save_json)
from models import BaselinePipeline, build_backbone, build_head
from models.weights import set_cache_dir

SPLIT_ALIASES = {"val": {"val", "valid", "validation", "dev"}, "train": {"train"}, "test": {"test"}}


class FrozenPipeline(nn.Module):
    """PTM backbone -> z-score (train statistics) -> head, mirroring frozen-mode training."""

    def __init__(self, backbone, head, mean, std):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return self.head((self.backbone(x) - self.mean) / self.std)


def find_checkpoint(path):
    if os.path.isdir(path):
        for cand in (os.path.join(path, "checkpoints", "best.pt"), os.path.join(path, "best.pt")):
            if os.path.isfile(cand):
                return cand
        sys.exit(f"error: no checkpoints/best.pt found in {path} (was the run made with --save_checkpoints?)")
    if not os.path.isfile(path):
        sys.exit(f"error: checkpoint not found: {path}")
    return path


def load_model(ckpt_path, device):
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:  # checkpoints written by evaluate_baseline.py are plain dicts/tensors; fall back anyway
        log("[warn] safe loading failed, retrying with weights_only=False (only do this for files you trust)")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta, cfg, classes = ck["meta"], ck["config"], ck["classes"]
    mode = meta["mode"]
    head_kw = {k: cfg[k] for k in ("dropout", "activation", "norm")}
    head = build_head(meta["downstream"], meta["embedding_dim"], len(classes), **head_kw)
    if mode == "finetune":
        backbone = build_backbone(meta["model"], pretrained=False, embedding_layer=meta["embedding_layer"])
        model = BaselinePipeline(backbone, head)
    else:
        if not meta.get("pretrained", True):
            log("[warn] this frozen run used a randomly initialised backbone (--no_pretrained); "
                "it cannot be reproduced exactly, predictions will be unreliable.")
        backbone = build_backbone(meta["model"], pretrained=meta.get("pretrained", True),
                                  embedding_layer=meta["embedding_layer"])
        model = FrozenPipeline(backbone, head, ck["emb_mean"].float(), ck["emb_std"].float())
        state_head = ck["state_dict"]
        head.load_state_dict(state_head)
        return model.to(device).eval(), backbone, meta, cfg, classes
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), backbone, meta, cfg, classes


def load_metadata(a):
    df = pd.read_csv(a.metadata)
    need = [a.filepath] + ([a.target] if a.target else []) + ([a.split_col] if a.split_col else [])
    for c in need:
        if c not in df.columns:
            sys.exit(f"error: column '{c}' not in metadata. Columns: {list(df.columns)}")
    if a.split and not a.split_col:
        sys.exit("error: --split requires --split_col")
    if a.split_col and a.split:
        vals = df[a.split_col].astype(str).str.lower()
        df = df[vals.isin(SPLIT_ALIASES[a.split])]
    df = df.dropna(subset=[a.filepath] + ([a.target] if a.target else [])).reset_index(drop=True)
    if df.empty:
        sys.exit("error: no rows left to process after filtering")
    csv_dir = os.path.dirname(os.path.abspath(a.metadata))

    def resolve(p):
        if a.root_dir:
            return os.path.join(a.root_dir, p)
        if os.path.isabs(p) or os.path.isfile(p):
            return p
        alt = os.path.join(csv_dir, p)
        return alt if os.path.isfile(alt) else p

    df["_path"] = [resolve(p) for p in df[a.filepath].astype(str)]
    missing = [p for p in df["_path"] if not os.path.isfile(p)]
    if missing:
        sys.exit(f"error: {len(missing)} image files not found, e.g. {missing[:3]} (use --root_dir?)")
    return df


@torch.no_grad()
def run_inference(model, backbone, loader, device, show, keep_embeddings, cfg=None):
    logits, embs = [], []
    with make_bar(len(loader), "inference", show, unit="batch") as bar:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with autocast_context(device, cfg):
                if keep_embeddings:
                    e = backbone(x)
                    embs.append(e.float())
                    out = model.head((e.float() - model.mean) / model.std) if isinstance(model, FrozenPipeline) \
                        else model.head(e.float())
                else:
                    out = model(x)
            logits.append(out.float())
            bar.update(1)
    return torch.cat(logits, dim=0).cpu().numpy(), (torch.cat(embs, dim=0).cpu().numpy() if keep_embeddings else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="best.pt file, or a run directory containing checkpoints/best.pt")
    p.add_argument("--metadata", required=True, help="CSV with the images to run on")
    p.add_argument("--filepath", required=True, help="metadata column holding image paths")
    p.add_argument("--target", default=None, help="metadata column with true labels (enables performance metrics)")
    p.add_argument("--root_dir", default=None, help="optional prefix prepended to image paths")
    p.add_argument("--split_col", default=None, help="optional split column in the metadata")
    p.add_argument("--split", default=None, choices=["train", "val", "test"], help="only use rows of this split")
    p.add_argument("--out_dir", default="inference_results")
    p.add_argument("--metrics", nargs="+", default=["accuracy", "eer", "loss"], choices=ALL_METRICS)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--cache_dir", default=None, help="pre-trained weights folder (frozen checkpoints only)")
    p.add_argument("--drop_unknown", action="store_true",
                   help="drop rows whose label is not one of the model's classes (default: error)")
    p.add_argument("--save_embeddings", action="store_true", help="also save backbone embeddings (embeddings.npz)")
    p.add_argument("--save_confusion_matrix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no_progress", dest="progress", action="store_false")
    a = p.parse_args()

    set_cache_dir(a.cache_dir)
    device = resolve_device(a.device)
    ckpt = find_checkpoint(a.checkpoint)
    model, backbone, meta, cfg, classes = load_model(ckpt, device)
    log(f"loaded {meta['run']} (mode={meta['mode']}, head={meta['downstream']}, "
        f"classes={classes}) on {device}")

    df = load_metadata(a)
    y = None
    if a.target:
        cmap = {c: i for i, c in enumerate(classes)}
        lab = df[a.target].astype(str)
        unknown = sorted(set(lab) - set(cmap))
        if unknown:
            if not a.drop_unknown:
                sys.exit(f"error: labels {unknown} in '{a.target}' are not model classes {classes} "
                         "(use --drop_unknown to ignore those rows)")
            log(f"[warn] dropping {int(lab.isin(unknown).sum())} rows with unknown labels {unknown}")
            df = df[~lab.isin(unknown)].reset_index(drop=True)
            lab = df[a.target].astype(str)
        y = lab.map(cmap).to_numpy()

    tf = make_transform(backbone.input_size, backbone.mean, backbone.std, augment=False)
    ds = ImageDataset(list(df["_path"]), y if y is not None else np.zeros(len(df), dtype=int), tf)
    loader = make_loader(ds, a.batch_size, False, a.num_workers)
    logits, emb = run_inference(model, backbone, loader, device, a.progress, a.save_embeddings, cfg)

    os.makedirs(a.out_dir, exist_ok=True)
    probs = F.softmax(torch.tensor(logits), 1).numpy()
    pred = probs.argmax(1)
    out = pd.DataFrame({"path": df[a.filepath], "pred": [classes[i] for i in pred],
                        "confidence": probs.max(1)})
    for i, c in enumerate(classes):
        out[f"prob_{c}"] = probs[:, i]
    if y is not None:
        out.insert(1, "true", df[a.target].astype(str))
        out["correct"] = (pred == y)
    out.to_csv(os.path.join(a.out_dir, "predictions.csv"), index=False)
    if emb is not None:
        np.savez_compressed(os.path.join(a.out_dir, "embeddings.npz"), embeddings=emb,
                            labels=(y if y is not None else -np.ones(len(df), dtype=int)),
                            paths=df["_path"].to_numpy(), classes=np.array(classes))

    result = {"checkpoint": ckpt, "run": meta["run"], "n_samples": int(len(df)), "classes": classes}
    if y is not None:
        m = compute_metrics(logits, y)
        result["metrics"] = m
        cm = confusion_matrix(y, pred, labels=range(len(classes)))
        result["classification_report"] = classification_report(
            y, pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
        if a.save_confusion_matrix:
            save_confusion(cm, classes, os.path.join(a.out_dir, "confusion_matrix"), f"{meta['run']} (inference)")
        log("\n=========== PERFORMANCE ===========")
        log(f"samples: {len(df)}   (true labels from '{a.target}')")
        for k in a.metrics:
            log(f"  {k:18s} {m[k]:.4f}")
        log("\nconfusion matrix (rows=true, cols=pred):")
        log(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    else:
        log(f"\n{len(df)} predictions written (no --target given, so no performance metrics).")
    save_json(result, os.path.join(a.out_dir, "metrics.json"))
    log(f"\nresults in: {os.path.abspath(a.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
