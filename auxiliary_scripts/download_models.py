#!/usr/bin/env python
"""Download (and verify) the pre-trained weights needed by evaluate_baseline.py.

Run this once while online; afterwards evaluation works from the local cache.

Examples
--------
python download_models.py --config config.json                     # every enabled model in the config
python download_models.py --models resnet18 vit_b_16               # specific models
python download_models.py --all --cache_dir ./pretrained           # everything, into a custom folder
python evaluate_baseline.py ... --cache_dir ./pretrained           # use that same folder later
"""
import argparse
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root (holds models/, config.json)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from models import MODEL_REGISTRY, build_backbone
from models.weights import cache_locations, set_cache_dir

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

SOURCE = {"dinov2_small": "HuggingFace facebook/dinov2-small", "clip_vit_b32": "HuggingFace openai/clip-vit-base-patch32"}


def say(msg):
    tqdm.write(msg) if tqdm else print(msg, flush=True)


def pick_models(a):
    if a.models:
        names = a.models
    elif a.all or not os.path.isfile(a.config):
        if not a.all:
            say(f"[info] config '{a.config}' not found -> downloading all models")
        names = sorted(MODEL_REGISTRY)
    else:
        with open(a.config) as f:
            cfg = json.load(f)
        names = [n for n, m in cfg.get("models", {}).items() if m.get("enabled", True)]
    unknown = [n for n in names if n not in MODEL_REGISTRY]
    if unknown:
        sys.exit(f"error: unknown model(s) {unknown}. Available: {sorted(MODEL_REGISTRY)}")
    return names


def fetch(name, retries):
    """Build the model with pre-trained weights (this triggers the download) and verify it runs."""
    last = None
    for attempt in range(1, retries + 2):
        try:
            bb = build_backbone(name, pretrained=True, embedding_layer="final")  # downloads if not cached
            bb.eval()
            with torch.no_grad():
                out = bb(torch.zeros(1, 3, bb.input_size, bb.input_size))
            n = sum(p.numel() for p in bb.parameters())
            return {"params_M": n / 1e6, "emb_dim": out.shape[1]}
        except ImportError as e:
            raise RuntimeError(f"missing dependency ({e}). Try: pip install transformers") from e
        except Exception as e:  # network errors, corrupt files, ...
            last = e
            if attempt <= retries:
                say(f"  [retry {attempt}/{retries}] {name}: {type(e).__name__}: {str(e)[:150]}")
                time.sleep(2 * attempt)
    raise last


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "config.json"), help="download the enabled models of this config")
    p.add_argument("--models", nargs="+", default=None, help="explicit model names (overrides --config)")
    p.add_argument("--all", action="store_true", help="download every registered model")
    p.add_argument("--cache_dir", default=None, help="store weights here (default: ~/.cache/torch and ~/.cache/huggingface)")
    p.add_argument("--retries", type=int, default=2, help="retries per model on failure")
    p.add_argument("--list", action="store_true", help="only list the models and exit")
    a = p.parse_args()

    set_cache_dir(a.cache_dir)
    names = pick_models(a)
    if a.list:
        for n in names:
            print(f"{n:22s} {SOURCE.get(n, 'torchvision (download.pytorch.org)')}")
        return 0
    loc = cache_locations()
    say(f"cache: torchvision -> {loc['torchvision']}\n       huggingface -> {loc['huggingface']}")

    results = []
    bar = tqdm(names, desc="models", unit="model", dynamic_ncols=True) if tqdm else names
    for n in bar:
        if tqdm:
            bar.set_description(f"models [{n}]")
        t0 = time.time()
        try:
            info = fetch(n, a.retries)
            results.append((n, "ok", f"{info['params_M']:.1f}M params, emb_dim={info['emb_dim']}", time.time() - t0))
        except Exception as e:
            say(traceback.format_exc(limit=2))
            results.append((n, "FAILED", f"{type(e).__name__}: {str(e)[:120]}", time.time() - t0))

    print("\n=========== download summary ===========")
    for n, st, info, dt in results:
        print(f"{n:22s} {st:7s} {dt:6.1f}s  {info}")
    failed = [r[0] for r in results if r[1] != "ok"]
    if failed:
        print(f"\n{len(failed)} failed: {failed}\nCheck your internet connection / proxy and re-run (finished models are cached).")
        return 1
    hint = f" --cache_dir {a.cache_dir}" if a.cache_dir else ""
    print(f"\nAll weights available. Evaluate with:  python evaluate_baseline.py ...{hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
