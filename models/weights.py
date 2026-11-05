"""Where pre-trained weights are cached (shared by evaluate_baseline.py and download_models.py)."""
import os

import torch


def set_cache_dir(path):
    """Use one folder for torchvision (<path>/torch) and HuggingFace (<path>/huggingface) weights.

    Must be called before the first model is built. `None` keeps the default caches
    (~/.cache/torch, ~/.cache/huggingface).
    """
    if not path:
        return
    path = os.path.abspath(path)
    os.environ["TORCH_HOME"] = os.path.join(path, "torch")
    os.environ["HF_HOME"] = os.path.join(path, "huggingface")
    torch.hub.set_dir(os.path.join(path, "torch", "hub"))


def cache_locations():
    hf = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return {"torchvision": os.path.join(torch.hub.get_dir(), "checkpoints"), "huggingface": hf}
