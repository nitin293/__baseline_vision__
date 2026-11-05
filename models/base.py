"""Common backbone interface shared by every pre-trained model (PTM) script."""
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _StopForward(Exception):
    """Raised from a forward hook to stop the forward pass once the requested layer has run."""

    def __init__(self, out):
        super().__init__("stop")
        self.out = out


def pool_to_vector(out):
    """Turn whatever a layer returns into a (B, D) embedding."""
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    if out.dim() == 4:      # (B, C, H, W) feature map -> global average pool
        return out.mean(dim=(2, 3))
    if out.dim() == 3:      # (B, T, C) token sequence -> mean over tokens
        return out.mean(dim=1)
    return out.flatten(1)


class BaseBackbone(nn.Module):
    """Wraps a network `self.net` whose forward returns the final (B, D) embedding.

    embedding_layer:
        "final"          -> output of the network with its classification head removed
        "<module.path>"  -> output of any named sub-module (see `--list_layers`), pooled to a vector.
    """
    name = "base"
    input_size = 224
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    def __init__(self, pretrained=True, embedding_layer="final", **kwargs):
        super().__init__()
        self.embedding_layer = embedding_layer or "final"
        self.net = self.build(pretrained, **kwargs)
        self._intermediate = self.embedding_layer != "final"
        if self._intermediate:
            mods = dict(self.net.named_modules())
            if self.embedding_layer not in mods:
                raise ValueError(
                    f"[{self.name}] unknown embedding_layer '{self.embedding_layer}'. "
                    f"Examples: {self.available_layers()[:12]} ... (use --list_layers {self.name})")

            def _hook(module, inp, out):
                raise _StopForward(out)

            mods[self.embedding_layer].register_forward_hook(_hook)
        self.embedding_dim = self._probe_dim()

    def build(self, pretrained, **kwargs):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def available_layers(self, max_depth=2):
        return [n for n, _ in self.net.named_modules() if n and n.count(".") < max_depth]

    def forward(self, x):
        if self._intermediate:
            try:
                self.net(x)
            except _StopForward as e:
                return pool_to_vector(e.out)
            raise RuntimeError(f"layer '{self.embedding_layer}' was never executed")
        return pool_to_vector(self.net(x))

    @torch.no_grad()
    def _probe_dim(self):
        was_training = self.training
        self.eval()
        d = self.forward(torch.zeros(2, 3, self.input_size, self.input_size)).shape[1]
        self.train(was_training)
        return int(d)
