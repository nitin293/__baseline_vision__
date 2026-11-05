"""Model zoo. Each PTM lives in its own script and registers itself in MODEL_REGISTRY."""
from .registry import MODEL_REGISTRY
from . import resnet, vgg, densenet, efficientnet, mobilenet, vit, dinov2, clip  # noqa: F401
from .fcn import FCNHead
from .cnn_maxpool import CNNMaxPoolHead
from .pipeline import BaselinePipeline

DOWNSTREAMS = {"fcn": FCNHead, "cnn_maxpool": CNNMaxPoolHead}


def build_backbone(name, pretrained=True, embedding_layer="final"):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](pretrained=pretrained, embedding_layer=embedding_layer)


def build_head(kind, in_dim, n_classes, **kw):
    if kind not in DOWNSTREAMS:
        raise KeyError(f"Unknown downstream '{kind}'. Available: {sorted(DOWNSTREAMS)}")
    return DOWNSTREAMS[kind](in_dim, n_classes, **kw)
