import torch.nn as nn
from .registry import register
from .torchvision_base import TorchvisionBackbone


@register("vit_b_16")
class ViTB16(TorchvisionBackbone):
    tv_name = "vit_b_16"

    def strip_head(self, net):
        net.heads = nn.Identity()   # CLS-token embedding (768-d)
        return net
