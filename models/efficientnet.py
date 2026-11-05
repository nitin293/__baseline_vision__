import torch.nn as nn
from .registry import register
from .torchvision_base import TorchvisionBackbone


@register("efficientnet_b0")
class EfficientNetB0(TorchvisionBackbone):
    tv_name = "efficientnet_b0"

    def strip_head(self, net):
        net.classifier = nn.Identity()
        return net
