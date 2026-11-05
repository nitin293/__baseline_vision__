import torch.nn as nn
from .registry import register
from .torchvision_base import TorchvisionBackbone


@register("mobilenet_v3_small")
class MobileNetV3Small(TorchvisionBackbone):
    tv_name = "mobilenet_v3_small"

    def strip_head(self, net):
        net.classifier = nn.Identity()
        return net
