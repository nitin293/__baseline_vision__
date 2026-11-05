import torch.nn as nn
from .registry import register
from .torchvision_base import TorchvisionBackbone


class _ResNet(TorchvisionBackbone):
    def strip_head(self, net):
        net.fc = nn.Identity()
        return net


@register("resnet18")
class ResNet18(_ResNet):
    tv_name = "resnet18"


@register("resnet50")
class ResNet50(_ResNet):
    tv_name = "resnet50"
