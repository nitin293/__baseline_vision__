import torch.nn as nn
from .registry import register
from .torchvision_base import TorchvisionBackbone


@register("vgg16")
class VGG16(TorchvisionBackbone):
    tv_name = "vgg16"

    def strip_head(self, net):
        net.classifier[-1] = nn.Identity()   # keep the 4096-d fc7 embedding
        return net
