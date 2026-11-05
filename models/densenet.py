import torch.nn as nn
from .registry import register
from .torchvision_base import TorchvisionBackbone


@register("densenet121")
class DenseNet121(TorchvisionBackbone):
    tv_name = "densenet121"

    def strip_head(self, net):
        net.classifier = nn.Identity()
        return net
