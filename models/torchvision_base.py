import torch.nn as nn
import torchvision.models as tvm
from .base import BaseBackbone


class TorchvisionBackbone(BaseBackbone):
    tv_name = None

    def build(self, pretrained, **kwargs):
        net = tvm.get_model(self.tv_name, weights="DEFAULT" if pretrained else None)
        return self.strip_head(net)

    def strip_head(self, net):  # pragma: no cover
        raise NotImplementedError


IDENTITY = nn.Identity
