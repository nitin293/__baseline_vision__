"""DINOv2-small via HuggingFace `transformers` (optional dependency)."""
import torch.nn as nn
from .base import BaseBackbone
from .registry import register


class _Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).pooler_output


@register("dinov2_small")
class DINOv2Small(BaseBackbone):
    hf_id = "facebook/dinov2-small"

    def build(self, pretrained, **kwargs):
        from transformers import Dinov2Config, Dinov2Model
        if pretrained:
            m = Dinov2Model.from_pretrained(self.hf_id)
        else:  # same architecture, random init, no download
            m = Dinov2Model(Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6))
        return _Wrap(m)
