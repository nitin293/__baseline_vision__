"""CLIP ViT-B/32 image encoder via HuggingFace `transformers` (optional dependency)."""
import torch.nn as nn
from .base import BaseBackbone
from .registry import register


class _Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).pooler_output


@register("clip_vit_b32")
class CLIPViTB32(BaseBackbone):
    hf_id = "openai/clip-vit-base-patch32"
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    def build(self, pretrained, **kwargs):
        from transformers import CLIPVisionConfig, CLIPVisionModel
        m = CLIPVisionModel.from_pretrained(self.hf_id) if pretrained else CLIPVisionModel(CLIPVisionConfig())
        return _Wrap(m)
