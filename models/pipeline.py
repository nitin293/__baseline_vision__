import torch.nn as nn


class BaselinePipeline(nn.Module):
    """backbone (PTM) -> downstream head. Used for end-to-end fine-tuning."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return self.head(self.backbone(x))
