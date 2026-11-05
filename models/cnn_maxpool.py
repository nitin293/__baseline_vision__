"""CNN-maxpool downstream.

The (B, D) embedding is zero-padded and reshaped to a 1 x s x s map (s = ceil(sqrt(D))), then:
Conv2d(1,64,3) -> maxpool -> Conv2d(64,128,3) -> maxpool -> flatten -> Linear(n_classes).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fcn import ACTS


class CNNMaxPoolHead(nn.Module):
    def __init__(self, in_dim, n_classes, dropout=0.2, activation="relu", norm="batch", **_):
        super().__init__()
        self.in_dim = in_dim
        self.side = max(4, math.ceil(math.sqrt(in_dim)))
        act = ACTS[activation]

        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1)]
            if norm == "batch":
                layers.append(nn.BatchNorm2d(cout))
            layers += [act(), nn.MaxPool2d(2)]
            return layers

        self.conv = nn.Sequential(*block(1, 64), *block(64, 128))
        self.flatten = nn.Flatten()
        with torch.no_grad():
            flat = self.conv.eval()(torch.zeros(2, 1, self.side, self.side)).flatten(1).shape[1]
        self.conv.train()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = F.pad(x, (0, self.side * self.side - self.in_dim)).view(x.shape[0], 1, self.side, self.side)
        return self.classifier(self.dropout(self.flatten(self.conv(x))))
