"""Fully-connected downstream: D -> 128 -> 256 -> 128 -> n_classes."""
import torch.nn as nn

ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}


class FCNHead(nn.Module):
    HIDDEN = (128, 256, 128)

    def __init__(self, in_dim, n_classes, dropout=0.2, activation="relu", norm="batch", **_):
        super().__init__()
        layers, prev = [], in_dim
        for h in self.HIDDEN:
            layers.append(nn.Linear(prev, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            layers += [ACTS[activation](), nn.Dropout(dropout)]
            prev = h
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x))
