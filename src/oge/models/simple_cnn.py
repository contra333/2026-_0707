"""Small CNN architectures for optimizer-geometry experiments."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A compact CNN for CIFAR-like ``[B, 3, 32, 32]`` inputs."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_projection = nn.Linear(128, feature_dim)
        self.activation = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.features(x)
        hidden = torch.flatten(hidden, 1)
        features = self.activation(self.feature_projection(hidden))
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits
