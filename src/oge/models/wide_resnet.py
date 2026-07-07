"""Wide ResNet backbones for CIFAR-like inputs."""

from __future__ import annotations

import torch
from torch import nn


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        identity = self.shortcut(x)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + identity


class WideNetworkBlock(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        layers = []
        for index in range(num_blocks):
            block_stride = stride if index == 0 else 1
            block_in = in_channels if index == 0 else out_channels
            layers.append(WideBasicBlock(block_in, out_channels, block_stride, dropout_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class WideResNet(nn.Module):
    """Wide ResNet with native ``64 * widen_factor`` penultimate features."""

    def __init__(
        self,
        num_classes: int = 10,
        *,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy depth = 6n + 4")
        if widen_factor <= 0:
            raise ValueError("widen_factor must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")

        blocks_per_group = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.num_classes = num_classes
        self.depth = depth
        self.widen_factor = widen_factor
        self.feature_dim = widths[3]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = WideNetworkBlock(
            blocks_per_group,
            widths[0],
            widths[1],
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.block2 = WideNetworkBlock(
            blocks_per_group,
            widths[1],
            widths[2],
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.block3 = WideNetworkBlock(
            blocks_per_group,
            widths[2],
            widths[3],
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits
