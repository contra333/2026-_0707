"""ResNet backbones with explicit architecture variants."""

from __future__ import annotations

import torch
from torch import nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module | None = None

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                _conv1x1(in_channels, out_channels * self.expansion, stride),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 with explicit ``cifar`` or ``imagenet`` stem variants."""

    feature_dim = 512

    def __init__(self, num_classes: int = 10, *, variant: str) -> None:
        super().__init__()
        if variant not in {"cifar", "imagenet"}:
            raise ValueError("ResNet18 variant must be 'cifar' or 'imagenet'")

        self.num_classes = num_classes
        self.variant = variant
        self.in_channels = 64

        if variant == "cifar":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._initialize_weights()

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


class ResNet9(nn.Module):
    """The 512-dimensional ResNet9 used by Zhao et al. for 32x32 inputs.

    The computational graph follows ``jydzhao/neural_collapse_optimizer`` at
    commit ``7cab4a59bc28da6e356cee1e793ec67a694933b9``.  In particular, the
    convolutions retain PyTorch's default bias, the final residual convolution
    uses the separately constructed 512-channel BatchNorm module, and no
    project-specific initialization is applied.
    """

    feature_dim = 512

    def __init__(self, num_classes: int = 10, *, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels not in {1, 3}:
            raise ValueError("ResNet9 in_channels must be 1 or 3")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.last_residual_bn = nn.BatchNorm2d(512)

        def conv_block(
            input_channels: int,
            output_channels: int,
            *,
            pool: bool = False,
            batch_norm: nn.Module | None = None,
        ) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                batch_norm if batch_norm is not None else nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512, batch_norm=self.last_residual_bn),
        )
        self.feature_pool = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    @property
    def fc(self) -> nn.Linear:
        """Pinned-upstream alias for the project's classifier endpoint."""
        return self.classifier

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        features = self.feature_pool(out)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits
