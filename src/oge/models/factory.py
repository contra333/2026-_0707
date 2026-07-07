"""Model factory utilities."""

from __future__ import annotations

from torch import nn

from .resnet import ResNet18
from .toy_cnn import ToyCifarCNN
from .wide_resnet import WideResNet


def _model_name(config: dict) -> str:
    if "name" not in config:
        raise ValueError("Model config must include 'name'")
    name = config["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("Model config 'name' must be a non-empty string")
    return name.lower()


def _reject_research_feature_dim(config: dict, name: str) -> None:
    if "feature_dim" in config:
        raise ValueError(f"{name} uses its native feature_dim; do not configure feature_dim")


def make_model(config: dict) -> nn.Module:
    """Build a model from a configuration dictionary."""
    name = _model_name(config)
    num_classes = config.get("num_classes", 10)

    if name == "toy_cifar_cnn":
        return ToyCifarCNN(
            num_classes=num_classes,
            feature_dim=config.get("feature_dim", 128),
        )

    if name == "resnet18":
        _reject_research_feature_dim(config, name)
        if "variant" not in config:
            raise ValueError("resnet18 requires explicit 'variant'")
        return ResNet18(num_classes=num_classes, variant=config["variant"])

    if name == "wrn28_10":
        _reject_research_feature_dim(config, name)
        if "variant" in config:
            raise ValueError("wrn28_10 is fixed by name and does not accept 'variant'")
        depth = config.get("depth", 28)
        widen_factor = config.get("widen_factor", 10)
        if depth != 28 or widen_factor != 10:
            raise ValueError("wrn28_10 requires depth=28 and widen_factor=10")
        return WideResNet(num_classes=num_classes, depth=28, widen_factor=10)

    raise ValueError(f"Unknown model name: {name}")
