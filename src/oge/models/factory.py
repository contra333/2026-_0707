"""Model factory utilities."""

from __future__ import annotations

from .simple_cnn import SimpleCNN


def make_model(config: dict) -> SimpleCNN:
    """Build a model from a configuration dictionary."""
    name = config.get("name", "simple_cnn")
    num_classes = config.get("num_classes", 10)
    feature_dim = config.get("feature_dim", 128)

    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, feature_dim=feature_dim)

    raise ValueError(f"Unknown model name: {name}")
