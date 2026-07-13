from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from oge.models import WideResNet, make_model
from oge.models.wide_resnet import WideBasicBlock


MODEL_CONFIG_DIR = Path(__file__).parents[1] / "configs" / "models"


def _load_model_config(filename: str) -> dict:
    with (MODEL_CONFIG_DIR / filename).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict)
    return config


def test_projection_shortcut_receives_preactivated_tensor():
    block = WideBasicBlock(16, 32, stride=2, dropout_rate=0.0).eval()
    x = torch.randn(2, 16, 8, 8)
    shortcut_inputs = []
    handle = block.shortcut.register_forward_pre_hook(
        lambda _module, inputs: shortcut_inputs.append(inputs[0].detach().clone())
    )

    with torch.no_grad():
        expected = block.relu1(block.bn1(x)).detach().clone()
        block(x)
    handle.remove()

    assert len(shortcut_inputs) == 1
    torch.testing.assert_close(shortcut_inputs[0], expected)


def test_same_shape_block_identity_shortcut_receives_original_input():
    block = WideBasicBlock(16, 16, stride=1, dropout_rate=0.0).eval()
    x = torch.randn(2, 16, 8, 8)
    shortcut_inputs = []
    handle = block.shortcut.register_forward_pre_hook(
        lambda _module, inputs: shortcut_inputs.append(inputs[0])
    )

    with torch.no_grad():
        block(x)
    handle.remove()

    assert len(shortcut_inputs) == 1
    assert shortcut_inputs[0] is x


@pytest.mark.parametrize(
    "filename,dropout_rate,expected_type",
    [
        ("wrn28_10.yaml", 0.0, nn.Identity),
        ("wrn28_10_dropout.yaml", 0.3, nn.Dropout),
    ],
)
def test_wrn_presets_configure_every_dropout_location(
    filename, dropout_rate, expected_type
):
    model = make_model(_load_model_config(filename))
    dropouts = [
        module.dropout
        for module in model.modules()
        if isinstance(module, WideBasicBlock)
    ]

    assert model.dropout_rate == pytest.approx(dropout_rate)
    assert len(dropouts) == 12
    assert all(isinstance(dropout, expected_type) for dropout in dropouts)
    if dropout_rate > 0:
        assert all(dropout.p == pytest.approx(dropout_rate) for dropout in dropouts)


def test_wrn28_10_requires_explicit_dropout_rate():
    with pytest.raises(ValueError, match="explicit 'dropout_rate'"):
        make_model(
            {
                "name": "wrn28_10",
                "num_classes": 10,
                "depth": 28,
                "widen_factor": 10,
            }
        )


@pytest.mark.parametrize(
    "dropout_rate",
    [True, False, None, "0.3", -0.1, 1.0, 1.1, float("nan")],
)
def test_wrn28_10_rejects_invalid_dropout_rate(dropout_rate):
    with pytest.raises(ValueError, match="dropout_rate"):
        make_model(
            {
                "name": "wrn28_10",
                "num_classes": 10,
                "depth": 28,
                "widen_factor": 10,
                "dropout_rate": dropout_rate,
            }
        )


@pytest.mark.parametrize("filename", ["wrn28_10.yaml", "wrn28_10_dropout.yaml"])
def test_wrn_presets_preserve_model_api_and_eval_determinism(filename):
    model = make_model(_load_model_config(filename)).eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        logits = model(x)
        logits_with_features, features = model(x, return_features=True)
        repeated_logits, repeated_features = model(x, return_features=True)

    assert isinstance(model, WideResNet)
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.in_features == 640
    assert model.classifier.out_features == 10
    assert list(logits.shape) == [2, 10]
    assert list(features.shape) == [2, 640]
    torch.testing.assert_close(logits, logits_with_features)
    torch.testing.assert_close(logits_with_features, repeated_logits)
    torch.testing.assert_close(features, repeated_features)
