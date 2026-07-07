import pytest
import torch
from torch import nn

from oge.models import ResNet18, ToyCifarCNN, WideResNet, make_model
from oge.train_utils import make_weight_decay_param_groups


MODEL_CASES = [
    ("toy_cifar_cnn", {"feature_dim": 32}, ToyCifarCNN, 32),
    ("resnet18", {"variant": "cifar"}, ResNet18, 512),
    ("wrn28_10", {}, WideResNet, 640),
]


def _assert_model_api_contract(model, *, batch_size, num_classes, feature_dim):
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32)

    with torch.no_grad():
        logits = model(x)
        logits_with_features, features = model(x, return_features=True)

    assert isinstance(logits, torch.Tensor)
    torch.testing.assert_close(logits, logits_with_features)
    assert list(logits.shape) == [batch_size, num_classes]
    assert list(features.shape) == [batch_size, feature_dim]
    assert model.feature_dim == feature_dim
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.in_features == feature_dim
    assert model.classifier.out_features == num_classes


def _names_in(group, model):
    ids = {id(p) for p in group["params"]}
    return {name for name, p in model.named_parameters() if id(p) in ids}


def test_toy_cifar_cnn_forward_api_contract():
    _assert_model_api_contract(
        ToyCifarCNN(num_classes=7, feature_dim=32),
        batch_size=4,
        num_classes=7,
        feature_dim=32,
    )


@pytest.mark.parametrize("name,extra,expected_cls,feature_dim", MODEL_CASES)
def test_make_model_implemented_endpoints_api_contract(name, extra, expected_cls, feature_dim):
    model = make_model({"name": name, "num_classes": 5, **extra})

    assert isinstance(model, expected_cls)
    _assert_model_api_contract(
        model,
        batch_size=2,
        num_classes=5,
        feature_dim=feature_dim,
    )


def test_make_model_requires_name():
    with pytest.raises(ValueError, match="name"):
        make_model({"num_classes": 5})


@pytest.mark.parametrize("name", ["simple_cnn", "vgg16", "convnext_tiny", "does_not_exist"])
def test_make_model_unknown_or_unimplemented_names_fail(name):
    with pytest.raises(ValueError, match="Unknown model name"):
        make_model({"name": name, "num_classes": 5})


def test_resnet18_requires_explicit_variant():
    with pytest.raises(ValueError, match="variant"):
        make_model({"name": "resnet18", "num_classes": 5})


@pytest.mark.parametrize(
    "config",
    [
        {"name": "resnet18", "variant": "cifar", "feature_dim": 128},
        {"name": "wrn28_10", "feature_dim": 128},
    ],
)
def test_research_backbones_reject_configured_feature_dim(config):
    with pytest.raises(ValueError, match="native feature_dim"):
        make_model(config)


@pytest.mark.parametrize(
    "config,expected_bn_name",
    [
        ({"name": "resnet18", "variant": "cifar", "num_classes": 5}, "bn1.weight"),
        ({"name": "wrn28_10", "num_classes": 5}, "bn.weight"),
    ],
)
def test_model_param_groups_exclude_bias_and_batchnorm(config, expected_bn_name):
    model = make_model(config)
    groups = make_weight_decay_param_groups(model, weight_decay=0.01)
    decay_names = _names_in(groups[0], model)
    no_decay_names = _names_in(groups[1], model)
    bias_names = {name for name, _ in model.named_parameters() if name.endswith(".bias")}
    batchnorm_names = {
        f"{module_name}.{param_name}" if module_name else param_name
        for module_name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for param_name, _ in module.named_parameters(recurse=False)
    }

    assert "classifier.weight" in decay_names
    assert "classifier.bias" in no_decay_names
    assert expected_bn_name in no_decay_names
    assert bias_names.issubset(no_decay_names)
    assert batchnorm_names.issubset(no_decay_names)
    assert decay_names.isdisjoint(no_decay_names)
