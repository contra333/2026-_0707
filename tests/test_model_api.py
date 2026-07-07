import torch

from oge.models import SimpleCNN, make_model


def _assert_model_api_contract(model, *, batch_size, num_classes, feature_dim):
    x = torch.randn(batch_size, 3, 32, 32)

    logits = model(x)
    logits_with_features, features = model(x, return_features=True)

    assert isinstance(logits, torch.Tensor)
    torch.testing.assert_close(logits, logits_with_features)
    assert list(logits.shape) == [batch_size, num_classes]
    assert list(features.shape) == [batch_size, feature_dim]
    assert list(model.classifier.weight.shape) == [num_classes, feature_dim]


def test_simple_cnn_forward_api_contract():
    _assert_model_api_contract(
        SimpleCNN(num_classes=7, feature_dim=32),
        batch_size=4,
        num_classes=7,
        feature_dim=32,
    )


def test_make_model_simple_cnn_api_contract():
    model = make_model({"name": "simple_cnn", "num_classes": 5, "feature_dim": 64})

    _assert_model_api_contract(
        model,
        batch_size=3,
        num_classes=5,
        feature_dim=64,
    )
