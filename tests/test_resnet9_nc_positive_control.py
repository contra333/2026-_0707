import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torchvision import transforms

from oge.data import (
    MNIST_POSITIVE_CONTROL_MEAN,
    MNIST_POSITIVE_CONTROL_SIZE,
    MNIST_POSITIVE_CONTROL_STD,
    RecordingRandomSampler,
    make_mnist_positive_control_transform,
)
from oge.evaluation.geometry import (
    fit_geometry_statistics,
    neural_collapse_metrics,
    neural_collapse_structure_metrics,
)
from oge.models import ResNet9, make_model
from oge.optimizers import AdamCoupledDecoupled, make_optimizer
from oge.train_utils import make_weight_decay_param_groups
from oge.validation.resnet9_nc_positive_control import (
    EXPECTED_RATIOS,
    STRUCTURE_KEYS,
    load_positive_control_config,
    validate_positive_control_config,
)
from oge.validation.resnet9_nc_summary import summarize_resnet9_nc_positive_control


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs/validation/resnet9_mnist_nc_positive_control_v1.yaml"
)


class PinnedUpstreamResNet9(nn.Module):
    """Independent transcription of upstream resnet.py at pinned commit."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.llBN = nn.BatchNorm2d(512)

        def conv_block(input_channels, output_channels, pool=False):
            layers = [
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        def conv_block_bn(input_channels, output_channels, pool=False):
            layers = [
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                self.llBN,
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
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block_bn(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
        self.fc = nn.Linear(512, num_classes)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return self.fc(out)


def _copy_config():
    return load_positive_control_config(CONFIG_PATH)


def test_resnet9_forward_matches_independent_pinned_upstream_transcription():
    torch.manual_seed(17)
    reference = PinnedUpstreamResNet9(1, 10).eval()
    torch.manual_seed(17)
    model = ResNet9(num_classes=10, in_channels=1).eval()
    inputs = torch.randn(3, 1, 32, 32)

    with torch.no_grad():
        expected = reference(inputs)
        actual, features = model(inputs, return_features=True)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, model.classifier(features), rtol=0.0, atol=0.0)
    assert features.shape == (3, 512)
    assert model.fc is model.classifier
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(
        parameter.numel() for parameter in reference.parameters()
    )


def test_resnet9_factory_contract_is_one_channel_and_native_512_dimensional():
    model = make_model({"name": "resnet9", "num_classes": 10, "in_channels": 1})
    assert isinstance(model, ResNet9)
    assert model.feature_dim == 512
    assert model.conv1[0].in_channels == 1
    assert model.classifier.weight.shape == (10, 512)
    with pytest.raises(ValueError, match="native feature_dim"):
        make_model({"name": "resnet9", "feature_dim": 128})


def test_mnist_transform_is_exact_resize_tensor_normalization_without_augmentation():
    transform = make_mnist_positive_control_transform()
    assert [type(item) for item in transform.transforms] == [
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert transform.transforms[0].size == MNIST_POSITIVE_CONTROL_SIZE
    assert tuple(transform.transforms[2].mean) == MNIST_POSITIVE_CONTROL_MEAN
    assert tuple(transform.transforms[2].std) == MNIST_POSITIVE_CONTROL_STD


def test_positive_control_config_freezes_ratio_orientation_and_recipe():
    config = _copy_config()
    assert tuple(config["optimizer"]["coupled_ratios"]) == EXPECTED_RATIOS
    assert config["optimizer"]["total_weight_decay"] == 5.0e-4
    assert config["optimizer"]["weight_decay_policy"] == "all_parameters"
    assert config["scheduler"]["milestones"] == [66, 133]
    assert config["training"]["seeds"] == [3141]

    changed = copy.deepcopy(config)
    changed["optimizer"]["coupled_ratios"] = list(reversed(EXPECTED_RATIOS))
    with pytest.raises(ValueError, match="optimizer mapping"):
        validate_positive_control_config(changed)


def test_all_parameter_decay_policy_includes_bias_and_batchnorm():
    model = make_model({"name": "resnet9", "num_classes": 10, "in_channels": 1})
    groups = make_weight_decay_param_groups(model, 5.0e-4, policy="all_parameters")
    assert len(groups) == 1
    grouped = {id(parameter) for parameter in groups[0]["params"]}
    assert grouped == {id(parameter) for parameter in model.parameters()}
    assert groups[0]["weight_decay"] == 5.0e-4
    names = {name for name, parameter in model.named_parameters() if id(parameter) in grouped}
    assert "classifier.bias" in names
    assert "conv1.1.weight" in names


def test_same_seed_gives_same_initialization_and_full_first_epoch_order():
    torch.manual_seed(3141)
    first = make_model({"name": "resnet9", "num_classes": 10, "in_channels": 1})
    torch.manual_seed(3141)
    second = make_model({"name": "resnet9", "num_classes": 10, "in_channels": 1})
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    dataset = list(range(31))
    left_sampler = RecordingRandomSampler(
        dataset, generator=torch.Generator().manual_seed(3141)
    )
    right_sampler = RecordingRandomSampler(
        dataset, generator=torch.Generator().manual_seed(3141)
    )
    assert list(left_sampler) == list(right_sampler)
    assert left_sampler.last_order_digest == right_sampler.last_order_digest


def _torch_adam_kwargs():
    signature = inspect.signature(torch.optim.Adam)
    return {name: False for name in ("foreach", "fused") if name in signature.parameters}


def test_mixed_adam_two_step_float64_matches_independent_equation_with_lr_change():
    initial = torch.tensor([2.0, -3.0], dtype=torch.float64)
    parameter = nn.Parameter(initial.clone())
    ratio = 0.4
    total = 5.0e-4
    optimizer = make_optimizer(
        [parameter],
        {
            "name": "adam_coupled_decoupled",
            "lr": 1.0e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-8,
            "total_weight_decay": total,
            "coupled_ratio": ratio,
        },
    )
    assert isinstance(optimizer, AdamCoupledDecoupled)

    expected = initial.clone()
    exp_avg = torch.zeros_like(expected)
    exp_avg_sq = torch.zeros_like(expected)
    gradients = (
        torch.tensor([0.4, -0.6], dtype=torch.float64),
        torch.tensor([-0.2, 0.3], dtype=torch.float64),
    )
    learning_rates = (1.0e-3, 3.0e-4)
    for step, (gradient, lr) in enumerate(zip(gradients, learning_rates), 1):
        optimizer.param_groups[0]["lr"] = lr
        parameter.grad = gradient.clone()
        optimizer.step()

        coupled_gradient = gradient + ratio * total * expected
        exp_avg = 0.9 * exp_avg + 0.1 * coupled_gradient
        exp_avg_sq = 0.999 * exp_avg_sq + 0.001 * coupled_gradient.square()
        expected = expected * (1.0 - lr * (1.0 - ratio) * total)
        expected = expected - (
            lr
            * (exp_avg / (1.0 - 0.9**step))
            / ((exp_avg_sq / (1.0 - 0.999**step)).sqrt() + 1.0e-8)
        )

    torch.testing.assert_close(parameter.detach(), expected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], exp_avg)
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], exp_avg_sq)


def test_lightweight_structure_metrics_are_identical_to_full_geometry_path():
    means = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ],
        dtype=np.float64,
    )
    features = np.repeat(means, 2, axis=0)
    labels = np.repeat(np.arange(3), 2)
    statistics = fit_geometry_statistics(features, labels, num_classes=3)
    light = neural_collapse_structure_metrics(means, means)
    full = neural_collapse_metrics(statistics, means)
    for key in STRUCTURE_KEYS:
        assert light[key]["value"] == pytest.approx(full[key]["value"], abs=1.0e-14)


def test_structure_metrics_are_invariant_to_joint_orthogonal_feature_rotation():
    rng = np.random.default_rng(42)
    means = rng.normal(size=(4, 6))
    weights = rng.normal(size=(4, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    base = neural_collapse_structure_metrics(means, weights)
    rotated = neural_collapse_structure_metrics(means @ q, weights @ q)
    for key in STRUCTURE_KEYS:
        assert rotated[key]["value"] == pytest.approx(base[key]["value"], abs=1.0e-12)


def _write_six_arm_summaries(tmp_path, *, matching_directions=3, dirty=False):
    for index, ratio in enumerate(EXPECTED_RATIOS):
        metrics = {
            "nc0_row_sum_raw": {"value": 1.0 - 0.1 * index},
            "nc2_etf_raw": {
                "value": 1.0 - 0.1 * index if matching_directions >= 2 else 1.0 + index
            },
            "nc3_self_duality_raw": {
                "value": 1.0 - 0.1 * index if matching_directions == 3 else 1.0 + index
            },
        }
        run_dir = tmp_path / f"ratio-{ratio}"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "seed": 3141,
                    "repository_sha": "execution-sha",
                    "repository_dirty": dirty,
                    "smoke_only": False,
                    "completed_epoch": 200,
                    "coupled_ratio": ratio,
                    "initial_model_state_sha256": "same-model",
                    "first_epoch_train_order_sha256": "same-order",
                    "terminal": {"test_accuracy": 0.99, "metrics": metrics},
                }
            ),
            encoding="utf-8",
        )


@pytest.mark.parametrize(
    ("matching_directions", "expected"),
    [(3, "PASS"), (2, "PARTIAL"), (1, "FAIL")],
)
def test_six_arm_summary_applies_frozen_directional_gate(
    tmp_path, matching_directions, expected
):
    _write_six_arm_summaries(tmp_path, matching_directions=matching_directions)
    summary = summarize_resnet9_nc_positive_control(tmp_path)
    assert summary["verdict"] == expected
    assert summary["endpoint_direction_count"] == matching_directions


def test_six_arm_summary_blocks_dirty_execution(tmp_path):
    _write_six_arm_summaries(tmp_path, dirty=True)
    summary = summarize_resnet9_nc_positive_control(tmp_path)
    assert summary["verdict"] == "BLOCKED"
    assert summary["blocker_reasons"] == ["dirty_execution_repository"]
