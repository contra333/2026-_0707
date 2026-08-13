import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from oge.optimizers import make_optimizer
from oge.training import (
    UpdateTelemetryRecorder,
    counterfactual_candidate_updates,
    train_one_epoch,
)


class OneBatchDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "image": torch.tensor([index + 1.0, 2.0 - index]),
            "class_label": index,
            "sample_id": f"fixture:{index}",
        }


class TinyBatchNormClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 3)
        self.bn = nn.BatchNorm1d(3)
        self.classifier = nn.Linear(3, 2)

    def forward(self, inputs):
        return self.classifier(torch.relu(self.bn(self.input(inputs))))


def _optimizer(model):
    return make_optimizer(
        model,
        {
            "name": "adam",
            "lr": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "weight_decay_policy": "weights_only_no_bias_norm",
        },
    )


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def test_update_telemetry_is_passive_and_records_required_fields():
    torch.manual_seed(31)
    model_without = TinyBatchNormClassifier()
    model_with = copy.deepcopy(model_without)
    optimizer_without = _optimizer(model_without)
    optimizer_with = _optimizer(model_with)
    loader = DataLoader(OneBatchDataset(), batch_size=2, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    recorder = UpdateTelemetryRecorder(
        model=model_with,
        optimizer=optimizer_with,
        audit_steps=[1],
        include_batch_norm=True,
        activation_residual_provider=lambda: {
            "activation_scale": {"status": "fixture"},
            "residual_shortcut_norm_ratio": {"status": "fixture"},
        },
    )

    metrics_without = train_one_epoch(
        model_without,
        loader,
        optimizer_without,
        criterion,
        device="cpu",
    )
    metrics_with = train_one_epoch(
        model_with,
        loader,
        optimizer_with,
        criterion,
        device="cpu",
        update_telemetry=recorder,
    )

    assert metrics_without == metrics_with
    _assert_nested_equal(model_without.state_dict(), model_with.state_dict())
    _assert_nested_equal(optimizer_without.state_dict(), optimizer_with.state_dict())
    parameter_records = [
        record for record in recorder.records if record["record_type"] == "parameter_update"
    ]
    assert parameter_records
    required = {
        "parameter_name",
        "layer_name",
        "parameter_group_index",
        "parameter_group_identity",
        "parameter_group_decay",
        "parameter_norm",
        "gradient_norm",
        "optimizer_update_norm",
        "update_parameter_norm_ratio",
        "update_weight_cosine",
        "radial_update_signed_magnitude",
        "radial_update_norm",
        "tangential_update_norm",
        "angular_update_radians",
        "adam_moments",
    }
    assert required.issubset(parameter_records[0])
    assert any(record["adam_moments"]["exp_avg"] is not None for record in parameter_records)
    diagnostics = [
        record for record in recorder.records if record["record_type"] == "step_diagnostics"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0]["batch_norm"][0]["module_name"] == "bn"
    assert diagnostics[0]["activation_residual_diagnostics"] == {
        "activation_scale": {"status": "fixture"},
        "residual_shortcut_norm_ratio": {"status": "fixture"},
    }


def test_counterfactual_candidate_updates_never_touch_live_parameter_or_gradient():
    parameter = torch.tensor([1.0, -2.0], dtype=torch.float64)
    gradient = torch.tensor([0.2, -0.4], dtype=torch.float64)
    parameter_before = parameter.clone()
    gradient_before = gradient.clone()
    common = {
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
    }
    updates = counterfactual_candidate_updates(
        parameter=parameter,
        gradient=gradient,
        candidates={
            "zero": {"name": "adam", **common, "weight_decay": 0.0},
            "decoupled": {"name": "adamw", **common, "weight_decay": 1e-4},
            "coupled": {"name": "adam", **common, "weight_decay": 1e-4},
            "mixed": {
                "name": "adam_coupled_decoupled",
                **common,
                "total_weight_decay": 1e-4,
                "coupled_ratio": 0.5,
            },
        },
    )

    torch.testing.assert_close(parameter, parameter_before, rtol=0, atol=0)
    torch.testing.assert_close(gradient, gradient_before, rtol=0, atol=0)
    assert set(updates) == {"zero", "decoupled", "coupled", "mixed"}
    assert all(update.shape == parameter.shape for update in updates.values())
    assert not torch.equal(updates["zero"], updates["decoupled"])


def test_counterfactual_candidates_reuse_cloned_populated_adam_state_only():
    live_parameter = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    live_optimizer = make_optimizer(
        [live_parameter],
        {"name": "adam", "lr": 1e-3, "weight_decay": 0.0},
    )
    live_parameter.grad = torch.tensor([0.2, -0.4], dtype=torch.float64)
    live_optimizer.step()
    live_parameter.grad = torch.tensor([0.1, -0.2], dtype=torch.float64)
    parameter_before = live_parameter.detach().clone()
    gradient_before = live_parameter.grad.detach().clone()
    state_before = copy.deepcopy(live_optimizer.state[live_parameter])
    common = {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}

    updates = counterfactual_candidate_updates(
        parameter=live_parameter,
        gradient=live_parameter.grad,
        optimizer_state=live_optimizer.state[live_parameter],
        candidates={
            "zero": {"name": "adam", **common, "weight_decay": 0.0},
            "decoupled": {"name": "adamw", **common, "weight_decay": 1e-4},
            "coupled": {"name": "adam", **common, "weight_decay": 1e-4},
            "mixed": {
                "name": "adam_coupled_decoupled",
                **common,
                "total_weight_decay": 1e-4,
                "coupled_ratio": 0.5,
            },
        },
    )

    torch.testing.assert_close(live_parameter, parameter_before, rtol=0, atol=0)
    torch.testing.assert_close(live_parameter.grad, gradient_before, rtol=0, atol=0)
    _assert_nested_equal(live_optimizer.state[live_parameter], state_before)
    assert set(updates) == {"zero", "decoupled", "coupled", "mixed"}


def test_audit_schedule_is_config_driven_and_rejects_implicit_thresholds():
    model = TinyBatchNormClassifier()
    optimizer = _optimizer(model)
    recorder = UpdateTelemetryRecorder.from_config(
        model=model,
        optimizer=optimizer,
        config={
            "schema_version": "task_f_update_telemetry_v1",
            "audit_steps": [2, 7],
            "include_batch_norm": False,
            "activation_residual_diagnostics": [],
        },
    )
    assert recorder.before_step(global_step=1) is None
    assert recorder.before_step(global_step=2) is not None
    assert not hasattr(recorder, "threshold")
