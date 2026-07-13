import copy
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from oge.models import make_model
from oge.optimizers import make_optimizer
from oge.training import (
    capture_rng_state,
    evaluate_classifier,
    is_better_validation,
    make_scheduler,
    restore_rng_state,
    train_one_epoch,
)


class DictTensorDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "class_label": self.labels[index],
        }


def test_wrn_common_factories_train_one_finite_batch_and_change_parameter():
    torch.manual_seed(3)
    model = make_model(
        {
            "name": "wrn28_10",
            "num_classes": 10,
            "depth": 28,
            "widen_factor": 10,
            "dropout_rate": 0.0,
        }
    )
    optimizer = make_optimizer(
        model,
        {
            "name": "sgd",
            "lr": 0.01,
            "momentum": 0.0,
            "nesterov": False,
            "weight_decay": 0.0005,
            "weight_decay_policy": "weights_only_no_bias_norm",
        },
    )
    loader = DataLoader(
        DictTensorDataset(torch.randn(1, 3, 8, 8), torch.tensor([2])),
        batch_size=1,
    )
    before = model.classifier.weight.detach().clone()

    metrics = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        device="cpu",
    )

    assert np.isfinite(metrics["loss"])
    assert metrics["sample_count"] == 1
    assert metrics["step_count"] == 1
    assert not torch.equal(before, model.classifier.weight.detach())


class EvaluationGuard(nn.Module):
    def forward(self, inputs):
        assert self.training is False
        assert torch.is_grad_enabled() is False
        return torch.stack([inputs[:, 0], -inputs[:, 0]], dim=1)


def test_validation_uses_eval_mode_without_gradients():
    loader = DataLoader(
        DictTensorDataset(torch.tensor([[1.0], [-1.0]]), torch.tensor([0, 1])),
        batch_size=2,
    )

    metrics = evaluate_classifier(
        EvaluationGuard(),
        loader,
        nn.CrossEntropyLoss(),
        device="cpu",
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["sample_count"] == 2


def test_multistep_scheduler_matches_one_based_epoch_boundaries():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = make_scheduler(
        optimizer,
        {
            "name": "multistep",
            "milestones": [60, 120, 160],
            "gamma": 0.2,
            "step_timing": "end_of_epoch",
        },
    )
    used_learning_rates = []
    for _epoch in range(1, 201):
        used_learning_rates.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()

    assert used_learning_rates[:60] == pytest.approx([0.1] * 60)
    assert used_learning_rates[60:120] == pytest.approx([0.02] * 60)
    assert used_learning_rates[120:160] == pytest.approx([0.004] * 40)
    assert used_learning_rates[160:] == pytest.approx([0.0008] * 40)


def test_none_scheduler_returns_none():
    optimizer = torch.optim.SGD([nn.Parameter(torch.tensor(1.0))], lr=0.1)
    assert make_scheduler(optimizer, {"name": "none"}) is None


@pytest.mark.parametrize(
    "candidate,best,expected",
    [
        ({"epoch": 2, "accuracy": 0.9, "nll": 1.2}, None, True),
        (
            {"epoch": 2, "accuracy": 0.91, "nll": 2.0},
            {"epoch": 1, "accuracy": 0.9, "nll": 1.0},
            True,
        ),
        (
            {"epoch": 2, "accuracy": 0.9, "nll": 0.9},
            {"epoch": 1, "accuracy": 0.9, "nll": 1.0},
            True,
        ),
        (
            {"epoch": 2, "accuracy": 0.9, "nll": 1.0},
            {"epoch": 1, "accuracy": 0.9, "nll": 1.0},
            False,
        ),
        (
            {"epoch": 2, "accuracy": 0.9, "nll": 1.1},
            {"epoch": 1, "accuracy": 0.9, "nll": 1.0},
            False,
        ),
        (
            {"epoch": 2, "accuracy": 0.89, "nll": 0.1},
            {"epoch": 1, "accuracy": 0.9, "nll": 1.0},
            False,
        ),
    ],
)
def test_best_validation_uses_accuracy_nll_then_earliest(candidate, best, expected):
    assert is_better_validation(candidate, best) is expected


def test_rng_capture_and_restore_covers_python_numpy_torch_and_train_generator():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    generator = torch.Generator().manual_seed(11)
    state = capture_rng_state(generator)
    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )
    for _ in range(5):
        random.random()
        np.random.random()
        torch.rand(())
        torch.rand((), generator=generator)

    restore_rng_state(copy.deepcopy(state), generator)
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )

    assert actual == expected
