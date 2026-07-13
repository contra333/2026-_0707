"""Common classifier training and evaluation primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, object],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build one of the schedulers allowed by the training protocol."""
    name = str(config.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "multistep":
        if config.get("step_timing", "end_of_epoch") != "end_of_epoch":
            raise ValueError("MultiStepLR requires step_timing='end_of_epoch'")
        milestones = config.get("milestones")
        if not isinstance(milestones, list) or not milestones:
            raise ValueError("MultiStepLR requires a non-empty milestones list")
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(value) for value in milestones],
            gamma=float(config.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {name!r}")


def current_learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: str | torch.device,
) -> dict[str, float | int]:
    """Train for exactly one complete epoch through the common logits API."""
    model.train()
    target_device = torch.device(device)
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    step_count = 0

    for batch in loader:
        images = batch["image"].to(target_device)
        labels = batch["class_label"].to(target_device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("training loss is not finite")
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        loss_sum += float(loss.detach().item()) * batch_size
        correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        sample_count += batch_size
        step_count += 1

    if sample_count == 0:
        raise ValueError("training loader must contain at least one sample")
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "sample_count": sample_count,
        "step_count": step_count,
    }


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: str | torch.device,
) -> dict[str, float | int]:
    """Evaluate classification NLL and accuracy without creating gradients."""
    model.eval()
    target_device = torch.device(device)
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(target_device)
            labels = batch["class_label"].to(target_device)
            logits = model(images)
            loss = criterion(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("evaluation loss is not finite")
            batch_size = int(labels.shape[0])
            loss_sum += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("evaluation loader must contain at least one sample")
    nll = loss_sum / sample_count
    if not math.isfinite(nll):
        raise FloatingPointError("evaluation NLL is not finite")
    return {
        "nll": nll,
        "accuracy": correct / sample_count,
        "sample_count": sample_count,
    }


def is_better_validation(
    candidate: Mapping[str, float | int],
    best: Mapping[str, float | int] | None,
) -> bool:
    """Apply accuracy, NLL, then earliest-epoch best-validation ordering."""
    if best is None:
        return True
    candidate_accuracy = float(candidate["accuracy"])
    best_accuracy = float(best["accuracy"])
    if candidate_accuracy != best_accuracy:
        return candidate_accuracy > best_accuracy
    candidate_nll = float(candidate["nll"])
    best_nll = float(best["nll"])
    if candidate_nll != best_nll:
        return candidate_nll < best_nll
    return False
