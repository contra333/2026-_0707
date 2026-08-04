"""ID performance, calibration, failure-ranking, and logit controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn import metrics

from .common import MetricValue, finite_float64, finite_labels


def _logsumexp(values: np.ndarray, *, axis: int = 1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(values - maximum), axis=axis)
    )


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return float(implementation(y, x))


def softmax_probabilities(logits: Any) -> np.ndarray:
    values = finite_float64(logits, name="logits", ndim=2)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def top1_accuracy(logits: Any, labels: Any) -> dict[str, Any]:
    values = finite_float64(logits, name="logits", ndim=2)
    targets = finite_labels(labels, count=len(values), num_classes=values.shape[1])
    predictions = np.argmax(values, axis=1).astype(np.int64)
    correct = predictions == targets
    per_class = {}
    for class_index in range(values.shape[1]):
        mask = targets == class_index
        per_class[str(class_index)] = {
            "correct": int(np.count_nonzero(correct & mask)),
            "count": int(np.count_nonzero(mask)),
        }
    return {
        "metric": MetricValue(float(np.mean(correct))).to_dict(),
        "num_correct": int(np.count_nonzero(correct)),
        "count": int(len(values)),
        "predictions": predictions,
        "correct": correct,
        "per_class": per_class,
    }


def negative_log_likelihood(logits: Any, labels: Any) -> dict[str, Any]:
    values = finite_float64(logits, name="logits", ndim=2)
    targets = finite_labels(labels, count=len(values), num_classes=values.shape[1])
    per_sample = _logsumexp(values) - values[np.arange(len(values)), targets]
    return {
        "metric": MetricValue(float(np.mean(per_sample))).to_dict(),
        "per_sample": per_sample,
        "count": int(len(values)),
    }


def expected_calibration_error(logits: Any, labels: Any, *, bins: int = 15) -> dict[str, Any]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    probabilities = softmax_probabilities(logits)
    targets = finite_labels(labels, count=len(probabilities), num_classes=probabilities.shape[1])
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = predictions == targets
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    bin_records: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        if index == 0:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        count = int(np.count_nonzero(mask))
        accuracy = float(np.mean(correct[mask])) if count else None
        mean_confidence = float(np.mean(confidence[mask])) if count else None
        weighted_gap = (
            count / len(confidence) * abs(accuracy - mean_confidence) if count else 0.0
        )
        ece += weighted_gap
        bin_records.append(
            {
                "left": float(edges[index]),
                "right": float(edges[index + 1]),
                "left_inclusive": index == 0,
                "right_inclusive": True,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "weighted_gap": float(weighted_gap),
            }
        )
    return {
        "metric": MetricValue(float(ece)).to_dict(),
        "bins": bin_records,
        "confidence": confidence,
        "predictions": predictions.astype(np.int64),
    }


def logit_ood_scores(logits: Any) -> dict[str, np.ndarray]:
    values = finite_float64(logits, name="logits", ndim=2)
    probabilities = softmax_probabilities(values)
    return {
        "msp": np.max(probabilities, axis=1),
        "max_logit": np.max(values, axis=1),
        "energy_t1": _logsumexp(values),
    }


def misclassification_auroc(logits: Any, labels: Any) -> dict[str, Any]:
    accuracy = top1_accuracy(logits, labels)
    correct = accuracy["correct"].astype(np.int64)
    confidence = logit_ood_scores(logits)["msp"]
    if np.unique(correct).size < 2:
        return {
            "metric": MetricValue(
                None,
                status="degenerate",
                reason_codes=("single_correctness_class",),
            ).to_dict(),
            "confidence": confidence,
            "correct": correct,
        }
    return {
        "metric": MetricValue(float(metrics.roc_auc_score(correct, confidence))).to_dict(),
        "confidence": confidence,
        "correct": correct,
    }


def augrc(logits: Any, labels: Any) -> dict[str, Any]:
    accuracy_payload = top1_accuracy(logits, labels)
    ranking = misclassification_auroc(logits, labels)
    accuracy = float(accuracy_payload["metric"]["value"])
    auroc = ranking["metric"]["value"]
    if auroc is None:
        return {
            "metric": MetricValue(
                None,
                status="degenerate",
                reason_codes=("misclassification_auroc_degenerate",),
            ).to_dict(),
            "accuracy": accuracy,
            "auroc_correct": None,
        }
    ranking_term = (1.0 - float(auroc)) * accuracy * (1.0 - accuracy)
    prevalence_term = 0.5 * (1.0 - accuracy) ** 2
    return {
        "metric": MetricValue(float(ranking_term + prevalence_term)).to_dict(),
        "accuracy": accuracy,
        "auroc_correct": float(auroc),
        "ranking_term": float(ranking_term),
        "prevalence_term": float(prevalence_term),
    }


def aurc_tie_grouped(logits: Any, labels: Any) -> dict[str, Any]:
    accuracy_payload = top1_accuracy(logits, labels)
    errors = (~accuracy_payload["correct"]).astype(np.float64)
    confidence = logit_ood_scores(logits)["msp"]
    thresholds = np.unique(confidence)
    thresholds.sort()
    coverage = [1.0]
    risk = [float(np.mean(errors))]
    group_counts: list[int] = []
    accepted = np.ones(len(errors), dtype=bool)
    for threshold in thresholds:
        group = confidence == threshold
        group_counts.append(int(np.count_nonzero(group)))
        accepted[group] = False
        current_coverage = float(np.mean(accepted))
        coverage.append(current_coverage)
        risk.append(float(np.mean(errors[accepted])) if np.any(accepted) else risk[-1])
    area = -_trapezoid(np.asarray(risk), np.asarray(coverage))
    return {
        "metric": MetricValue(area).to_dict(),
        "thresholds": thresholds,
        "coverage": np.asarray(coverage),
        "risk": np.asarray(risk),
        "group_counts": np.asarray(group_counts, dtype=np.int64),
        "display_scale": 1000,
    }


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    fallback_used: bool
    optimizer_evaluations: int
    validation_nll_before: float
    validation_nll_after: float


def fit_temperature(logits: Any, labels: Any) -> TemperatureFit:
    values = finite_float64(logits, name="validation_logits", ndim=2)
    targets = finite_labels(labels, count=len(values), num_classes=values.shape[1])
    logits_tensor = torch.as_tensor(values, dtype=torch.float64)
    labels_tensor = torch.as_tensor(targets, dtype=torch.long)
    baseline = float(torch.nn.functional.cross_entropy(logits_tensor, labels_tensor).item())
    log_temperature = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.01,
        max_iter=200,
        max_eval=250,
        history_size=100,
        tolerance_grad=1.0e-7,
        tolerance_change=1.0e-9,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0
    failed = False

    def closure() -> torch.Tensor:
        nonlocal evaluations
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = torch.nn.functional.cross_entropy(logits_tensor / temperature, labels_tensor)
        evaluations += 1
        if torch.isfinite(loss):
            loss.backward()
        return loss

    try:
        optimizer.step(closure)
        candidate = float(torch.exp(log_temperature.detach()).item())
        candidate_loss = float(
            torch.nn.functional.cross_entropy(logits_tensor / candidate, labels_tensor).item()
        )
        gradient = log_temperature.grad
        failed = (
            not np.isfinite(candidate)
            or not np.isfinite(candidate_loss)
            or gradient is None
            or not torch.isfinite(gradient).all()
            or not (0.01 <= candidate <= 100.0)
            or candidate_loss > baseline + 1.0e-12
        )
    except (RuntimeError, ValueError, FloatingPointError):
        candidate = 1.0
        candidate_loss = baseline
        failed = True

    if failed:
        grid = np.geomspace(0.01, 100.0, 2001)
        grid_losses = np.asarray(
            [negative_log_likelihood(values / temperature, targets)["metric"]["value"] for temperature in grid],
            dtype=np.float64,
        )
        if not np.isfinite(grid_losses).all():
            raise ValueError("temperature fallback produced non-finite objectives")
        index = int(np.argmin(grid_losses))
        candidate = float(grid[index])
        candidate_loss = float(grid_losses[index])
    return TemperatureFit(
        temperature=candidate,
        fallback_used=failed,
        optimizer_evaluations=evaluations,
        validation_nll_before=baseline,
        validation_nll_after=candidate_loss,
    )


def evaluate_temperature_scaled(logits: Any, labels: Any, fit: TemperatureFit) -> dict[str, Any]:
    values = finite_float64(logits, name="test_logits", ndim=2)
    targets = finite_labels(labels, count=len(values), num_classes=values.shape[1])
    raw_predictions = np.argmax(values, axis=1)
    scaled = values / fit.temperature
    if not np.array_equal(raw_predictions, np.argmax(scaled, axis=1)):
        raise ValueError("temperature scaling changed argmax predictions")
    return {
        "temperature": MetricValue(fit.temperature).to_dict(),
        "nll": negative_log_likelihood(scaled, targets),
        "ece_m15": expected_calibration_error(scaled, targets, bins=15),
        "fit": {
            "fallback_used": fit.fallback_used,
            "optimizer_evaluations": fit.optimizer_evaluations,
            "validation_nll_before": fit.validation_nll_before,
            "validation_nll_after": fit.validation_nll_after,
        },
    }
