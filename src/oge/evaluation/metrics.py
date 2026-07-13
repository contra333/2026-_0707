"""Explicit ID-positive and OpenOOD-compatible OOD metric definitions."""

from __future__ import annotations

import math

import numpy as np
from sklearn import metrics


def _validated_scores(
    id_like_scores: np.ndarray | list[float],
    ood_like_scores: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    id_scores = np.asarray(id_like_scores, dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(ood_like_scores, dtype=np.float64).reshape(-1)
    if id_scores.size == 0:
        raise ValueError("ID score array must not be empty")
    if ood_scores.size == 0:
        raise ValueError("OOD score array must not be empty")
    if not np.isfinite(id_scores).all() or not np.isfinite(ood_scores).all():
        raise ValueError("score arrays must contain only finite values")
    return id_scores, ood_scores


def fpr95_id_tpr(
    id_like_scores: np.ndarray | list[float],
    ood_like_scores: np.ndarray | list[float],
) -> tuple[float, float]:
    """Return OOD acceptance rate and threshold at the first ID TPR >= 0.95."""
    id_scores, ood_scores = _validated_scores(id_like_scores, ood_like_scores)
    required_id = math.ceil(0.95 * id_scores.size)
    threshold = float(np.sort(id_scores)[::-1][required_id - 1])
    fpr = float(np.mean(ood_scores >= threshold))
    return fpr, threshold


def fpr95_openood_ood_tpr(
    id_like_scores: np.ndarray | list[float],
    ood_like_scores: np.ndarray | list[float],
) -> tuple[float, float]:
    """Reproduce pinned OpenOOD OOD-positive ``roc_curve`` behavior."""
    id_scores, ood_scores = _validated_scores(id_like_scores, ood_like_scores)
    labels = np.concatenate(
        [np.zeros(id_scores.size, dtype=np.int64), np.ones(ood_scores.size, dtype=np.int64)]
    )
    negated_scores = -np.concatenate([id_scores, ood_scores])
    fpr_values, tpr_values, thresholds = metrics.roc_curve(labels, negated_scores)
    index = int(np.argmax(tpr_values >= 0.95))
    return float(fpr_values[index]), float(thresholds[index])


def compute_ood_metrics(
    id_like_scores: np.ndarray | list[float],
    ood_like_scores: np.ndarray | list[float],
) -> dict[str, float]:
    id_scores, ood_scores = _validated_scores(id_like_scores, ood_like_scores)
    labels = np.concatenate(
        [np.ones(id_scores.size, dtype=np.int64), np.zeros(ood_scores.size, dtype=np.int64)]
    )
    scores = np.concatenate([id_scores, ood_scores])
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    project_fpr, _ = fpr95_id_tpr(id_scores, ood_scores)
    openood_fpr, _ = fpr95_openood_ood_tpr(id_scores, ood_scores)
    return {
        "auroc": float(metrics.roc_auc_score(labels, scores)),
        "aupr_in_ap": float(metrics.average_precision_score(labels, scores)),
        "aupr_in_openood_auc": float(metrics.auc(recall, precision)),
        "fpr95_id_tpr": project_fpr,
        "fpr95_openood_ood_tpr": openood_fpr,
    }


METRIC_DEFINITIONS = {
    "score_direction": "higher_is_more_id_like",
    "id_binary_label": 1,
    "ood_binary_label": 0,
    "fpr95_id_tpr": (
        "Highest inclusive ID-like threshold that first reaches ID TPR >= 0.95; "
        "value is the fraction of OOD scores >= threshold; equal-score ties are grouped."
    ),
    "fpr95_openood_ood_tpr": (
        "Pinned OpenOOD OOD-positive sklearn roc_curve on negated ID-like scores; "
        "first returned point with OOD TPR >= 0.95."
    ),
    "aupr_in_ap": "sklearn.metrics.average_precision_score with ID positive.",
    "aupr_in_openood_auc": (
        "sklearn.metrics.precision_recall_curve followed by sklearn.metrics.auc, "
        "with ID positive."
    ),
}
