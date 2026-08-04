"""Metric Contract v1.2 ID-positive OOD metric definitions."""

from __future__ import annotations

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
) -> tuple[float, float, float]:
    """Return OOD acceptance, linear ID 5th percentile, and achieved ID TPR."""
    id_scores, ood_scores = _validated_scores(id_like_scores, ood_like_scores)
    threshold = float(np.quantile(id_scores, 0.05, method="linear"))
    fpr = float(np.mean(ood_scores >= threshold))
    achieved_id_tpr = float(np.mean(id_scores >= threshold))
    return fpr, threshold, achieved_id_tpr


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
    precision_in, recall_in, _ = metrics.precision_recall_curve(labels, scores)
    ood_labels = 1 - labels
    ood_scores_oriented = -scores
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        ood_labels, ood_scores_oriented
    )
    project_fpr, threshold, achieved_id_tpr = fpr95_id_tpr(id_scores, ood_scores)
    return {
        "auroc": float(metrics.roc_auc_score(labels, scores)),
        "ap_in": float(metrics.average_precision_score(labels, scores)),
        "aupr_in_openood_auc": float(metrics.auc(recall_in, precision_in)),
        "ap_out": float(metrics.average_precision_score(ood_labels, ood_scores_oriented)),
        "aupr_out_openood_auc": float(metrics.auc(recall_out, precision_out)),
        "fpr95_id_tpr": project_fpr,
        "fpr95_threshold": threshold,
        "fpr95_achieved_id_tpr": achieved_id_tpr,
    }


def macro_average_ood_metrics(
    per_dataset: dict[str, dict[str, float]],
    *,
    near_datasets: tuple[str, ...] = ("cifar100", "tin"),
    far_datasets: tuple[str, ...] = ("mnist", "svhn", "texture", "places365"),
) -> dict[str, dict[str, float]]:
    expected = near_datasets + far_datasets
    missing = set(expected).difference(per_dataset)
    if missing:
        raise ValueError(f"per-dataset metrics are missing: {sorted(missing)}")
    metric_names = set(per_dataset[expected[0]])
    if any(set(per_dataset[name]) != metric_names for name in expected):
        raise ValueError("all datasets must provide the same metric keys")

    def average(names: tuple[str, ...]) -> dict[str, float]:
        return {
            metric_name: float(np.mean([per_dataset[name][metric_name] for name in names]))
            for metric_name in sorted(metric_names)
        }

    return {
        "near_macro_mean": average(near_datasets),
        "far_macro_mean": average(far_datasets),
        "overall_macro_mean": average(expected),
    }


METRIC_DEFINITIONS = {
    "score_direction": "higher_is_more_id_like",
    "id_binary_label": 1,
    "ood_binary_label": 0,
    "fpr95_id_tpr": (
        "Linear 5th percentile of ID-like scores; value is the fraction of OOD "
        "scores >= threshold and achieved inclusive ID TPR is stored separately."
    ),
    "fpr95_openood_ood_tpr": (
        "Pinned OpenOOD OOD-positive sklearn roc_curve on negated ID-like scores; "
        "first returned point with OOD TPR >= 0.95."
    ),
    "ap_in": "sklearn.metrics.average_precision_score with ID positive.",
    "aupr_in_openood_auc": (
        "sklearn.metrics.precision_recall_curve followed by sklearn.metrics.auc, "
        "with ID positive."
    ),
    "ap_out": "sklearn.metrics.average_precision_score with OOD positive and negated score.",
    "aupr_out_openood_auc": (
        "sklearn.metrics.precision_recall_curve followed by sklearn.metrics.auc, "
        "with OOD positive and negated ID-like score."
    ),
}
