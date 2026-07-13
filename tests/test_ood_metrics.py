import numpy as np
import pytest
from sklearn import metrics

from oge.evaluation.metrics import (
    compute_ood_metrics,
    fpr95_id_tpr,
    fpr95_openood_ood_tpr,
)


def _pinned_openood_values(id_scores, ood_scores):
    confidence = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate(
        [np.zeros(len(id_scores), dtype=int), -np.ones(len(ood_scores), dtype=int)]
    )
    ood_indicator = np.zeros_like(labels)
    ood_indicator[labels == -1] = 1
    fpr_values, tpr_values, _ = metrics.roc_curve(ood_indicator, -confidence)
    fpr = fpr_values[np.argmax(tpr_values >= 0.95)]
    precision_in, recall_in, _ = metrics.precision_recall_curve(1 - ood_indicator, confidence)
    return float(fpr), float(metrics.auc(recall_in, precision_in))


def test_openood_compatible_metrics_match_pinned_code_on_synthetic_arrays():
    id_scores = np.array([0.95, 0.8, 0.7, 0.55, 0.5])
    ood_scores = np.array([0.75, 0.45, 0.4, 0.2, 0.1])
    expected_fpr, expected_aupr = _pinned_openood_values(id_scores, ood_scores)

    actual = compute_ood_metrics(id_scores, ood_scores)

    assert actual["fpr95_openood_ood_tpr"] == expected_fpr
    assert actual["aupr_in_openood_auc"] == expected_aupr
    assert "fpr95" not in actual
    assert "aupr_in" not in actual


def test_project_fpr_uses_highest_inclusive_threshold_and_does_not_split_ties():
    id_scores = np.array([1.0] * 18 + [0.5, 0.5])
    ood_scores = np.array([0.6, 0.5, 0.49, 0.1])

    fpr, threshold = fpr95_id_tpr(id_scores, ood_scores)

    assert threshold == 0.5
    assert fpr == 0.5


def test_project_and_openood_fpr95_are_distinct_definitions():
    id_scores = np.array([0.99, 0.98, 0.97, 0.2])
    ood_scores = np.array([0.96, 0.95, 0.1, 0.05])

    project_fpr, _ = fpr95_id_tpr(id_scores, ood_scores)
    openood_fpr, _ = fpr95_openood_ood_tpr(id_scores, ood_scores)

    assert project_fpr != openood_fpr


def test_ap_and_openood_auc_use_the_declared_sklearn_backends():
    id_scores = np.array([0.9, 0.6, 0.4])
    ood_scores = np.array([0.8, 0.3, 0.2])
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.concatenate([id_scores, ood_scores])
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)

    actual = compute_ood_metrics(id_scores, ood_scores)

    assert actual["aupr_in_ap"] == metrics.average_precision_score(labels, scores)
    assert actual["aupr_in_openood_auc"] == metrics.auc(recall, precision)


@pytest.mark.parametrize("id_scores,ood_scores", [([], [0.1]), ([0.9], [])])
def test_metrics_reject_empty_id_or_ood_arrays(id_scores, ood_scores):
    with pytest.raises(ValueError, match="must not be empty"):
        compute_ood_metrics(id_scores, ood_scores)


def test_metrics_reject_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        compute_ood_metrics([0.9, np.nan], [0.1])
