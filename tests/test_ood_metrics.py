import numpy as np
import pytest
from sklearn import metrics

from oge.evaluation.metrics import (
    compute_ood_metrics,
    fpr95_id_tpr,
    fpr95_openood_ood_tpr,
    macro_average_ood_metrics,
)


def test_contract_fpr_uses_linear_quantile_and_records_achieved_tpr():
    id_scores = np.array([0.0, 1.0, 2.0, 3.0])
    ood_scores = np.array([0.14, 0.151, 0.16, 4.0])

    fpr, threshold, achieved = fpr95_id_tpr(id_scores, ood_scores)

    assert threshold == pytest.approx(0.15)
    assert fpr == 0.75
    assert achieved == 0.75


def test_contract_fpr_uses_inclusive_threshold_for_ties():
    id_scores = np.array([1.0] * 18 + [0.5, 0.5])
    ood_scores = np.array([0.6, 0.5, 0.49, 0.1])

    fpr, threshold, achieved = fpr95_id_tpr(id_scores, ood_scores)

    assert threshold == 0.5
    assert fpr == 0.5
    assert achieved == 1.0


def test_openood_compatibility_function_remains_distinct_but_is_not_emitted():
    id_scores = np.array([0.99, 0.98, 0.97, 0.2])
    ood_scores = np.array([0.96, 0.95, 0.1, 0.05])

    contract_fpr, _, _ = fpr95_id_tpr(id_scores, ood_scores)
    compatibility_fpr, _ = fpr95_openood_ood_tpr(id_scores, ood_scores)
    payload = compute_ood_metrics(id_scores, ood_scores)

    assert contract_fpr != compatibility_fpr
    assert "fpr95_openood_ood_tpr" not in payload


def test_ood_metrics_match_declared_id_and_ood_positive_backends():
    id_scores = np.array([0.9, 0.6, 0.4])
    ood_scores = np.array([0.8, 0.3, 0.2])
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.concatenate([id_scores, ood_scores])
    precision_in, recall_in, _ = metrics.precision_recall_curve(labels, scores)
    precision_out, recall_out, _ = metrics.precision_recall_curve(1 - labels, -scores)

    actual = compute_ood_metrics(id_scores, ood_scores)

    assert actual["auroc"] == metrics.roc_auc_score(labels, scores)
    assert actual["ap_in"] == metrics.average_precision_score(labels, scores)
    assert actual["aupr_in_openood_auc"] == metrics.auc(recall_in, precision_in)
    assert actual["ap_out"] == metrics.average_precision_score(1 - labels, -scores)
    assert actual["aupr_out_openood_auc"] == metrics.auc(recall_out, precision_out)


def test_perfect_reversed_and_tied_auroc_fixtures():
    assert compute_ood_metrics([2, 3], [0, 1])["auroc"] == 1.0
    assert compute_ood_metrics([0, 1], [2, 3])["auroc"] == 0.0
    assert compute_ood_metrics([1, 1], [1, 1])["auroc"] == 0.5


def test_macro_averages_datasets_without_size_weighting():
    names = ("cifar100", "tin", "mnist", "svhn", "texture", "places365")
    per_dataset = {
        name: {"auroc": float(index), "fpr95_id_tpr": float(index) / 10.0}
        for index, name in enumerate(names)
    }

    payload = macro_average_ood_metrics(per_dataset)

    assert payload["near_macro_mean"]["auroc"] == 0.5
    assert payload["far_macro_mean"]["auroc"] == 3.5
    assert payload["overall_macro_mean"]["auroc"] == 2.5


@pytest.mark.parametrize("id_scores,ood_scores", [([], [0.1]), ([0.9], [])])
def test_metrics_reject_empty_id_or_ood_arrays(id_scores, ood_scores):
    with pytest.raises(ValueError, match="must not be empty"):
        compute_ood_metrics(id_scores, ood_scores)


def test_metrics_reject_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        compute_ood_metrics([0.9, np.nan], [0.1])
