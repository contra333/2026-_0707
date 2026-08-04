import numpy as np
import pytest
import torch

from oge.evaluation.classification import (
    augrc,
    aurc_tie_grouped,
    evaluate_temperature_scaled,
    expected_calibration_error,
    fit_temperature,
    logit_ood_scores,
    misclassification_auroc,
    negative_log_likelihood,
    top1_accuracy,
)


def test_perfect_and_all_wrong_classification_hand_fixtures():
    perfect = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
    labels = np.array([0, 1])
    wrong = perfect[:, ::-1]

    assert top1_accuracy(perfect, labels)["metric"]["value"] == 1.0
    assert negative_log_likelihood(perfect, labels)["metric"]["value"] == 0.0
    assert expected_calibration_error(perfect, labels, bins=15)["metric"]["value"] == 0.0
    assert top1_accuracy(wrong, labels)["metric"]["value"] == 0.0
    assert misclassification_auroc(perfect, labels)["metric"]["status"] == "degenerate"
    assert misclassification_auroc(wrong, labels)["metric"]["status"] == "degenerate"
    assert augrc(perfect, labels)["metric"]["status"] == "degenerate"


def test_identical_confidence_is_one_aurc_threshold_group():
    logits = np.zeros((4, 2), dtype=np.float64)
    labels = np.array([0, 1, 1, 0])

    payload = aurc_tie_grouped(logits, labels)

    assert payload["thresholds"].tolist() == [0.5]
    assert payload["group_counts"].tolist() == [4]
    assert payload["coverage"].tolist() == [1.0, 0.0]
    assert payload["metric"]["value"] == 0.5


def test_ece_bin_boundary_is_right_inclusive_after_first_bin():
    confidence = 0.75
    logit = np.log(confidence / (1.0 - confidence))
    payload = expected_calibration_error([[logit, 0.0]], [0], bins=4)

    assert payload["bins"][2]["count"] == 1
    assert payload["bins"][3]["count"] == 0


def test_logit_controls_use_raw_logits_and_energy_t1():
    logits = np.array([[1.0, 2.0], [3.0, -1.0]])
    scores = logit_ood_scores(logits)

    np.testing.assert_allclose(scores["max_logit"], [2.0, 3.0])
    np.testing.assert_allclose(scores["energy_t1"], np.log(np.exp(logits).sum(axis=1)))


def test_temperature_fit_is_positive_and_preserves_argmax():
    logits = np.array([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 0, 1])

    fit = fit_temperature(logits, labels)
    evaluated = evaluate_temperature_scaled(logits, labels, fit)

    assert 0.01 <= fit.temperature <= 100.0
    assert evaluated["temperature"]["value"] == fit.temperature
    np.testing.assert_array_equal(np.argmax(logits, axis=1), np.argmax(logits / fit.temperature, axis=1))


def test_temperature_lbfgs_exception_uses_geomspace_fallback(monkeypatch):
    def fail_step(self, closure):
        raise RuntimeError("fixture")

    monkeypatch.setattr(torch.optim.LBFGS, "step", fail_step)
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    fit = fit_temperature(logits, [0, 1])

    assert fit.fallback_used is True
    assert fit.temperature in np.geomspace(0.01, 100.0, 2001)
