import numpy as np

from oge.evaluation.classification import expected_calibration_error, top1_accuracy
from oge.evaluation.detectors import fit_tied_gaussians, score_tied_gaussians
from oge.evaluation.geometry import (
    covariance_spectrum,
    fit_geometry_statistics,
    neural_collapse_metrics,
)


def test_classification_metrics_are_sample_permutation_invariant():
    logits = np.array([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.2, 0.3]])
    labels = np.array([0, 1, 1, 1])
    permutation = np.array([2, 0, 3, 1])

    assert top1_accuracy(logits, labels)["metric"]["value"] == top1_accuracy(
        logits[permutation], labels[permutation]
    )["metric"]["value"]
    assert expected_calibration_error(logits, labels)["metric"]["value"] == expected_calibration_error(
        logits[permutation], labels[permutation]
    )["metric"]["value"]


def test_gaussian_scores_are_sample_class_permutation_and_batch_invariant():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(3), 8)
    features = rng.normal(size=(24, 5)) + np.eye(3, 5)[labels]
    queries = rng.normal(size=(7, 5))
    fit = fit_tied_gaussians(features, labels, num_classes=3)
    expected = score_tied_gaussians(fit, queries)

    sample_permutation = rng.permutation(len(features))
    permuted_fit = fit_tied_gaussians(
        features[sample_permutation], labels[sample_permutation], num_classes=3
    )
    np.testing.assert_allclose(
        score_tied_gaussians(permuted_fit, queries)["mahalanobis"],
        expected["mahalanobis"],
        atol=1e-12,
        rtol=1e-10,
    )

    class_map = np.array([2, 0, 1])
    class_fit = fit_tied_gaussians(features, class_map[labels], num_classes=3)
    np.testing.assert_allclose(
        score_tied_gaussians(class_fit, queries)["relative_mahalanobis"],
        expected["relative_mahalanobis"],
        atol=1e-12,
        rtol=1e-10,
    )

    batched = np.concatenate(
        [
            score_tied_gaussians(fit, queries[:3])["mahalanobis"],
            score_tied_gaussians(fit, queries[3:])["mahalanobis"],
        ]
    )
    np.testing.assert_allclose(batched, expected["mahalanobis"], atol=1e-12, rtol=1e-10)


def test_nc1_and_covariance_spectrum_are_orthogonal_rotation_invariant():
    rng = np.random.default_rng(8)
    labels = np.repeat(np.arange(3), 10)
    features = rng.normal(size=(30, 4)) + np.eye(3, 4)[labels]
    weight = rng.normal(size=(3, 4))
    orthogonal, _ = np.linalg.qr(rng.normal(size=(4, 4)))

    original = fit_geometry_statistics(features, labels, num_classes=3)
    rotated = fit_geometry_statistics(features @ orthogonal, labels, num_classes=3)
    original_nc = neural_collapse_metrics(original, weight)
    rotated_nc = neural_collapse_metrics(rotated, weight @ orthogonal)

    np.testing.assert_allclose(
        original_nc["nc1_pinv"]["value"],
        rotated_nc["nc1_pinv"]["value"],
        atol=1e-12,
        rtol=1e-10,
    )
    original_spectrum = covariance_spectrum(original.within_covariance)
    rotated_spectrum = covariance_spectrum(rotated.within_covariance)
    np.testing.assert_allclose(
        original_spectrum["eigenvalues"],
        rotated_spectrum["eigenvalues"],
        atol=1e-12,
        rtol=1e-10,
    )
