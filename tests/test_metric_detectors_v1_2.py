import math

import numpy as np
import pytest

from oge.evaluation.detectors import (
    exact_knn_l2_scores,
    fit_gda_class_density,
    fit_neco,
    NECOFit,
    fit_tied_gaussians,
    fit_vim,
    gda_jitter_candidates,
    prototype_scores,
    score_gda_class_density,
    score_neco,
    score_tied_gaussians,
    score_vim,
)


def test_one_dimensional_two_class_mahalanobis_and_relative_components():
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([0, 0, 1, 1])
    fit = fit_tied_gaussians(features, labels)

    payload = score_tied_gaussians(fit, [[-1.5]])

    np.testing.assert_allclose(fit.class_means[:, 0], [-1.5, 1.5])
    np.testing.assert_allclose(fit.within_covariance, [[0.25]])
    assert payload["mahalanobis"][0] == pytest.approx(0.0)
    assert payload["global_distances"][0] == pytest.approx(0.9)
    assert payload["relative_mahalanobis"][0] == pytest.approx(0.9)


def test_gda_uses_unbiased_class_covariance_priors_and_full_log_density():
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels = np.array([0, 0, 1, 1])
    fit = fit_gda_class_density(features, labels)

    payload = score_gda_class_density(fit, [[-1.5]])
    class0_density = 1.0 / math.sqrt(2.0 * math.pi * 0.5)
    class1_density = class0_density * math.exp(-0.5 * 3.0**2 / 0.5)
    expected = math.log(0.5 * class0_density + 0.5 * class1_density)

    np.testing.assert_allclose(fit.class_covariances[:, 0, 0], [0.5, 0.5])
    assert fit.selected_jitter == 0.0
    assert payload["score"][0] == pytest.approx(expected)


def test_gda_jitter_order_matches_pinned_contract():
    candidates = gda_jitter_candidates()

    assert candidates[0] == 0.0
    assert candidates[1] == np.finfo(np.float64).tiny
    assert candidates[2] == 1.0e-308
    assert candidates[-1] == 1.0e-1
    assert len(candidates) == 310


def test_mahalanobis_pp_refits_in_normalized_space_and_reports_near_zero():
    features = np.array([[1.0, 0.0], [2.0, 1.0], [-1.0, 0.0], [-2.0, -1.0]])
    labels = np.array([0, 0, 1, 1])
    fit = fit_tied_gaussians(features, labels, normalize=True)

    assert fit.normalized is True
    assert fit.normalization_status == "success"
    assert not np.allclose(fit.class_means, [[1.5, 0.5], [-1.5, -0.5]])

    near = features.copy()
    near[0] = [1.0e-13, 0.0]
    near_fit = fit_tied_gaussians(near, labels, normalize=True)
    assert near_fit.normalization_status == "degenerate"


def test_normalized_detectors_fail_on_exact_zero_before_epsilon():
    features = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    with pytest.raises(ValueError, match="exact-zero"):
        fit_tied_gaussians(features, [0, 0, 1, 1], normalize=True)

    with pytest.raises(ValueError, match="exact-zero"):
        exact_knn_l2_scores(
            features,
            [[1.0, 0.0]],
            fit_sample_ids=["a", "b", "c", "d"],
            k=2,
        )


def test_knn_is_negative_exact_kth_squared_distance_with_sample_id_tie_break():
    bank = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    payload = exact_knn_l2_scores(
        bank,
        [[1.0, 0.0]],
        fit_sample_ids=["z", "b", "a"],
        k=2,
    )

    assert payload["kth_squared_distance"][0] == pytest.approx(2.0)
    assert payload["score"][0] == pytest.approx(-2.0)
    assert payload["kth_neighbor_sample_id"][0] == "a"


def test_prototype_cosine_and_weight_cosine_keep_sources_distinct():
    fit_features = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    labels = np.array([0, 0, 1, 1])
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    payload = prototype_scores(
        fit_features,
        labels,
        [[1.0, 0.0]],
        classifier_weight=weights,
    )

    assert payload["ctm_class"][0] == 0
    assert payload["weight_cosine_class"][0] == 1
    assert payload["ncp_class"][0] == 0


def test_vim_covariance_has_no_additional_sample_mean_centering():
    features = np.array(
        [[2.0, 0.0, 1.0, 0.0], [2.0, 1.0, 0.0, 0.0], [3.0, 0.0, 0.0, 1.0], [3.0, 1.0, 1.0, 1.0]]
    )
    weight = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    bias = np.array([-1.0, 0.5])
    logits = features @ weight.T + bias

    fit = fit_vim(features, logits, weight, bias, dim=2)
    shifted = features - fit.origin
    author_covariance = shifted.T @ shifted / len(shifted)
    centered_covariance = np.cov(shifted, rowvar=False, bias=True)

    np.testing.assert_allclose(fit.covariance, author_covariance)
    assert not np.allclose(fit.covariance, centered_covariance)
    assert fit.residual_subspace.shape == (4, 2)
    scored = score_vim(fit, features, logits)
    np.testing.assert_allclose(
        scored["score"], scored["energy"] - scored["alpha_residual"]
    )


def test_neco_matches_standard_scaler_then_fixed_pca_ratio_without_maxlogit():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(120, 100)) + np.linspace(-2.0, 2.0, 100)
    query = rng.normal(size=(3, 100))
    fit = fit_neco(features, dim=100)

    payload = score_neco(fit, query)
    standardized = (query - fit.scaler_mean) / fit.scaler_scale
    coordinates = (standardized - fit.pca_mean) @ fit.pca_components.T
    expected = np.linalg.norm(coordinates, axis=1) / np.linalg.norm(standardized, axis=1)

    np.testing.assert_allclose(payload["score"], expected, atol=1e-12, rtol=1e-10)
    assert payload["max_logit_used"] is False


def test_neco_rejects_too_few_dimensions_but_standardscaler_allows_constant_feature():
    with pytest.raises(ValueError, match="N and p"):
        fit_neco(np.ones((101, 99)), dim=100)
    features = np.arange(10100, dtype=np.float64).reshape(101, 100)
    features[:, 0] = 1.0
    fit = fit_neco(features, dim=100)
    assert fit.scaler_scale[0] == 1.0


def test_neco_rejects_an_invalid_fitted_zero_scale():
    invalid = NECOFit(
        scaler_mean=np.zeros(2),
        scaler_scale=np.array([1.0, 0.0]),
        scaler_variance=np.ones(2),
        pca_mean=np.zeros(2),
        pca_components=np.eye(2),
        pca_explained_variance=np.ones(2),
        dim=2,
    )
    with pytest.raises(ValueError, match="zero"):
        score_neco(invalid, [[1.0, 1.0]])
