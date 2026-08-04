import numpy as np
import pytest
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from oge.evaluation.classification import aurc_tie_grouped
from oge.evaluation.detectors import fit_neco, fit_vim, score_neco
from oge.evaluation.geometry import twonn


VIM_COMMIT = "dabf9e5b242dbd31c15e09ff12af3d11f009f79c"
NECO_COMMIT = "6a55640669f0aad3e23f45ce2f6a8e6400c929ba"
FD_SHIFTS_COMMIT = "c4467aec134e99691359da209f811d91283fc1e3"
DADAPY_COMMIT = "c37e52ca3cbe00107696720e5e3efa09f8370bbb"


def test_vim_matches_pinned_assume_centered_covariance_primitive():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(12, 6)) + 2.0
    weight = rng.normal(size=(3, 6))
    bias = rng.normal(size=3)
    logits = features @ weight.T + bias
    fit = fit_vim(features, logits, weight, bias, dim=3)

    origin = -np.linalg.pinv(weight) @ bias
    pinned_covariance = EmpiricalCovariance(assume_centered=True).fit(
        features - origin
    ).covariance_

    assert VIM_COMMIT.startswith("dabf9e5b")
    np.testing.assert_allclose(fit.origin, origin, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(fit.covariance, pinned_covariance, atol=1e-12, rtol=1e-10)


def test_neco_matches_pinned_standardscaler_pca_ratio_primitive():
    rng = np.random.default_rng(9)
    train = rng.normal(size=(140, 100))
    query = rng.normal(size=(5, 100))
    fit = fit_neco(train, dim=100)
    actual = score_neco(fit, query)["score"]

    scaler = StandardScaler().fit(train)
    standardized_train = scaler.transform(train)
    pca = PCA(n_components=100, svd_solver="full").fit(standardized_train)
    standardized_query = scaler.transform(query)
    expected = np.linalg.norm(pca.transform(standardized_query), axis=1) / np.linalg.norm(
        standardized_query, axis=1
    )

    assert NECO_COMMIT.startswith("6a556406")
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)


def test_fd_shifts_tie_grouped_aurc_parity_after_removing_display_scale():
    logits = np.array(
        [[2.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    )
    labels = np.array([0, 1, 0, 1, 1])
    actual = aurc_tie_grouped(logits, labels)

    confidence = actual["thresholds"]
    coverage = actual["coverage"]
    risk = actual["risk"]
    pinned_display_value = -1000.0 * np.trapz(risk, coverage)

    assert FD_SHIFTS_COMMIT.startswith("c4467aec")
    assert confidence.size == 3
    assert actual["metric"]["value"] == pytest.approx(pinned_display_value / 1000.0)


def test_twonn_matches_pinned_base_origin_ols_formula():
    points = np.array([0.0, 0.8, 2.1, 4.7, 9.8, 20.2, 41.0, 82.5, 166.0, 333.0])[:, None]
    sample_ids = np.asarray([f"p-{index}" for index in range(len(points))])
    actual = twonn(points, sample_ids=sample_ids, mu_fraction=0.9)

    pairwise = np.abs(points - points.T)
    np.fill_diagonal(pairwise, np.inf)
    nearest = np.sort(pairwise, axis=1)[:, :2]
    log_mu = np.sort(np.log(nearest[:, 1] / nearest[:, 0]))
    n_eff = int(np.floor(0.9 * len(points)))
    x = log_mu[:n_eff]
    y = -np.log(1.0 - np.arange(1, n_eff + 1) / len(points))
    expected = float((x @ y) / (x @ x))

    assert DADAPY_COMMIT.startswith("c37e52ca")
    assert actual["metric"]["value"] == pytest.approx(expected)
