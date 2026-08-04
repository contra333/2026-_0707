import numpy as np
import pytest

from oge.evaluation.geometry import (
    all_covariance_spectra,
    cdnv,
    class_pair_geometry,
    covariance_spectrum,
    exact_self_neighbor_distances,
    feature_norm_distribution,
    fit_geometry_statistics,
    hypersphere_alignment_uniformity,
    local_intrinsic_dimensionality,
    neural_collapse_metrics,
    rankme,
    twonn,
)


def _simplex_fixture():
    means = np.array(
        [
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [-0.5, -np.sqrt(3.0) / 2.0],
        ]
    )
    features = np.repeat(means, 2, axis=0)
    labels = np.repeat(np.arange(3), 2)
    return features, labels, means


def test_regular_simplex_validates_nc1_through_nc4():
    features, labels, weight = _simplex_fixture()
    statistics = fit_geometry_statistics(features, labels, num_classes=3)
    logits = features @ weight.T

    payload = neural_collapse_metrics(
        statistics,
        weight,
        np.zeros(3),
        query_features=features,
        query_logits=logits,
        query_labels=labels,
    )

    assert payload["nc0_row_sum_raw"]["value"] < 1e-12
    assert payload["nc1_pinv"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert payload["nc2_equinorm"]["value"] < 1e-12
    assert payload["nc2_equiangular"]["value"] < 1e-8
    assert payload["nc2_etf_raw"]["value"] < 1e-12
    assert payload["nc3_self_duality_raw"]["value"] < 1e-12
    assert payload["nc4"]["agreement_with_bias"]["value"] == 1.0
    assert payload["nc4"]["agreement_without_bias"]["value"] == 1.0


def test_nc2_equinorm_uses_sample_standard_deviation_correction_one():
    means = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -2.0]])
    features = np.repeat(means, 2, axis=0)
    labels = np.repeat(np.arange(3), 2)
    statistics = fit_geometry_statistics(features, labels, num_classes=3)
    payload = neural_collapse_metrics(statistics, statistics.centered_class_means)
    norms = np.linalg.norm(statistics.centered_class_means, axis=1)

    expected = np.std(norms, ddof=1) / np.mean(norms)
    assert payload["nc2_equinorm"]["value"] == pytest.approx(expected)


def test_diagonal_covariance_has_three_distinct_rank_values():
    payload = covariance_spectrum(np.diag([4.0, 1.0]))
    probabilities = np.array([0.8, 0.2])
    expected_entropy = np.exp(-np.sum(probabilities * np.log(probabilities)))

    assert payload["entropy_rank"]["value"] == pytest.approx(expected_entropy)
    assert payload["trace_top_rank"]["value"] == pytest.approx(5.0 / 4.0)
    assert payload["participation_ratio"]["value"] == pytest.approx(25.0 / 17.0)
    assert len(
        {
            payload["entropy_rank"]["value"],
            payload["trace_top_rank"]["value"],
            payload["participation_ratio"]["value"],
        }
    ) == 3


def test_covariance_spectrum_failure_and_small_negative_clipping():
    failed = covariance_spectrum(np.zeros((2, 2)))
    assert failed["status"] == "failed"
    assert failed["reason_codes"] == ["all_zero_spectrum"]

    clipped = covariance_spectrum(np.diag([1.0, -1.0e-17]))
    assert clipped["status"] == "success"
    assert clipped["negative_clip_count"] == 1

    indefinite = covariance_spectrum(np.diag([1.0, -1.0e-3]))
    assert indefinite["status"] == "failed"


def test_rankme_is_uncentered_singular_value_entropy():
    equal = rankme(np.eye(2))
    rank_one = rankme(np.array([[1.0, 0.0], [1.0, 0.0]]))

    assert equal["metric"]["value"] > rank_one["metric"]["value"]
    assert equal["centered"] is False


def test_feature_norm_cdnv_and_centered_pair_angle_invariances():
    features = np.array([[1.0, 0.0], [1.2, 0.0], [-1.0, 0.0], [-1.2, 0.0]])
    labels = np.array([0, 0, 1, 1])
    statistics = fit_geometry_statistics(features, labels)
    shifted = fit_geometry_statistics(features + np.array([4.0, -7.0]), labels)
    scaled = fit_geometry_statistics(features * 3.0, labels)

    assert feature_norm_distribution(statistics)["metric"]["status"] == "success"
    assert cdnv(statistics)["metric"]["value"] == pytest.approx(cdnv(shifted)["metric"]["value"])
    assert cdnv(statistics)["metric"]["value"] == pytest.approx(cdnv(scaled)["metric"]["value"])
    np.testing.assert_allclose(
        class_pair_geometry(statistics)["centered_cosine"],
        class_pair_geometry(shifted)["centered_cosine"],
    )


def test_lid_and_twonn_duplicate_failure_semantics():
    line = np.array([0.0, 1.0, 3.0, 7.0, 15.0, 31.0, 63.0, 127.0])[:, None]
    ids = np.asarray([f"sample-{index}" for index in range(len(line))])
    lid = local_intrinsic_dimensionality(line, sample_ids=ids, k_values=(2,))
    base_twonn = twonn(line, sample_ids=ids)

    assert lid["2"]["metric"]["status"] == "success"
    assert np.isfinite(lid["2"]["metric"]["value"])
    assert base_twonn["metric"]["status"] == "success"

    duplicate = line.copy()
    duplicate[1] = duplicate[0]
    duplicate_lid = local_intrinsic_dimensionality(duplicate, sample_ids=ids, k_values=(2,))
    duplicate_twonn = twonn(duplicate, sample_ids=ids)
    assert duplicate_lid["2"]["metric"]["status"] == "degenerate"
    assert duplicate_twonn["metric"]["status"] == "degenerate"
    assert duplicate_twonn["metric"]["value"] is None


def test_lid_and_twonn_precomputed_exact_neighbors_match_independent_calls():
    rng = np.random.default_rng(21)
    features = rng.normal(size=(18, 4))
    ids = np.asarray([f"sample-{index:02d}" for index in range(len(features))])
    neighbors = exact_self_neighbor_distances(
        features,
        k=5,
        sample_ids=ids,
        query_chunk_size=3,
        bank_chunk_size=4,
    )
    independent_lid = local_intrinsic_dimensionality(
        features, sample_ids=ids, k_values=(2, 5), query_chunk_size=7, bank_chunk_size=8
    )
    cached_lid = local_intrinsic_dimensionality(
        features, sample_ids=ids, k_values=(2, 5), precomputed_neighbors=neighbors
    )
    for k in ("2", "5"):
        assert cached_lid[k]["metric"] == independent_lid[k]["metric"]
        np.testing.assert_allclose(cached_lid[k]["per_sample"], independent_lid[k]["per_sample"])

    independent_twonn = twonn(
        features, sample_ids=ids, query_chunk_size=7, bank_chunk_size=8
    )
    cached_twonn = twonn(features, sample_ids=ids, precomputed_neighbors=neighbors)
    assert cached_twonn["metric"] == independent_twonn["metric"]
    np.testing.assert_allclose(cached_twonn["mu"], independent_twonn["mu"])


def test_hypersphere_alignment_and_uniformity_are_fixed_seed_deterministic():
    features, labels, _ = _simplex_fixture()
    statistics = fit_geometry_statistics(features, labels, num_classes=3)

    first = hypersphere_alignment_uniformity(statistics, pair_count=500, seeds=(0, 1, 2))
    second = hypersphere_alignment_uniformity(statistics, pair_count=500, seeds=(0, 1, 2))

    assert first["class_alignment"]["value"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_array_equal(first["uniformity_repeats"], second["uniformity_repeats"])


def test_all_covariance_spectra_keeps_within_total_between_separate():
    features, labels, _ = _simplex_fixture()
    statistics = fit_geometry_statistics(features, labels, num_classes=3)
    payload = all_covariance_spectra(statistics)

    assert set(payload) == {"sw", "total", "between"}
    assert payload["sw"]["status"] == "failed"
    assert payload["between"]["status"] == "success"
