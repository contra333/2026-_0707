import json

import numpy as np
import pytest

from oge.analysis.discriminant_residual_preflight import (
    _contract_scaled_max,
    _gate2_summary,
    _json_safe,
    _raw_norm_component_correlations,
    _rmd_cancellation_scaled_max,
    build_v1_v2_diff,
    deterministic_class_stratified_twofold,
    fit_discriminant_geometry,
    fit_residual_nu,
    gaussian_residual_nll,
    residual_t_nll,
    rtmd_residual_penalty,
    run_preflight,
    score_discriminant_components,
)


def _full_rank_fixture(seed=20260812, sample_count=1800, dimension=8, classes=3):
    rng = np.random.default_rng(seed)
    labels = np.arange(sample_count, dtype=np.int64) % classes
    directions = rng.normal(size=(classes, dimension))
    directions -= np.mean(directions, axis=0, keepdims=True)
    noise = rng.normal(size=(sample_count, dimension))
    transform = rng.normal(size=(dimension, dimension))
    features = noise @ transform + directions[labels]
    return features.astype(np.float64), labels


def test_stable_twofold_is_deterministic_unique_and_class_balanced():
    labels = np.repeat(np.arange(4), [11, 12, 13, 14])
    sample_ids = np.asarray([f"c{label}/sample-{index}" for index, label in enumerate(labels)])

    first = deterministic_class_stratified_twofold(sample_ids, labels)
    second = deterministic_class_stratified_twofold(sample_ids[::-1], labels[::-1])[::-1]

    np.testing.assert_array_equal(first, second)
    for class_index in range(4):
        counts = np.bincount(first[labels == class_index], minlength=2)
        assert abs(int(counts[0]) - int(counts[1])) <= 1
    with pytest.raises(ValueError, match="unique"):
        deterministic_class_stratified_twofold(
            np.asarray(["duplicate", "duplicate"]), np.asarray([0, 0])
        )


def test_twofold_partition_keeps_the_v1_namespace_snapshot():
    sample_ids = np.asarray([f"id-{index}" for index in range(12)])
    labels = np.repeat(np.arange(3), 4)

    folds = deterministic_class_stratified_twofold(sample_ids, labels)

    assert folds.tolist() == [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]


def test_discriminant_components_reconstruct_scores_covariance_and_energy():
    features, labels = _full_rank_fixture()
    fit = fit_discriminant_geometry(features, labels)
    record, arrays = score_discriminant_components(fit, features)

    assert fit.applicable
    assert fit.dim <= 2
    assert fit.residual_dim == features.shape[1] - fit.dim
    assert all(fit.numerical["checks"].values())
    assert all(record["checks"].values())
    assert np.mean(arrays["q_perp"]) == pytest.approx(
        fit.residual_dim, rel=fit.tau_alg, abs=fit.tau_alg
    )
    np.testing.assert_allclose(
        arrays["md"], np.sum(arrays["components"], axis=1), rtol=fit.tau_alg
    )
    np.testing.assert_allclose(
        arrays["md"], arrays["marginal"] + arrays["rmd"], rtol=fit.tau_alg
    )
    json.dumps(fit.numerical, allow_nan=False)
    json.dumps(record, allow_nan=False)


def test_rank_deficient_fit_is_preserved_but_not_theorem_applicable():
    features, labels = _full_rank_fixture(dimension=6)
    duplicated = np.column_stack((features, features[:, 0]))
    fit = fit_discriminant_geometry(duplicated, labels)

    assert not fit.applicable
    assert fit.numerical["within_spectrum"]["rank"] < duplicated.shape[1]


def test_residual_t_likelihood_and_penalty_match_frozen_formulas():
    q = np.asarray([0.0, 1.0, 5.0, 20.0])
    k = 7
    nu = 5.5

    t_nll = residual_t_nll(q, k, nu)
    gaussian_nll = gaussian_residual_nll(q, k)
    assert np.isfinite(t_nll).all()
    assert np.isfinite(gaussian_nll).all()
    np.testing.assert_array_equal(rtmd_residual_penalty(q, k, None), q)
    np.testing.assert_array_equal(rtmd_residual_penalty(q, k, np.inf), q)
    np.testing.assert_allclose(
        rtmd_residual_penalty(q, k, nu),
        (nu + k) * np.log1p(q / (nu - 2.0)),
    )


def test_deterministic_nu_fitter_selects_heavy_tail_and_falls_back_for_gaussian():
    rng = np.random.default_rng(20260812)
    k = 12
    nu = 5.0
    t_vectors = (
        rng.standard_t(nu, size=(30_000, k)) * np.sqrt((nu - 2.0) / nu)
    )
    heavy = fit_residual_nu(np.sum(t_vectors * t_vectors, axis=1), k)
    gaussian = fit_residual_nu(rng.chisquare(k, size=30_000), k)

    assert heavy["selected_model"] == "finite_t"
    assert 2.05 < heavy["selected_nu"] < 1000.0
    assert heavy["likelihood_ratio"] > heavy["bic_threshold_log_n"]
    assert gaussian["selected_model"] == "gaussian_infinity"
    assert gaussian["selected_nu"] == "infinity"


def test_nu_fitter_is_bitwise_deterministic_for_identical_input():
    rng = np.random.default_rng(73)
    q = rng.chisquare(9, size=4000)

    assert fit_residual_nu(q, 9) == fit_residual_nu(q.copy(), 9)


def test_gate2_is_inconclusive_when_any_required_fit_is_numerically_inapplicable():
    complete = {
        "transform": "raw",
        "status": "PASS",
        "nu_fit": {"selected_model": "finite_t", "selected_nu": 8.0},
        "heldout_id_test": {
            "selected_gain_per_sample": 0.1,
            "chi2_ks_statistic": 0.2,
            "variance_to_chi2_ratio": 1.5,
            "chi2_upper_tail_exceedance": {"0.99": 0.02},
        },
    }
    records = [
        complete,
        {
            **complete,
            "nu_fit": {"selected_model": "finite_t", "selected_nu": 10.0},
        },
        {
            "transform": "raw",
            "status": "INCONCLUSIVE_NUMERICAL",
            "nu_fit": None,
            "heldout_id_test": None,
        },
        {**complete, "transform": "l2"},
        {
            **complete,
            "transform": "l2",
            "nu_fit": {"selected_model": "finite_t", "selected_nu": 12.0},
        },
    ]

    summary = _gate2_summary(records)

    assert summary["status"] == "DESCRIPTIVE_ONLY"
    assert summary["gate_status"] == "INCONCLUSIVE"
    assert summary["aggregate_decision_rule"] == "PERMANENTLY_NOT_ADJUDICATED"
    assert not summary["retroactive_threshold_or_readjudication"]
    assert summary["by_transform"]["raw"]["inconclusive_fit_count"] == 1
    assert "pass" not in summary["by_transform"]["raw"]


def test_reconstruction_scale_uses_sum_of_absolute_components():
    residual = np.asarray([1.0, 1.0])
    direct = np.asarray([1.0, 10.0])
    first = np.asarray([2.0, 3.0])
    second = np.asarray([3.0, 4.0])

    result = _contract_scaled_max(residual, direct, first, second)

    assert result == pytest.approx(1.0 / 5.0)


def test_rmd_cancellation_uses_only_the_card13_v11_operand_scale():
    residual = np.asarray([1.0, 2.0])
    direct_global = np.asarray([100.0, 10.0])
    direct_class = np.asarray([99.0, 5.0])
    q_parallel_global = np.asarray([4.0, 20.0])
    q_parallel_class = np.asarray([3.0, 10.0])

    result = _rmd_cancellation_scaled_max(
        residual,
        direct_global,
        direct_class,
        q_parallel_global,
        q_parallel_class,
    )

    assert result == pytest.approx(2.0 / 30.0)


def test_raw_norm_correlations_are_recorded_for_train_and_test():
    train = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    test = np.asarray([[1.0, 1.0], [0.0, 3.0], [4.0, 0.0]])
    train_components = np.column_stack((np.arange(3), np.arange(3) ** 2, -np.arange(3)))
    test_components = np.column_stack((np.arange(3) + 1, np.arange(3) ** 2, -np.arange(3)))

    correlations = _raw_norm_component_correlations(
        train, test, train_components, test_components
    )

    assert set(correlations) == {"id_train", "id_test"}
    assert set(correlations["id_train"]) == {
        "s_perp",
        "s_parallel_marginal",
        "s_rmd",
    }
    assert set(correlations["id_test"]) == set(correlations["id_train"])


def test_v1_v2_diff_requires_and_conserves_exactly_60_fit_keys():
    def record(index, *, v2, rescued=False):
        passed = bool(rescued)
        return {
            "bundle_id": f"{index:064x}",
            "transform": "raw" if index % 2 == 0 else "l2",
            "status": "PASS" if passed else "INAPPLICABLE",
            "applicable": True,
            "required_numerical_pass": passed,
            "primary_channel": "S-perp" if passed else "none",
            "id_train": {
                "rmd_cancellation_scaled_max": 1.0e-12 if passed else 1.0e-5,
                "rmd_cancellation_scale": "operand_aware_card13_v11" if v2 else None,
                "checks": {"rmd_cancellation": passed},
            },
            "id_test": {
                "rmd_cancellation_scaled_max": 1.0e-12 if passed else 1.0e-5,
                "checks": {"rmd_cancellation": passed},
            },
            "cached_metric_v1_2_id_test_parity": {
                "pass": passed,
                "scaled_max_residuals": {"rmd": 1.0e-12 if passed else 1.0e-5},
                **(
                    {"legacy_output_scaled_max_residuals": {"rmd": 1.0e-5}}
                    if v2
                    else {}
                ),
            },
            "raw_norm_component_correlations": (
                {"id_train": {}, "id_test": {}} if index % 2 == 0 else None
            ),
        }

    before = [record(index, v2=False) for index in range(60)]
    after = [
        record(index, v2=True, rescued=index == 0) for index in range(60)
    ]

    records, summary = build_v1_v2_diff(before, after)

    assert len(records) == 60
    assert summary["fit_count"] == 60
    assert summary["changed_count_by_field"]["required_numerical_pass"] == 1
    assert summary["changed_count_by_field"]["primary_channel"] == 1
    assert not summary["historical_gate_2_re_adjudicated"]


def test_json_safe_preserves_nonfinite_numeric_sentinels_without_invalid_json():
    value = _json_safe(
        {"positive": np.inf, "negative": -np.inf, "missing": np.nan, "ok": 1.5}
    )

    assert value == {
        "positive": "infinity",
        "negative": "-infinity",
        "missing": "nan",
        "ok": 1.5,
    }
    json.dumps(value, allow_nan=False)


def test_missing_required_cache_stops_with_compact_inconclusive(tmp_path):
    manifest = {
        "schema_version": "discriminant_residual_preflight_inputs_v1",
        "source_reuse_manifest_sha256": (
            "f814f1ab3070a2dc4f0d541746bbd05c3ad74d761bd5c8fa88bc859f308318dc"
        ),
        "expected_bundle_count": 1,
        "bundles": [
            {
                "bundle_id": "0" * 64,
                "bundle_checksums_sha256": "1" * 64,
                "config_hash": "2" * 64,
                "optimizer_family": "adam",
                "training_seed": 0,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_preflight(
        input_manifest_path=manifest_path,
        cache_root=tmp_path / "absent-cache",
        output_dir=tmp_path / "output",
    )

    assert result["status"] == "INCONCLUSIVE_INPUT"
    assert result["gates"]["gate_2"] == "INCONCLUSIVE"
    assert "required bundle is absent" in result["reason"]
    assert not (tmp_path / "output" / "bundle_records.jsonl").exists()
    assert (tmp_path / "output" / "summary.json.sha256").is_file()
