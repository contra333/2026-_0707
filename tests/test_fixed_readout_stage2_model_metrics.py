from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from oge.evaluation.classification import logit_ood_scores
from oge.evaluation.common import normalize_rows
from oge.evaluation.detectors import fit_tied_gaussians, score_tied_gaussians
from oge.evaluation.metrics import compute_ood_metrics


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "extract_fixed_readout_stage2_model_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("stage2_model_metrics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE2
SPEC.loader.exec_module(STAGE2)


def _brute_transitions(raw_id, raw_ood, transformed_id, transformed_ood):
    state = lambda left, right: (
        "correct" if left > right else "tie" if left == right else "misordered"
    )
    counts = {key: 0 for key in STAGE2.TRANSITION_FIELDS}
    for raw_left, transformed_left in zip(raw_id, transformed_id):
        for raw_right, transformed_right in zip(raw_ood, transformed_ood):
            counts[
                f"{state(raw_left, raw_right)}_to_"
                f"{state(transformed_left, transformed_right)}"
            ] += 1
    return counts


@pytest.mark.parametrize("seed", range(12))
def test_exact_pair_order_transition_sweep_matches_brute_force_with_ties(seed):
    rng = np.random.default_rng(seed)
    raw_id = rng.integers(-2, 3, size=7).astype(np.float64)
    raw_ood = rng.integers(-2, 3, size=9).astype(np.float64)
    transformed_id = rng.integers(-2, 3, size=7).astype(np.float64)
    transformed_ood = rng.integers(-2, 3, size=9).astype(np.float64)

    observed = STAGE2.exact_pair_order_transitions(
        raw_id, raw_ood, transformed_id, transformed_ood
    )
    expected = _brute_transitions(
        raw_id, raw_ood, transformed_id, transformed_ood
    )

    assert observed["counts"] == expected
    assert observed["pair_count"] == len(raw_id) * len(raw_ood)
    assert observed["pair_credit_identity_pass"] is True
    raw_metrics = compute_ood_metrics(raw_id, raw_ood)
    transformed_metrics = compute_ood_metrics(transformed_id, transformed_ood)
    assert observed["raw_auroc_from_pairs"] == pytest.approx(
        raw_metrics["auroc"], abs=1e-12
    )
    assert observed["transformed_auroc_from_pairs"] == pytest.approx(
        transformed_metrics["auroc"], abs=1e-12
    )


def _metric_values(id_scores, ood_scores):
    values = compute_ood_metrics(id_scores, ood_scores)
    return {
        "auroc": values["auroc"],
        "fpr95": values["fpr95_id_tpr"],
        "fpr95_threshold": values["fpr95_threshold"],
        "achieved_id_tpr": values["fpr95_achieved_id_tpr"],
    }


def _baseline_rows(values):
    return {name: {"value": value} for name, value in values.items()}


def _synthetic_bundle(seed: int, bundle_label: str):
    rng = np.random.default_rng(seed)
    feature_dim = 6
    class_count = 2
    counts = {"id_train": 60, "id_test": 11, "near_ood": 9, "far_ood": 10}
    centers = np.asarray(
        [[-1.3, 0.2, 0.5, -0.1, 0.4, -0.6], [1.2, -0.3, -0.4, 0.2, -0.5, 0.7]],
        dtype=np.float64,
    )
    weight = rng.normal(size=(class_count, feature_dim))
    bias = rng.normal(size=class_count)
    splits = {}
    for split_index, (split_name, count) in enumerate(counts.items()):
        if split_name.startswith("id_"):
            labels = np.resize(np.arange(class_count), count)
            features = centers[labels] + rng.normal(scale=0.45, size=(count, feature_dim))
        else:
            labels = None
            features = rng.normal(
                loc=0.35 + 0.7 * split_index,
                scale=0.9,
                size=(count, feature_dim),
            )
        # Deterministic radial variation ensures the required non-invariant
        # witness has a meaningful synthetic signal.
        features *= np.linspace(0.65, 1.45, count)[:, None]
        logits = features @ weight.T + bias
        arrays = {
            "features": features.astype(np.float64),
            "logits": logits.astype(np.float64),
            "sample_ids": np.asarray(
                [f"{bundle_label}:{split_name}:{index:04d}" for index in range(count)]
            ),
        }
        if labels is not None:
            arrays["class_labels"] = labels.astype(np.int64)
        splits[split_name] = arrays

    train = splits["id_train"]
    raw_fit = fit_tied_gaussians(
        train["features"], train["class_labels"], num_classes=class_count
    )
    pp_fit = fit_tied_gaussians(
        train["features"],
        train["class_labels"],
        num_classes=class_count,
        normalize=True,
    )
    vim_fit = STAGE2.fit_vim(
        train["features"],
        train["logits"],
        weight,
        bias,
        dim=None,
    )
    normalized_bank, _, _, _ = normalize_rows(
        train["features"], denominator_epsilon=1e-10
    )
    scores = {}
    diagnostics = {}
    for split_name in ("id_test", "near_ood", "far_ood"):
        query = splits[split_name]
        raw = score_tied_gaussians(raw_fit, query["features"])
        pp = score_tied_gaussians(pp_fit, query["features"])
        normalized_query, _, _, _ = normalize_rows(
            query["features"], denominator_epsilon=1e-10
        )
        knn = STAGE2.exact_raw_knn_topk(
            normalized_bank,
            normalized_query,
            fit_sample_ids=train["sample_ids"],
            k=50,
            query_chunk_size=4,
            bank_chunk_size=53,
        )
        energy = logit_ood_scores(query["logits"])["energy_t1"]
        vim = STAGE2.score_vim(vim_fit, query["features"], query["logits"])
        ctm = STAGE2.prototype_scores(
            train["features"],
            train["class_labels"],
            query["features"],
            classifier_weight=weight,
            num_classes=class_count,
        )
        scores[split_name] = {
            "detector/mahalanobis_raw": raw["mahalanobis"],
            "detector/mahalanobis_pp": pp["mahalanobis"],
            "detector/knn_l2_k50": knn["score"],
            "detector/ctm_prototype_cosine": ctm["ctm_score"],
            "detector/vim_author_dim": vim["score"],
            "ood_score/energy_t1": energy,
        }
        diagnostics[split_name] = {
            "raw_class_distances": raw["class_distances"],
            "raw_nearest_class": raw["nearest_class"],
            "normalized_class_distances": pp["class_distances"],
            "knn_kth_squared_distance": knn["neighbor_squared_distances"][:, -1],
            "knn_kth_neighbor_sample_id": knn["neighbor_sample_ids"][:, -1],
            "vim_residual": vim["residual"],
        }

    bundle_id = hashlib.sha256(f"bundle:{bundle_label}".encode()).hexdigest()
    config_hash = hashlib.sha256(f"config:{bundle_label}".encode()).hexdigest()
    allowed_files = {}
    raw_manifest = {"dataset": {"splits": {}}}
    for split_name, arrays in splits.items():
        relative_directory = f"raw/{split_name}"
        raw_manifest["dataset"]["splits"][split_name] = {
            "relative_directory": relative_directory
        }
        for array_name in ("features", "logits", "class_labels"):
            if array_name not in arrays:
                continue
            logical = f"{relative_directory}/{array_name}.npy"
            allowed_files[logical] = {
                "sha256": hashlib.sha256(
                    f"file:{bundle_label}:{logical}".encode()
                ).hexdigest(),
                "size": max(1, int(arrays[array_name].nbytes)),
            }
    for array_name, array in (("weight", weight), ("bias", bias)):
        logical = f"classifier_{array_name}.npy"
        allowed_files[logical] = {
            "sha256": hashlib.sha256(
                f"file:{bundle_label}:{logical}".encode()
            ).hexdigest(),
            "size": max(1, int(array.nbytes)),
        }
    result = {"scores": {}}
    for split_name, detector_scores in scores.items():
        result["scores"][split_name] = {}
        for detector, array in detector_scores.items():
            array_path = (
                f"scores/{split_name}/{detector.replace('/', '__')}.npy"
            )
            logical = f"metrics/{array_path}"
            file_sha256 = hashlib.sha256(
                f"file:{bundle_label}:{logical}".encode()
            ).hexdigest()
            allowed_files[logical] = {
                "sha256": file_sha256,
                "size": max(1, int(array.nbytes)),
            }
            result["scores"][split_name][detector] = {
                "array_path": array_path,
                "sha256": file_sha256,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        residual = diagnostics[split_name]["vim_residual"]
        residual_path = f"scores/{split_name}/diagnostics/vim/residual.npy"
        residual_logical = f"metrics/{residual_path}"
        residual_file_sha256 = hashlib.sha256(
            f"file:{bundle_label}:{residual_logical}".encode()
        ).hexdigest()
        allowed_files[residual_logical] = {
            "sha256": residual_file_sha256,
            "size": max(1, int(residual.nbytes)),
        }
        result["scores"][split_name]["diagnostics"] = {
            "vim": {
                "residual": {
                    "array_path": residual_path,
                    "sha256": residual_file_sha256,
                    "shape": list(residual.shape),
                    "dtype": str(residual.dtype),
                }
            }
        }
    bundle = STAGE2.SelectiveBundle(
        bundle_id=bundle_id,
        bundle_spec={
            "bundle_id": bundle_id,
            "config_hash": config_hash,
            "training_seed": seed % 3,
            "optimizer_family": "fixture",
            "labels": [f"role:fixture:{bundle_label}"],
            "identity": {
                "checkpoint_role": "last",
                "evaluation_git_sha": "e" * 40,
                "inventory_hash": "f" * 64,
                "dataset_policy_hash": "d" * 64,
            },
        },
        access=SimpleNamespace(allowed_files=allowed_files),
        raw_manifest=raw_manifest,
        result=result,
        splits=splits,
        scores=scores,
        diagnostics=diagnostics,
        classifier_weight=weight,
        classifier_bias=bias,
        query_keys=("id_test", "near_ood", "far_ood"),
        ood_keys=("near_ood", "far_ood"),
        source_allowlist_sha256=hashlib.sha256(
            f"allowlist:{bundle_label}".encode()
        ).hexdigest(),
        reuse_manifest_sha256="a" * 64,
        analysis_policy={},
    )
    baseline = {}
    for dataset in ("near_ood", "far_ood"):
        for detector in (
            "detector/mahalanobis_raw",
            "detector/mahalanobis_pp",
            "detector/knn_l2_k50",
            "detector/ctm_prototype_cosine",
            "ood_score/energy_t1",
        ):
            baseline[(bundle_id, detector, dataset)] = _baseline_rows(
                _metric_values(scores["id_test"][detector], scores[dataset][detector])
            )
    return bundle, baseline


def _extract(
    bundle,
    baseline,
    cache_root,
    *,
    require_existing=False,
    bank_chunk_size=53,
    numeric_runtime_fingerprint=None,
):
    return STAGE2.extract_bundle_rows(
        bundle,
        baseline_metrics=baseline,
        id_metrics={bundle.bundle_id: {"accuracy": 0.8, "nll": 0.4, "ece": 0.03}},
        ood_datasets=("near_ood", "far_ood"),
        ood_groups={"near_ood": "near", "far_ood": "far"},
        analysis_id="a" * 64,
        cache_root=cache_root,
        require_existing_knn_cache=require_existing,
        query_chunk_size=4,
        bank_chunk_size=bank_chunk_size,
        numeric_runtime_fingerprint=numeric_runtime_fingerprint,
    )


def _extract_diagnostics(bundle, baseline, *, numeric_runtime_fingerprint=None):
    return STAGE2.extract_bundle_diagnostic_rows(
        bundle,
        baseline_metrics=baseline,
        ood_datasets=("near_ood", "far_ood"),
        ood_groups={"near_ood": "near", "far_ood": "far"},
        analysis_id="a" * 64,
        numeric_runtime_fingerprint=numeric_runtime_fingerprint,
    )


def test_synthetic_diagnostic_rows_are_complete_formula_bound_and_selection_excluded(
    tmp_path,
):
    bundle, baseline = _synthetic_bundle(5, "diagnostics")
    rows = _extract_diagnostics(bundle, baseline)

    assert len(rows) == 4
    assert {
        (row["diagnostic_id"], row["ood_dataset"]) for row in rows
    } == {
        (diagnostic_id, dataset)
        for diagnostic_id in STAGE2.DIAGNOSTIC_IDS
        for dataset in ("near_ood", "far_ood")
    }
    assert set(STAGE2.DIAGNOSTIC_IDS).isdisjoint(STAGE2.CANDIDATES)
    for row in rows:
        assert row["selection_status"] == "diagnostic_only"
        assert row["candidate_eligibility"] is False
        assert row["ranking_use"] == "forbidden"
        assert row["status"] == "PASS"
        assert row["formula_reconstruction_pass"] is True
        assert row["stored_artifact_parity"]["status"] == "PASS"
        assert row["score_distribution"]["id_test"]["count"] == 11
        assert row["source_references"]["source_allowlist_sha256"] == (
            bundle.source_allowlist_sha256
        )

    pure = next(row for row in rows if row["diagnostic_id"] == "pure_residual_channel")
    assert pure["score_components"]["fit"]["feature_dim"] == 6
    assert pure["score_components"]["fit"]["author_dim"] == 3
    assert pure["score_components"]["fit"]["residual_subspace_dim"] == 3
    assert pure["score_components"]["fit"]["author_dim_rule"] == (
        "floor(feature_dim/2)"
    )
    assert pure["channel_contract"] == {
        "pure_residual_uses_energy": False,
        "pure_residual_uses_alpha": False,
        "score_orientation": "negative_residual_id_like",
        "author_dim_rule": "floor(feature_dim/2)",
    }
    assert pure["scalar_parity"]["status"] == "NOT_APPLICABLE"
    assert pure["scalar_parity"]["reason_codes"] == [
        "V2_ONLY_PURE_RESIDUAL_HAS_NO_V1_2_SCALAR_BASELINE"
    ]
    assert pure["stored_artifact_parity"]["full_vim_score"]["status"] == "PASS"
    assert pure["stored_artifact_parity"]["vim_residual"]["status"] == "PASS"
    assert {
        value["source_kind"]
        for value in pure["source_references"]["derived_score_arrays"].values()
    } == {"derived_negative_of_selective_reuse_diagnostic_array"}

    ctm = next(row for row in rows if row["diagnostic_id"] == "ctm_angular_channel")
    assert ctm["scalar_parity"]["status"] == "PASS"
    assert ctm["score_components"]["fit"]["normalization_status"]["status"] == (
        "success"
    )
    assert len(ctm["score_components"]["fit"][
        "classwise_id_train_angular_concentration"
    ]["classes"]) == 2
    assert sum(ctm["score_components"]["id_test"]["matched_class"]["class_counts"]) == 11
    assert ctm["score_components"]["id_test"]["normalization_status"]["status"] == (
        "success"
    )
    assert ctm["score_components"]["ood"]["normalization_status"]["status"] == (
        "success"
    )
    assert ctm["source_references"]["class_label_arrays"]["id_train"][
        "source_kind"
    ] == "selective_reuse_class_label_array"

    output = tmp_path / "diagnostic-evidence"
    written = STAGE2.write_evidence_directory_no_overwrite(
        {
            "schema_version": STAGE2.SELECTION_MANIFEST_SCHEMA,
            "analysis_id": "a" * 64,
            "files": [],
            "diagnostic_population": {"row_count": len(rows)},
            "scientific_gate": {"status": "NOT_RUN"},
        },
        [],
        output,
        diagnostic_rows=rows,
    )
    assert written["diagnostic_metric_row_count"] == 4
    assert sum(
        1
        for line in (output / "diagnostic_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ) == 4


def test_frozen_diagnostic_population_and_author_dimensions_are_exact():
    assert (
        STAGE2.STAGE2_BUNDLE_COUNT
        * STAGE2.STAGE2_OOD_DATASET_COUNT
        * len(STAGE2.DIAGNOSTIC_IDS)
    ) == STAGE2.EXPECTED_DIAGNOSTIC_ROW_COUNT == 360
    assert STAGE2._vim_author_dimensions(640) == (320, 320)
    assert STAGE2._vim_author_dimensions(512) == (256, 256)
    with pytest.raises(ValueError, match="feature_dim >= 2"):
        STAGE2._vim_author_dimensions(1)


def test_full_array_parity_uses_frozen_allclose_and_records_exact_sha_separately():
    stored = np.asarray([1.0], dtype=np.float64)
    observed = np.nextafter(stored, np.asarray([2.0], dtype=np.float64))

    record = STAGE2._full_array_parity_record(stored, observed, label="fixture")

    assert record["status"] == "PASS"
    assert record["exact_array_sha256_equal"] is False
    with pytest.raises(AssertionError, match="tolerance-bound full-array parity"):
        STAGE2._full_array_parity_record(
            stored, stored + 1.0e-4, label="corrupted fixture"
        )


def test_diagnostic_rows_fail_closed_on_residual_corruption_or_missing_label_source():
    bundle, baseline = _synthetic_bundle(9, "corruption")
    bundle.diagnostics["id_test"]["vim_residual"] = (
        bundle.diagnostics["id_test"]["vim_residual"] + 1.0e-4
    )
    with pytest.raises(AssertionError, match="full-array parity failed"):
        _extract_diagnostics(bundle, baseline)

    bundle, baseline = _synthetic_bundle(9, "missing-label-source")
    id_train_directory = bundle.raw_manifest["dataset"]["splits"]["id_train"][
        "relative_directory"
    ]
    del bundle.access.allowed_files[f"{id_train_directory}/class_labels.npy"]
    with pytest.raises(ValueError, match="absent from allowlist"):
        _extract_diagnostics(bundle, baseline)


def test_ctm_diagnostic_fails_closed_on_near_zero_query_norm():
    bundle, baseline = _synthetic_bundle(15, "ctm-near-zero")
    bundle.splits["near_ood"]["features"][0] = np.asarray(
        [0.5 * STAGE2.EPSILON_NORM, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    with pytest.raises(AssertionError, match="CTM near_ood normalization status"):
        _extract_diagnostics(bundle, baseline)


def test_synthetic_worker_rows_cover_both_candidates_states_and_components(tmp_path):
    bundle, baseline = _synthetic_bundle(7, "left")
    cache_root = tmp_path / "cache"
    rows = _extract(bundle, baseline, cache_root)

    assert len(rows) == 8
    assert {
        (row["candidate_id"], row["state"], row["ood_dataset"]) for row in rows
    } == {
        (candidate, state, dataset)
        for candidate in STAGE2.CANDIDATES
        for state in ("raw", "transformed")
        for dataset in ("near_ood", "far_ood")
    }
    assert len(list((cache_root / ("a" * 64) / bundle.bundle_id).iterdir())) == 7
    for row in rows:
        assert row["formula_reconstruction_pass"] is True
        assert row["pair_credit_identity_pass"] is True
        assert row["exact_null_oracle"]["score_sha256_equal"] is True
        assert row["exact_null_oracle"]["raw_auroc"] == row[
            "exact_null_oracle"
        ]["transformed_auroc"]
        assert row["status"] == "PASS"
        assert 1.0 - row["auroc"] == pytest.approx(
            row["misordering_mass"] + 0.5 * row["tie_mass"], abs=1e-12
        )
        assert row["score_distribution"]["quantile_probabilities"] == {
            "q01": 0.01,
            "q05": 0.05,
            "q25": 0.25,
            "q50": 0.5,
            "q75": 0.75,
            "q95": 0.95,
            "q99": 0.99,
        }
        assert row["score_distribution"]["id_test"]["array_sha256"] == row[
            "source_references"
        ]["score_arrays"]["id_test"]["array_sha256"]
        assert row["score_distribution"]["ood"]["array_sha256"] == row[
            "source_references"
        ]["score_arrays"]["ood"]["array_sha256"]
        geometry = row["geometry_channel"]
        assert geometry["active_ood_split"] == row["ood_dataset"]
        assert set(geometry["splits"]) == {
            "id_train",
            "id_test",
            "near_ood",
            "far_ood",
        }
        for split_record in geometry["splits"].values():
            assert set(split_record["feature_norm"]["quantiles"]) == {
                "q01",
                "q05",
                "q25",
                "q50",
                "q75",
                "q95",
                "q99",
            }
            assert split_record["normalization"]["status"] == "success"
            assert split_record["normalization"]["exact_zero_count"] == 0
            assert split_record["normalization"]["near_zero_count"] == 0
        preservation = row["non_invariant_witness"][
            "normalization_status_preservation"
        ]
        assert preservation["status"] == "PASS"
        assert all(item["preserved"] for item in preservation["splits"].values())
        assert set(row["source_references"]["feature_arrays"]) == {
            "id_train",
            "id_test",
            "ood",
        }
        assert set(row["source_references"]["null_logit_arrays"]) == {
            "id_test",
            "ood",
        }
        if row["candidate_id"] == "radial_l2_knn":
            overlap = row["score_components"]["raw_to_l2_neighbor_overlap"]
            assert overlap["id_test"]["full_query_array_evaluated"] is True
            assert overlap["ood"]["k"] == 50
            assert row["score_components"]["exact_bank_mapping"][
                "lossless_from_neighbor_component"
            ] is True
            assert row["score_components"]["id_test"][
                "kth_squared_distance_distribution"
            ]["count"] == 11
            assert row["score_components"]["id_test"][
                "full_query_array_evaluated"
            ] is True
        else:
            for split in ("id_test", "ood"):
                components = row["score_components"][split]
                assert components["full_query_array_evaluated"] is True
                assert components["covariance_eigendirection_contributions"][
                    "max_abs_distance_reconstruction_error"
                ] < 1e-9
                assert set(
                    components["covariance_eigendirection_contributions"][
                        "precision_eigenvalue_tertile_distribution"
                    ]
                ) == {"low", "middle", "high"}
                assert "selected_class_distance_distribution" in components
                assert "nearest_vs_second_distance_margin_distribution" in components
            scale_oracle = next(
                item
                for item in row["invariant_oracles"]
                if item["oracle_id"] == "mahalanobis_common_positive_scale"
            )
            assert scale_oracle["numerical_rank_before"] == scale_oracle[
                "numerical_rank_after"
            ]

    # A second call is a checksum-verified resume and must be deterministic.
    assert _extract(bundle, baseline, cache_root, require_existing=True) == rows


def test_two_bundle_cache_shards_merge_without_overwrite(tmp_path):
    cache_root = tmp_path / "merged-cache"
    first_bundle, first_baseline = _synthetic_bundle(11, "first")
    second_bundle, second_baseline = _synthetic_bundle(13, "second")
    for bundle in (first_bundle, second_bundle):
        result = STAGE2.precompute_bundle_knn_variants(
            bundle,
            analysis_id="a" * 64,
            cache_root=cache_root,
            query_chunk_size=4,
            bank_chunk_size=53,
            memory_budget_bytes=None,
        )
        assert result["variant_count"] == 7
        assert result["status"] == "PASS"

    assert len(list((cache_root / ("a" * 64)).iterdir())) == 2
    first = _extract(
        first_bundle, first_baseline, cache_root, require_existing=True
    )
    second = _extract(
        second_bundle, second_baseline, cache_root, require_existing=True
    )
    assert len(first) == len(second) == 8

    cache_file = next(
        (cache_root / ("a" * 64) / first_bundle.bundle_id).rglob("score.npy")
    )
    cache_file.write_bytes(cache_file.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _extract(first_bundle, first_baseline, cache_root, require_existing=True)


def test_concurrent_variant_writers_stage_outside_the_shared_cache_tree(
    tmp_path, monkeypatch
):
    cache_root = tmp_path / "shared-cache"
    analysis_id = "a" * 64
    first_bundle = hashlib.sha256(b"concurrent-first").hexdigest()
    second_bundle = hashlib.sha256(b"concurrent-second").hexdigest()
    first_staged = threading.Event()
    release_first = threading.Event()
    original_write_checksums = STAGE2._write_cache_checksums

    def hold_first_staging_directory(root):
        original_write_checksums(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["bundle_id"] == first_bundle:
            first_staged.set()
            assert release_first.wait(timeout=10.0)

    monkeypatch.setattr(STAGE2, "_write_cache_checksums", hold_first_staging_directory)

    def publish(bundle_id, variant_id):
        verified_root = STAGE2._prepare_cache_root(cache_root)
        destination = verified_root / analysis_id / bundle_id / variant_id
        identity = {
            "schema_version": STAGE2.KNN_CACHE_SCHEMA,
            "analysis_id": analysis_id,
            "bundle_id": bundle_id,
            "variant_id": variant_id,
            "include_top50_components": False,
        }
        STAGE2._write_knn_variant_cache(
            destination,
            cache_root=verified_root,
            identity=identity,
            scores={"id_test": np.asarray([0.25, -0.5], dtype=np.float64)},
            backend_metadata={"backend": "synthetic-concurrency-fixture"},
            components={},
            train_sample_ids=np.asarray(["train-0"]),
        )
        return destination

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(publish, first_bundle, "raw_identity")
        assert first_staged.wait(timeout=10.0)
        second_future = executor.submit(publish, second_bundle, "l2_targeted")
        try:
            second_destination = second_future.result(timeout=10.0)
        finally:
            release_first.set()
        first_destination = first_future.result(timeout=10.0)

    assert first_destination.is_dir()
    assert second_destination.is_dir()
    STAGE2._prepare_cache_root(cache_root)
    assert not any(path.name.endswith(".tmp") for path in cache_root.rglob("*"))
    staging_root = STAGE2._prepare_cache_staging_root(cache_root)
    assert staging_root.stat().st_dev == cache_root.stat().st_dev
    assert list(staging_root.iterdir()) == []


def test_bank_chunk_smaller_than_k_has_identical_cache_create_resume_semantics(
    tmp_path,
):
    bundle, baseline = _synthetic_bundle(17, "small-bank-chunk")
    cache_root = tmp_path / "cache"
    rows = _extract(bundle, baseline, cache_root, bank_chunk_size=7)
    assert all(row["raw_knn_backend"]["bank_chunk_size"] == 50 for row in rows)
    resumed = _extract(
        bundle,
        baseline,
        cache_root,
        require_existing=True,
        bank_chunk_size=7,
    )
    assert resumed == rows


def test_cache_runtime_fingerprint_is_exact_and_mismatch_fails_closed(tmp_path):
    bundle, baseline = _synthetic_bundle(19, "runtime")
    cache_root = tmp_path / "cache"
    runtime = STAGE2._numeric_runtime_fingerprint()
    rows = _extract(
        bundle,
        baseline,
        cache_root,
        numeric_runtime_fingerprint=runtime,
    )
    assert {
        row["numeric_runtime_fingerprint_sha256"] for row in rows
    } == {runtime["canonical_lock_sha256"]}
    cache_manifest = json.loads(
        (
            cache_root
            / ("a" * 64)
            / bundle.bundle_id
            / "raw_identity"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert cache_manifest["numeric_runtime_fingerprint"] == runtime
    assert cache_manifest["numeric_runtime_fingerprint_sha256"] == runtime[
        "canonical_lock_sha256"
    ]

    changed = json.loads(json.dumps(runtime))
    changed["thread_environment"]["OMP_NUM_THREADS"] = "different"
    changed.pop("canonical_lock_sha256")
    changed["canonical_lock_sha256"] = STAGE2._sha256_bytes(
        STAGE2._canonical_bytes(changed)
    )
    with pytest.raises(ValueError, match="cache identity mismatch"):
        _extract(
            bundle,
            baseline,
            cache_root,
            require_existing=True,
            numeric_runtime_fingerprint=changed,
        )


def test_cache_root_and_descendants_reject_symlinks_special_temp_and_extra_dirs(
    tmp_path,
):
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        STAGE2._prepare_cache_root(linked_root)

    bundle, _ = _synthetic_bundle(23, "safe-tree")
    cache_root = tmp_path / "cache"
    STAGE2.precompute_bundle_knn_variants(
        bundle,
        analysis_id="a" * 64,
        cache_root=cache_root,
        query_chunk_size=4,
        bank_chunk_size=53,
        memory_budget_bytes=None,
    )
    variant = cache_root / ("a" * 64) / bundle.bundle_id / "raw_identity"

    rogue_link = variant / "rogue-link"
    rogue_link.symlink_to(variant / "manifest.json")
    with pytest.raises(ValueError, match="symlink"):
        STAGE2._prepare_cache_root(cache_root)
    rogue_link.unlink()

    stale = cache_root / ".stale.tmp"
    stale.mkdir()
    with pytest.raises(ValueError, match="temporary cache entry"):
        STAGE2._prepare_cache_root(cache_root)
    stale.rmdir()

    unexpected = variant / "unexpected-empty-directory"
    unexpected.mkdir()
    with pytest.raises(ValueError, match="unexpected directories"):
        STAGE2._prepare_cache_root(cache_root)
    unexpected.rmdir()

    fifo = cache_root / "rogue-fifo"
    os.mkfifo(fifo)
    try:
        with pytest.raises(ValueError, match="non-regular cache file"):
            STAGE2._prepare_cache_root(cache_root)
    finally:
        fifo.unlink()


def test_targeted_witness_rules_and_norm_status_preservation_are_fail_closed():
    assert STAGE2._targeted_witness_accepts(
        "radial_l2_mahalanobis",
        score_allclose=True,
        rank_disagreement_fraction=0.0,
        auroc_abs_drift=STAGE2.EXACT_ATOL,
    )
    allclose_but_reordered_reference = np.asarray([0.0, 0.5e-12])
    allclose_but_reordered_observed = allclose_but_reordered_reference[::-1]
    assert np.allclose(
        allclose_but_reordered_reference,
        allclose_but_reordered_observed,
        atol=STAGE2.EXACT_ATOL,
        rtol=STAGE2.EXACT_RTOL,
    )
    reordered_rank = STAGE2._rank_signature_disagreement(
        allclose_but_reordered_reference, allclose_but_reordered_observed
    )
    assert reordered_rank > 0.0
    assert not STAGE2._targeted_witness_accepts(
        "radial_l2_mahalanobis",
        score_allclose=True,
        rank_disagreement_fraction=reordered_rank,
        auroc_abs_drift=0.0,
    )
    assert STAGE2._targeted_witness_accepts(
        "radial_l2_knn",
        score_allclose=False,
        rank_disagreement_fraction=0.5,
        auroc_abs_drift=STAGE2.KNN_APPROXIMATE_AUROC_TOLERANCE,
    )

    base = {"id_train": np.asarray([[0.75 * STAGE2.EPSILON_NORM, 0.0]])}
    witness = {"id_train": base["id_train"] * 2.0}
    with pytest.raises(AssertionError, match="normalization status changed"):
        STAGE2._normalization_status_preservation(
            base, witness, denominator_epsilon=0.0
        )


def test_runtime_fingerprint_rejects_oge_import_outside_repository(monkeypatch):
    original = STAGE2.importlib.import_module

    def redirected(module_name):
        if module_name == "oge.evaluation.metrics":
            return SimpleNamespace(__file__="/tmp/site-packages/oge/evaluation/metrics.py")
        return original(module_name)

    monkeypatch.setattr(STAGE2.importlib, "import_module", redirected)
    with pytest.raises(ValueError, match="outside repository src"):
        STAGE2._numeric_runtime_fingerprint()


def _relocated_baseline_fixture(tmp_path):
    root = tmp_path / "relocated-baseline"
    root.mkdir()
    checkpoint = hashlib.sha256(b"baseline-checkpoint").hexdigest()
    config_hash = hashlib.sha256(b"baseline-config").hexdigest()
    bundle = {
        "config_hash": config_hash,
        "training_seed": 1,
        "optimizer_family": "fixture_optimizer",
        "labels": ["fixture:relocated-baseline"],
        "identity": {
            "checkpoint_role": "last",
            "evaluation_git_sha": "e" * 40,
        },
    }
    common = {
        "checkpoint_sha256": checkpoint,
        "checkpoint_role": "last",
        "config_hash": config_hash,
        "training_seed": 1,
        "evaluation_git_sha": "e" * 40,
        "optimizer_family": bundle["optimizer_family"],
        "labels": bundle["labels"],
        "status": "success",
        "reason_codes": [],
    }
    rows = []
    id_values = {"accuracy": 0.8, "nll": 0.4, "ece": 0.03}
    for name, artifact_key in STAGE2.ID_METRIC_ARTIFACTS.items():
        rows.append({**common, "artifact_key": artifact_key, "value": id_values[name]})
    metric_values = {
        "auroc": 0.75,
        "fpr95": 0.25,
        "fpr95_threshold": 0.1,
        "achieved_id_tpr": 0.95,
    }
    for dataset in ("near_ood", "far_ood"):
        for detector in STAGE2.PARITY_DETECTORS:
            for name, artifact_key in STAGE2.METRIC_ARTIFACTS.items():
                rows.append(
                    {
                        **common,
                        "artifact_key": artifact_key,
                        "detector": detector,
                        "ood_dataset": dataset,
                        "value": metric_values[name],
                    }
                )
    (root / "per_checkpoint_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    for logical in STAGE2.PARITY_BASELINE_LOGICAL_PATHS:
        path = root / logical
        if not path.exists():
            path.write_bytes(f"fixture:{logical}\n".encode())
    historical = tmp_path / "absent-historical-baseline"
    manifest = {
        "parity_baseline": {
            "role": "completed_metric_contract_v1.2_scalar_parity_baseline",
            "files": [
                {
                    "logical_path": logical,
                    "local_path": str((historical / logical).absolute()),
                    "sha256": STAGE2.sha256_file(root / logical),
                    "size": (root / logical).stat().st_size,
                }
                for logical in STAGE2.PARITY_BASELINE_LOGICAL_PATHS
            ],
        },
        "path_contract": {
            "query_keys": ["id_test", "near_ood", "far_ood"]
        },
    }
    context = SimpleNamespace(manifest=manifest, bundles={checkpoint: bundle})
    return root, manifest, context


def test_relocated_parity_baseline_is_exact_and_records_actual_absolute_paths(
    tmp_path,
):
    root, manifest, context = _relocated_baseline_fixture(tmp_path)

    metrics, id_metrics, sources = STAGE2.load_frozen_baseline(
        context, parity_baseline_root=root
    )

    assert len(metrics) == 2 * len(STAGE2.PARITY_DETECTORS)
    assert len(id_metrics) == 1
    frozen = {
        item["logical_path"]: item for item in manifest["parity_baseline"]["files"]
    }
    assert {item["logical_path"] for item in sources} == set(
        STAGE2.PARITY_BASELINE_LOGICAL_PATHS
    )
    for item in sources:
        assert item["local_path"] == str((root / item["logical_path"]).absolute())
        assert item["sha256"] == frozen[item["logical_path"]]["sha256"]
        assert item["size"] == frozen[item["logical_path"]]["size"]


def test_relocated_parity_baseline_root_fails_closed_on_tree_and_content_drift(
    tmp_path,
):
    root, manifest, _ = _relocated_baseline_fixture(tmp_path)
    STAGE2._verify_baseline_files(manifest, root)

    unexpected = root / "unexpected.txt"
    unexpected.write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected file"):
        STAGE2._verify_baseline_files(manifest, root)
    unexpected.unlink()

    selected = root / "aggregate.json"
    original = selected.read_bytes()
    selected.unlink()
    with pytest.raises(FileNotFoundError, match="baseline file is absent"):
        STAGE2._verify_baseline_files(manifest, root)
    selected.write_bytes(original)

    external = tmp_path / "external-aggregate.json"
    external.write_bytes(original)
    selected.unlink()
    selected.symlink_to(external)
    with pytest.raises(ValueError, match="non-regular entry"):
        STAGE2._verify_baseline_files(manifest, root)
    selected.unlink()
    selected.write_bytes(original)

    selected.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="checksum mismatch"):
        STAGE2._verify_baseline_files(manifest, root)
    selected.write_bytes(original)

    linked_root = tmp_path / "linked-baseline"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="root chain.*symlink"):
        STAGE2._verify_baseline_files(manifest, linked_root)
    with pytest.raises(ValueError, match="root must be absolute"):
        STAGE2._verify_baseline_files(manifest, Path("relative-baseline"))


def test_parity_baseline_cli_help_forwarding_and_precompute_exclusion(
    tmp_path, monkeypatch, capsys
):
    with pytest.raises(SystemExit) as help_exit:
        STAGE2.parse_args(["--help"])
    assert help_exit.value.code == 0
    assert "--parity-baseline-root" in capsys.readouterr().out

    with pytest.raises(ValueError, match="precompute forbids"):
        STAGE2.main(
            [
                "--reuse-manifest",
                str(tmp_path / "missing-reuse.json"),
                "--retrieval-root",
                str(tmp_path / "retrieval"),
                "--cache-root",
                str(tmp_path / "cache"),
                "--precompute-bundle-id",
                "b" * 64,
                "--parity-baseline-root",
                str(tmp_path / "baseline"),
            ]
        )

    root, manifest, _ = _relocated_baseline_fixture(tmp_path)
    reuse_path = tmp_path / "reuse.json"
    reuse_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return {"files": []}, [], []

    monkeypatch.setattr(STAGE2, "build_stage2_evidence", fake_build)
    monkeypatch.setattr(
        STAGE2,
        "write_evidence_directory_no_overwrite",
        lambda *args, **kwargs: {"status": "fixture"},
    )
    assert (
        STAGE2.main(
            [
                "--reuse-manifest",
                str(reuse_path),
                "--retrieval-root",
                str(tmp_path / "retrieval"),
                "--cache-root",
                str(tmp_path / "cache"),
                "--output-directory",
                str(tmp_path / "evidence"),
                "--parity-baseline-root",
                str(root),
            ]
        )
        == 0
    )
    assert captured["parity_baseline_root"] == root


@pytest.mark.parametrize("use_override", (False, True))
def test_full_cli_rejects_output_inside_effective_parity_baseline_before_build(
    tmp_path, monkeypatch, use_override
):
    root, manifest, _ = _relocated_baseline_fixture(tmp_path)
    reuse_path = tmp_path / "reuse-overlap.json"
    if not use_override:
        for item in manifest["parity_baseline"]["files"]:
            item["local_path"] = str(root / item["logical_path"])
    reuse_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        STAGE2,
        "build_stage2_evidence",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("protected arrays must not be opened")
        ),
    )
    arguments = [
        "--reuse-manifest",
        str(reuse_path),
        "--retrieval-root",
        str(tmp_path / "retrieval"),
        "--cache-root",
        str(tmp_path / "cache"),
        "--output-directory",
        str(root / "evidence"),
    ]
    if use_override:
        arguments.extend(("--parity-baseline-root", str(root)))
    with pytest.raises(ValueError, match="parity baseline root"):
        STAGE2.main(arguments)


def test_evidence_directory_is_atomic_checksummed_and_no_overwrite(tmp_path):
    output = tmp_path / "evidence"
    selection = {
        "schema_version": STAGE2.SELECTION_MANIFEST_SCHEMA,
        "analysis_id": "a" * 64,
        "files": [],
        "scientific_gate": {"status": "NOT_RUN"},
    }
    rows = [
        {
            "schema_version": STAGE2.MODEL_METRIC_SCHEMA,
            "analysis_id": "a" * 64,
            "candidate_id": "fixture",
        }
    ]
    result = STAGE2.write_evidence_directory_no_overwrite(selection, rows, output)

    assert result["model_metric_row_count"] == 1
    assert result["diagnostic_metric_row_count"] == 0
    assert result["scientific_gate"] == "NOT_RUN"
    assert (output / "checksums.sha256").is_file()
    assert (output / "diagnostic_metrics.jsonl").is_file()
    manifest = __import__("json").loads(
        (output / "selection_manifest.json").read_text(encoding="utf-8")
    )
    assert {record["logical_path"]: record["row_count"] for record in manifest["files"]} == {
        "model_metrics.jsonl": 1,
        "diagnostic_metrics.jsonl": 0,
    }
    checksum_text = (output / "checksums.sha256").read_text(encoding="utf-8")
    assert "diagnostic_metrics.jsonl" in checksum_text
    with pytest.raises(FileExistsError, match="no-overwrite"):
        STAGE2.write_evidence_directory_no_overwrite(selection, rows, output)


def test_evidence_directory_rejects_dangling_symlink_without_following_it(tmp_path):
    requested = tmp_path / "evidence"
    redirected = tmp_path / "redirected-target"
    requested.symlink_to(redirected, target_is_directory=True)

    with pytest.raises(FileExistsError, match="no-overwrite"):
        STAGE2.write_evidence_directory_no_overwrite({"files": []}, [], requested)

    assert requested.is_symlink()
    assert not redirected.exists()


def test_evidence_directory_rejects_symlink_or_nondirectory_parent_chain(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="parent chain.*symlink"):
        STAGE2.write_evidence_directory_no_overwrite(
            {"files": []}, [], linked_parent / "evidence"
        )
    assert not (real_parent / "evidence").exists()

    non_directory = tmp_path / "not-a-directory"
    non_directory.write_text("sentinel", encoding="utf-8")
    with pytest.raises(ValueError, match="parent chain.*directories"):
        STAGE2.write_evidence_directory_no_overwrite(
            {"files": []}, [], non_directory / "evidence"
        )
    assert non_directory.read_text(encoding="utf-8") == "sentinel"


@pytest.mark.parametrize("root_name", ("retrieval", "cache"))
def test_full_cli_rejects_output_under_protected_input_root_before_build(
    tmp_path, monkeypatch, root_name
):
    retrieval = tmp_path / "retrieval"
    cache = tmp_path / "cache"
    retrieval.mkdir()
    cache.mkdir()
    sentinel = retrieval / "source-sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")
    protected_root = retrieval if root_name == "retrieval" else cache

    def forbidden_build(**_kwargs):
        raise AssertionError("protected arrays must not be opened")

    monkeypatch.setattr(STAGE2, "build_stage2_evidence", forbidden_build)
    with pytest.raises(ValueError, match=f"disjoint from {root_name} root"):
        STAGE2.main(
            [
                "--reuse-manifest",
                str(tmp_path / "reuse.json"),
                "--retrieval-root",
                str(retrieval),
                "--cache-root",
                str(cache),
                "--output-directory",
                str(protected_root / "evidence"),
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not (protected_root / "evidence").exists()


def test_full_cli_rejects_output_ancestor_of_retrieval_before_build(
    tmp_path, monkeypatch
):
    output = tmp_path / "would-be-output"
    retrieval = output / "retrieval"
    retrieval.mkdir(parents=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    sentinel = retrieval / "source-sentinel"
    sentinel.write_text("unchanged", encoding="utf-8")

    monkeypatch.setattr(
        STAGE2,
        "build_stage2_evidence",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("protected arrays must not be opened")
        ),
    )
    with pytest.raises(FileExistsError, match="no-overwrite"):
        STAGE2.main(
            [
                "--reuse-manifest",
                str(tmp_path / "reuse.json"),
                "--retrieval-root",
                str(retrieval),
                "--cache-root",
                str(cache),
                "--output-directory",
                str(output),
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_single_bundle_precompute_path_never_loads_scalar_baseline(
    tmp_path, monkeypatch
):
    bundle_id = "b" * 64
    context = SimpleNamespace(
        manifest_path=tmp_path / "reuse.json",
        manifest_sha256="m" * 64,
        retrieval_root=tmp_path / "retrieval",
        manifest={
            "bundles": [{"bundle_id": bundle_id}],
            "analysis_policy": {
                "candidate_registry": {"sha256": "c" * 64},
                "gate_policy": {"sha256": "g" * 64},
            },
        },
        bundles={bundle_id: {"bundle_id": bundle_id}},
        accesses={},
    )
    monkeypatch.setattr(STAGE2, "load_single_bundle_reuse_context", lambda *args: context)
    monkeypatch.setattr(
        STAGE2,
        "_validate_registry_and_gate",
        lambda value: ({}, {"required_output_artifacts": {}}, (), {}),
    )
    monkeypatch.setattr(
        STAGE2,
        "load_frozen_baseline",
        lambda value: (_ for _ in ()).throw(AssertionError("baseline must not load")),
    )
    monkeypatch.setattr(
        STAGE2, "load_knn_precompute_bundle", lambda value, selected: object()
    )
    monkeypatch.setattr(
        STAGE2,
        "precompute_bundle_knn_variants",
        lambda *args, **kwargs: {
            "bundle_id": bundle_id,
            "variant_count": 7,
            "status": "PASS",
        },
    )

    report, rows, diagnostic_rows = STAGE2.build_stage2_evidence(
        reuse_manifest_path=tmp_path / "reuse.json",
        retrieval_root=tmp_path / "single",
        analysis_git_sha="1" * 40,
        code_sources=[],
        validate_git_sources=False,
        cache_root=tmp_path / "cache",
        precompute_bundle_id=bundle_id,
    )

    assert rows == []
    assert diagnostic_rows == []
    assert report["bundle_ids"] == [bundle_id]
    assert report["variant_count_per_bundle"] == 7
    assert report["numeric_runtime_fingerprint"]["schema_version"] == (
        STAGE2.NUMERIC_RUNTIME_SCHEMA
    )
    assert report["scientific_gate"] == "NOT_RUN"
    with pytest.raises(ValueError, match="precompute forbids"):
        STAGE2.build_stage2_evidence(
            reuse_manifest_path=tmp_path / "missing-reuse.json",
            retrieval_root=tmp_path / "single",
            analysis_git_sha="1" * 40,
            code_sources=[],
            validate_git_sources=False,
            cache_root=tmp_path / "cache",
            precompute_bundle_id=bundle_id,
            parity_baseline_root=tmp_path / "baseline",
        )
