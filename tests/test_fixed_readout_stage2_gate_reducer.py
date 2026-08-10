import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Callable

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts/reduce_fixed_readout_stage2_gate.py"
)
SPEC = importlib.util.spec_from_file_location("stage2_gate_reducer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
STAGE2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE2
SPEC.loader.exec_module(STAGE2)

REPOSITORY_ROOT = Path(__file__).parents[1]
CANDIDATE_REGISTRY = (
    REPOSITORY_ROOT
    / "configs/evaluation/fixed_readout_stage2/candidate_registry.json"
)
GATE_POLICY = (
    REPOSITORY_ROOT / "configs/evaluation/fixed_readout_stage2/gate_policy.json"
)
REUSE_MANIFEST = (
    REPOSITORY_ROOT / "configs/evaluation/fixed_readout_stage2/reuse_manifest.json"
)
REGISTRY = json.loads(CANDIDATE_REGISTRY.read_text(encoding="utf-8"))
POLICY = json.loads(GATE_POLICY.read_text(encoding="utf-8"))
REUSE = json.loads(REUSE_MANIFEST.read_text(encoding="utf-8"))
DATASETS = tuple(POLICY["ood_partition"]["dataset_order"])
NEAR = set(POLICY["ood_partition"]["near"])
ANALYSIS_ID = hashlib.sha256(b"synthetic-stage2-gate").hexdigest()
NUMERIC_RUNTIME_SHA256 = hashlib.sha256(
    b"synthetic-stage2-numeric-runtime"
).hexdigest()
TEST_GIT_SHA = "f" * 40


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


CHECKPOINTS = sorted(
    [
        {
            "checkpoint_sha256": bundle["bundle_id"],
            "config_hash": bundle["config_hash"],
            "training_seed": bundle["training_seed"],
            "checkpoint_role": bundle["identity"]["checkpoint_role"],
            "optimizer_family": bundle["optimizer_family"],
            "role_labels": bundle["labels"],
            "source_allowlist_sha256": bundle["remote_integrity"][
                "allowed_payload_catalog_sha256"
            ],
        }
        for bundle in REUSE["bundles"]
    ],
    key=lambda item: (
        item["config_hash"],
        item["training_seed"],
        item["checkpoint_sha256"],
    ),
)
CONFIGS = sorted({item["config_hash"] for item in CHECKPOINTS})
assert len(CHECKPOINTS) == 30
assert len(CONFIGS) == 10
BUNDLE_BY_ID = {bundle["bundle_id"]: bundle for bundle in REUSE["bundles"]}
SPLIT_COUNTS = {
    "id_train": 45_000,
    "id_test": 10_000,
    "cifar100": 9_000,
    "tin": 7_793,
    "mnist": 70_000,
    "svhn": 26_032,
    "texture": 5_640,
    "places365": 35_195,
}
FEATURE_DIMENSION = 640
CLASS_COUNT = 10


PASS_PROFILE = {
    "radial_l2_mahalanobis": {
        "raw": {0: 0.020, 1: 0.020, 2: 0.020},
        "transformed": {0: 0.005, 1: 0.005, 2: 0.005},
    },
    "radial_l2_knn": {
        "raw": {0: 0.015, 1: 0.015, 2: 0.015},
        "transformed": {0: 0.014, 1: 0.014, 2: 0.014},
    },
}


def _candidate_specs() -> dict[str, dict[str, Any]]:
    return {item["candidate_id"]: item for item in REGISTRY["candidates"]}


def _distribution(count: int, value: float, identity: str) -> dict[str, Any]:
    return {
        "summary_schema": STAGE2.DISTRIBUTION_SUMMARY_SCHEMA,
        "count": count,
        "minimum": value,
        "maximum": value,
        "mean": value,
        "std_ddof0": 0.0,
        "quantiles": {
            key: value for key in STAGE2.FIXED_QUANTILE_PROBABILITIES
        },
        "array_sha256": _sha(identity),
    }


def _normalization(count: int, denominator_epsilon: float) -> dict[str, Any]:
    return {
        "status": "success",
        "reason_codes": [],
        "epsilon_norm": STAGE2.EPSILON_NORM,
        "denominator_epsilon": denominator_epsilon,
        "exact_zero_count": 0,
        "near_zero_count": 0,
        "sample_count": count,
    }


def _geometry_channel(
    checkpoint_sha: str, candidate_id: str, dataset: str
) -> dict[str, Any]:
    denominator = 0.0 if candidate_id == "radial_l2_mahalanobis" else 1.0e-10
    splits = {}
    for split_name, count in SPLIT_COUNTS.items():
        normalization = _normalization(count, denominator)
        splits[split_name] = {
            "feature_norm": _distribution(
                count, 2.0, f"{checkpoint_sha}:feature-norm:{split_name}"
            ),
            "normalization": normalization,
        }
    return {
        "channel_id": STAGE2.GEOMETRY_CHANNEL_ID,
        "normalization_contract": STAGE2.GEOMETRY_NORMALIZATION_CONTRACT,
        "normalization_denominator_epsilon": denominator,
        "active_ood_split": dataset,
        "splits": splits,
    }


def _normalization_preservation(
    geometry: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    denominator = 0.0 if candidate_id == "radial_l2_mahalanobis" else 1.0e-10
    return {
        "status": "PASS",
        "reason_codes": [],
        "denominator_epsilon": denominator,
        "preserved_fields": list(STAGE2.NORMALIZATION_PRESERVED_FIELDS),
        "splits": {
            split_name: {
                "preserved": True,
                "base": dict(split["normalization"]),
                "witness": dict(split["normalization"]),
            }
            for split_name, split in geometry["splits"].items()
        },
    }


def _allowlist_record(checkpoint_sha: str, logical_path: str) -> dict[str, Any]:
    records = {
        item["logical_path"]: item
        for item in BUNDLE_BY_ID[checkpoint_sha]["remote_integrity"]["files"]
    }
    return records[logical_path]


def _selective_source_reference(
    *,
    checkpoint: dict[str, Any],
    logical_path: str,
    source_kind: str,
    shape: list[int],
    dtype: str,
    array_identity: str,
    score: bool = False,
) -> dict[str, Any]:
    allowed = _allowlist_record(checkpoint["checkpoint_sha256"], logical_path)
    array_sha = _sha(array_identity)
    result = {
        "source_kind": source_kind,
        "logical_path": logical_path,
        "file_sha256": allowed["sha256"],
        "file_size": allowed["size"],
        "array_sha256": array_sha,
        "shape": shape,
        "dtype": dtype,
        "bundle_id": checkpoint["checkpoint_sha256"],
        "source_allowlist_sha256": checkpoint["source_allowlist_sha256"],
    }
    if score:
        result.update(
            {
                "stored_array_sha256": array_sha,
                "source_declared_shape": shape,
                "source_declared_dtype": dtype,
                "parity_status": "PASS",
            }
        )
    return result


def _source_references(
    *,
    checkpoint: dict[str, Any],
    detector: str,
    dataset: str,
) -> dict[str, Any]:
    checkpoint_sha = checkpoint["checkpoint_sha256"]
    num_id = SPLIT_COUNTS["id_test"]
    num_ood = SPLIT_COUNTS[dataset]
    features = {
        "id_train": _selective_source_reference(
            checkpoint=checkpoint,
            logical_path="cifar10/train/features.npy",
            source_kind="selective_reuse_feature_array",
            shape=[SPLIT_COUNTS["id_train"], FEATURE_DIMENSION],
            dtype="float32",
            array_identity=f"{checkpoint_sha}:cifar10/train/features.npy",
        ),
        "id_test": _selective_source_reference(
            checkpoint=checkpoint,
            logical_path="cifar10/test/features.npy",
            source_kind="selective_reuse_feature_array",
            shape=[num_id, FEATURE_DIMENSION],
            dtype="float32",
            array_identity=f"{checkpoint_sha}:cifar10/test/features.npy",
        ),
        "ood": _selective_source_reference(
            checkpoint=checkpoint,
            logical_path=f"{dataset}/test/features.npy",
            source_kind="selective_reuse_feature_array",
            shape=[num_ood, FEATURE_DIMENSION],
            dtype="float32",
            array_identity=f"{checkpoint_sha}:{dataset}/test/features.npy",
        ),
    }
    logits = {
        "id_test": _selective_source_reference(
            checkpoint=checkpoint,
            logical_path="cifar10/test/logits.npy",
            source_kind="selective_reuse_logit_array",
            shape=[num_id, CLASS_COUNT],
            dtype="float32",
            array_identity=f"{checkpoint_sha}:cifar10/test/logits.npy",
        ),
        "ood": _selective_source_reference(
            checkpoint=checkpoint,
            logical_path=f"{dataset}/test/logits.npy",
            source_kind="selective_reuse_logit_array",
            shape=[num_ood, CLASS_COUNT],
            dtype="float32",
            array_identity=f"{checkpoint_sha}:{dataset}/test/logits.npy",
        ),
    }
    scores = {}
    for key, split_name, count in (
        ("id_test", "id_test", num_id),
        ("ood", dataset, num_ood),
    ):
        score_identity = f"{checkpoint_sha}:{detector}:{split_name}:scores"
        if detector == "detector/knn_raw_k50":
            logical_path = f"splits/{split_name}/score.npy"
            scores[key] = {
                "source_kind": "stage2_knn_cache_score_array",
                "logical_path": logical_path,
                "file_sha256": _sha(f"cache-file:{score_identity}"),
                "file_size": count * 8 + 128,
                "array_sha256": _sha(score_identity),
                "shape": [count],
                "dtype": "float64",
                "bundle_id": checkpoint_sha,
                "source_allowlist_sha256": checkpoint["source_allowlist_sha256"],
                "analysis_id": ANALYSIS_ID,
                "variant_id": "raw_identity",
                "cache_schema_version": "fixed_readout_stage2_knn_variant_cache_v1",
                "cache_manifest_sha256": _sha(
                    f"{checkpoint_sha}:raw_identity:cache-manifest"
                ),
            }
        else:
            logical_path = (
                f"metrics/arrays/scores/{split_name}/"
                f"{detector.replace('/', '__')}.npy"
            )
            scores[key] = _selective_source_reference(
                checkpoint=checkpoint,
                logical_path=logical_path,
                source_kind="selective_reuse_score_array",
                shape=[count],
                dtype="float64",
                array_identity=score_identity,
                score=True,
            )
    return {
        "source_allowlist_sha256": checkpoint["source_allowlist_sha256"],
        "score_arrays": scores,
        "feature_arrays": features,
        "null_logit_arrays": logits,
    }


def _mahalanobis_component(
    *, checkpoint_sha: str, detector: str, split_name: str, count: int
) -> dict[str, Any]:
    identity = f"{checkpoint_sha}:{detector}:{split_name}"
    band_values = {"low": 1.0, "middle": 1.0, "high": 2.0}
    return {
        "class_distances": {
            "shape": [count, CLASS_COUNT],
            "sha256": _sha(f"{identity}:class-distances"),
        },
        "nearest_class": {
            "shape": [count],
            "sha256": _sha(f"{identity}:nearest-class"),
        },
        "class_means": {
            "shape": [CLASS_COUNT, FEATURE_DIMENSION],
            "sha256": _sha(f"{checkpoint_sha}:{detector}:class-means"),
        },
        "within_covariance": {
            "shape": [FEATURE_DIMENSION, FEATURE_DIMENSION],
            "sha256": _sha(f"{checkpoint_sha}:{detector}:covariance"),
        },
        "within_precision": {
            "shape": [FEATURE_DIMENSION, FEATURE_DIMENSION],
            "sha256": _sha(f"{checkpoint_sha}:{detector}:precision"),
        },
        "covariance_eigenvalues": {
            "shape": [FEATURE_DIMENSION],
            "sha256": _sha(f"{checkpoint_sha}:{detector}:covariance-eigenvalues"),
            "minimum": 0.1,
            "maximum": 2.0,
        },
        "precision_eigenvalues": {
            "shape": [FEATURE_DIMENSION],
            "sha256": _sha(f"{checkpoint_sha}:{detector}:precision-eigenvalues"),
            "minimum": 0.5,
            "maximum": 10.0,
        },
        "covariance_eigendirection_contributions": {
            "shape": [count, FEATURE_DIMENSION],
            "sha256": _sha(f"{identity}:contributions"),
            "sha256_encoding": "dtype_shape_prefix_then_row_major_float64_chunks",
            "max_abs_distance_reconstruction_error": 0.0,
            "precision_eigenvalue_tertile_mean_contributions": band_values,
            "precision_eigenvalue_tertile_distribution": {
                band: _distribution(
                    count, value, f"{identity}:contribution-band:{band}"
                )
                for band, value in band_values.items()
            },
        },
        "selected_class_distance_distribution": _distribution(
            count, 4.0, f"{identity}:selected-distance"
        ),
        "nearest_vs_second_distance_margin_distribution": _distribution(
            count, 1.0, f"{identity}:distance-margin"
        ),
        "nearest_class_counts": {
            str(index): count if index == 0 else 0
            for index in range(CLASS_COUNT)
        },
        "numerical_rank": FEATURE_DIMENSION,
        "condition_number": 20.0,
        "stored_component_parity_pass": True,
        "stored_component_max_abs_error": 0.0,
        "full_query_array_evaluated": True,
    }


def _knn_component(
    *, checkpoint_sha: str, detector: str, split_name: str, count: int
) -> dict[str, Any]:
    identity = f"{checkpoint_sha}:{detector}:{split_name}"
    return {
        "kth_squared_distance_sha256": _sha(f"{identity}:kth-distance"),
        "kth_neighbor_sample_id_sha256": _sha(f"{identity}:kth-neighbor"),
        "top50_squared_distances_sha256": _sha(f"{identity}:top50-distance"),
        "top50_neighbor_bank_indices_sha256": _sha(f"{identity}:top50-indices"),
        "top50_neighbor_identity_mapping_sha256": _sha(f"{identity}:mapping"),
        "sample_count": count,
        "k": 50,
        "middle_rank": 25,
        "neighbor_identity_encoding": "exact_train_sample_id_strings",
        "train_sample_id_mapping_is_lossless": True,
        "first_squared_distance_distribution": _distribution(
            count, 1.0, f"{identity}:first-distance"
        ),
        "middle_rank_squared_distance_distribution": _distribution(
            count, 2.0, f"{identity}:middle-distance"
        ),
        "kth_squared_distance_distribution": _distribution(
            count, 4.0, f"{identity}:kth-distance"
        ),
        "mean_top50_squared_distance_distribution": _distribution(
            count, 2.5, f"{identity}:mean-top50-distance"
        ),
        "full_query_array_evaluated": True,
    }


def _overlap(checkpoint_sha: str, split_name: str, count: int) -> dict[str, Any]:
    total = count * 25
    return {
        "sample_count": count,
        "k": 50,
        "total_shared_neighbor_count": total,
        "mean_overlap_fraction": 0.5,
        "minimum_overlap_fraction": 0.4,
        "maximum_overlap_fraction": 0.6,
        "per_sample_shared_count_sha256": _sha(
            f"{checkpoint_sha}:{split_name}:overlap-counts"
        ),
        "identity_encoding": "exact_bank_indices_bound_to_train_sample_id_digest",
        "full_query_array_evaluated": True,
    }


def _score_components(
    *,
    checkpoint_sha: str,
    candidate_id: str,
    detector: str,
    dataset: str,
) -> dict[str, Any]:
    num_id = SPLIT_COUNTS["id_test"]
    num_ood = SPLIT_COUNTS[dataset]
    if candidate_id == "radial_l2_mahalanobis":
        return {
            "id_test": _mahalanobis_component(
                checkpoint_sha=checkpoint_sha,
                detector=detector,
                split_name="id_test",
                count=num_id,
            ),
            "ood": _mahalanobis_component(
                checkpoint_sha=checkpoint_sha,
                detector=detector,
                split_name=dataset,
                count=num_ood,
            ),
        }
    return {
        "id_test": _knn_component(
            checkpoint_sha=checkpoint_sha,
            detector=detector,
            split_name="id_test",
            count=num_id,
        ),
        "ood": _knn_component(
            checkpoint_sha=checkpoint_sha,
            detector=detector,
            split_name=dataset,
            count=num_ood,
        ),
        "raw_to_l2_neighbor_overlap": {
            "id_test": _overlap(checkpoint_sha, "id_test", num_id),
            "ood": _overlap(checkpoint_sha, dataset, num_ood),
        },
        "exact_bank_mapping": {
            "encoding": "exact_train_sample_id_strings",
            "train_sample_id_sha256": _sha(f"{checkpoint_sha}:train-sample-ids"),
            "lossless_from_neighbor_component": True,
        },
    }


def _transition(
    raw_auroc: float, transformed_auroc: float
) -> tuple[dict[str, float], float, float, float]:
    keys = POLICY["score_overlap"]["required_transition_mass_fields"]
    masses = {key: 0.0 for key in keys}
    delta = transformed_auroc - raw_auroc
    if delta >= 0.0:
        masses["correct_to_correct"] = raw_auroc
        masses["misordered_to_correct"] = delta
        masses["misordered_to_misordered"] = 1.0 - transformed_auroc
        corrected = delta
        harmed = 0.0
    else:
        masses["correct_to_correct"] = transformed_auroc
        masses["correct_to_misordered"] = -delta
        masses["misordered_to_misordered"] = 1.0 - raw_auroc
        corrected = 0.0
        harmed = -delta
    return masses, corrected, harmed, delta


def _reconcile_pair(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["candidate_id"], row["checkpoint_sha256"], row["ood_dataset"])
        grouped.setdefault(key, {})[row["state"]] = row
    for pair in grouped.values():
        raw = pair["raw"]
        transformed = pair["transformed"]
        for row in (raw, transformed):
            row["fpr95"] = 1.0 - row["auroc"]
            row["misordering_mass"] = 1.0 - row["auroc"]
            row["scalar_parity"]["metrics"]["auroc"] = row["auroc"]
            row["scalar_parity"]["metrics"]["fpr95"] = row["fpr95"]
        transition, corrected, harmed, net = _transition(
            raw["auroc"], transformed["auroc"]
        )
        for row in (raw, transformed):
            row["pair_order_transition_mass"] = transition.copy()
            row["corrected_pair_mass"] = corrected
            row["harmed_pair_mass"] = harmed
            row["net_corrected_pair_mass"] = net


RowMutation = Callable[[list[dict[str, Any]]], None]


def _model_rows(
    *,
    profile: dict[str, dict[str, dict[int, float]]] | None = None,
    mutate: RowMutation | None = None,
) -> list[dict[str, Any]]:
    profile = profile or PASS_PROFILE
    specs = _candidate_specs()
    checkpoint_by_config_seed = {
        (item["config_hash"], item["training_seed"]): item for item in CHECKPOINTS
    }
    rows: list[dict[str, Any]] = []
    seed_offset = {0: 0.0000, 1: -0.0001, 2: 0.0001}
    for candidate_id in REGISTRY["selection_boundary"]["eligible_candidate_ids"]:
        candidate = specs[candidate_id]
        invariant_oracles = candidate["formula_invariant_oracles"]
        witness_id = candidate["non_invariant_witnesses"][0]["witness_id"]
        for config_index, config_hash in enumerate(CONFIGS):
            for seed in (0, 1, 2):
                checkpoint = checkpoint_by_config_seed[(config_hash, seed)]
                for dataset_index, dataset in enumerate(DATASETS):
                    base = 0.56 + 0.002 * dataset_index
                    raw_auroc = (
                        base
                        + profile[candidate_id]["raw"][seed] * config_index
                        + seed_offset[seed]
                    )
                    transformed_auroc = (
                        base
                        + profile[candidate_id]["transformed"][seed] * config_index
                        + seed_offset[seed]
                    )
                    null_auroc = 0.67 + 0.001 * config_index + 0.00001 * seed
                    for state, auroc in (
                        ("raw", raw_auroc),
                        ("transformed", transformed_auroc),
                    ):
                        raw_state = state == "raw"
                        detector = (
                            candidate["raw_detector"]["artifact_key"]
                            if raw_state
                            else candidate["transformed_detector"]["artifact_key"]
                        )
                        transform_id = (
                            "raw_identity"
                            if raw_state
                            else candidate["targeted_transform"]["transform_id"]
                        )
                        parity_status = (
                            "NOT_APPLICABLE"
                            if candidate_id == "radial_l2_knn" and raw_state
                            else "PASS"
                        )
                        geometry = _geometry_channel(
                            checkpoint["checkpoint_sha256"], candidate_id, dataset
                        )
                        sources = _source_references(
                            checkpoint=checkpoint,
                            detector=detector,
                            dataset=dataset,
                        )
                        num_id = SPLIT_COUNTS["id_test"]
                        num_ood = SPLIT_COUNTS[dataset]
                        id_score_identity = (
                            f"{checkpoint['checkpoint_sha256']}:{detector}:id_test:scores"
                        )
                        ood_score_identity = (
                            f"{checkpoint['checkpoint_sha256']}:{detector}:{dataset}:scores"
                        )
                        row = {
                            "schema_version": STAGE2.MODEL_METRIC_SCHEMA,
                            "analysis_id": ANALYSIS_ID,
                            "candidate_id": candidate_id,
                            "config_hash": config_hash,
                            "optimizer_family": checkpoint["optimizer_family"],
                            "role_labels": checkpoint["role_labels"],
                            "training_seed": seed,
                            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                            "checkpoint_role": "last",
                            "ood_dataset": dataset,
                            "ood_group": "near" if dataset in NEAR else "far",
                            "detector": detector,
                            "transform_id": transform_id,
                            "state": state,
                            "paired_detector": (
                                candidate["transformed_detector"]["artifact_key"]
                                if raw_state
                                else candidate["raw_detector"]["artifact_key"]
                            ),
                            "paired_transform_id": (
                                candidate["targeted_transform"]["transform_id"]
                                if raw_state
                                else "raw_identity"
                            ),
                            "num_id": num_id,
                            "num_ood": num_ood,
                            "auroc": auroc,
                            "fpr95": 1.0 - auroc,
                            "fpr95_threshold": 0.0,
                            "achieved_id_tpr": 0.95,
                            "misordering_mass": 1.0 - auroc,
                            "tie_mass": 0.0,
                            "pair_order_transition_mass": {},
                            "corrected_pair_mass": 0.0,
                            "harmed_pair_mass": 0.0,
                            "net_corrected_pair_mass": 0.0,
                            "score_reconstruction_max_abs_error": 0.0,
                            "score_reconstruction_max_rel_error": 0.0,
                            "formula_reconstruction_pass": True,
                            "pair_credit_identity_pass": True,
                            "rank_disagreement_fraction": 0.1,
                            "scalar_parity": {
                                "status": parity_status,
                                "reason_codes": (
                                    ["V2_ONLY_RAW_KNN_HAS_NO_V1_2_BASELINE"]
                                    if parity_status == "NOT_APPLICABLE"
                                    else []
                                ),
                                "metrics": {
                                    "auroc": auroc,
                                    "fpr95": 1.0 - auroc,
                                    "fpr95_threshold": 0.0,
                                    "achieved_id_tpr": 0.95,
                                },
                            },
                            "exact_null_oracle": {
                                "status": "PASS",
                                "reason_codes": [],
                                "score_sha256_equal": True,
                                "rank_disagreement_fraction": 0.0,
                                "auroc_abs_drift": 0.0,
                                "raw_auroc": null_auroc,
                                "transformed_auroc": null_auroc,
                            },
                            "invariant_oracles": [
                                {
                                    "oracle_id": oracle["oracle_id"],
                                    "status": "PASS",
                                    "reason_codes": [],
                                    "score_max_abs_error": 0.0,
                                    "score_max_rel_error": 0.0,
                                    "rank_disagreement_fraction": 0.0,
                                    "auroc_abs_drift": 0.0,
                                }
                                for oracle in invariant_oracles
                            ],
                            "non_invariant_witness": {
                                "witness_id": witness_id,
                                "status": "PASS",
                                "reason_codes": [],
                                "raw_score_max_abs_change": 0.1,
                                "raw_rank_disagreement_fraction": 0.1,
                                "targeted_score_max_abs_error": 0.0,
                                "targeted_score_allclose": True,
                                "targeted_rank_disagreement_fraction": 0.0,
                                "targeted_auroc_abs_drift": 0.0,
                                "targeted_acceptance_rule": (
                                    "allclose_and_exact_rank_and_exact_auroc"
                                    if candidate_id == "radial_l2_mahalanobis"
                                    else "approximate_auroc_only"
                                ),
                                "targeted_exact_atol": POLICY[
                                    "invariant_and_witness_policy"
                                ]["exact_score_tolerance"]["absolute"],
                                "targeted_exact_rtol": POLICY[
                                    "invariant_and_witness_policy"
                                ]["exact_score_tolerance"]["relative"],
                                "targeted_approximate_auroc_tolerance": (
                                    POLICY["invariant_and_witness_policy"][
                                        "approximate_knn_l2_auroc_drift_tolerance"
                                    ]
                                    if candidate_id == "radial_l2_knn"
                                    else None
                                ),
                                "normalization_status_preservation": (
                                    _normalization_preservation(
                                        geometry, candidate_id
                                    )
                                ),
                            },
                            "id_metrics": {
                                "accuracy": 0.84 + 0.001 * config_index,
                                "nll": 0.30 - 0.001 * config_index,
                                "ece": 0.04 + 0.0001 * config_index,
                            },
                            "score_sha256": _sha(
                                f"{checkpoint['checkpoint_sha256']}:{detector}:"
                                f"{dataset}:combined-scores"
                            ),
                            "score_distribution": {
                                "summary_schema": STAGE2.SCORE_DISTRIBUTION_SCHEMA,
                                "score_direction": "higher_is_more_id_like",
                                "quantile_method": "numpy_linear",
                                "quantile_probabilities": dict(
                                    STAGE2.FIXED_QUANTILE_PROBABILITIES
                                ),
                                "id_test": _distribution(
                                    num_id, -4.0, id_score_identity
                                ),
                                "ood": _distribution(
                                    num_ood, -4.0, ood_score_identity
                                ),
                            },
                            "geometry_channel": geometry,
                            "score_components": _score_components(
                                checkpoint_sha=checkpoint["checkpoint_sha256"],
                                candidate_id=candidate_id,
                                detector=detector,
                                dataset=dataset,
                            ),
                            "source_references": sources,
                            "source_allowlist_sha256": checkpoint[
                                "source_allowlist_sha256"
                            ],
                            "status": "PASS",
                            "reason_codes": [],
                        }
                        rows.append(row)
    _reconcile_pair(rows)
    if mutate is not None:
        mutate(rows)
        _reconcile_pair(rows)
    rows.sort(
        key=lambda row: (
            row["candidate_id"],
            row["config_hash"],
            row["training_seed"],
            DATASETS.index(row["ood_dataset"]),
            0 if row["state"] == "raw" else 1,
        )
    )
    assert len(rows) == 720
    return rows


def _diagnostic_component(
    *, shape: list[int], value: float, identity: str
) -> dict[str, Any]:
    count = 1
    for dimension in shape:
        count *= dimension
    return {
        "shape": shape,
        "dtype": "float64",
        "array_sha256": _sha(identity),
        "distribution": _distribution(count, value, identity),
    }


def _diagnostic_formula(
    *, reference_identity: str, reconstructed_identity: str | None = None
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "reason_codes": [],
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "reference_array_sha256": _sha(reference_identity),
        "reconstructed_array_sha256": _sha(
            reconstructed_identity or reference_identity
        ),
        "absolute_tolerance": 1.0e-12,
        "relative_tolerance": 1.0e-10,
    }


def _stored_diagnostic_source(
    *,
    checkpoint: dict[str, Any],
    logical_path: str,
    source_kind: str,
    count: int,
    observed_identity: str,
) -> dict[str, Any]:
    source = _selective_source_reference(
        checkpoint=checkpoint,
        logical_path=logical_path,
        source_kind=source_kind,
        shape=[count],
        dtype="float64",
        array_identity=observed_identity,
        score=True,
    )
    # The production worker records numerical allclose separately from exact
    # byte identity.  Synthetic evidence exercises the allowed non-exact case.
    source["stored_array_sha256"] = _sha(f"{observed_identity}:stored")
    return source


def _full_array_parity(
    source: dict[str, Any], *, count: int
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "count": count,
        "max_abs_error": 1.0e-13,
        "max_rel_error": 1.0e-13,
        "observed_array_sha256": source["array_sha256"],
        "stored_array_sha256": source["stored_array_sha256"],
        "exact_array_sha256_equal": False,
        "absolute_tolerance": 1.0e-12,
        "relative_tolerance": 1.0e-10,
    }


def _classifier_source(
    checkpoint: dict[str, Any], array_name: str, shape: list[int]
) -> dict[str, Any]:
    logical_path = f"classifier_{array_name}.npy"
    allowed = _allowlist_record(checkpoint["checkpoint_sha256"], logical_path)
    return {
        "source_kind": f"selective_reuse_classifier_{array_name}_array",
        "logical_path": logical_path,
        "file_sha256": allowed["sha256"],
        "file_size": allowed["size"],
        "analysis_array_sha256": _sha(
            f"{checkpoint['checkpoint_sha256']}:{logical_path}:analysis"
        ),
        "analysis_shape": shape,
        "analysis_dtype": "float64",
        "bundle_id": checkpoint["checkpoint_sha256"],
        "source_allowlist_sha256": checkpoint["source_allowlist_sha256"],
    }


def _diagnostic_input_sources(
    checkpoint: dict[str, Any], dataset: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    checkpoint_sha = checkpoint["checkpoint_sha256"]
    counts = {
        "id_train": SPLIT_COUNTS["id_train"],
        "id_test": SPLIT_COUNTS["id_test"],
        "ood": SPLIT_COUNTS[dataset],
    }
    paths = {
        "id_train": "cifar10/train",
        "id_test": "cifar10/test",
        "ood": f"{dataset}/test",
    }
    features = {
        split: _selective_source_reference(
            checkpoint=checkpoint,
            logical_path=f"{paths[split]}/features.npy",
            source_kind="selective_reuse_feature_array",
            shape=[counts[split], FEATURE_DIMENSION],
            dtype="float32",
            array_identity=f"{checkpoint_sha}:{paths[split]}/features.npy",
        )
        for split in ("id_train", "id_test", "ood")
    }
    logits = {
        split: _selective_source_reference(
            checkpoint=checkpoint,
            logical_path=f"{paths[split]}/logits.npy",
            source_kind="selective_reuse_logit_array",
            shape=[counts[split], CLASS_COUNT],
            dtype="float32",
            array_identity=f"{checkpoint_sha}:{paths[split]}/logits.npy",
        )
        for split in ("id_train", "id_test", "ood")
    }
    labels = {
        "id_train": _selective_source_reference(
            checkpoint=checkpoint,
            logical_path="cifar10/train/class_labels.npy",
            source_kind="selective_reuse_class_label_array",
            shape=[counts["id_train"]],
            dtype="int64",
            array_identity=f"{checkpoint_sha}:cifar10/train/class_labels.npy",
        )
    }
    return features, logits, labels


def _ctm_normalization_status() -> dict[str, Any]:
    diagnostic = {
        "epsilon_norm": STAGE2.EPSILON_NORM,
        "exact_zero_count": 0,
        "near_zero_count": 0,
    }
    return {
        "status": "success",
        "reason_codes": [],
        "normalization_diagnostics": {
            "query": dict(diagnostic),
            "class_mean": dict(diagnostic),
        },
    }


def _balanced_class_counts(count: int) -> list[int]:
    quotient, remainder = divmod(count, CLASS_COUNT)
    return [quotient + (index < remainder) for index in range(CLASS_COUNT)]


def _diagnostic_score_distribution(
    *,
    num_id: int,
    num_ood: int,
    value: float,
    id_identity: str,
    ood_identity: str,
) -> dict[str, Any]:
    return {
        "summary_schema": STAGE2.SCORE_DISTRIBUTION_SCHEMA,
        "score_direction": "higher_is_more_id_like",
        "quantile_method": "numpy_linear",
        "quantile_probabilities": dict(STAGE2.FIXED_QUANTILE_PROBABILITIES),
        "id_test": _distribution(num_id, value, id_identity),
        "ood": _distribution(num_ood, value, ood_identity),
    }


def _diagnostic_identity(
    checkpoint: dict[str, Any], dataset: str
) -> dict[str, Any]:
    return {
        "schema_version": STAGE2.DIAGNOSTIC_METRIC_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "config_hash": checkpoint["config_hash"],
        "optimizer_family": checkpoint["optimizer_family"],
        "role_labels": checkpoint["role_labels"],
        "training_seed": checkpoint["training_seed"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_role": checkpoint["checkpoint_role"],
        "selection_status": "diagnostic_only",
        "candidate_eligibility": False,
        "ranking_use": "forbidden",
        "ood_dataset": dataset,
        "ood_group": "near" if dataset in NEAR else "far",
        "num_id": SPLIT_COUNTS["id_test"],
        "num_ood": SPLIT_COUNTS[dataset],
        "auroc": 0.61,
        "fpr95": 0.71,
        "fpr95_threshold": 0.0,
        "achieved_id_tpr": 0.95,
        "formula_reconstruction_max_abs_error": 0.0,
        "formula_reconstruction_max_rel_error": 0.0,
        "formula_reconstruction_pass": True,
        "source_allowlist_sha256": checkpoint["source_allowlist_sha256"],
        "numeric_runtime_fingerprint_sha256": NUMERIC_RUNTIME_SHA256,
        "status": "PASS",
        "reason_codes": [],
    }


def _pure_residual_diagnostic_row(
    checkpoint: dict[str, Any], dataset: str
) -> dict[str, Any]:
    checkpoint_sha = checkpoint["checkpoint_sha256"]
    num_id = SPLIT_COUNTS["id_test"]
    num_ood = SPLIT_COUNTS[dataset]
    features, logits, _ = _diagnostic_input_sources(checkpoint, dataset)
    split_records: dict[str, dict[str, Any]] = {}
    full_sources: dict[str, dict[str, Any]] = {}
    residual_sources: dict[str, dict[str, Any]] = {}
    derived_sources: dict[str, dict[str, Any]] = {}
    for alias, split_name, count in (
        ("id_test", "id_test", num_id),
        ("ood", dataset, num_ood),
    ):
        prefix = f"{checkpoint_sha}:pure-residual:{split_name}"
        residual_identity = f"{prefix}:residual"
        score_identity = f"{prefix}:score"
        full_identity = f"{prefix}:full-vim"
        split_records[alias] = {
            "residual": _diagnostic_component(
                shape=[count], value=4.0, identity=residual_identity
            ),
            "pure_residual_score": _diagnostic_component(
                shape=[count], value=-4.0, identity=score_identity
            ),
            "vim_energy": _diagnostic_component(
                shape=[count], value=3.0, identity=f"{prefix}:energy"
            ),
            "vim_alpha_residual": _diagnostic_component(
                shape=[count], value=1.0, identity=f"{prefix}:alpha-residual"
            ),
            "vim_full_score": _diagnostic_component(
                shape=[count], value=2.0, identity=full_identity
            ),
            "pure_residual_formula": _diagnostic_formula(
                reference_identity=score_identity
            ),
            "full_vim_formula": _diagnostic_formula(
                reference_identity=full_identity
            ),
        }
        full_sources[alias] = _stored_diagnostic_source(
            checkpoint=checkpoint,
            logical_path=(
                f"metrics/arrays/scores/{split_name}/"
                "detector__vim_author_dim.npy"
            ),
            source_kind="selective_reuse_score_array",
            count=count,
            observed_identity=full_identity,
        )
        residual_sources[alias] = _stored_diagnostic_source(
            checkpoint=checkpoint,
            logical_path=(
                f"metrics/arrays/scores/{split_name}/diagnostics/vim/residual.npy"
            ),
            source_kind="selective_reuse_diagnostic_array",
            count=count,
            observed_identity=residual_identity,
        )
        derived_sources[alias] = {
            "source_kind": (
                "derived_negative_of_selective_reuse_diagnostic_array"
            ),
            "derivation": "score_equals_negative_stored_vim_residual",
            "component_logical_path": residual_sources[alias]["logical_path"],
            "component_file_sha256": residual_sources[alias]["file_sha256"],
            "component_array_sha256": residual_sources[alias]["array_sha256"],
            "array_sha256": _sha(score_identity),
            "shape": [count],
            "dtype": "float64",
            "bundle_id": checkpoint_sha,
            "source_allowlist_sha256": checkpoint["source_allowlist_sha256"],
        }
    fit_prefix = f"{checkpoint_sha}:pure-residual:fit"
    metric = {
        "auroc": 0.61,
        "fpr95": 0.71,
        "fpr95_threshold": 0.0,
        "achieved_id_tpr": 0.95,
    }
    row = {
        **_diagnostic_identity(checkpoint, dataset),
        "diagnostic_id": "pure_residual_channel",
        "detector": "detector/vim_residual_author_dim",
        "channel_contract": {
            "pure_residual_uses_energy": False,
            "pure_residual_uses_alpha": False,
            "score_orientation": "negative_residual_id_like",
            "author_dim_rule": "floor(feature_dim/2)",
        },
        "score_sha256": _sha(f"{checkpoint_sha}:pure-residual:{dataset}:combined"),
        "score_distribution": _diagnostic_score_distribution(
            num_id=num_id,
            num_ood=num_ood,
            value=-4.0,
            id_identity=f"{checkpoint_sha}:pure-residual:id_test:score",
            ood_identity=f"{checkpoint_sha}:pure-residual:{dataset}:score",
        ),
        "stored_artifact_parity": {
            "status": "PASS",
            "reason_codes": [],
            "full_vim_score": {
                "status": "PASS",
                "reason_codes": [],
                "id_test": _full_array_parity(full_sources["id_test"], count=num_id),
                "ood": _full_array_parity(full_sources["ood"], count=num_ood),
            },
            "vim_residual": {
                "status": "PASS",
                "reason_codes": [],
                "id_test": _full_array_parity(
                    residual_sources["id_test"], count=num_id
                ),
                "ood": _full_array_parity(residual_sources["ood"], count=num_ood),
            },
        },
        "scalar_parity": {
            "status": "NOT_APPLICABLE",
            "reason_codes": [
                "V2_ONLY_PURE_RESIDUAL_HAS_NO_V1_2_SCALAR_BASELINE"
            ],
            "metrics": metric,
        },
        "score_components": {
            "fit": {
                "feature_dim": 640,
                "author_dim": 320,
                "residual_subspace_dim": 320,
                "author_dim_rule": "floor(feature_dim/2)",
                "origin": _diagnostic_component(
                    shape=[640], value=0.0, identity=f"{fit_prefix}:origin"
                ),
                "covariance_eigenvalues": _diagnostic_component(
                    shape=[640], value=1.0, identity=f"{fit_prefix}:eigenvalues"
                ),
                "residual_subspace": _diagnostic_component(
                    shape=[640, 320],
                    value=0.0,
                    identity=f"{fit_prefix}:subspace",
                ),
                "alpha": 1.0,
                "residual_mean": 4.0,
            },
            **split_records,
        },
        "source_references": {
            "source_allowlist_sha256": checkpoint["source_allowlist_sha256"],
            "feature_arrays": features,
            "logit_arrays": logits,
            "classifier_arrays": {
                "weight": _classifier_source(
                    checkpoint, "weight", [CLASS_COUNT, FEATURE_DIMENSION]
                ),
                "bias": _classifier_source(checkpoint, "bias", [CLASS_COUNT]),
            },
            "stored_full_vim_score_arrays": full_sources,
            "stored_vim_residual_arrays": residual_sources,
            "derived_score_arrays": derived_sources,
        },
    }
    return row


def _ctm_diagnostic_row(
    checkpoint: dict[str, Any], dataset: str
) -> dict[str, Any]:
    checkpoint_sha = checkpoint["checkpoint_sha256"]
    num_id = SPLIT_COUNTS["id_test"]
    num_ood = SPLIT_COUNTS[dataset]
    features, _, labels = _diagnostic_input_sources(checkpoint, dataset)
    split_records: dict[str, dict[str, Any]] = {}
    stored_sources: dict[str, dict[str, Any]] = {}
    for alias, split_name, count in (
        ("id_test", "id_test", num_id),
        ("ood", dataset, num_ood),
    ):
        prefix = f"{checkpoint_sha}:ctm:{split_name}"
        score_identity = f"{prefix}:score"
        split_records[alias] = {
            "matched_cosine": _diagnostic_component(
                shape=[count], value=0.5, identity=score_identity
            ),
            "matched_class": {
                "count": count,
                "dtype": "int64",
                "array_sha256": _sha(f"{prefix}:matched-class"),
                "class_counts": _balanced_class_counts(count),
            },
            "prototype_cosine_matrix": _diagnostic_component(
                shape=[count, CLASS_COUNT],
                value=0.5,
                identity=f"{prefix}:cosine-matrix",
            ),
            "normalization_status": _ctm_normalization_status(),
            "ctm_formula": _diagnostic_formula(
                reference_identity=score_identity
            ),
        }
        stored_sources[alias] = _stored_diagnostic_source(
            checkpoint=checkpoint,
            logical_path=(
                f"metrics/arrays/scores/{split_name}/"
                "detector__ctm_prototype_cosine.npy"
            ),
            source_kind="selective_reuse_score_array",
            count=count,
            observed_identity=score_identity,
        )
    train_counts = _balanced_class_counts(SPLIT_COUNTS["id_train"])
    fit_prefix = f"{checkpoint_sha}:ctm:fit"
    metric = {
        "auroc": 0.61,
        "fpr95": 0.71,
        "fpr95_threshold": 0.0,
        "achieved_id_tpr": 0.95,
    }
    row = {
        **_diagnostic_identity(checkpoint, dataset),
        "diagnostic_id": "ctm_angular_channel",
        "detector": "detector/ctm_prototype_cosine",
        "score_sha256": _sha(f"{checkpoint_sha}:ctm:{dataset}:combined"),
        "score_distribution": _diagnostic_score_distribution(
            num_id=num_id,
            num_ood=num_ood,
            value=0.5,
            id_identity=f"{checkpoint_sha}:ctm:id_test:score",
            ood_identity=f"{checkpoint_sha}:ctm:{dataset}:score",
        ),
        "stored_artifact_parity": {
            "status": "PASS",
            "reason_codes": [],
            "ctm_score": {
                "status": "PASS",
                "reason_codes": [],
                "id_test": _full_array_parity(
                    stored_sources["id_test"], count=num_id
                ),
                "ood": _full_array_parity(stored_sources["ood"], count=num_ood),
            },
        },
        "scalar_parity": {
            "status": "PASS",
            "reason_codes": [],
            "metrics": metric,
            "baseline_metrics": dict(metric),
            "absolute_tolerance": 1.0e-12,
            "relative_tolerance": 1.0e-10,
        },
        "score_components": {
            "fit": {
                "class_means": _diagnostic_component(
                    shape=[CLASS_COUNT, FEATURE_DIMENSION],
                    value=0.5,
                    identity=f"{fit_prefix}:class-means",
                ),
                "class_counts": train_counts,
                "normalization_status": _ctm_normalization_status(),
                "classwise_id_train_angular_concentration": {
                    "definition": (
                        "cosine_of_each_raw_id_train_feature_to_its_own_raw_"
                        "before_average_class_mean"
                    ),
                    "all_id_train": _distribution(
                        SPLIT_COUNTS["id_train"],
                        0.5,
                        f"{fit_prefix}:all-concentration",
                    ),
                    "classes": [
                        {
                            "class_index": class_index,
                            "sample_count": count,
                            "own_class_cosine": _distribution(
                                count,
                                0.5,
                                f"{fit_prefix}:class-{class_index}:concentration",
                            ),
                        }
                        for class_index, count in enumerate(train_counts)
                    ],
                },
            },
            **split_records,
        },
        "source_references": {
            "source_allowlist_sha256": checkpoint["source_allowlist_sha256"],
            "feature_arrays": features,
            "class_label_arrays": labels,
            "stored_score_arrays": stored_sources,
        },
    }
    return row


DiagnosticMutation = Callable[[list[dict[str, Any]]], None]


def _diagnostic_rows(
    *, mutate: DiagnosticMutation | None = None
) -> list[dict[str, Any]]:
    rows = [
        builder(checkpoint, dataset)
        for checkpoint in CHECKPOINTS
        for dataset in DATASETS
        for builder in (_pure_residual_diagnostic_row, _ctm_diagnostic_row)
    ]
    if mutate is not None:
        mutate(rows)
    rows.sort(
        key=lambda row: (
            row["diagnostic_id"],
            row["config_hash"],
            row["training_seed"],
            DATASETS.index(row["ood_dataset"]),
        )
    )
    assert len(rows) == 360
    return rows


def _worker_code_sources() -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "sha256": hashlib.sha256(
                (REPOSITORY_ROOT / relative).read_bytes()
            ).hexdigest(),
            "size": (REPOSITORY_ROOT / relative).stat().st_size,
        }
        for relative in STAGE2.WORKER_CODE_SOURCE_PATHS
    ]


def _baseline_source_files() -> list[dict[str, Any]]:
    return [
        {
            key: item[key]
            for key in ("logical_path", "local_path", "sha256", "size")
        }
        for item in sorted(
            REUSE["parity_baseline"]["files"],
            key=lambda row: row["logical_path"],
        )
    ]


def _write_inputs(
    root: Path,
    rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    root.mkdir(parents=True)
    metrics_path = root / "model_metrics.jsonl"
    metrics_data = _jsonl_bytes(rows)
    metrics_path.write_bytes(metrics_data)
    metrics_sha = hashlib.sha256(metrics_data).hexdigest()
    diagnostics_path = root / "diagnostic_metrics.jsonl"
    diagnostic_rows = _diagnostic_rows() if diagnostic_rows is None else diagnostic_rows
    diagnostics_data = _jsonl_bytes(diagnostic_rows)
    diagnostics_path.write_bytes(diagnostics_data)
    diagnostics_sha = hashlib.sha256(diagnostics_data).hexdigest()
    manifest = {
        "schema_version": STAGE2.SELECTION_SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "contract_version": "fixed_readout_training_rule_intervention_v2",
        "analysis_role": STAGE2.WORKER_ANALYSIS_ROLE,
        "analysis_git_sha": TEST_GIT_SHA,
        "checkpoint_role": "last",
        "inventory_hash": POLICY["authorities"]["v1_2_checkpoint_inventory"][
            "inventory_hash"
        ],
        "dataset_policy_hash": POLICY["authorities"]["v1_2_checkpoint_inventory"][
            "dataset_policy_hash"
        ],
        "evaluation_git_sha": REUSE["bundles"][0]["identity"][
            "evaluation_git_sha"
        ],
        "checkpoint_count": 30,
        "unique_config_count": 10,
        "seed_set": [0, 1, 2],
        "candidate_selection_seed_set": [1, 2],
        "seed_zero_role_selection_bias": True,
        "ood_datasets": list(DATASETS),
        "candidate_registry_sha256": hashlib.sha256(
            CANDIDATE_REGISTRY.read_bytes()
        ).hexdigest(),
        "gate_policy_sha256": hashlib.sha256(GATE_POLICY.read_bytes()).hexdigest(),
        "candidate_ids": list(
            REGISTRY["selection_boundary"]["eligible_candidate_ids"]
        ),
        "diagnostic_ids": list(STAGE2.DIAGNOSTIC_IDS),
        "diagnostic_population": {
            "row_count": len(diagnostic_rows),
            "diagnostic_count": len(STAGE2.DIAGNOSTIC_IDS),
            "rows_per_diagnostic": len(CHECKPOINTS) * len(DATASETS),
            "selection_status": "diagnostic_only",
            "candidate_eligibility": False,
            "ranking_use": "forbidden",
            "status": "PASS",
            "reason_codes": [],
        },
        "checkpoints": CHECKPOINTS,
        "files": [
            {
                "logical_path": "model_metrics.jsonl",
                "sha256": metrics_sha,
                "size": len(metrics_data),
                "row_count": len(rows),
                "schema_version": STAGE2.MODEL_METRIC_SCHEMA,
            },
            {
                "logical_path": "diagnostic_metrics.jsonl",
                "sha256": diagnostics_sha,
                "size": len(diagnostics_data),
                "row_count": len(diagnostic_rows),
                "schema_version": STAGE2.DIAGNOSTIC_METRIC_SCHEMA,
            },
        ],
        "destination_policy": "atomic_new_directory_no_overwrite",
        "scientific_gate": {
            "status": "NOT_RUN",
            "reason_codes": ["STAGE2_GATE_REDUCER_NOT_RUN"],
        },
        "source_reuse_manifest_sha256": hashlib.sha256(
            REUSE_MANIFEST.read_bytes()
        ).hexdigest(),
        "numeric_runtime_fingerprint": {
            "canonical_lock_sha256": NUMERIC_RUNTIME_SHA256,
        },
        "source_files": _baseline_source_files(),
        "code_files": _worker_code_sources(),
    }
    selection_path = root / "selection_manifest.json"
    selection_data = _json_bytes(manifest)
    selection_path.write_bytes(selection_data)
    selection_sha = hashlib.sha256(selection_data).hexdigest()
    checksums_path = root / "checksums.sha256"
    checksums_path.write_text(
        f"{selection_sha}  selection_manifest.json\n"
        f"{metrics_sha}  model_metrics.jsonl\n"
        f"{diagnostics_sha}  diagnostic_metrics.jsonl\n",
        encoding="utf-8",
    )
    return selection_path, metrics_path, diagnostics_path, checksums_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _test_reducer_sources() -> list[dict[str, Any]]:
    paths = (
        SCRIPT_PATH,
        CANDIDATE_REGISTRY,
        GATE_POLICY,
        REUSE_MANIFEST,
        STAGE2.DEFAULT_CHECKPOINT_INVENTORY,
    )
    return [
        {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in paths
    ]


def _execute(
    root: Path,
    rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path]:
    selection, metrics, diagnostics, checksums = _write_inputs(
        root / "input", rows, diagnostic_rows
    )
    output = root / "output"
    result = STAGE2.run_reducer(
        selection_manifest_path=selection,
        model_metrics_path=metrics,
        diagnostic_metrics_path=diagnostics,
        input_checksums_path=checksums,
        output_directory=output,
        validate_git_sources=False,
        reducer_git_sha="f" * 40,
        reducer_code_sources=_test_reducer_sources(),
    )
    return result, output


def test_pass_exact_estimands_outputs_and_no_overwrite(tmp_path: Path) -> None:
    result, output = _execute(tmp_path, _model_rows())
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    summaries = _read_jsonl(output / "dataset_summaries.jsonl")
    contrasts = _read_jsonl(output / "contrast_metrics.jsonl")
    folds = _read_jsonl(output / "cv_folds.jsonl")
    diagnostic_summaries = _read_jsonl(output / "diagnostic_summaries.jsonl")

    assert result["status"] == "PASS"
    assert decision["selected_candidate_id"] == "radial_l2_mahalanobis"
    assert decision["scientific_launch_allowed"] is False
    assert decision["provenance"]["source_validation_mode"] == (
        "test_only_explicit_bypass"
    )
    assert decision["provenance"]["evidence_provenance_validation_mode"] == (
        "test_only_working_tree_and_manifest_metadata"
    )
    assert decision["provenance"]["ctm_baseline_validation_mode"] == (
        "test_only_embedded_scalar_parity"
    )
    assert {
        item["path"] for item in decision["provenance"]["reducer_code_sources"]
    } >= {
        "scripts/reduce_fixed_readout_stage2_gate.py",
        "configs/evaluation/fixed_readout_stage2/candidate_registry.json",
        "configs/evaluation/fixed_readout_stage2/gate_policy.json",
    }
    assert decision["lodo_selection_count"] == 6
    assert decision["loco_selection_count"] == 10
    assert len(summaries) == 2 * 6 * 3
    assert len(folds) == 6 + 10
    assert len(diagnostic_summaries) == 2 * 6
    assert {
        row["diagnostic_id"] for row in diagnostic_summaries
    } == {"pure_residual_channel", "ctm_angular_channel"}
    assert all(
        row["selection_status"] == "diagnostic_only"
        and row["candidate_eligibility"] is False
        and row["ranking_use"] == "forbidden"
        and row["checkpoint_count"] == 30
        for row in diagnostic_summaries
    )
    eligible = set(REGISTRY["selection_boundary"]["eligible_candidate_ids"])
    assert {
        row["candidate_id"] for row in decision["candidate_ranking"]
    }.issubset(eligible)
    assert eligible.isdisjoint(
        row["diagnostic_id"] for row in diagnostic_summaries
    )
    assert {row["role_code"] for row in contrasts} == {"C1", "C3", "C4"}
    required_outputs = POLICY["required_output_artifacts"]
    assert all(
        set(required_outputs["contrast_metrics.jsonl"]).issubset(row)
        for row in contrasts
    )
    assert all(
        set(required_outputs["dataset_summaries.jsonl"]).issubset(row)
        for row in summaries
    )
    assert all(
        set(required_outputs["cv_folds.jsonl"]).issubset(row) for row in folds
    )
    assert set(required_outputs["gate_decision.json"]).issubset(decision)
    combined = next(
        row
        for row in summaries
        if row["candidate_id"] == "radial_l2_mahalanobis"
        and row["ood_dataset"] == "cifar100"
        and row["selection_seed_set"] == [1, 2]
    )
    assert combined["config_pair_count"] == 45
    assert combined["raw_gap"] == pytest.approx(0.06)
    assert combined["transformed_gap"] == pytest.approx(0.015)
    assert combined["absolute_attenuation"] == pytest.approx(0.045)
    assert combined["seed_noise_q90"] == pytest.approx(0.0002)
    assert not (output / "preconfirmation_freeze.json").exists()

    checksums = {
        line.split()[1]: line.split()[0]
        for line in (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    }
    assert set(checksums) == set(STAGE2.OUTPUT_FILENAMES)
    for filename, expected in checksums.items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    with pytest.raises(FileExistsError):
        STAGE2.write_outputs_no_overwrite(
            {
                "contrast_metrics.jsonl": contrasts,
                "dataset_summaries.jsonl": summaries,
                "cv_folds.jsonl": folds,
                "gate_decision.json": decision,
            },
            output,
        )


def test_diagnostic_summaries_are_descriptive_and_cannot_change_ranking(
    tmp_path: Path,
) -> None:
    selection, metrics, diagnostics, checksums = _write_inputs(
        tmp_path / "input", _model_rows(), _diagnostic_rows()
    )
    context = STAGE2.load_inputs(
        selection_manifest_path=selection,
        model_metrics_path=metrics,
        diagnostic_metrics_path=diagnostics,
        input_checksums_path=checksums,
        validate_git_sources=False,
        reducer_git_sha=TEST_GIT_SHA,
        reducer_code_sources=_test_reducer_sources(),
    )
    baseline = STAGE2.reduce_gate(context)
    changed_diagnostics = json.loads(json.dumps(context.diagnostic_rows))
    for row in changed_diagnostics:
        row["auroc"] = 0.99 if row["diagnostic_id"] == "ctm_angular_channel" else 0.01
        row["fpr95"] = 0.01 if row["diagnostic_id"] == "ctm_angular_channel" else 0.99
        row["scalar_parity"]["metrics"]["auroc"] = row["auroc"]
        row["scalar_parity"]["metrics"]["fpr95"] = row["fpr95"]
    changed = STAGE2.reduce_gate(
        replace(context, diagnostic_rows=tuple(changed_diagnostics))
    )

    baseline_decision = baseline["gate_decision.json"]
    changed_decision = changed["gate_decision.json"]
    assert baseline_decision["status"] == changed_decision["status"] == "PASS"
    for field in (
        "status",
        "selected_candidate_id",
        "full_data_candidate_id",
        "candidate_ranking",
        "selection_score",
        "lodo_selection_count",
        "loco_selection_count",
    ):
        assert changed_decision[field] == baseline_decision[field]
    for filename in (
        "contrast_metrics.jsonl",
        "dataset_summaries.jsonl",
        "cv_folds.jsonl",
    ):
        assert changed[filename] == baseline[filename]
    assert changed["diagnostic_summaries.jsonl"] != baseline[
        "diagnostic_summaries.jsonl"
    ]
    assert {
        row["candidate_id"] for row in changed_decision["candidate_ranking"]
    } == set(REGISTRY["selection_boundary"]["eligible_candidate_ids"])


def test_non_exact_diagnostic_array_sha_can_pass_frozen_allclose_contract(
    tmp_path: Path,
) -> None:
    diagnostics = _diagnostic_rows()
    parity_records = []
    for row in diagnostics:
        parity = row["stored_artifact_parity"]
        groups = (
            (parity["full_vim_score"], parity["vim_residual"])
            if row["diagnostic_id"] == "pure_residual_channel"
            else (parity["ctm_score"],)
        )
        parity_records.extend(
            group[split] for group in groups for split in ("id_test", "ood")
        )
    assert all(
        record["status"] == "PASS"
        and record["exact_array_sha256_equal"] is False
        and record["observed_array_sha256"] != record["stored_array_sha256"]
        for record in parity_records
    )
    result, _ = _execute(tmp_path, _model_rows(), diagnostics)
    assert result["status"] == "PASS"


def test_output_inside_evidence_input_tree_is_rejected_before_reduction(
    tmp_path: Path,
) -> None:
    selection, metrics, diagnostics, checksums = _write_inputs(
        tmp_path / "input", _model_rows()
    )
    unsafe_output = selection.parent / "gate-output"
    with pytest.raises(ValueError, match="input trees must be disjoint"):
        STAGE2.run_reducer(
            selection_manifest_path=selection,
            model_metrics_path=metrics,
            diagnostic_metrics_path=diagnostics,
            input_checksums_path=checksums,
            output_directory=unsafe_output,
            validate_git_sources=False,
            reducer_git_sha="f" * 40,
            reducer_code_sources=_test_reducer_sources(),
        )
    assert not unsafe_output.exists()


def test_output_parent_symlink_is_rejected_before_input_read(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="parent chain.*symlink"):
        STAGE2.run_reducer(
            selection_manifest_path=tmp_path / "missing-selection.json",
            model_metrics_path=tmp_path / "missing-metrics.jsonl",
            diagnostic_metrics_path=tmp_path / "missing-diagnostics.jsonl",
            input_checksums_path=tmp_path / "missing-checksums.sha256",
            output_directory=linked_parent / "gate-output",
            validate_git_sources=False,
            reducer_git_sha="f" * 40,
            reducer_code_sources=_test_reducer_sources(),
        )
    assert not (real_parent / "gate-output").exists()


def test_broken_output_symlink_and_missing_parent_are_rejected(tmp_path: Path) -> None:
    broken_output = tmp_path / "broken-output"
    broken_output.symlink_to(tmp_path / "absent-target", target_is_directory=True)
    with pytest.raises(FileExistsError):
        STAGE2.validate_output_destination(broken_output)
    with pytest.raises(FileNotFoundError, match="parent must already exist"):
        STAGE2.validate_output_destination(
            tmp_path / "absent-parent" / "gate-output"
        )


def test_reducer_output_inside_git_worktree_is_rejected(tmp_path: Path) -> None:
    selection, metrics, diagnostics, checksums = _write_inputs(
        tmp_path / "input", _model_rows()
    )
    unsafe_output = REPOSITORY_ROOT / f".stage2-gate-test-{_sha(str(tmp_path))}"
    assert not unsafe_output.exists()
    with pytest.raises(ValueError, match="outside the Git worktree"):
        STAGE2.run_reducer(
            selection_manifest_path=selection,
            model_metrics_path=metrics,
            diagnostic_metrics_path=diagnostics,
            input_checksums_path=checksums,
            output_directory=unsafe_output,
            validate_git_sources=False,
            reducer_git_sha="f" * 40,
            reducer_code_sources=_test_reducer_sources(),
        )
    assert not unsafe_output.exists()


def test_no_go_when_no_candidate_meets_attenuation(tmp_path: Path) -> None:
    profile = {
        candidate: {
            "raw": {seed: 0.018 for seed in (0, 1, 2)},
            "transformed": {seed: 0.017 for seed in (0, 1, 2)},
        }
        for candidate in REGISTRY["selection_boundary"]["eligible_candidate_ids"]
    }
    result, output = _execute(tmp_path, _model_rows(profile=profile))
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "NO_GO"
    assert decision["full_data_candidate_id"] is None
    assert decision["selected_candidate_id"] is None


def test_inconclusive_when_single_seed_view_disagrees(tmp_path: Path) -> None:
    profile = json.loads(json.dumps(PASS_PROFILE))
    profile["radial_l2_mahalanobis"]["transformed"] = {
        "0": 0.005,
        "1": 0.005,
        "2": 0.020,
    }
    profile = {
        candidate: {
            state: {int(seed): value for seed, value in by_seed.items()}
            for state, by_seed in by_state.items()
        }
        for candidate, by_state in profile.items()
    }
    result, output = _execute(tmp_path, _model_rows(profile=profile))
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "INCONCLUSIVE"
    assert decision["full_data_candidate_id"] == "radial_l2_mahalanobis"
    assert decision["seed1_selected_candidate"] == "radial_l2_mahalanobis"
    assert decision["seed2_selected_candidate"] is None
    assert decision["selected_candidate_id"] is None


def test_failed_precedes_no_go_on_hard_validity_failure(tmp_path: Path) -> None:
    profile = {
        candidate: {
            "raw": {seed: 0.018 for seed in (0, 1, 2)},
            "transformed": {seed: 0.017 for seed in (0, 1, 2)},
        }
        for candidate in REGISTRY["selection_boundary"]["eligible_candidate_ids"]
    }

    def fail_formula(rows: list[dict[str, Any]]) -> None:
        rows[0]["formula_reconstruction_pass"] = False

    result, output = _execute(
        tmp_path, _model_rows(profile=profile, mutate=fail_formula)
    )
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert "FORMULA_RECONSTRUCTION_FAILED" in decision["criteria"]["hard_validity"][
        "reason_codes"
    ]
    assert decision["selected_candidate_id"] is None


def test_deterministic_priority_tie_break(tmp_path: Path) -> None:
    tied = {
        candidate: {
            "raw": {seed: 0.020 for seed in (0, 1, 2)},
            "transformed": {seed: 0.005 for seed in (0, 1, 2)},
        }
        for candidate in REGISTRY["selection_boundary"]["eligible_candidate_ids"]
    }
    _, output = _execute(tmp_path, _model_rows(profile=tied))
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "PASS"
    assert decision["selected_candidate_id"] == "radial_l2_mahalanobis"
    assert [item["candidate_id"] for item in decision["candidate_ranking"][:2]] == [
        "radial_l2_mahalanobis",
        "radial_l2_knn",
    ]


def test_seed_zero_cannot_leak_into_selection_or_margins(tmp_path: Path) -> None:
    baseline_result, baseline_output = _execute(tmp_path / "baseline", _model_rows())

    def mutate_seed_zero(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["training_seed"] == 0:
                config_index = CONFIGS.index(row["config_hash"])
                row["auroc"] = (
                    0.08 + 0.07 * config_index
                    if row["state"] == "raw"
                    else 0.92 - 0.07 * config_index
                )

    changed_result, changed_output = _execute(
        tmp_path / "changed", _model_rows(mutate=mutate_seed_zero)
    )
    assert baseline_result["status"] == changed_result["status"] == "PASS"
    baseline_decision = json.loads(
        (baseline_output / "gate_decision.json").read_text(encoding="utf-8")
    )
    changed_decision = json.loads(
        (changed_output / "gate_decision.json").read_text(encoding="utf-8")
    )
    for field in (
        "status",
        "selected_candidate_id",
        "full_data_candidate_id",
        "lodo_selection_count",
        "loco_selection_count",
        "seed1_selected_candidate",
        "seed2_selected_candidate",
        "candidate_ranking",
        "selection_score",
    ):
        assert changed_decision[field] == baseline_decision[field]
    assert (changed_output / "dataset_summaries.jsonl").read_bytes() == (
        baseline_output / "dataset_summaries.jsonl"
    ).read_bytes()
    assert (changed_output / "cv_folds.jsonl").read_bytes() == (
        baseline_output / "cv_folds.jsonl"
    ).read_bytes()
    assert (changed_output / "contrast_metrics.jsonl").read_bytes() != (
        baseline_output / "contrast_metrics.jsonl"
    ).read_bytes()


def test_lodo_training_selection_excludes_heldout_dataset(tmp_path: Path) -> None:
    _, baseline_output = _execute(tmp_path / "baseline", _model_rows())

    def remove_heldout_attenuation(rows: list[dict[str, Any]]) -> None:
        raw_by_key = {
            (row["candidate_id"], row["checkpoint_sha256"], row["ood_dataset"]): row
            for row in rows
            if row["state"] == "raw"
        }
        for row in rows:
            if (
                row["candidate_id"] == "radial_l2_mahalanobis"
                and row["ood_dataset"] == "cifar100"
                and row["state"] == "transformed"
            ):
                row["auroc"] = raw_by_key[
                    (row["candidate_id"], row["checkpoint_sha256"], row["ood_dataset"])
                ]["auroc"]

    _, changed_output = _execute(
        tmp_path / "changed", _model_rows(mutate=remove_heldout_attenuation)
    )
    baseline = next(
        row
        for row in _read_jsonl(baseline_output / "cv_folds.jsonl")
        if row["fold_id"] == "lodo:cifar100"
    )
    changed = next(
        row
        for row in _read_jsonl(changed_output / "cv_folds.jsonl")
        if row["fold_id"] == "lodo:cifar100"
    )
    assert "cifar100" not in changed["training_ood_datasets"]
    for field in (
        "training_ood_datasets",
        "selected_candidate_id",
        "selection_score",
        "tie_break_trace",
        "training_summaries_sha256",
    ):
        assert changed[field] == baseline[field]
    assert changed["heldout_absolute_attenuation"] != baseline[
        "heldout_absolute_attenuation"
    ]
    assert changed["heldout_pass"] is False


def test_loco_training_selection_excludes_all_heldout_config_seeds(
    tmp_path: Path,
) -> None:
    heldout_config = CONFIGS[-1]
    _, baseline_output = _execute(tmp_path / "baseline", _model_rows())

    def remove_heldout_config_attenuation(rows: list[dict[str, Any]]) -> None:
        raw_by_key = {
            (row["candidate_id"], row["checkpoint_sha256"], row["ood_dataset"]): row
            for row in rows
            if row["state"] == "raw"
        }
        for row in rows:
            if (
                row["candidate_id"] == "radial_l2_mahalanobis"
                and row["config_hash"] == heldout_config
                and row["state"] == "transformed"
            ):
                row["auroc"] = raw_by_key[
                    (row["candidate_id"], row["checkpoint_sha256"], row["ood_dataset"])
                ]["auroc"]

    _, changed_output = _execute(
        tmp_path / "changed",
        _model_rows(mutate=remove_heldout_config_attenuation),
    )
    fold_id = f"loco:{heldout_config}"
    baseline = next(
        row
        for row in _read_jsonl(baseline_output / "cv_folds.jsonl")
        if row["fold_id"] == fold_id
    )
    changed = next(
        row
        for row in _read_jsonl(changed_output / "cv_folds.jsonl")
        if row["fold_id"] == fold_id
    )
    assert heldout_config not in changed["training_config_hashes"]
    for field in (
        "training_config_hashes",
        "selected_candidate_id",
        "selection_score",
        "tie_break_trace",
        "training_summaries_sha256",
    ):
        assert changed[field] == baseline[field]
    assert changed["heldout_absolute_attenuation"] != baseline[
        "heldout_absolute_attenuation"
    ]


def test_lodo_heldout_null_cannot_enter_training_fold_margin(tmp_path: Path) -> None:
    _, baseline_output = _execute(tmp_path / "baseline", _model_rows())

    def mutate_heldout_null(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["ood_dataset"] != "cifar100":
                continue
            config_index = CONFIGS.index(row["config_hash"])
            oracle = row["exact_null_oracle"]
            oracle["transformed_auroc"] = oracle["raw_auroc"] + 0.002 * config_index
            oracle["auroc_abs_drift"] = abs(
                oracle["transformed_auroc"] - oracle["raw_auroc"]
            )

    _, changed_output = _execute(
        tmp_path / "changed", _model_rows(mutate=mutate_heldout_null)
    )
    baseline = next(
        row
        for row in _read_jsonl(baseline_output / "cv_folds.jsonl")
        if row["fold_id"] == "lodo:cifar100"
    )
    changed = next(
        row
        for row in _read_jsonl(changed_output / "cv_folds.jsonl")
        if row["fold_id"] == "lodo:cifar100"
    )
    for field in (
        "selected_candidate_id",
        "selection_score",
        "tie_break_trace",
        "training_summaries_sha256",
        "training_null_attenuation_q90",
        "heldout_practical_attenuation_margin",
    ):
        assert changed[field] == baseline[field]


def test_scalar_parity_row_corruption_forces_failed(tmp_path: Path) -> None:
    rows = _model_rows()
    rows[0]["scalar_parity"]["metrics"]["auroc"] += 0.01
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert "SCALAR_PARITY_ROW_MISMATCH" in decision["criteria"]["hard_validity"][
        "reason_codes"
    ]


def test_transition_credit_corruption_forces_failed(tmp_path: Path) -> None:
    rows = _model_rows()
    first = rows[0]
    pair_key = (
        first["candidate_id"],
        first["checkpoint_sha256"],
        first["ood_dataset"],
    )
    for row in rows:
        if (
            row["candidate_id"],
            row["checkpoint_sha256"],
            row["ood_dataset"],
        ) == pair_key:
            row["corrected_pair_mass"] += 0.01
            row["harmed_pair_mass"] += 0.01
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert "PAIR_CREDIT_MASS_RECONSTRUCTION_FAILED" in decision["criteria"][
        "hard_validity"
    ]["reason_codes"]


def test_manifest_authority_and_exact_reuse_identity_are_fail_closed(
    tmp_path: Path,
) -> None:
    selection, metrics, diagnostics, checksums = _write_inputs(
        tmp_path / "input", _model_rows()
    )
    manifest = json.loads(selection.read_text(encoding="utf-8"))
    manifest["checkpoints"][0]["source_allowlist_sha256"] = "0" * 64
    selection_data = _json_bytes(manifest)
    selection.write_bytes(selection_data)
    checksums.write_text(
        f"{hashlib.sha256(selection_data).hexdigest()}  selection_manifest.json\n"
        f"{hashlib.sha256(metrics.read_bytes()).hexdigest()}  model_metrics.jsonl\n"
        f"{hashlib.sha256(diagnostics.read_bytes()).hexdigest()}  diagnostic_metrics.jsonl\n",
        encoding="utf-8",
    )
    output = tmp_path / "failed-output"
    result = STAGE2.run_reducer(
        selection_manifest_path=selection,
        model_metrics_path=metrics,
        diagnostic_metrics_path=diagnostics,
        input_checksums_path=checksums,
        output_directory=output,
        validate_git_sources=False,
        reducer_git_sha="f" * 40,
        reducer_code_sources=_test_reducer_sources(),
    )
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["criteria"]["hard_validity"]["reason_codes"] == [
        "ARTIFACT_IDENTITY_FAILED"
    ]
    assert _read_jsonl(output / "contrast_metrics.jsonl") == []
    assert _read_jsonl(output / "dataset_summaries.jsonl") == []
    assert _read_jsonl(output / "cv_folds.jsonl") == []


@pytest.mark.parametrize(
    "corruption", ("analysis_git_sha", "worker_code_file", "baseline_source_file")
)
def test_self_consistent_manifest_rewrite_cannot_bypass_evidence_provenance(
    tmp_path: Path, corruption: str
) -> None:
    selection, metrics, diagnostics, checksums = _write_inputs(
        tmp_path / "input", _model_rows()
    )
    manifest = json.loads(selection.read_text(encoding="utf-8"))
    if corruption == "analysis_git_sha":
        manifest["analysis_git_sha"] = "e" * 40
    elif corruption == "worker_code_file":
        manifest["code_files"][0]["sha256"] = "0" * 64
    else:
        manifest["source_files"][0]["sha256"] = "0" * 64
    selection_data = _json_bytes(manifest)
    selection.write_bytes(selection_data)
    checksums.write_text(
        f"{hashlib.sha256(selection_data).hexdigest()}  selection_manifest.json\n"
        f"{hashlib.sha256(metrics.read_bytes()).hexdigest()}  model_metrics.jsonl\n"
        f"{hashlib.sha256(diagnostics.read_bytes()).hexdigest()}  diagnostic_metrics.jsonl\n",
        encoding="utf-8",
    )
    output = tmp_path / "failed-output"
    result = STAGE2.run_reducer(
        selection_manifest_path=selection,
        model_metrics_path=metrics,
        diagnostic_metrics_path=diagnostics,
        input_checksums_path=checksums,
        output_directory=output,
        validate_git_sources=False,
        reducer_git_sha=TEST_GIT_SHA,
        reducer_code_sources=_test_reducer_sources(),
    )
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["scientific_launch_allowed"] is False
    assert decision["candidate_ranking"] == []


def test_production_worker_source_validation_rejects_worktree_head_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = STAGE2.WORKER_CODE_SOURCE_PATHS[0]

    def committed_bytes(relative: str) -> bytes:
        if relative == drifted:
            return b"different committed worker source"
        return (REPOSITORY_ROOT / relative).read_bytes()

    monkeypatch.setattr(STAGE2, "_git_bytes", committed_bytes)
    with pytest.raises(ValueError, match="worker source differs from HEAD"):
        STAGE2._expected_worker_code_sources(validate_git_sources=True)


@pytest.mark.parametrize(
    "corruption",
    (None, "hash", "size", "relative_path", "symlink", "file_content"),
)
def test_relocated_baseline_source_is_verified_fail_closed(
    tmp_path: Path, corruption: str | None
) -> None:
    payload = b"synthetic frozen scalar baseline\n"
    relocated = tmp_path / "relocated" / "per_checkpoint_records.jsonl"
    relocated.parent.mkdir()
    relocated.write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()
    expected = {
        "logical_path": "per_checkpoint_records.jsonl",
        "local_path": "/original/host/path/per_checkpoint_records.jsonl",
        "sha256": checksum,
        "size": len(payload),
    }
    observed = {
        **expected,
        "local_path": str(relocated),
    }
    if corruption == "hash":
        observed["sha256"] = "0" * 64
    elif corruption == "size":
        observed["size"] += 1
    elif corruption == "relative_path":
        observed["local_path"] = "relative/per_checkpoint_records.jsonl"
    elif corruption == "symlink":
        linked = tmp_path / "relocated-link.jsonl"
        linked.symlink_to(relocated)
        observed["local_path"] = str(linked)
    elif corruption == "file_content":
        relocated.write_bytes(b"x" * len(payload))

    if corruption is None:
        validated = STAGE2._validate_relocated_baseline_source(
            observed,
            expected=expected,
            label="relocated baseline",
            verify_file=True,
        )
        assert validated["local_path"] == str(relocated)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            STAGE2._validate_relocated_baseline_source(
                observed,
                expected=expected,
                label="relocated baseline",
                verify_file=True,
            )


def test_input_checksum_failure_emits_atomic_failed_decision(tmp_path: Path) -> None:
    selection, metrics, diagnostics, checksums = _write_inputs(
        tmp_path / "input", _model_rows()
    )
    checksums.write_text(
        f"{'0' * 64}  selection_manifest.json\n"
        f"{hashlib.sha256(metrics.read_bytes()).hexdigest()}  model_metrics.jsonl\n"
        f"{hashlib.sha256(diagnostics.read_bytes()).hexdigest()}  diagnostic_metrics.jsonl\n",
        encoding="utf-8",
    )
    output = tmp_path / "failed-output"
    result = STAGE2.run_reducer(
        selection_manifest_path=selection,
        model_metrics_path=metrics,
        diagnostic_metrics_path=diagnostics,
        input_checksums_path=checksums,
        output_directory=output,
        validate_git_sources=False,
        reducer_git_sha="f" * 40,
        reducer_code_sources=_test_reducer_sources(),
    )
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert decision["criteria"]["hard_validity"]["reason_codes"] == [
        "ARTIFACT_CHECKSUM_FAILED"
    ]
    checksum_records = (output / "checksums.sha256").read_text(encoding="utf-8")
    assert "gate_decision.json" in checksum_records


def test_allclose_sized_target_change_cannot_hide_rank_change(tmp_path: Path) -> None:
    rows = _model_rows()
    target = next(
        row for row in rows if row["candidate_id"] == "radial_l2_mahalanobis"
    )
    pair_key = (
        target["candidate_id"],
        target["checkpoint_sha256"],
        target["ood_dataset"],
    )
    for row in rows:
        if (
            row["candidate_id"],
            row["checkpoint_sha256"],
            row["ood_dataset"],
        ) == pair_key:
            row["non_invariant_witness"]["targeted_score_max_abs_error"] = 1.0e-13
            row["non_invariant_witness"][
                "targeted_rank_disagreement_fraction"
            ] = 0.01
            row["non_invariant_witness"]["targeted_auroc_abs_drift"] = 0.0
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert "TARGETED_WITNESS_EXACT_RELATION_FAILED" in decision["criteria"][
        "hard_validity"
    ]["reason_codes"]


@pytest.mark.parametrize("corruption", ["missing", "malformed", "count"])
def test_distribution_evidence_corruption_hard_fails(
    tmp_path: Path, corruption: str
) -> None:
    rows = _model_rows()
    if corruption == "missing":
        rows[0].pop("score_distribution")
    elif corruption == "malformed":
        rows[0]["score_distribution"]["id_test"]["mean"] = "not-finite"
    else:
        rows[0]["score_distribution"]["id_test"]["count"] += 1
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["scientific_launch_allowed"] is False
    assert _read_jsonl(output / "contrast_metrics.jsonl") == []
    assert _read_jsonl(output / "dataset_summaries.jsonl") == []
    assert _read_jsonl(output / "cv_folds.jsonl") == []


def test_checksum_bound_source_reference_corruption_hard_fails(
    tmp_path: Path,
) -> None:
    rows = _model_rows()
    rows[0]["source_references"]["feature_arrays"]["id_test"][
        "file_sha256"
    ] = "0" * 64
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert decision["criteria"]["hard_validity"]["reason_codes"] == [
        "ARTIFACT_CHECKSUM_FAILED"
    ]


def test_digest_only_component_evidence_cannot_pass(tmp_path: Path) -> None:
    rows = _model_rows()
    rows[0]["score_components"]["id_test"] = {"sha256": _sha("digest-only")}
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["scientific_launch_allowed"] is False


def test_unsuccessful_witness_norm_status_hard_fails(tmp_path: Path) -> None:
    rows = _model_rows()
    preservation = rows[0]["non_invariant_witness"][
        "normalization_status_preservation"
    ]
    for side in ("base", "witness"):
        preservation["splits"]["id_test"][side]["status"] = "degenerate"
        preservation["splits"]["id_test"][side]["reason_codes"] = [
            "near_zero_norm"
        ]
        preservation["splits"]["id_test"][side]["near_zero_count"] = 1
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["scientific_launch_allowed"] is False


def test_raw_transformed_geometry_duplicate_mismatch_forces_failed(
    tmp_path: Path,
) -> None:
    rows = _model_rows()
    target = next(row for row in rows if row["state"] == "raw")
    target["geometry_channel"]["splits"]["id_test"]["feature_norm"][
        "array_sha256"
    ] = _sha("different-geometry-array")
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    reasons = decision["criteria"]["hard_validity"]["reason_codes"]
    assert "RAW_TRANSFORMED_GEOMETRY_CHANNEL_MISMATCH" in reasons
    assert "GEOMETRY_CHANNEL_POPULATION_MISMATCH" in reasons


def test_component_distribution_must_reconstruct_score_distribution(
    tmp_path: Path,
) -> None:
    rows = _model_rows()
    target_row = next(
        row for row in rows if row["candidate_id"] == "radial_l2_mahalanobis"
    )
    target = target_row["score_components"]["id_test"][
        "selected_class_distance_distribution"
    ]
    for field in ("minimum", "maximum", "mean"):
        target[field] = 5.0
    target["quantiles"] = {
        key: 5.0 for key in STAGE2.FIXED_QUANTILE_PROBABILITIES
    }
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["scientific_launch_allowed"] is False


def test_knn_targeted_witness_uses_frozen_approximate_auroc_tolerance(
    tmp_path: Path,
) -> None:
    rows = _model_rows()
    tolerance = POLICY["invariant_and_witness_policy"][
        "approximate_knn_l2_auroc_drift_tolerance"
    ]
    target = next(row for row in rows if row["candidate_id"] == "radial_l2_knn")
    pair_key = (
        target["candidate_id"],
        target["checkpoint_sha256"],
        target["ood_dataset"],
    )
    for row in rows:
        if (
            row["candidate_id"],
            row["checkpoint_sha256"],
            row["ood_dataset"],
        ) == pair_key:
            row["non_invariant_witness"]["targeted_auroc_abs_drift"] = (
                tolerance + 1.0e-6
            )
    result, output = _execute(tmp_path, rows)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert "TARGETED_WITNESS_APPROXIMATE_RELATION_FAILED" in decision["criteria"][
        "hard_validity"
    ]["reason_codes"]


@pytest.mark.parametrize(
    "corruption",
    ("missing_population_row", "candidate_eligibility", "ranking_use"),
)
def test_diagnostic_population_and_selection_boundary_fail_closed(
    tmp_path: Path, corruption: str
) -> None:
    diagnostics = _diagnostic_rows()
    if corruption == "missing_population_row":
        diagnostics.pop()
    elif corruption == "candidate_eligibility":
        diagnostics[0]["candidate_eligibility"] = True
    else:
        diagnostics[0]["ranking_use"] = "allowed"

    result, output = _execute(tmp_path, _model_rows(), diagnostics)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["candidate_ranking"] == []
    assert _read_jsonl(output / "diagnostic_summaries.jsonl") == []


@pytest.mark.parametrize("corruption", ("author_dim", "channel_contract"))
def test_pure_residual_frozen_dimension_and_channel_contract_fail_closed(
    tmp_path: Path, corruption: str
) -> None:
    diagnostics = _diagnostic_rows()
    pure = next(
        row for row in diagnostics if row["diagnostic_id"] == "pure_residual_channel"
    )
    if corruption == "author_dim":
        pure["score_components"]["fit"]["author_dim"] = 319
    else:
        pure["channel_contract"]["pure_residual_uses_energy"] = True

    result, output = _execute(tmp_path, _model_rows(), diagnostics)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["scientific_launch_allowed"] is False


@pytest.mark.parametrize("diagnostic_id", ("pure_residual_channel", "ctm_angular_channel"))
def test_diagnostic_scalar_parity_contract_fail_closed(
    tmp_path: Path, diagnostic_id: str
) -> None:
    diagnostics = _diagnostic_rows()
    row = next(item for item in diagnostics if item["diagnostic_id"] == diagnostic_id)
    if diagnostic_id == "pure_residual_channel":
        row["scalar_parity"]["status"] = "PASS"
        row["scalar_parity"]["reason_codes"] = []
    else:
        row["scalar_parity"]["metrics"]["auroc"] += 0.01

    result, output = _execute(tmp_path, _model_rows(), diagnostics)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert _read_jsonl(output / "diagnostic_summaries.jsonl") == []


def test_ctm_row_and_computed_metrics_cannot_be_rewritten_away_from_baseline(
    tmp_path: Path,
) -> None:
    diagnostics = _diagnostic_rows()
    ctm = next(
        row for row in diagnostics if row["diagnostic_id"] == "ctm_angular_channel"
    )
    ctm["auroc"] += 0.01
    ctm["fpr95"] -= 0.01
    ctm["scalar_parity"]["metrics"]["auroc"] = ctm["auroc"]
    ctm["scalar_parity"]["metrics"]["fpr95"] = ctm["fpr95"]

    result, output = _execute(tmp_path, _model_rows(), diagnostics)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["candidate_ranking"] == []
    assert _read_jsonl(output / "diagnostic_summaries.jsonl") == []


def test_ctm_embedded_baseline_must_equal_authoritative_v1_2_values() -> None:
    row = _ctm_diagnostic_row(CHECKPOINTS[0], DATASETS[0])
    authoritative = dict(row["scalar_parity"]["baseline_metrics"])
    STAGE2._validate_ctm_scalar_parity(
        row["scalar_parity"],
        row=row,
        label="synthetic CTM",
        authoritative_baseline=authoritative,
    )
    authoritative["auroc"] += 0.01
    with pytest.raises(ValueError, match="authoritative v1.2 baseline"):
        STAGE2._validate_ctm_scalar_parity(
            row["scalar_parity"],
            row=row,
            label="synthetic CTM",
            authoritative_baseline=authoritative,
        )


@pytest.mark.parametrize("corruption", ("status", "tolerance"))
def test_diagnostic_full_array_parity_contract_fail_closed(
    tmp_path: Path, corruption: str
) -> None:
    diagnostics = _diagnostic_rows()
    pure = next(
        row for row in diagnostics if row["diagnostic_id"] == "pure_residual_channel"
    )
    parity = pure["stored_artifact_parity"]["vim_residual"]["id_test"]
    if corruption == "status":
        parity["status"] = "FAIL"
    else:
        parity["absolute_tolerance"] = 1.0e-11

    result, output = _execute(tmp_path, _model_rows(), diagnostics)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"


def test_diagnostic_component_source_binding_corruption_fail_closed(
    tmp_path: Path,
) -> None:
    diagnostics = _diagnostic_rows()
    pure = next(
        row for row in diagnostics if row["diagnostic_id"] == "pure_residual_channel"
    )
    pure["source_references"]["stored_vim_residual_arrays"]["id_test"][
        "array_sha256"
    ] = _sha("corrupt-observed-residual")

    result, output = _execute(tmp_path, _model_rows(), diagnostics)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["candidate_ranking"] == []


def test_ctm_near_zero_normalization_status_fail_closed(tmp_path: Path) -> None:
    diagnostics = _diagnostic_rows()
    ctm = next(
        row for row in diagnostics if row["diagnostic_id"] == "ctm_angular_channel"
    )
    status = ctm["score_components"]["id_test"]["normalization_status"]
    status["status"] = "degenerate"
    status["reason_codes"] = ["near_zero_norm"]
    status["normalization_diagnostics"]["query"]["near_zero_count"] = 1

    result, output = _execute(tmp_path, _model_rows(), diagnostics)
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["scientific_launch_allowed"] is False


@pytest.mark.parametrize(
    "corruption", ("missing_file", "checksum_binding", "manifest_binding")
)
def test_diagnostic_input_file_checksum_and_manifest_binding_fail_closed(
    tmp_path: Path, corruption: str
) -> None:
    selection, metrics, diagnostics, checksums = _write_inputs(
        tmp_path / "input", _model_rows()
    )
    if corruption == "missing_file":
        diagnostics.unlink()
    elif corruption == "checksum_binding":
        checksums.write_text(
            f"{hashlib.sha256(selection.read_bytes()).hexdigest()}  selection_manifest.json\n"
            f"{hashlib.sha256(metrics.read_bytes()).hexdigest()}  model_metrics.jsonl\n",
            encoding="utf-8",
        )
    else:
        manifest = json.loads(selection.read_text(encoding="utf-8"))
        manifest["files"] = [
            item
            for item in manifest["files"]
            if item["logical_path"] != "diagnostic_metrics.jsonl"
        ]
        selection_data = _json_bytes(manifest)
        selection.write_bytes(selection_data)
        checksums.write_text(
            f"{hashlib.sha256(selection_data).hexdigest()}  selection_manifest.json\n"
            f"{hashlib.sha256(metrics.read_bytes()).hexdigest()}  model_metrics.jsonl\n"
            f"{hashlib.sha256(diagnostics.read_bytes()).hexdigest()}  diagnostic_metrics.jsonl\n",
            encoding="utf-8",
        )
    output = tmp_path / "failed-output"
    result = STAGE2.run_reducer(
        selection_manifest_path=selection,
        model_metrics_path=metrics,
        diagnostic_metrics_path=diagnostics,
        input_checksums_path=checksums,
        output_directory=output,
        validate_git_sources=False,
        reducer_git_sha="f" * 40,
        reducer_code_sources=_test_reducer_sources(),
    )
    decision = json.loads((output / "gate_decision.json").read_text(encoding="utf-8"))
    assert result["status"] == decision["status"] == "FAILED"
    assert decision["criteria"]["hard_validity"]["reason_codes"] in (
        ["ARTIFACT_CHECKSUM_FAILED"],
        ["ARTIFACT_FILE_MISSING"],
        ["ARTIFACT_IDENTITY_FAILED"],
    )
    assert _read_jsonl(output / "diagnostic_summaries.jsonl") == []


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    (
        ("PASS", 0),
        ("NO_GO", 0),
        ("INCONCLUSIVE", 0),
        ("FAILED", 1),
        ("UNEXPECTED", 1),
    ),
)
def test_cli_exit_code_is_nonzero_for_failed_or_unknown_decisions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(
        STAGE2,
        "run_reducer",
        lambda **_: {
            "status": status,
            "selected_candidate_id": None,
            "digests": {},
        },
    )
    exit_code = STAGE2.main(
        [
            "--selection-manifest",
            "selection_manifest.json",
            "--model-metrics",
            "model_metrics.jsonl",
            "--diagnostic-metrics",
            "diagnostic_metrics.jsonl",
            "--output-dir",
            "gate-output",
        ]
    )
    assert exit_code == expected_exit
    assert json.loads(capsys.readouterr().out)["status"] == status
