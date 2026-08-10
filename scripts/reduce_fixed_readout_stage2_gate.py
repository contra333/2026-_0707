#!/usr/bin/env python3
"""Reduce frozen Stage-2 model metrics into one scientific gate decision.

This is deliberately a pure metadata/scalar reducer.  It reads the frozen
candidate registry, gate policy, one checksum-bound ``selection_manifest``,
and its ``model_metrics.jsonl``.  It never reads features, logits, scores, or
any other raw array.  The reducer implements the prespecified config-cluster,
seed, dataset-support, LODO, and LOCO estimands and writes a new atomic output
directory.  It does not create the later ``preconfirmation_freeze.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_REGISTRY = (
    REPOSITORY_ROOT
    / "configs/evaluation/fixed_readout_stage2/candidate_registry.json"
)
DEFAULT_GATE_POLICY = (
    REPOSITORY_ROOT / "configs/evaluation/fixed_readout_stage2/gate_policy.json"
)
DEFAULT_CHECKPOINT_INVENTORY = (
    REPOSITORY_ROOT
    / "configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json"
)
DEFAULT_REUSE_MANIFEST = (
    REPOSITORY_ROOT / "configs/evaluation/fixed_readout_stage2/reuse_manifest.json"
)

SELECTION_SCHEMA = "fixed_readout_stage2_selection_manifest_v1"
MODEL_METRIC_SCHEMA = "fixed_readout_stage2_model_metric_v1"
DIAGNOSTIC_METRIC_SCHEMA = "fixed_readout_stage2_diagnostic_metric_v1"
CONTRAST_SCHEMA = "fixed_readout_stage2_contrast_metric_v1"
DATASET_SUMMARY_SCHEMA = "fixed_readout_stage2_dataset_summary_v1"
CV_FOLD_SCHEMA = "fixed_readout_stage2_cv_fold_v1"
GATE_DECISION_SCHEMA = "fixed_readout_stage2_gate_decision_v1"
DIAGNOSTIC_SUMMARY_SCHEMA = "fixed_readout_stage2_diagnostic_summary_v1"
RAW_TRANSFORM_ID = "raw_identity"
OUTPUT_FILENAMES = (
    "contrast_metrics.jsonl",
    "dataset_summaries.jsonl",
    "cv_folds.jsonl",
    "diagnostic_summaries.jsonl",
    "gate_decision.json",
)
FIXED_QUANTILE_PROBABILITIES = {
    "q01": 0.01,
    "q05": 0.05,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q95": 0.95,
    "q99": 0.99,
}
DISTRIBUTION_SUMMARY_SCHEMA = "fixed_linear_quantiles_v1"
SCORE_DISTRIBUTION_SCHEMA = "fixed_score_distribution_v1"
GEOMETRY_CHANNEL_ID = "penultimate_feature_l2_norm"
GEOMETRY_NORMALIZATION_CONTRACT = "oge.evaluation.common.normalize_rows"
EPSILON_NORM = 1.0e-12
NORMALIZATION_PRESERVED_FIELDS = (
    "status",
    "reason_codes",
    "epsilon_norm",
    "denominator_epsilon",
    "exact_zero_count",
    "near_zero_count",
    "sample_count",
)
SELECTIVE_SOURCE_KINDS = {
    "selective_reuse_score_array",
    "selective_reuse_feature_array",
    "selective_reuse_logit_array",
    "selective_reuse_class_label_array",
    "selective_reuse_diagnostic_array",
}
ALL_SOURCE_KINDS = SELECTIVE_SOURCE_KINDS | {"stage2_knn_cache_score_array"}
PRIMARY_FEATURE_DIMENSION = 640
PRIMARY_CLASS_COUNT = 10
WORKER_ANALYSIS_ROLE = "protected_discovery_not_confirmation_or_causality"
WORKER_CODE_SOURCE_PATHS = (
    "scripts/extract_fixed_readout_stage2_model_metrics.py",
    "scripts/analyze_fixed_readout_stage2_mechanism.py",
    "scripts/fixed_readout_stage2_raw_knn_backend.py",
    "src/oge/evaluation/classification.py",
    "src/oge/evaluation/common.py",
    "src/oge/evaluation/detectors.py",
    "src/oge/evaluation/extraction.py",
    "src/oge/evaluation/metrics.py",
    "pyproject.toml",
    "configs/environments/wrn_v1_2_common.yaml",
    "configs/evaluation/fixed_readout_stage2/reuse_manifest.json",
    "configs/evaluation/fixed_readout_stage2/candidate_registry.json",
    "configs/evaluation/fixed_readout_stage2/gate_policy.json",
)
DIAGNOSTIC_IDS = ("pure_residual_channel", "ctm_angular_channel")
CTM_DETECTOR = "detector/ctm_prototype_cosine"
SCALAR_METRIC_FIELDS = (
    "auroc",
    "fpr95",
    "fpr95_threshold",
    "achieved_id_tpr",
)
SCALAR_ARTIFACT_KEYS = {
    "auroc": "ood_metric/auroc_id_positive",
    "fpr95": "ood_metric/fpr95_id_tpr",
    "fpr95_threshold": "ood_metric/fpr95_threshold",
    "achieved_id_tpr": "ood_metric/fpr95_achieved_id_tpr",
}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    priority: int
    raw_detector: str
    transformed_detector: str
    transform_id: str
    invariant_oracle_ids: tuple[str, ...]
    approximate_oracle_ids: tuple[str, ...]
    witness_ids: tuple[str, ...]


@dataclass(frozen=True)
class InputContext:
    analysis_id: str
    selection_manifest_sha256: str
    model_metrics_sha256: str
    diagnostic_metrics_sha256: str
    candidate_registry_sha256: str
    gate_policy_sha256: str
    reducer_git_sha: str
    reducer_code_sources: tuple[Mapping[str, Any], ...]
    source_validation_mode: str
    evidence_provenance_validation_mode: str
    ctm_baseline_validation_mode: str
    evidence_git_sha: str
    checkpoints: tuple[Mapping[str, Any], ...]
    rows: tuple[Mapping[str, Any], ...]
    diagnostic_rows: tuple[Mapping[str, Any], ...]
    candidates: tuple[CandidateSpec, ...]
    policy: Mapping[str, Any]
    hard_validity: Mapping[str, Any]


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    missing = sorted(set(fields).difference(value))
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _unit_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return result


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{label} must be a {length}-character lowercase hex string")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a {length}-character lowercase hex string")
    return value


def _require_reason_codes(value: Any, label: str) -> list[str]:
    codes = _require_list(value, label)
    if any(not isinstance(code, str) or not code for code in codes):
        raise ValueError(f"{label} must contain non-empty strings")
    return codes


def _safe_logical_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{label} is unsafe: {value!r}")
    return value


def _require_shape(
    value: Any, label: str, *, rank: int | None = None
) -> tuple[int, ...]:
    shape = _require_list(value, label)
    if rank is not None and len(shape) != rank:
        raise ValueError(f"{label} must have rank {rank}")
    dimensions = tuple(_require_int(item, f"{label} dimension") for item in shape)
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise ValueError(f"{label} dimensions must be positive")
    return dimensions


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-12)


def _validate_distribution_summary(
    value: Any,
    *,
    label: str,
    expected_count: int | None = None,
    nonnegative: bool = False,
) -> Mapping[str, Any]:
    summary = _require_mapping(value, label)
    _require_fields(
        summary,
        (
            "summary_schema",
            "count",
            "minimum",
            "maximum",
            "mean",
            "std_ddof0",
            "quantiles",
            "array_sha256",
        ),
        label,
    )
    if summary["summary_schema"] != DISTRIBUTION_SUMMARY_SCHEMA:
        raise ValueError(f"{label} schema mismatch")
    count = _require_int(summary["count"], f"{label} count")
    if count <= 0 or (expected_count is not None and count != expected_count):
        raise ValueError(f"{label} count mismatch")
    minimum = _finite_float(summary["minimum"], f"{label} minimum")
    maximum = _finite_float(summary["maximum"], f"{label} maximum")
    mean = _finite_float(summary["mean"], f"{label} mean")
    standard_deviation = _finite_float(summary["std_ddof0"], f"{label} std")
    if minimum > maximum or standard_deviation < 0.0:
        raise ValueError(f"{label} has an invalid range or standard deviation")
    if mean < minimum - 1.0e-12 or mean > maximum + 1.0e-12:
        raise ValueError(f"{label} mean lies outside its range")
    if nonnegative and minimum < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    quantiles = _require_mapping(summary["quantiles"], f"{label} quantiles")
    if set(quantiles) != set(FIXED_QUANTILE_PROBABILITIES):
        raise ValueError(f"{label} quantile population mismatch")
    ordered_quantiles = [
        _finite_float(quantiles[key], f"{label} quantile {key}")
        for key in FIXED_QUANTILE_PROBABILITIES
    ]
    if any(
        value < minimum - 1.0e-12 or value > maximum + 1.0e-12
        for value in ordered_quantiles
    ) or any(
        left > right for left, right in zip(ordered_quantiles, ordered_quantiles[1:])
    ):
        raise ValueError(f"{label} quantiles are inconsistent with its range")
    _require_hex(summary["array_sha256"], 64, f"{label} array SHA256")
    return summary


def _validate_score_distribution(
    value: Any,
    *,
    label: str,
    num_id: int,
    num_ood: int,
) -> Mapping[str, Any]:
    distribution = _require_mapping(value, label)
    _require_fields(
        distribution,
        (
            "summary_schema",
            "score_direction",
            "quantile_method",
            "quantile_probabilities",
            "id_test",
            "ood",
        ),
        label,
    )
    if (
        distribution["summary_schema"] != SCORE_DISTRIBUTION_SCHEMA
        or distribution["score_direction"] != "higher_is_more_id_like"
        or distribution["quantile_method"] != "numpy_linear"
    ):
        raise ValueError(f"{label} frozen distribution contract mismatch")
    probabilities = _require_mapping(
        distribution["quantile_probabilities"], f"{label} probabilities"
    )
    if set(probabilities) != set(FIXED_QUANTILE_PROBABILITIES) or any(
        _finite_float(probabilities[key], f"{label} probability {key}") != expected
        for key, expected in FIXED_QUANTILE_PROBABILITIES.items()
    ):
        raise ValueError(f"{label} frozen quantile probabilities mismatch")
    _validate_distribution_summary(
        distribution["id_test"],
        label=f"{label} ID-test",
        expected_count=num_id,
    )
    _validate_distribution_summary(
        distribution["ood"], label=f"{label} OOD", expected_count=num_ood
    )
    return distribution


def _validate_normalization_record(
    value: Any,
    *,
    label: str,
    expected_count: int,
    denominator_epsilon: float,
) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    _require_fields(record, NORMALIZATION_PRESERVED_FIELDS, label)
    if record["status"] != "success" or _require_reason_codes(
        record["reason_codes"], f"{label} reason codes"
    ):
        raise ValueError(f"{label} normalization status is not successful")
    if _finite_float(record["epsilon_norm"], f"{label} epsilon_norm") != EPSILON_NORM:
        raise ValueError(f"{label} epsilon_norm mismatch")
    if _finite_float(
        record["denominator_epsilon"], f"{label} denominator epsilon"
    ) != denominator_epsilon:
        raise ValueError(f"{label} denominator epsilon mismatch")
    if (
        _require_int(record["exact_zero_count"], f"{label} exact-zero count") != 0
        or _require_int(record["near_zero_count"], f"{label} near-zero count") != 0
        or _require_int(record["sample_count"], f"{label} sample count")
        != expected_count
    ):
        raise ValueError(f"{label} zero-norm or sample-count mismatch")
    return record


def _validate_geometry_channel(
    value: Any,
    *,
    label: str,
    candidate_id: str,
    active_dataset: str,
    datasets: Sequence[str],
    active_feature_shapes: Mapping[str, tuple[int, ...]],
) -> Mapping[str, Any]:
    geometry = _require_mapping(value, label)
    _require_fields(
        geometry,
        (
            "channel_id",
            "normalization_contract",
            "normalization_denominator_epsilon",
            "active_ood_split",
            "splits",
        ),
        label,
    )
    expected_denominator = 0.0 if candidate_id == "radial_l2_mahalanobis" else 1.0e-10
    if (
        geometry["channel_id"] != GEOMETRY_CHANNEL_ID
        or geometry["normalization_contract"] != GEOMETRY_NORMALIZATION_CONTRACT
        or _finite_float(
            geometry["normalization_denominator_epsilon"],
            f"{label} denominator epsilon",
        )
        != expected_denominator
        or geometry["active_ood_split"] != active_dataset
    ):
        raise ValueError(f"{label} frozen channel contract mismatch")
    splits = _require_mapping(geometry["splits"], f"{label} splits")
    expected_splits = {"id_train", "id_test", *datasets}
    if set(splits) != expected_splits:
        raise ValueError(f"{label} split population mismatch")
    for split_name, item in splits.items():
        split = _require_mapping(item, f"{label} split {split_name}")
        _require_fields(split, ("feature_norm", "normalization"), f"{label} split")
        if split_name == "id_train":
            expected_count = active_feature_shapes["id_train"][0]
        elif split_name == "id_test":
            expected_count = active_feature_shapes["id_test"][0]
        elif split_name == active_dataset:
            expected_count = active_feature_shapes["ood"][0]
        else:
            expected_count = None
        distribution = _validate_distribution_summary(
            split["feature_norm"],
            label=f"{label} {split_name} feature norm",
            expected_count=expected_count,
            nonnegative=True,
        )
        _validate_normalization_record(
            split["normalization"],
            label=f"{label} {split_name} normalization",
            expected_count=int(distribution["count"]),
            denominator_epsilon=expected_denominator,
        )
    return geometry


def _validate_source_reference(
    value: Any,
    *,
    label: str,
    expected_kind: str,
    expected_bundle_id: str,
    expected_allowlist_sha256: str,
    expected_shape: tuple[int, ...],
    allowed_files: Mapping[str, Mapping[str, Any]],
    expected_logical_path: str,
    analysis_id: str,
    expected_variant_id: str | None = None,
) -> Mapping[str, Any]:
    reference = _require_mapping(value, label)
    common_fields = (
        "source_kind",
        "logical_path",
        "file_sha256",
        "file_size",
        "array_sha256",
        "shape",
        "dtype",
        "bundle_id",
        "source_allowlist_sha256",
    )
    _require_fields(reference, common_fields, label)
    if expected_kind not in ALL_SOURCE_KINDS or reference["source_kind"] != expected_kind:
        raise ValueError(f"{label} source kind mismatch")
    logical_path = _safe_logical_path(reference["logical_path"], f"{label} path")
    if logical_path != expected_logical_path:
        raise ValueError(f"{label} logical path mismatch")
    file_sha = _require_hex(reference["file_sha256"], 64, f"{label} file SHA256")
    file_size = _require_int(reference["file_size"], f"{label} file size")
    if file_size <= 0:
        raise ValueError(f"{label} file size must be positive")
    _require_hex(reference["array_sha256"], 64, f"{label} array SHA256")
    if _require_shape(reference["shape"], f"{label} shape") != expected_shape:
        raise ValueError(f"{label} array shape mismatch")
    if not isinstance(reference["dtype"], str) or not reference["dtype"]:
        raise ValueError(f"{label} dtype must be a non-empty string")
    if (
        reference["bundle_id"] != expected_bundle_id
        or reference["source_allowlist_sha256"] != expected_allowlist_sha256
    ):
        raise ValueError(f"{label} bundle/source-allowlist binding mismatch")
    if expected_kind in SELECTIVE_SOURCE_KINDS:
        if logical_path not in allowed_files:
            raise ValueError(f"{label} source is absent from the bundle allowlist")
        allowed = _require_mapping(allowed_files[logical_path], f"{label} allowlist row")
        if file_sha != allowed.get("sha256") or file_size != allowed.get("size"):
            raise ValueError(f"{label} allowlist checksum/size mismatch")
    if expected_kind in {
        "selective_reuse_score_array",
        "selective_reuse_diagnostic_array",
    }:
        _require_fields(
            reference,
            (
                "stored_array_sha256",
                "source_declared_shape",
                "source_declared_dtype",
                "parity_status",
            ),
            label,
        )
        _require_hex(
            reference["stored_array_sha256"],
            64,
            f"{label} stored array SHA256",
        )
        if (
            _require_shape(
                reference["source_declared_shape"], f"{label} declared shape"
            )
            != expected_shape
            or not isinstance(reference["source_declared_dtype"], str)
            or not reference["source_declared_dtype"]
            or reference["source_declared_dtype"] != reference["dtype"]
            or reference["parity_status"] != "PASS"
        ):
            raise ValueError(f"{label} stored-score parity binding mismatch")
    elif expected_kind == "stage2_knn_cache_score_array":
        _require_fields(
            reference,
            (
                "analysis_id",
                "variant_id",
                "cache_schema_version",
                "cache_manifest_sha256",
            ),
            label,
        )
        if (
            reference["analysis_id"] != analysis_id
            or reference["variant_id"] != expected_variant_id
            or reference["cache_schema_version"]
            != "fixed_readout_stage2_knn_variant_cache_v1"
        ):
            raise ValueError(f"{label} cache identity mismatch")
        _require_hex(
            reference["cache_manifest_sha256"],
            64,
            f"{label} cache manifest SHA256",
        )
    return reference


def _detector_score_logical_path(split_name: str, detector: str) -> str:
    return f"metrics/arrays/scores/{split_name}/{detector.replace('/', '__')}.npy"


def _validate_source_references(
    value: Any,
    *,
    label: str,
    analysis_id: str,
    candidate_id: str,
    detector: str,
    dataset: str,
    num_id: int,
    num_ood: int,
    checkpoint: Mapping[str, Any],
    allowed_files: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, tuple[int, ...]]]:
    sources = _require_mapping(value, label)
    _require_fields(
        sources,
        (
            "source_allowlist_sha256",
            "score_arrays",
            "feature_arrays",
            "null_logit_arrays",
        ),
        label,
    )
    allowlist_sha = str(checkpoint["source_allowlist_sha256"])
    bundle_id = str(checkpoint["checkpoint_sha256"])
    if sources["source_allowlist_sha256"] != allowlist_sha:
        raise ValueError(f"{label} source allowlist binding mismatch")
    score_arrays = _require_mapping(sources["score_arrays"], f"{label} scores")
    feature_arrays = _require_mapping(sources["feature_arrays"], f"{label} features")
    logit_arrays = _require_mapping(sources["null_logit_arrays"], f"{label} logits")
    if set(score_arrays) != {"id_test", "ood"}:
        raise ValueError(f"{label} score source population mismatch")
    if set(feature_arrays) != {"id_train", "id_test", "ood"}:
        raise ValueError(f"{label} feature source population mismatch")
    if set(logit_arrays) != {"id_test", "ood"}:
        raise ValueError(f"{label} logit source population mismatch")

    feature_shapes = {
        key: _require_shape(
            _require_mapping(reference, f"{label} feature {key}").get("shape"),
            f"{label} feature {key} shape",
            rank=2,
        )
        for key, reference in feature_arrays.items()
    }
    if (
        feature_shapes["id_test"][0] != num_id
        or feature_shapes["ood"][0] != num_ood
        or len({shape[1] for shape in feature_shapes.values()}) != 1
        or next(iter(feature_shapes.values()))[1] != PRIMARY_FEATURE_DIMENSION
    ):
        raise ValueError(f"{label} feature shape/count mismatch")
    feature_paths = {
        "id_train": "cifar10/train/features.npy",
        "id_test": "cifar10/test/features.npy",
        "ood": f"{dataset}/test/features.npy",
    }
    for key, reference in feature_arrays.items():
        validated = _validate_source_reference(
            reference,
            label=f"{label} feature {key}",
            expected_kind="selective_reuse_feature_array",
            expected_bundle_id=bundle_id,
            expected_allowlist_sha256=allowlist_sha,
            expected_shape=feature_shapes[key],
            allowed_files=allowed_files,
            expected_logical_path=feature_paths[key],
            analysis_id=analysis_id,
        )
        if validated["dtype"] != "float32":
            raise ValueError(f"{label} feature dtype mismatch")

    logit_shapes = {
        key: _require_shape(
            _require_mapping(reference, f"{label} logit {key}").get("shape"),
            f"{label} logit {key} shape",
            rank=2,
        )
        for key, reference in logit_arrays.items()
    }
    if (
        logit_shapes["id_test"][0] != num_id
        or logit_shapes["ood"][0] != num_ood
        or logit_shapes["id_test"][1] != logit_shapes["ood"][1]
        or logit_shapes["id_test"][1] != PRIMARY_CLASS_COUNT
    ):
        raise ValueError(f"{label} logit shape/count mismatch")
    logit_paths = {
        "id_test": "cifar10/test/logits.npy",
        "ood": f"{dataset}/test/logits.npy",
    }
    for key, reference in logit_arrays.items():
        validated = _validate_source_reference(
            reference,
            label=f"{label} logit {key}",
            expected_kind="selective_reuse_logit_array",
            expected_bundle_id=bundle_id,
            expected_allowlist_sha256=allowlist_sha,
            expected_shape=logit_shapes[key],
            allowed_files=allowed_files,
            expected_logical_path=logit_paths[key],
            analysis_id=analysis_id,
        )
        if validated["dtype"] != "float32":
            raise ValueError(f"{label} logit dtype mismatch")

    score_shapes = {"id_test": (num_id,), "ood": (num_ood,)}
    split_names = {"id_test": "id_test", "ood": dataset}
    cache_score = candidate_id == "radial_l2_knn" and detector == "detector/knn_raw_k50"
    for key, reference in score_arrays.items():
        if cache_score:
            source_kind = "stage2_knn_cache_score_array"
            logical_path = f"splits/{split_names[key]}/score.npy"
            variant_id = "raw_identity"
        else:
            source_kind = "selective_reuse_score_array"
            logical_path = _detector_score_logical_path(split_names[key], detector)
            variant_id = None
        validated = _validate_source_reference(
            reference,
            label=f"{label} score {key}",
            expected_kind=source_kind,
            expected_bundle_id=bundle_id,
            expected_allowlist_sha256=allowlist_sha,
            expected_shape=score_shapes[key],
            allowed_files=allowed_files,
            expected_logical_path=logical_path,
            analysis_id=analysis_id,
            expected_variant_id=variant_id,
        )
        if validated["dtype"] != "float64":
            raise ValueError(f"{label} score dtype mismatch")
    return sources, feature_shapes


def _validate_witness_normalization_status(
    witness: Mapping[str, Any],
    *,
    label: str,
    candidate_id: str,
    geometry: Mapping[str, Any],
    datasets: Sequence[str],
    exact_atol: float,
    exact_rtol: float,
    approximate_auroc_tolerance: float,
) -> None:
    _require_fields(
        witness,
        (
            "targeted_acceptance_rule",
            "targeted_score_allclose",
            "targeted_exact_atol",
            "targeted_exact_rtol",
            "targeted_approximate_auroc_tolerance",
            "normalization_status_preservation",
        ),
        label,
    )
    expected_rule = (
        "allclose_and_exact_rank_and_exact_auroc"
        if candidate_id == "radial_l2_mahalanobis"
        else "approximate_auroc_only"
    )
    if witness["targeted_acceptance_rule"] != expected_rule:
        raise ValueError(f"{label} targeted acceptance rule mismatch")
    if (
        _finite_float(witness["targeted_exact_atol"], f"{label} targeted atol")
        != exact_atol
        or _finite_float(witness["targeted_exact_rtol"], f"{label} targeted rtol")
        != exact_rtol
        or not isinstance(witness["targeted_score_allclose"], bool)
    ):
        raise ValueError(f"{label} targeted exact tolerance binding mismatch")
    if candidate_id == "radial_l2_mahalanobis":
        if (
            witness["targeted_score_allclose"] is not True
            or witness["targeted_approximate_auroc_tolerance"] is not None
        ):
            raise ValueError(f"{label} exact targeted witness relation failed")
    elif _finite_float(
        witness["targeted_approximate_auroc_tolerance"],
        f"{label} approximate AUROC tolerance",
    ) != approximate_auroc_tolerance:
        raise ValueError(f"{label} approximate AUROC tolerance binding mismatch")
    preservation = _require_mapping(
        witness["normalization_status_preservation"],
        f"{label} normalization preservation",
    )
    _require_fields(
        preservation,
        ("status", "reason_codes", "denominator_epsilon", "preserved_fields", "splits"),
        f"{label} normalization preservation",
    )
    expected_denominator = 0.0 if candidate_id == "radial_l2_mahalanobis" else 1.0e-10
    if (
        preservation["status"] != "PASS"
        or _require_reason_codes(
            preservation["reason_codes"], f"{label} normalization reasons"
        )
        or _finite_float(
            preservation["denominator_epsilon"],
            f"{label} normalization denominator epsilon",
        )
        != expected_denominator
        or preservation["preserved_fields"] != list(NORMALIZATION_PRESERVED_FIELDS)
    ):
        raise ValueError(f"{label} normalization preservation contract mismatch")
    split_rows = _require_mapping(
        preservation["splits"], f"{label} normalization preservation splits"
    )
    expected_splits = {"id_train", "id_test", *datasets}
    if set(split_rows) != expected_splits:
        raise ValueError(f"{label} witness normalization split population mismatch")
    geometry_splits = _require_mapping(geometry["splits"], f"{label} geometry splits")
    for split_name, value in split_rows.items():
        split = _require_mapping(value, f"{label} witness split {split_name}")
        _require_fields(split, ("preserved", "base", "witness"), f"{label} witness split")
        geometry_normalization = _require_mapping(
            geometry_splits[split_name]["normalization"],
            f"{label} geometry normalization {split_name}",
        )
        expected_count = int(geometry_normalization["sample_count"])
        base = _validate_normalization_record(
            split["base"],
            label=f"{label} witness base {split_name}",
            expected_count=expected_count,
            denominator_epsilon=expected_denominator,
        )
        observed = _validate_normalization_record(
            split["witness"],
            label=f"{label} witness transformed {split_name}",
            expected_count=expected_count,
            denominator_epsilon=expected_denominator,
        )
        if split["preserved"] is not True or base != observed or base != geometry_normalization:
            raise ValueError(f"{label} witness normalization status was not preserved")


def _validate_digest_array_record(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    record = _require_mapping(value, label)
    _require_fields(record, ("shape", "sha256"), label)
    shape = _require_shape(record["shape"], f"{label} shape")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"{label} shape mismatch")
    _require_hex(record["sha256"], 64, f"{label} SHA256")
    return shape


def _validate_negative_distance_distribution(
    score: Mapping[str, Any], distance: Mapping[str, Any], *, label: str
) -> None:
    pairs = (
        (float(score["minimum"]), -float(distance["maximum"])),
        (float(score["maximum"]), -float(distance["minimum"])),
        (float(score["mean"]), -float(distance["mean"])),
        (float(score["std_ddof0"]), float(distance["std_ddof0"])),
    )
    reverse_keys = list(FIXED_QUANTILE_PROBABILITIES)[::-1]
    score_quantiles = _require_mapping(score["quantiles"], f"{label} score quantiles")
    distance_quantiles = _require_mapping(
        distance["quantiles"], f"{label} distance quantiles"
    )
    pairs += tuple(
        (
            float(score_quantiles[score_key]),
            -float(distance_quantiles[distance_key]),
        )
        for score_key, distance_key in zip(FIXED_QUANTILE_PROBABILITIES, reverse_keys)
    )
    if any(not _same_float(left, right) for left, right in pairs):
        raise ValueError(f"{label} score is not the negative distance distribution")


def _validate_mahalanobis_components(
    value: Any,
    *,
    label: str,
    expected_count: int,
    score_distribution: Mapping[str, Any],
    exact_atol: float,
    exact_rtol: float,
) -> None:
    component = _require_mapping(value, label)
    required = (
        "class_distances",
        "nearest_class",
        "class_means",
        "within_covariance",
        "within_precision",
        "covariance_eigenvalues",
        "precision_eigenvalues",
        "covariance_eigendirection_contributions",
        "selected_class_distance_distribution",
        "nearest_vs_second_distance_margin_distribution",
        "nearest_class_counts",
        "numerical_rank",
        "condition_number",
        "stored_component_parity_pass",
        "stored_component_max_abs_error",
        "full_query_array_evaluated",
    )
    _require_fields(component, required, label)
    class_distance_shape = _validate_digest_array_record(
        component["class_distances"], label=f"{label} class distances"
    )
    if len(class_distance_shape) != 2 or class_distance_shape[0] != expected_count:
        raise ValueError(f"{label} class-distance population mismatch")
    class_count = class_distance_shape[1]
    if class_count != PRIMARY_CLASS_COUNT:
        raise ValueError(f"{label} class count mismatch")
    nearest_counts = _require_mapping(
        component["nearest_class_counts"], f"{label} nearest-class counts"
    )
    if set(nearest_counts) != {str(index) for index in range(class_count)}:
        raise ValueError(f"{label} nearest-class count population mismatch")
    count_values = [
        _require_int(nearest_counts[str(index)], f"{label} class {index} count")
        for index in range(class_count)
    ]
    if any(count < 0 for count in count_values) or sum(count_values) != expected_count:
        raise ValueError(f"{label} nearest-class counts do not cover the population")
    _validate_digest_array_record(
        component["nearest_class"],
        label=f"{label} nearest class",
        expected_shape=(expected_count,),
    )
    class_means_shape = _validate_digest_array_record(
        component["class_means"], label=f"{label} class means"
    )
    if len(class_means_shape) != 2 or class_means_shape[0] != class_count:
        raise ValueError(f"{label} class-mean population mismatch")
    feature_dimension = class_means_shape[1]
    if feature_dimension != PRIMARY_FEATURE_DIMENSION:
        raise ValueError(f"{label} feature dimension mismatch")
    for field in ("within_covariance", "within_precision"):
        _validate_digest_array_record(
            component[field],
            label=f"{label} {field}",
            expected_shape=(feature_dimension, feature_dimension),
        )
    for field in ("covariance_eigenvalues", "precision_eigenvalues"):
        eigen = _require_mapping(component[field], f"{label} {field}")
        _validate_digest_array_record(
            eigen, label=f"{label} {field}", expected_shape=(feature_dimension,)
        )
        minimum = _finite_float(eigen.get("minimum"), f"{label} {field} minimum")
        maximum = _finite_float(eigen.get("maximum"), f"{label} {field} maximum")
        if minimum > maximum:
            raise ValueError(f"{label} {field} range mismatch")
    selected = _validate_distribution_summary(
        component["selected_class_distance_distribution"],
        label=f"{label} selected class distance",
        expected_count=expected_count,
        nonnegative=True,
    )
    _validate_distribution_summary(
        component["nearest_vs_second_distance_margin_distribution"],
        label=f"{label} nearest/second margin",
        expected_count=expected_count,
        nonnegative=True,
    )
    _validate_negative_distance_distribution(
        score_distribution, selected, label=f"{label} selected distance"
    )
    contributions = _require_mapping(
        component["covariance_eigendirection_contributions"],
        f"{label} eigendirection contributions",
    )
    _require_fields(
        contributions,
        (
            "shape",
            "sha256",
            "sha256_encoding",
            "max_abs_distance_reconstruction_error",
            "precision_eigenvalue_tertile_mean_contributions",
            "precision_eigenvalue_tertile_distribution",
        ),
        f"{label} eigendirection contributions",
    )
    if _require_shape(
        contributions["shape"], f"{label} contribution shape"
    ) != (expected_count, feature_dimension):
        raise ValueError(f"{label} contribution shape mismatch")
    _require_hex(contributions["sha256"], 64, f"{label} contribution SHA256")
    if contributions["sha256_encoding"] != "dtype_shape_prefix_then_row_major_float64_chunks":
        raise ValueError(f"{label} contribution digest encoding mismatch")
    reconstruction_error = _finite_float(
        contributions["max_abs_distance_reconstruction_error"],
        f"{label} contribution reconstruction error",
    )
    if reconstruction_error > exact_atol + exact_rtol * abs(float(selected["maximum"])):
        raise ValueError(f"{label} contribution reconstruction tolerance failed")
    band_means = _require_mapping(
        contributions["precision_eigenvalue_tertile_mean_contributions"],
        f"{label} band means",
    )
    band_distributions = _require_mapping(
        contributions["precision_eigenvalue_tertile_distribution"],
        f"{label} band distributions",
    )
    if set(band_means) != {"low", "middle", "high"} or set(band_distributions) != {
        "low",
        "middle",
        "high",
    }:
        raise ValueError(f"{label} eigendirection-band population mismatch")
    observed_band_means = []
    for band_name in ("low", "middle", "high"):
        band = _validate_distribution_summary(
            band_distributions[band_name],
            label=f"{label} {band_name} contribution",
            expected_count=expected_count,
        )
        band_mean = _finite_float(band_means[band_name], f"{label} {band_name} mean")
        if not _same_float(band_mean, float(band["mean"])):
            raise ValueError(f"{label} contribution band mean mismatch")
        observed_band_means.append(band_mean)
    if not _same_float(sum(observed_band_means), float(selected["mean"])):
        raise ValueError(f"{label} contribution bands do not reconstruct selected distance")
    rank = _require_int(component["numerical_rank"], f"{label} numerical rank")
    condition = _finite_float(component["condition_number"], f"{label} condition number")
    if rank <= 0 or rank > feature_dimension or condition < 0.0:
        raise ValueError(f"{label} invalid rank/condition diagnostic")
    _finite_float(
        component["stored_component_max_abs_error"], f"{label} stored parity error"
    )
    if component["stored_component_parity_pass"] is not True:
        raise ValueError(f"{label} stored component parity failed")
    if component["full_query_array_evaluated"] is not True:
        raise ValueError(f"{label} is digest-only rather than full-population evidence")


def _validate_knn_components(
    value: Any,
    *,
    label: str,
    expected_count: int,
    score_distribution: Mapping[str, Any],
) -> None:
    component = _require_mapping(value, label)
    required = (
        "kth_squared_distance_sha256",
        "kth_neighbor_sample_id_sha256",
        "top50_squared_distances_sha256",
        "top50_neighbor_bank_indices_sha256",
        "top50_neighbor_identity_mapping_sha256",
        "sample_count",
        "k",
        "middle_rank",
        "neighbor_identity_encoding",
        "train_sample_id_mapping_is_lossless",
        "first_squared_distance_distribution",
        "middle_rank_squared_distance_distribution",
        "kth_squared_distance_distribution",
        "mean_top50_squared_distance_distribution",
        "full_query_array_evaluated",
    )
    _require_fields(component, required, label)
    for field in (
        "kth_squared_distance_sha256",
        "kth_neighbor_sample_id_sha256",
        "top50_squared_distances_sha256",
        "top50_neighbor_bank_indices_sha256",
        "top50_neighbor_identity_mapping_sha256",
    ):
        _require_hex(component[field], 64, f"{label} {field}")
    if (
        _require_int(component["sample_count"], f"{label} sample count")
        != expected_count
        or _require_int(component["k"], f"{label} k") != 50
        or _require_int(component["middle_rank"], f"{label} middle rank") != 25
        or component["neighbor_identity_encoding"]
        != "exact_train_sample_id_strings"
        or component["train_sample_id_mapping_is_lossless"] is not True
        or component["full_query_array_evaluated"] is not True
    ):
        raise ValueError(f"{label} exact/full-population kNN contract mismatch")
    distributions = {
        name: _validate_distribution_summary(
            component[field],
            label=f"{label} {name} distance",
            expected_count=expected_count,
            nonnegative=True,
        )
        for name, field in (
            ("first", "first_squared_distance_distribution"),
            ("middle", "middle_rank_squared_distance_distribution"),
            ("kth", "kth_squared_distance_distribution"),
            ("mean_top50", "mean_top50_squared_distance_distribution"),
        )
    }
    _validate_negative_distance_distribution(
        score_distribution, distributions["kth"], label=f"{label} kth distance"
    )
    for statistic in ("minimum", "maximum", "mean", *FIXED_QUANTILE_PROBABILITIES):
        if statistic in FIXED_QUANTILE_PROBABILITIES:
            values = [
                float(distributions[name]["quantiles"][statistic])
                for name in ("first", "middle", "kth")
            ]
            top50_value = float(distributions["mean_top50"]["quantiles"][statistic])
        else:
            values = [float(distributions[name][statistic]) for name in ("first", "middle", "kth")]
            top50_value = float(distributions["mean_top50"][statistic])
        if values[0] > values[1] or values[1] > values[2] or not (
            values[0] <= top50_value <= values[2]
        ):
            raise ValueError(f"{label} neighbor-distance order mismatch")


def _validate_neighbor_overlap(
    value: Any, *, label: str, expected_count: int
) -> None:
    overlap = _require_mapping(value, label)
    _require_fields(
        overlap,
        (
            "sample_count",
            "k",
            "total_shared_neighbor_count",
            "mean_overlap_fraction",
            "minimum_overlap_fraction",
            "maximum_overlap_fraction",
            "per_sample_shared_count_sha256",
            "identity_encoding",
            "full_query_array_evaluated",
        ),
        label,
    )
    sample_count = _require_int(overlap["sample_count"], f"{label} sample count")
    k = _require_int(overlap["k"], f"{label} k")
    total = _require_int(overlap["total_shared_neighbor_count"], f"{label} total")
    mean = _unit_float(overlap["mean_overlap_fraction"], f"{label} mean")
    minimum = _unit_float(overlap["minimum_overlap_fraction"], f"{label} minimum")
    maximum = _unit_float(overlap["maximum_overlap_fraction"], f"{label} maximum")
    if (
        sample_count != expected_count
        or k != 50
        or total < 0
        or total > sample_count * k
        or minimum > mean
        or mean > maximum
        or not _same_float(mean, total / (sample_count * k))
        or overlap["identity_encoding"]
        != "exact_bank_indices_bound_to_train_sample_id_digest"
        or overlap["full_query_array_evaluated"] is not True
    ):
        raise ValueError(f"{label} overlap population/identity mismatch")
    _require_hex(
        overlap["per_sample_shared_count_sha256"], 64, f"{label} count SHA256"
    )


def _validate_score_components(
    value: Any,
    *,
    label: str,
    candidate_id: str,
    num_id: int,
    num_ood: int,
    score_distribution: Mapping[str, Any],
    exact_atol: float,
    exact_rtol: float,
) -> Mapping[str, Any]:
    components = _require_mapping(value, label)
    if candidate_id == "radial_l2_mahalanobis":
        if set(components) != {"id_test", "ood"}:
            raise ValueError(f"{label} Mahalanobis component population mismatch")
        _validate_mahalanobis_components(
            components["id_test"],
            label=f"{label} ID-test",
            expected_count=num_id,
            score_distribution=score_distribution["id_test"],
            exact_atol=exact_atol,
            exact_rtol=exact_rtol,
        )
        _validate_mahalanobis_components(
            components["ood"],
            label=f"{label} OOD",
            expected_count=num_ood,
            score_distribution=score_distribution["ood"],
            exact_atol=exact_atol,
            exact_rtol=exact_rtol,
        )
    else:
        expected_keys = {
            "id_test",
            "ood",
            "raw_to_l2_neighbor_overlap",
            "exact_bank_mapping",
        }
        if set(components) != expected_keys:
            raise ValueError(f"{label} kNN component population mismatch")
        _validate_knn_components(
            components["id_test"],
            label=f"{label} ID-test",
            expected_count=num_id,
            score_distribution=score_distribution["id_test"],
        )
        _validate_knn_components(
            components["ood"],
            label=f"{label} OOD",
            expected_count=num_ood,
            score_distribution=score_distribution["ood"],
        )
        overlap = _require_mapping(
            components["raw_to_l2_neighbor_overlap"], f"{label} overlap"
        )
        if set(overlap) != {"id_test", "ood"}:
            raise ValueError(f"{label} overlap split population mismatch")
        _validate_neighbor_overlap(
            overlap["id_test"], label=f"{label} ID-test overlap", expected_count=num_id
        )
        _validate_neighbor_overlap(
            overlap["ood"], label=f"{label} OOD overlap", expected_count=num_ood
        )
        mapping = _require_mapping(components["exact_bank_mapping"], f"{label} bank mapping")
        _require_fields(
            mapping,
            ("encoding", "train_sample_id_sha256", "lossless_from_neighbor_component"),
            f"{label} bank mapping",
        )
        if (
            mapping["encoding"] != "exact_train_sample_id_strings"
            or mapping["lossless_from_neighbor_component"] is not True
        ):
            raise ValueError(f"{label} bank mapping is not lossless")
        _require_hex(
            mapping["train_sample_id_sha256"], 64, f"{label} train sample ID SHA256"
        )
    return components


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _pairwise_absolute_median(values: Mapping[str, float]) -> float:
    keys = sorted(values)
    if len(keys) < 2:
        raise ValueError("cross-config gap requires at least two configurations")
    differences = [
        abs(values[left] - values[right])
        for left_index, left in enumerate(keys)
        for right in keys[left_index + 1 :]
    ]
    return float(median(differences))


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    return float(median(values))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return float(sum(values) / len(values))


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error
    return _require_mapping(payload, label)


def _load_jsonl(
    path: Path, *, label: str = "model metrics"
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"blank JSONL record at line {line_number}")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL record at line {line_number}: {error}"
                    ) from error
                rows.append(_require_mapping(value, f"{label} line {line_number}"))
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error
    if not rows:
        raise ValueError(f"{label} JSONL must not be empty")
    return tuple(rows)


def _git_bytes(relative_path: str, *, revision: str = "HEAD") -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def validate_committed_reducer_sources(
    candidate_registry_path: Path,
    gate_policy_path: Path,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    _require_hex(head, 40, "reducer Git SHA")
    paths = [
        Path(__file__).resolve(),
        candidate_registry_path.resolve(),
        gate_policy_path.resolve(),
        DEFAULT_CHECKPOINT_INVENTORY.resolve(),
        DEFAULT_REUSE_MANIFEST.resolve(),
    ]
    relative_paths: list[str] = []
    for path in paths:
        try:
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError as error:
            raise ValueError("production reducer sources must be inside the repository") from error
        if relative not in relative_paths:
            relative_paths.append(relative)
    sources: list[Mapping[str, Any]] = []
    for relative in relative_paths:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required reducer source is absent: {relative}")
        working = path.read_bytes()
        if working != _git_bytes(relative):
            raise ValueError(f"reducer source differs from HEAD: {relative}")
        sources.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(working),
                "size": len(working),
            }
        )
    return head, tuple(sources)


def _expected_worker_code_sources(
    *, validate_git_sources: bool
) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = []
    for relative in WORKER_CODE_SOURCE_PATHS:
        _safe_logical_path(relative, "worker source path")
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required worker source is absent: {relative}")
        working = path.read_bytes()
        if validate_git_sources and working != _git_bytes(relative):
            raise ValueError(f"worker source differs from HEAD: {relative}")
        sources.append(
            {
                "path": relative,
                "sha256": _sha256_bytes(working),
                "size": len(working),
            }
        )
    return tuple(sources)


def _expected_baseline_source_records() -> tuple[Mapping[str, Any], ...]:
    reuse = _load_json(DEFAULT_REUSE_MANIFEST, "Stage-2 reuse manifest")
    baseline = _require_mapping(reuse.get("parity_baseline"), "parity baseline")
    if baseline.get("role") != "completed_metric_contract_v1.2_scalar_parity_baseline":
        raise ValueError("parity baseline role mismatch")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(
        _require_list(baseline.get("files"), "parity baseline files")
    ):
        record = _require_mapping(item, f"parity baseline file {index}")
        logical_path = _safe_logical_path(
            record.get("logical_path"), "parity baseline logical path"
        )
        local_path = record.get("local_path")
        if not isinstance(local_path, str) or not Path(local_path).is_absolute():
            raise ValueError("parity baseline local path must be absolute")
        records.append(
            {
                "logical_path": logical_path,
                "local_path": local_path,
                "sha256": _require_hex(
                    record.get("sha256"), 64, "parity baseline SHA256"
                ),
                "size": _require_int(record.get("size"), "parity baseline size"),
            }
        )
    records.sort(key=lambda item: str(item["logical_path"]))
    expected_paths = {
        "aggregate.json",
        "artifact_manifest.json",
        "detector_rank_concordance.jsonl",
        "per_checkpoint_records.jsonl",
        "seed_aggregates.jsonl",
    }
    if {str(item["logical_path"]) for item in records} != expected_paths:
        raise ValueError("parity baseline source population mismatch")
    if any(int(item["size"]) <= 0 for item in records):
        raise ValueError("parity baseline source size must be positive")
    return tuple(records)


def _validate_relocated_baseline_source(
    value: Any,
    *,
    expected: Mapping[str, Any],
    label: str,
    verify_file: bool,
) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    if set(record) != {"logical_path", "local_path", "sha256", "size"}:
        raise ValueError(f"{label} source-file schema mismatch")
    logical_path = _safe_logical_path(
        record["logical_path"], f"{label} logical path"
    )
    checksum = _require_hex(record["sha256"], 64, f"{label} SHA256")
    size = _require_int(record["size"], f"{label} size")
    if size <= 0:
        raise ValueError(f"{label} size must be positive")
    if (
        logical_path != expected["logical_path"]
        or checksum != expected["sha256"]
        or size != expected["size"]
    ):
        raise ValueError(f"{label} frozen logical/checksum/size identity mismatch")
    local_path = record["local_path"]
    if not isinstance(local_path, str) or not Path(local_path).is_absolute():
        raise ValueError(f"{label} relocated local path must be absolute")
    path = Path(local_path)
    if verify_file:
        current = path
        while current != current.parent:
            if os.path.islink(current):
                raise ValueError(f"{label} relocated path contains a symlink")
            current = current.parent
        if not path.is_file():
            raise FileNotFoundError(f"{label} relocated file is absent")
        if path.stat().st_size != size:
            raise ValueError(f"{label} relocated file size mismatch")
        if _sha256_file(path) != checksum:
            raise ValueError(f"{label} relocated file checksum mismatch")
    return record


def _validate_evidence_provenance(
    manifest: Mapping[str, Any],
    *,
    expected_git_sha: str,
    validate_git_sources: bool,
) -> str:
    if manifest.get("analysis_git_sha") != expected_git_sha:
        raise ValueError("evidence analysis Git SHA differs from reducer HEAD")

    observed_code_files = _require_list(manifest.get("code_files"), "worker code files")
    for index, item in enumerate(observed_code_files):
        record = _require_mapping(item, f"worker code file {index}")
        if set(record) != {"path", "sha256", "size"}:
            raise ValueError("worker code-file schema mismatch")
        _safe_logical_path(record["path"], "worker code-file path")
        _require_hex(record["sha256"], 64, "worker code-file SHA256")
        if _require_int(record["size"], "worker code-file size") <= 0:
            raise ValueError("worker code-file size must be positive")
    expected_code_files = list(
        _expected_worker_code_sources(validate_git_sources=validate_git_sources)
    )
    if observed_code_files != expected_code_files:
        raise ValueError("evidence worker code files differ from the expected source set")

    observed_source_files = _require_list(
        manifest.get("source_files"), "baseline source files"
    )
    expected_source_files = list(_expected_baseline_source_records())
    if len(observed_source_files) != len(expected_source_files):
        raise ValueError("evidence baseline source population mismatch")
    for index, (observed, expected) in enumerate(
        zip(observed_source_files, expected_source_files)
    ):
        _validate_relocated_baseline_source(
            observed,
            expected=expected,
            label=f"baseline source file {index}",
            verify_file=validate_git_sources,
        )

    if validate_git_sources:
        return "committed_head_and_frozen_v1_2_sources"
    return "test_only_working_tree_and_manifest_metadata"


def _load_authoritative_ctm_baseline(
    manifest: Mapping[str, Any],
    *,
    checkpoints: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Mapping[tuple[str, str], Mapping[str, float]]:
    source_by_logical = {
        str(item["logical_path"]): _require_mapping(item, "baseline source")
        for item in _require_list(manifest["source_files"], "baseline source files")
    }
    source = source_by_logical.get("per_checkpoint_records.jsonl")
    if source is None:
        raise ValueError("authoritative per-checkpoint scalar baseline is absent")
    path = Path(str(source["local_path"]))
    checkpoint_by_sha = {
        str(item["checkpoint_sha256"]): item for item in checkpoints
    }
    datasets = tuple(
        _require_mapping(policy["ood_partition"], "OOD partition")["dataset_order"]
    )
    artifact_to_metric = {
        artifact: metric for metric, artifact in SCALAR_ARTIFACT_KEYS.items()
    }
    baseline: dict[tuple[str, str], dict[str, float]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = _require_mapping(
                        json.loads(line), f"baseline line {line_number}"
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid baseline JSONL at line {line_number}"
                    ) from error
                checkpoint_sha = row.get("checkpoint_sha256")
                if checkpoint_sha not in checkpoint_by_sha:
                    continue
                checkpoint = checkpoint_by_sha[str(checkpoint_sha)]
                expected_identity = {
                    "checkpoint_role": checkpoint["checkpoint_role"],
                    "config_hash": checkpoint["config_hash"],
                    "training_seed": checkpoint["training_seed"],
                    "evaluation_git_sha": manifest["evaluation_git_sha"],
                    "optimizer_family": checkpoint["optimizer_family"],
                    "labels": checkpoint["role_labels"],
                }
                if any(row.get(field) != expected for field, expected in expected_identity.items()):
                    raise ValueError(
                        f"baseline identity mismatch for checkpoint {checkpoint_sha}"
                    )
                artifact = row.get("artifact_key")
                dataset = row.get("ood_dataset")
                if (
                    artifact not in artifact_to_metric
                    or row.get("detector") != CTM_DETECTOR
                    or dataset not in datasets
                ):
                    continue
                if row.get("status") != "success" or row.get("reason_codes") != []:
                    raise ValueError("authoritative CTM scalar baseline is not successful")
                key = (str(checkpoint_sha), str(dataset))
                metric = artifact_to_metric[str(artifact)]
                values = baseline.setdefault(key, {})
                if metric in values:
                    raise ValueError("duplicate authoritative CTM scalar baseline row")
                values[metric] = _finite_float(
                    row.get("value"), f"authoritative CTM {metric}"
                )
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read authoritative CTM scalar baseline: {error}") from error
    expected_keys = {
        (str(checkpoint["checkpoint_sha256"]), dataset)
        for checkpoint in checkpoints
        for dataset in datasets
    }
    if set(baseline) != expected_keys or any(
        set(values) != set(SCALAR_METRIC_FIELDS) for values in baseline.values()
    ):
        raise ValueError("authoritative CTM scalar baseline population mismatch")
    return baseline


def _validate_test_source_bypass(
    reducer_git_sha: str | None,
    reducer_code_sources: Sequence[Mapping[str, Any]] | None,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    if reducer_git_sha is None or reducer_code_sources is None:
        raise ValueError(
            "tests bypassing committed-source validation must supply reducer Git SHA and code sources"
        )
    git_sha = _require_hex(reducer_git_sha, 40, "test reducer Git SHA")
    sources: list[Mapping[str, Any]] = []
    for index, item in enumerate(reducer_code_sources):
        source = _require_mapping(item, f"test reducer source {index}")
        _require_fields(source, ("path", "sha256", "size"), "test reducer source")
        _safe_logical_path(source["path"], "test reducer source path")
        _require_hex(source["sha256"], 64, "test reducer source SHA256")
        if _require_int(source["size"], "test reducer source size") <= 0:
            raise ValueError("test reducer source size must be positive")
        sources.append(dict(source))
    if not sources:
        raise ValueError("test reducer source list must not be empty")
    return git_sha, tuple(sources)


def _load_candidates(registry: Mapping[str, Any]) -> tuple[CandidateSpec, ...]:
    if registry.get("registry_id") != "fixed_readout_stage2_candidates_v1":
        raise ValueError("candidate registry identity mismatch")
    boundary = _require_mapping(registry.get("selection_boundary"), "selection boundary")
    expected_ids = _require_list(boundary.get("eligible_candidate_ids"), "eligible IDs")
    candidates: list[CandidateSpec] = []
    for item in _require_list(registry.get("candidates"), "candidate list"):
        row = _require_mapping(item, "candidate")
        if row.get("selection_status") != "eligible":
            continue
        candidate_id = str(row.get("candidate_id", ""))
        raw = _require_mapping(row.get("raw_detector"), f"{candidate_id} raw detector")
        transformed = _require_mapping(
            row.get("transformed_detector"), f"{candidate_id} transformed detector"
        )
        transform = _require_mapping(
            row.get("targeted_transform"), f"{candidate_id} transform"
        )
        oracle_ids: list[str] = []
        approximate: list[str] = []
        for oracle_item in _require_list(
            row.get("formula_invariant_oracles"), f"{candidate_id} invariant oracles"
        ):
            oracle = _require_mapping(oracle_item, "invariant oracle")
            oracle_id = str(oracle.get("oracle_id", ""))
            if not oracle_id:
                raise ValueError("invariant oracle ID must not be empty")
            oracle_ids.append(oracle_id)
            if oracle.get("control_class") == "approximate_not_exact":
                approximate.append(oracle_id)
        witnesses = tuple(
            str(_require_mapping(item, "witness").get("witness_id", ""))
            for item in _require_list(
                row.get("non_invariant_witnesses"), f"{candidate_id} witnesses"
            )
        )
        if any(not witness for witness in witnesses):
            raise ValueError("witness ID must not be empty")
        candidates.append(
            CandidateSpec(
                candidate_id=candidate_id,
                priority=_require_int(row.get("priority"), f"{candidate_id} priority"),
                raw_detector=str(raw.get("artifact_key", "")),
                transformed_detector=str(transformed.get("artifact_key", "")),
                transform_id=str(transform.get("transform_id", "")),
                invariant_oracle_ids=tuple(oracle_ids),
                approximate_oracle_ids=tuple(approximate),
                witness_ids=witnesses,
            )
        )
    if [candidate.candidate_id for candidate in candidates] != expected_ids:
        raise ValueError("eligible candidate order differs from the frozen registry")
    if len({candidate.priority for candidate in candidates}) != len(candidates):
        raise ValueError("candidate priorities must be unique")
    return tuple(candidates)


def _parse_checksum_file(path: Path) -> Mapping[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read input checksum file: {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid checksum record at line {line_number}")
        digest = _require_hex(parts[0], 64, f"checksum line {line_number}")
        logical_path = _safe_logical_path(parts[1].lstrip("*"), "checksum path")
        if logical_path in entries:
            raise ValueError(f"duplicate checksum path: {logical_path}")
        entries[logical_path] = digest
    return entries


def _validate_input_checksums(
    selection_path: Path,
    model_metrics_path: Path,
    diagnostic_metrics_path: Path,
    checksum_path: Path,
) -> tuple[str, str, str]:
    entries = _parse_checksum_file(checksum_path)
    expected_names = {
        selection_path.name,
        model_metrics_path.name,
        diagnostic_metrics_path.name,
    }
    if set(entries) != expected_names:
        raise ValueError(
            "input checksums must bind exactly the selection manifest, model metrics, "
            "and diagnostic metrics"
        )
    selection_sha = _sha256_file(selection_path)
    metric_sha = _sha256_file(model_metrics_path)
    diagnostic_sha = _sha256_file(diagnostic_metrics_path)
    if entries[selection_path.name] != selection_sha:
        raise ValueError("selection manifest checksum mismatch")
    if entries[model_metrics_path.name] != metric_sha:
        raise ValueError("model metrics checksum mismatch")
    if entries[diagnostic_metrics_path.name] != diagnostic_sha:
        raise ValueError("diagnostic metrics checksum mismatch")
    return selection_sha, metric_sha, diagnostic_sha


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    model_metrics_path: Path,
    model_metrics_sha256: str,
    model_metric_row_count: int,
    diagnostic_metrics_path: Path,
    diagnostic_metrics_sha256: str,
    diagnostic_metric_row_count: int,
    candidate_registry_sha256: str,
    gate_policy_sha256: str,
    candidates: Sequence[CandidateSpec],
    policy: Mapping[str, Any],
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    required = (
        "schema_version",
        "analysis_id",
        "contract_version",
        "analysis_role",
        "analysis_git_sha",
        "checkpoint_role",
        "inventory_hash",
        "dataset_policy_hash",
        "evaluation_git_sha",
        "checkpoint_count",
        "unique_config_count",
        "seed_set",
        "candidate_selection_seed_set",
        "seed_zero_role_selection_bias",
        "ood_datasets",
        "candidate_registry_sha256",
        "gate_policy_sha256",
        "files",
        "destination_policy",
        "checkpoints",
        "candidate_ids",
        "scientific_gate",
        "source_reuse_manifest_sha256",
        "source_files",
        "code_files",
        "numeric_runtime_fingerprint",
        "diagnostic_ids",
        "diagnostic_population",
    )
    _require_fields(manifest, required, "selection manifest")
    if manifest["schema_version"] != SELECTION_SCHEMA:
        raise ValueError("selection manifest schema mismatch")
    analysis_id = _require_hex(manifest["analysis_id"], 64, "analysis ID")
    if manifest["contract_version"] != "fixed_readout_training_rule_intervention_v2":
        raise ValueError("selection manifest contract version mismatch")
    if manifest["analysis_role"] != WORKER_ANALYSIS_ROLE:
        raise ValueError("selection manifest analysis role mismatch")
    _require_hex(manifest["analysis_git_sha"], 40, "analysis Git SHA")
    _require_hex(manifest["inventory_hash"], 64, "inventory hash")
    _require_hex(manifest["dataset_policy_hash"], 64, "dataset policy hash")
    _require_hex(manifest["evaluation_git_sha"], 40, "evaluation Git SHA")
    _require_hex(
        manifest["source_reuse_manifest_sha256"], 64, "source reuse manifest SHA256"
    )
    runtime = _require_mapping(
        manifest["numeric_runtime_fingerprint"], "numeric runtime fingerprint"
    )
    _require_hex(
        runtime.get("canonical_lock_sha256"),
        64,
        "numeric runtime fingerprint SHA256",
    )
    population = _require_mapping(policy.get("population"), "gate population")
    partition = _require_mapping(policy.get("ood_partition"), "OOD partition")
    if manifest["checkpoint_role"] != population["checkpoint_role"]:
        raise ValueError("selection checkpoint role mismatch")
    expected_population = {
        "checkpoint_count": population["full_bundle_count_for_integrity_and_parity"],
        "unique_config_count": population["unique_config_count"],
        "seed_set": population["seed_set_for_integrity_and_parity"],
        "candidate_selection_seed_set": population["seed_set_for_candidate_selection"],
        "seed_zero_role_selection_bias": population["seed_zero_role_selection_bias"],
        "ood_datasets": partition["dataset_order"],
    }
    for field, expected in expected_population.items():
        if manifest[field] != expected:
            raise ValueError(f"selection manifest frozen population mismatch: {field}")
    if manifest["candidate_registry_sha256"] != candidate_registry_sha256:
        raise ValueError("selection manifest candidate-registry checksum mismatch")
    if manifest["gate_policy_sha256"] != gate_policy_sha256:
        raise ValueError("selection manifest gate-policy checksum mismatch")
    inventory_authority = _require_mapping(
        _require_mapping(policy["authorities"], "gate authorities")[
            "v1_2_checkpoint_inventory"
        ],
        "checkpoint inventory authority",
    )
    if manifest["inventory_hash"] != inventory_authority["inventory_hash"]:
        raise ValueError("selection manifest inventory hash mismatch")
    if manifest["dataset_policy_hash"] != inventory_authority["dataset_policy_hash"]:
        raise ValueError("selection manifest dataset-policy hash mismatch")
    expected_candidate_ids = [candidate.candidate_id for candidate in candidates]
    if manifest["candidate_ids"] != expected_candidate_ids:
        raise ValueError("selection manifest candidate IDs mismatch")
    if manifest["diagnostic_ids"] != list(DIAGNOSTIC_IDS):
        raise ValueError("selection manifest diagnostic IDs mismatch")
    diagnostic_population = _require_mapping(
        manifest["diagnostic_population"], "diagnostic population"
    )
    expected_diagnostic_population = {
        "row_count": diagnostic_metric_row_count,
        "diagnostic_count": len(DIAGNOSTIC_IDS),
        "rows_per_diagnostic": int(expected_population["checkpoint_count"])
        * len(partition["dataset_order"]),
        "selection_status": "diagnostic_only",
        "candidate_eligibility": False,
        "ranking_use": "forbidden",
        "status": "PASS",
        "reason_codes": [],
    }
    if diagnostic_population != expected_diagnostic_population:
        raise ValueError("selection manifest diagnostic population mismatch")
    if manifest["destination_policy"] != "atomic_new_directory_no_overwrite":
        raise ValueError("selection manifest destination policy mismatch")
    scientific_gate = _require_mapping(manifest["scientific_gate"], "scientific gate")
    if scientific_gate.get("status") != "NOT_RUN":
        raise ValueError("selection manifest must precede the scientific gate")
    _require_reason_codes(scientific_gate.get("reason_codes"), "gate reason codes")

    files = _require_list(manifest["files"], "selection files")
    file_by_path: dict[str, Mapping[str, Any]] = {}
    for item in files:
        record = _require_mapping(item, "selection file")
        logical_path = _safe_logical_path(
            record.get("logical_path"), "selection file logical path"
        )
        if logical_path in file_by_path:
            raise ValueError(f"duplicate selection file binding: {logical_path}")
        file_by_path[logical_path] = record
    expected_files = {
        "model_metrics.jsonl": (
            model_metrics_path,
            model_metrics_sha256,
            model_metric_row_count,
            MODEL_METRIC_SCHEMA,
        ),
        "diagnostic_metrics.jsonl": (
            diagnostic_metrics_path,
            diagnostic_metrics_sha256,
            diagnostic_metric_row_count,
            DIAGNOSTIC_METRIC_SCHEMA,
        ),
    }
    if set(file_by_path) != set(expected_files):
        raise ValueError(
            "selection manifest must bind exactly model_metrics.jsonl and "
            "diagnostic_metrics.jsonl"
        )
    for logical_path, (path, checksum, row_count, schema) in expected_files.items():
        file_record = file_by_path[logical_path]
        _require_fields(
            file_record,
            ("logical_path", "sha256", "size", "row_count", "schema_version"),
            f"{logical_path} file record",
        )
        if file_record["schema_version"] != schema:
            raise ValueError(f"{logical_path} file-record schema mismatch")
        if file_record["sha256"] != checksum:
            raise ValueError(f"{logical_path} manifest checksum mismatch")
        if file_record["size"] != path.stat().st_size:
            raise ValueError(f"{logical_path} manifest size mismatch")
        if file_record["row_count"] != row_count:
            raise ValueError(f"{logical_path} manifest row-count mismatch")

    checkpoints_raw = _require_list(manifest["checkpoints"], "checkpoints")
    if len(checkpoints_raw) != expected_population["checkpoint_count"]:
        raise ValueError("selection manifest checkpoint count mismatch")
    checkpoints: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for index, item in enumerate(checkpoints_raw):
        checkpoint = _require_mapping(item, f"checkpoint {index}")
        _require_fields(
            checkpoint,
            (
                "checkpoint_sha256",
                "config_hash",
                "training_seed",
                "checkpoint_role",
                "optimizer_family",
                "role_labels",
                "source_allowlist_sha256",
            ),
            f"checkpoint {index}",
        )
        checkpoint_sha = _require_hex(
            checkpoint["checkpoint_sha256"], 64, "checkpoint SHA256"
        )
        config_hash = _require_hex(checkpoint["config_hash"], 64, "config hash")
        seed = _require_int(checkpoint["training_seed"], "training seed")
        role = str(checkpoint["checkpoint_role"])
        _require_hex(
            checkpoint["source_allowlist_sha256"], 64, "source allowlist SHA256"
        )
        if role != "last" or seed not in (0, 1, 2):
            raise ValueError("checkpoint identity is outside the frozen population")
        if not isinstance(checkpoint["optimizer_family"], str):
            raise ValueError("checkpoint optimizer_family must be a string")
        labels = _require_list(checkpoint["role_labels"], "checkpoint role labels")
        if any(not isinstance(label, str) for label in labels):
            raise ValueError("checkpoint role labels must be strings")
        identity = (checkpoint_sha, config_hash, seed, role)
        if identity in seen:
            raise ValueError("duplicate checkpoint identity")
        seen.add(identity)
        checkpoints.append(checkpoint)
    sort_key = lambda item: (
        str(item["config_hash"]),
        int(item["training_seed"]),
        str(item["checkpoint_sha256"]),
    )
    if checkpoints != sorted(checkpoints, key=sort_key):
        raise ValueError("selection checkpoints must use canonical sorted order")
    config_seeds: dict[str, set[int]] = {}
    for checkpoint in checkpoints:
        config_seeds.setdefault(str(checkpoint["config_hash"]), set()).add(
            int(checkpoint["training_seed"])
        )
    if len(config_seeds) != expected_population["unique_config_count"] or any(
        seeds != {0, 1, 2} for seeds in config_seeds.values()
    ):
        raise ValueError("selection checkpoints must be 10 configurations x seeds 0,1,2")
    return analysis_id, tuple(checkpoints)


def _validate_authoritative_checkpoint_population(
    manifest: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> None:
    authority = _require_mapping(policy["authorities"], "gate authorities")
    inventory_binding = _require_mapping(
        authority["v1_2_checkpoint_inventory"], "checkpoint inventory authority"
    )
    relative = _safe_logical_path(inventory_binding["path"], "checkpoint inventory path")
    inventory_path = (REPOSITORY_ROOT / relative).resolve()
    if inventory_path != DEFAULT_CHECKPOINT_INVENTORY.resolve():
        raise ValueError("checkpoint inventory authority path mismatch")
    if _sha256_file(inventory_path) != inventory_binding["sha256"]:
        raise ValueError("checkpoint inventory file checksum mismatch")
    inventory = _load_json(inventory_path, "checkpoint inventory")
    for field in ("inventory_hash", "dataset_policy_hash"):
        if inventory.get(field) != inventory_binding[field]:
            raise ValueError(f"checkpoint inventory authority mismatch: {field}")
        if manifest[field] != inventory_binding[field]:
            raise ValueError(f"selection manifest authority mismatch: {field}")
    expected: set[tuple[Any, ...]] = set()
    for item in _require_list(inventory.get("checkpoint_jobs"), "checkpoint jobs"):
        job = _require_mapping(item, "checkpoint job")
        if (
            job.get("checkpoint_role") == "last"
            and job.get("reporting_role") == "confirmatory_primary"
        ):
            expected.add(
                (
                    str(job["checkpoint_sha256"]),
                    str(job["config_hash"]),
                    int(job["training_seed"]),
                    str(job["checkpoint_role"]),
                    str(job["optimizer_family"]),
                    tuple(job["labels"]),
                )
            )
    observed = {
        (
            str(item["checkpoint_sha256"]),
            str(item["config_hash"]),
            int(item["training_seed"]),
            str(item["checkpoint_role"]),
            str(item["optimizer_family"]),
            tuple(item["role_labels"]),
        )
        for item in checkpoints
    }
    if len(expected) != 30 or observed != expected:
        raise ValueError("selection checkpoints differ from the frozen inventory population")

    reuse_sha = _sha256_file(DEFAULT_REUSE_MANIFEST)
    if manifest["source_reuse_manifest_sha256"] != reuse_sha:
        raise ValueError("selection manifest source-reuse checksum mismatch")
    reuse = _load_json(DEFAULT_REUSE_MANIFEST, "Stage-2 reuse manifest")
    if reuse.get("retrieval_ready") is not True:
        raise ValueError("Stage-2 reuse manifest is not retrieval-ready")
    bindings = _require_mapping(reuse.get("analysis_policy"), "reuse analysis policy")
    if (
        _require_mapping(bindings.get("candidate_registry"), "registry binding").get(
            "sha256"
        )
        != manifest["candidate_registry_sha256"]
        or _require_mapping(bindings.get("gate_policy"), "gate binding").get("sha256")
        != manifest["gate_policy_sha256"]
    ):
        raise ValueError("reuse manifest analysis-policy binding mismatch")
    reuse_expected: set[tuple[Any, ...]] = set()
    for item in _require_list(reuse.get("bundles"), "reuse bundles"):
        bundle = _require_mapping(item, "reuse bundle")
        identity = _require_mapping(bundle.get("identity"), "reuse bundle identity")
        integrity = _require_mapping(
            bundle.get("remote_integrity"), "reuse bundle integrity"
        )
        if (
            identity.get("checkpoint_role") != "last"
            or identity.get("reporting_role") != "confirmatory_primary"
            or identity.get("inventory_hash") != manifest["inventory_hash"]
            or identity.get("dataset_policy_hash") != manifest["dataset_policy_hash"]
            or identity.get("evaluation_git_sha") != manifest["evaluation_git_sha"]
        ):
            raise ValueError("reuse bundle identity is outside the selection population")
        reuse_expected.add(
            (
                str(bundle["bundle_id"]),
                str(bundle["config_hash"]),
                int(bundle["training_seed"]),
                str(identity["checkpoint_role"]),
                str(bundle["optimizer_family"]),
                tuple(bundle["labels"]),
                str(integrity["allowed_payload_catalog_sha256"]),
            )
        )
    reuse_observed = {
        (
            str(item["checkpoint_sha256"]),
            str(item["config_hash"]),
            int(item["training_seed"]),
            str(item["checkpoint_role"]),
            str(item["optimizer_family"]),
            tuple(item["role_labels"]),
            str(item["source_allowlist_sha256"]),
        )
        for item in checkpoints
    }
    if len(reuse_expected) != 30 or reuse_observed != reuse_expected:
        raise ValueError("selection checkpoints differ from the frozen reuse manifest")


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["candidate_id"]),
        str(row["checkpoint_sha256"]),
        str(row["ood_dataset"]),
        str(row["state"]),
    )


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_id: str,
    checkpoints: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateSpec],
    policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    required = (
        "schema_version",
        "analysis_id",
        "candidate_id",
        "config_hash",
        "optimizer_family",
        "role_labels",
        "training_seed",
        "checkpoint_sha256",
        "checkpoint_role",
        "ood_dataset",
        "ood_group",
        "detector",
        "transform_id",
        "state",
        "num_id",
        "num_ood",
        "auroc",
        "fpr95",
        "fpr95_threshold",
        "achieved_id_tpr",
        "misordering_mass",
        "tie_mass",
        "pair_order_transition_mass",
        "corrected_pair_mass",
        "harmed_pair_mass",
        "net_corrected_pair_mass",
        "score_reconstruction_max_abs_error",
        "score_reconstruction_max_rel_error",
        "formula_reconstruction_pass",
        "pair_credit_identity_pass",
        "rank_disagreement_fraction",
        "scalar_parity",
        "exact_null_oracle",
        "invariant_oracles",
        "non_invariant_witness",
        "id_metrics",
        "score_sha256",
        "score_distribution",
        "geometry_channel",
        "score_components",
        "source_references",
        "source_allowlist_sha256",
        "status",
        "reason_codes",
    )
    partition = _require_mapping(policy["ood_partition"], "OOD partition")
    datasets = list(partition["dataset_order"])
    group_by_dataset = {
        dataset: "near" if dataset in partition["near"] else "far"
        for dataset in datasets
    }
    checkpoint_by_sha = {str(item["checkpoint_sha256"]): item for item in checkpoints}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    expected_count = len(checkpoints) * len(candidates) * len(datasets) * 2
    if len(rows) != expected_count:
        raise ValueError(
            f"model metric population must contain exactly {expected_count} rows"
        )

    hard_reasons: set[str] = set()
    seen: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    witness_observed = {candidate.candidate_id: False for candidate in candidates}
    exact_tolerance = _require_mapping(
        policy["invariant_and_witness_policy"]["exact_score_tolerance"],
        "exact score tolerance",
    )
    exact_atol = float(exact_tolerance["absolute"])
    exact_rtol = float(exact_tolerance["relative"])
    exact_auroc_tol = float(
        policy["invariant_and_witness_policy"]["exact_auroc_drift_tolerance"]
    )
    approximate_auroc_tol = float(
        policy["invariant_and_witness_policy"][
            "approximate_knn_l2_auroc_drift_tolerance"
        ]
    )
    transition_keys = tuple(policy["score_overlap"]["required_transition_mass_fields"])
    inventory = _load_json(DEFAULT_CHECKPOINT_INVENTORY, "checkpoint inventory")
    dataset_policy = _require_mapping(
        inventory.get("dataset_policy"), "checkpoint-inventory dataset policy"
    )
    id_splits = _require_mapping(dataset_policy.get("id_splits"), "inventory ID splits")
    ood_splits = _require_mapping(
        dataset_policy.get("ood_splits"), "inventory OOD splits"
    )
    frozen_num_id = _require_int(
        _require_mapping(id_splits.get("id_test"), "inventory ID-test split").get(
            "count"
        ),
        "inventory ID-test count",
    )
    frozen_id_train_count = _require_int(
        _require_mapping(id_splits.get("id_train"), "inventory ID-train split").get(
            "count"
        ),
        "inventory ID-train count",
    )
    frozen_num_ood = {
        dataset: _require_int(
            _require_mapping(ood_splits.get(dataset), f"inventory OOD split {dataset}").get(
                "count"
            ),
            f"inventory OOD count {dataset}",
        )
        for dataset in datasets
    }
    reuse_manifest = _load_json(DEFAULT_REUSE_MANIFEST, "Stage-2 reuse manifest")
    reuse_bundle_by_id = {
        str(bundle["bundle_id"]): _require_mapping(bundle, "reuse bundle")
        for bundle in _require_list(reuse_manifest.get("bundles"), "reuse bundles")
    }
    geometry_norms_by_bundle: dict[str, Mapping[str, Any]] = {}
    shared_sources: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    detector_split_evidence: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    cache_manifest_by_variant: dict[tuple[str, str], str] = {}

    for index, row in enumerate(rows):
        label = f"model metric row {index}"
        _require_fields(row, required, label)
        if row["schema_version"] != MODEL_METRIC_SCHEMA:
            raise ValueError(f"{label} schema mismatch")
        if row["analysis_id"] != analysis_id:
            raise ValueError(f"{label} analysis identity mismatch")
        candidate_id = str(row["candidate_id"])
        if candidate_id not in candidate_by_id:
            raise ValueError(f"{label} has an ineligible candidate")
        candidate = candidate_by_id[candidate_id]
        checkpoint_sha = _require_hex(
            row["checkpoint_sha256"], 64, f"{label} checkpoint SHA256"
        )
        if checkpoint_sha not in checkpoint_by_sha:
            raise ValueError(f"{label} checkpoint is absent from the selection manifest")
        checkpoint = checkpoint_by_sha[checkpoint_sha]
        for field in (
            "config_hash",
            "optimizer_family",
            "role_labels",
            "training_seed",
            "checkpoint_role",
        ):
            if row[field] != checkpoint[field]:
                raise ValueError(f"{label} checkpoint binding mismatch: {field}")
        if row["source_allowlist_sha256"] != checkpoint["source_allowlist_sha256"]:
            raise ValueError(f"{label} source allowlist binding mismatch")
        dataset = str(row["ood_dataset"])
        if dataset not in group_by_dataset or row["ood_group"] != group_by_dataset[dataset]:
            raise ValueError(f"{label} OOD dataset/group mismatch")
        state = str(row["state"])
        if state == "raw":
            expected_detector = candidate.raw_detector
            expected_transform = RAW_TRANSFORM_ID
        elif state == "transformed":
            expected_detector = candidate.transformed_detector
            expected_transform = candidate.transform_id
        else:
            raise ValueError(f"{label} state must be raw or transformed")
        if row["detector"] != expected_detector or row["transform_id"] != expected_transform:
            raise ValueError(f"{label} detector/transform does not match the registry")
        key = _row_key(row)
        if key in seen:
            raise ValueError(f"duplicate model metric identity: {key}")
        seen[key] = row

        num_id = _require_int(row["num_id"], f"{label} num_id")
        num_ood = _require_int(row["num_ood"], f"{label} num_ood")
        if num_id != frozen_num_id or num_ood != frozen_num_ood[dataset]:
            raise ValueError(f"{label} sample counts differ from the frozen population")
        score_distribution = _validate_score_distribution(
            row["score_distribution"],
            label=f"{label} score distribution",
            num_id=num_id,
            num_ood=num_ood,
        )
        _require_hex(row["score_sha256"], 64, f"{label} combined score SHA256")
        if checkpoint_sha not in reuse_bundle_by_id:
            raise ValueError(f"{label} reuse bundle identity is absent")
        reuse_bundle = reuse_bundle_by_id[checkpoint_sha]
        allowed_files = {
            str(item["logical_path"]): _require_mapping(item, "allowlist file")
            for item in _require_list(
                _require_mapping(
                    reuse_bundle.get("remote_integrity"), "reuse remote integrity"
                ).get("files"),
                "reuse allowlist files",
            )
        }
        sources, feature_shapes = _validate_source_references(
            row["source_references"],
            label=f"{label} source references",
            analysis_id=analysis_id,
            candidate_id=candidate_id,
            detector=str(row["detector"]),
            dataset=dataset,
            num_id=num_id,
            num_ood=num_ood,
            checkpoint=checkpoint,
            allowed_files=allowed_files,
        )
        if feature_shapes["id_train"][0] != frozen_id_train_count:
            raise ValueError(f"{label} ID-train source count differs from the frozen population")
        source_scores = _require_mapping(
            sources["score_arrays"], f"{label} source score arrays"
        )
        if any(
            score_distribution[split_name]["array_sha256"]
            != source_scores[split_name]["array_sha256"]
            for split_name in ("id_test", "ood")
        ):
            raise ValueError(f"{label} score distribution/source digest mismatch")
        geometry = _validate_geometry_channel(
            row["geometry_channel"],
            label=f"{label} geometry channel",
            candidate_id=candidate_id,
            active_dataset=dataset,
            datasets=datasets,
            active_feature_shapes=feature_shapes,
        )
        geometry_norms = {
            split_name: split["feature_norm"]
            for split_name, split in _require_mapping(
                geometry["splits"], f"{label} geometry splits"
            ).items()
        }
        prior_geometry_norms = geometry_norms_by_bundle.setdefault(
            checkpoint_sha, geometry_norms
        )
        if prior_geometry_norms != geometry_norms:
            hard_reasons.add("GEOMETRY_CHANNEL_POPULATION_MISMATCH")

        feature_sources = _require_mapping(
            sources["feature_arrays"], f"{label} feature sources"
        )
        logit_sources = _require_mapping(
            sources["null_logit_arrays"], f"{label} logit sources"
        )
        for source_group, source_key, shared_name in (
            (feature_sources, "id_train", "feature:id_train"),
            (feature_sources, "id_test", "feature:id_test"),
            (feature_sources, "ood", f"feature:{dataset}"),
            (logit_sources, "id_test", "logit:id_test"),
            (logit_sources, "ood", f"logit:{dataset}"),
        ):
            shared_key = (checkpoint_sha, shared_name, "source")
            previous_source = shared_sources.setdefault(shared_key, source_group[source_key])
            if previous_source != source_group[source_key]:
                hard_reasons.add("SOURCE_REFERENCE_DUPLICATE_EVIDENCE_MISMATCH")
        for split_name, source_key in (("id_test", "id_test"), (dataset, "ood")):
            score_key = (checkpoint_sha, str(row["detector"]), split_name)
            previous_score_source = detector_split_evidence.setdefault(
                score_key, source_scores[source_key]
            )
            if previous_score_source != source_scores[source_key]:
                hard_reasons.add("SCORE_SOURCE_DUPLICATE_EVIDENCE_MISMATCH")
        for source_reference in source_scores.values():
            if source_reference["source_kind"] == "stage2_knn_cache_score_array":
                cache_key = (checkpoint_sha, str(source_reference["variant_id"]))
                cache_sha = str(source_reference["cache_manifest_sha256"])
                previous_cache_sha = cache_manifest_by_variant.setdefault(
                    cache_key, cache_sha
                )
                if previous_cache_sha != cache_sha:
                    hard_reasons.add("CACHE_MANIFEST_DUPLICATE_EVIDENCE_MISMATCH")

        score_components = _validate_score_components(
            row["score_components"],
            label=f"{label} score components",
            candidate_id=candidate_id,
            num_id=num_id,
            num_ood=num_ood,
            score_distribution=score_distribution,
            exact_atol=exact_atol,
            exact_rtol=exact_rtol,
        )
        id_component_key = (checkpoint_sha, str(row["detector"]), "component:id_test")
        previous_id_components = detector_split_evidence.setdefault(
            id_component_key, score_components["id_test"]
        )
        if previous_id_components != score_components["id_test"]:
            hard_reasons.add("SCORE_COMPONENT_DUPLICATE_EVIDENCE_MISMATCH")
        auroc = _unit_float(row["auroc"], f"{label} AUROC")
        _unit_float(row["fpr95"], f"{label} FPR95")
        _finite_float(row["fpr95_threshold"], f"{label} FPR95 threshold")
        _unit_float(row["achieved_id_tpr"], f"{label} achieved ID TPR")
        misordering = _unit_float(row["misordering_mass"], f"{label} misordering")
        tie_mass = _unit_float(row["tie_mass"], f"{label} tie mass")
        if not math.isclose(
            1.0 - auroc,
            misordering + 0.5 * tie_mass,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            hard_reasons.add("PAIRWISE_AUROC_IDENTITY_FAILED")
        _unit_float(row["rank_disagreement_fraction"], f"{label} rank disagreement")
        reconstruction_abs = _finite_float(
            row["score_reconstruction_max_abs_error"],
            f"{label} reconstruction absolute error",
        )
        reconstruction_rel = _finite_float(
            row["score_reconstruction_max_rel_error"],
            f"{label} reconstruction relative error",
        )
        # The worker's boolean is evaluated elementwise with the frozen
        # ``atol + rtol * |reference|`` rule.  The two maxima are retained for
        # audit, but applying independent thresholds to both maxima here would
        # be a stricter and different test.
        if row["formula_reconstruction_pass"] is not True:
            hard_reasons.add("FORMULA_RECONSTRUCTION_FAILED")
        if row["status"] != "PASS" or _require_reason_codes(
            row["reason_codes"], f"{label} reason codes"
        ):
            hard_reasons.add("MODEL_METRIC_STATUS_FAILED")

        parity = _require_mapping(row["scalar_parity"], f"{label} scalar parity")
        _require_fields(parity, ("status", "reason_codes", "metrics"), "scalar parity")
        expected_parity = (
            "NOT_APPLICABLE"
            if candidate_id == "radial_l2_knn" and state == "raw"
            else "PASS"
        )
        parity_reasons = _require_reason_codes(
            parity["reason_codes"], f"{label} parity reason codes"
        )
        expected_parity_reasons = (
            ["V2_ONLY_RAW_KNN_HAS_NO_V1_2_BASELINE"]
            if expected_parity == "NOT_APPLICABLE"
            else []
        )
        parity_reason_failure = parity_reasons != expected_parity_reasons
        if parity["status"] != expected_parity or parity_reason_failure:
            hard_reasons.add("SCALAR_PARITY_FAILED")
        parity_metrics = _require_mapping(parity["metrics"], "scalar parity metrics")
        for field in ("auroc", "fpr95", "fpr95_threshold", "achieved_id_tpr"):
            parity_value = _finite_float(parity_metrics[field], f"{label} parity {field}")
            if parity_value != float(row[field]):
                hard_reasons.add("SCALAR_PARITY_ROW_MISMATCH")

        null = _require_mapping(row["exact_null_oracle"], f"{label} exact null oracle")
        _require_fields(
            null,
            (
                "status",
                "reason_codes",
                "score_sha256_equal",
                "rank_disagreement_fraction",
                "auroc_abs_drift",
                "raw_auroc",
                "transformed_auroc",
            ),
            "exact null oracle",
        )
        null_rank = _unit_float(
            null["rank_disagreement_fraction"], f"{label} null rank disagreement"
        )
        null_drift = _unit_float(null["auroc_abs_drift"], f"{label} null AUROC drift")
        null_raw = _unit_float(null["raw_auroc"], f"{label} null raw AUROC")
        null_transformed = _unit_float(
            null["transformed_auroc"], f"{label} null transformed AUROC"
        )
        if (
            null["status"] != "PASS"
            or _require_reason_codes(null["reason_codes"], f"{label} null reasons")
            or null["score_sha256_equal"] is not True
            or null_rank != 0.0
            or null_drift > exact_auroc_tol
            or not math.isclose(
                abs(null_raw - null_transformed),
                null_drift,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            )
        ):
            hard_reasons.add("EXACT_NULL_ORACLE_FAILED")

        oracle_rows = _require_list(row["invariant_oracles"], f"{label} invariant oracles")
        oracle_by_id: dict[str, Mapping[str, Any]] = {}
        for oracle_item in oracle_rows:
            oracle = _require_mapping(oracle_item, "invariant oracle result")
            _require_fields(
                oracle,
                (
                    "oracle_id",
                    "status",
                    "reason_codes",
                    "score_max_abs_error",
                    "score_max_rel_error",
                    "rank_disagreement_fraction",
                    "auroc_abs_drift",
                ),
                "invariant oracle result",
            )
            oracle_id = str(oracle["oracle_id"])
            if oracle_id in oracle_by_id:
                raise ValueError(f"{label} contains duplicate invariant oracle IDs")
            oracle_by_id[oracle_id] = oracle
        if set(oracle_by_id) != set(candidate.invariant_oracle_ids):
            hard_reasons.add("INVARIANT_ORACLE_POPULATION_FAILED")
        for oracle_id in candidate.invariant_oracle_ids:
            if oracle_id not in oracle_by_id:
                continue
            oracle = oracle_by_id[oracle_id]
            rank_drift = _unit_float(
                oracle["rank_disagreement_fraction"], f"{label} {oracle_id} rank drift"
            )
            auroc_drift = _unit_float(
                oracle["auroc_abs_drift"], f"{label} {oracle_id} AUROC drift"
            )
            _finite_float(
                oracle["score_max_abs_error"], f"{label} {oracle_id} absolute error"
            )
            _finite_float(
                oracle["score_max_rel_error"], f"{label} {oracle_id} relative error"
            )
            tolerance = (
                approximate_auroc_tol
                if oracle_id in candidate.approximate_oracle_ids
                else exact_auroc_tol
            )
            if (
                oracle["status"] != "PASS"
                or _require_reason_codes(
                    oracle["reason_codes"], f"{label} {oracle_id} reasons"
                )
                or auroc_drift > tolerance
                or (
                    oracle_id not in candidate.approximate_oracle_ids
                    and rank_drift != 0.0
                )
            ):
                hard_reasons.add("INVARIANT_ORACLE_FAILED")

        witness = _require_mapping(
            row["non_invariant_witness"], f"{label} non-invariant witness"
        )
        _require_fields(
            witness,
            (
                "witness_id",
                "status",
                "reason_codes",
                "raw_score_max_abs_change",
                "raw_rank_disagreement_fraction",
                "targeted_score_max_abs_error",
                "targeted_rank_disagreement_fraction",
                "targeted_auroc_abs_drift",
            ),
            "non-invariant witness",
        )
        witness_id = str(witness["witness_id"])
        if witness_id not in candidate.witness_ids:
            hard_reasons.add("NON_INVARIANT_WITNESS_POPULATION_FAILED")
        raw_change = _finite_float(
            witness["raw_score_max_abs_change"], f"{label} witness raw change"
        )
        raw_rank_change = _unit_float(
            witness["raw_rank_disagreement_fraction"], f"{label} witness rank change"
        )
        targeted_abs = _finite_float(
            witness["targeted_score_max_abs_error"],
            f"{label} witness targeted absolute error",
        )
        targeted_rank = _unit_float(
            witness["targeted_rank_disagreement_fraction"],
            f"{label} witness targeted rank disagreement",
        )
        targeted_auroc_drift = _unit_float(
            witness["targeted_auroc_abs_drift"],
            f"{label} witness targeted AUROC drift",
        )
        if witness["status"] != "PASS" or _require_reason_codes(
            witness["reason_codes"], f"{label} witness reasons"
        ):
            hard_reasons.add("NON_INVARIANT_WITNESS_FAILED")
        _validate_witness_normalization_status(
            witness,
            label=f"{label} witness",
            candidate_id=candidate_id,
            geometry=geometry,
            datasets=datasets,
            exact_atol=exact_atol,
            exact_rtol=exact_rtol,
            approximate_auroc_tolerance=approximate_auroc_tol,
        )
        if raw_change > exact_atol or raw_rank_change > 0.0:
            witness_observed[candidate_id] = True
        if candidate_id == "radial_l2_mahalanobis":
            if (
                targeted_abs > exact_atol
                or targeted_rank != 0.0
                or targeted_auroc_drift > exact_auroc_tol
            ):
                hard_reasons.add("TARGETED_WITNESS_EXACT_RELATION_FAILED")
        elif targeted_auroc_drift > approximate_auroc_tol:
            hard_reasons.add("TARGETED_WITNESS_APPROXIMATE_RELATION_FAILED")

        id_metrics = _require_mapping(row["id_metrics"], f"{label} ID metrics")
        _require_fields(id_metrics, ("accuracy", "nll", "ece"), "ID metrics")
        _unit_float(id_metrics["accuracy"], f"{label} ID accuracy")
        if _finite_float(id_metrics["nll"], f"{label} ID NLL") < 0.0:
            raise ValueError(f"{label} ID NLL must be nonnegative")
        _unit_float(id_metrics["ece"], f"{label} ID ECE")

        transitions = _require_mapping(
            row["pair_order_transition_mass"], f"{label} transition mass"
        )
        if set(transitions) != set(transition_keys):
            raise ValueError(f"{label} transition-state population mismatch")
        transition_values = [
            _unit_float(transitions[key], f"{label} transition {key}")
            for key in transition_keys
        ]
        if not math.isclose(sum(transition_values), 1.0, rel_tol=1.0e-10, abs_tol=1.0e-12):
            hard_reasons.add("PAIR_TRANSITION_MASS_FAILED")
        corrected = _unit_float(row["corrected_pair_mass"], f"{label} corrected mass")
        harmed = _unit_float(row["harmed_pair_mass"], f"{label} harmed mass")
        net = _finite_float(row["net_corrected_pair_mass"], f"{label} net mass")
        credits = {"correct": 1.0, "tie": 0.5, "misordered": 0.0}
        recomputed_corrected = 0.0
        recomputed_harmed = 0.0
        for before in credits:
            for after in credits:
                mass = float(transitions[f"{before}_to_{after}"])
                recomputed_corrected += mass * max(
                    credits[after] - credits[before], 0.0
                )
                recomputed_harmed += mass * max(
                    credits[before] - credits[after], 0.0
                )
        recomputed_net = recomputed_corrected - recomputed_harmed
        strict_misordering = sum(
            float(value)
            for key, value in transitions.items()
            if (
                key.startswith("misordered_to_")
                if state == "raw"
                else key.endswith("_to_misordered")
            )
        )
        strict_tie = sum(
            float(value)
            for key, value in transitions.items()
            if (
                key.startswith("tie_to_")
                if state == "raw"
                else key.endswith("_to_tie")
            )
        )
        if not math.isclose(misordering, strict_misordering, rel_tol=1.0e-10, abs_tol=1.0e-12):
            hard_reasons.add("PAIR_STRICT_MISORDERING_MASS_FAILED")
        if not math.isclose(tie_mass, strict_tie, rel_tol=1.0e-10, abs_tol=1.0e-12):
            hard_reasons.add("PAIR_TIE_MASS_FAILED")
        if not (
            math.isclose(corrected, recomputed_corrected, rel_tol=1.0e-10, abs_tol=1.0e-12)
            and math.isclose(harmed, recomputed_harmed, rel_tol=1.0e-10, abs_tol=1.0e-12)
            and math.isclose(net, recomputed_net, rel_tol=1.0e-10, abs_tol=1.0e-12)
        ):
            hard_reasons.add("PAIR_CREDIT_MASS_RECONSTRUCTION_FAILED")
        if row["pair_credit_identity_pass"] is not True or not math.isclose(
            corrected - harmed, net, rel_tol=1.0e-10, abs_tol=1.0e-12
        ):
            hard_reasons.add("PAIR_CREDIT_IDENTITY_FAILED")

    expected_keys = {
        (candidate.candidate_id, str(checkpoint["checkpoint_sha256"]), dataset, state)
        for candidate in candidates
        for checkpoint in checkpoints
        for dataset in datasets
        for state in ("raw", "transformed")
    }
    if set(seen) != expected_keys:
        raise ValueError("model metric identities do not cover the frozen population")

    for candidate in candidates:
        if not witness_observed[candidate.candidate_id]:
            hard_reasons.add(
                f"NON_INVARIANT_WITNESS_NOT_OBSERVED:{candidate.candidate_id}"
            )
    for candidate in candidates:
        for checkpoint in checkpoints:
            checkpoint_sha = str(checkpoint["checkpoint_sha256"])
            for dataset in datasets:
                raw = seen[(candidate.candidate_id, checkpoint_sha, dataset, "raw")]
                transformed = seen[
                    (candidate.candidate_id, checkpoint_sha, dataset, "transformed")
                ]
                duplicate_fields = (
                    "pair_order_transition_mass",
                    "corrected_pair_mass",
                    "harmed_pair_mass",
                    "net_corrected_pair_mass",
                    "pair_credit_identity_pass",
                    "exact_null_oracle",
                    "id_metrics",
                )
                if any(raw[field] != transformed[field] for field in duplicate_fields):
                    hard_reasons.add("RAW_TRANSFORMED_PAIR_EVIDENCE_MISMATCH")
                raw_geometry = _require_mapping(
                    raw["geometry_channel"], "raw paired geometry"
                )
                transformed_geometry = _require_mapping(
                    transformed["geometry_channel"], "transformed paired geometry"
                )
                if raw_geometry != transformed_geometry:
                    hard_reasons.add("RAW_TRANSFORMED_GEOMETRY_CHANNEL_MISMATCH")
                raw_sources = _require_mapping(
                    raw["source_references"], "raw paired sources"
                )
                transformed_sources = _require_mapping(
                    transformed["source_references"], "transformed paired sources"
                )
                for field in (
                    "source_allowlist_sha256",
                    "feature_arrays",
                    "null_logit_arrays",
                ):
                    if raw_sources[field] != transformed_sources[field]:
                        hard_reasons.add("RAW_TRANSFORMED_SOURCE_EVIDENCE_MISMATCH")
                raw_preservation = _require_mapping(
                    raw["non_invariant_witness"], "raw paired witness"
                )["normalization_status_preservation"]
                transformed_preservation = _require_mapping(
                    transformed["non_invariant_witness"],
                    "transformed paired witness",
                )["normalization_status_preservation"]
                if raw_preservation != transformed_preservation:
                    hard_reasons.add(
                        "RAW_TRANSFORMED_WITNESS_NORMALIZATION_MISMATCH"
                    )
                if candidate.candidate_id == "radial_l2_knn":
                    raw_components = _require_mapping(
                        raw["score_components"], "raw paired kNN components"
                    )
                    transformed_components = _require_mapping(
                        transformed["score_components"],
                        "transformed paired kNN components",
                    )
                    for field in (
                        "raw_to_l2_neighbor_overlap",
                        "exact_bank_mapping",
                    ):
                        if raw_components[field] != transformed_components[field]:
                            hard_reasons.add(
                                "RAW_TRANSFORMED_KNN_MAPPING_EVIDENCE_MISMATCH"
                            )
                transitions = _require_mapping(
                    transformed["pair_order_transition_mass"], "paired transition mass"
                )
                credits = {"correct": 1.0, "tie": 0.5, "misordered": 0.0}
                transition_raw_auroc = sum(
                    float(transitions[f"{before}_to_{after}"]) * credits[before]
                    for before in credits
                    for after in credits
                )
                transition_transformed_auroc = sum(
                    float(transitions[f"{before}_to_{after}"]) * credits[after]
                    for before in credits
                    for after in credits
                )
                if not math.isclose(
                    float(raw["auroc"]),
                    transition_raw_auroc,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                ) or not math.isclose(
                    float(transformed["auroc"]),
                    transition_transformed_auroc,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                ):
                    hard_reasons.add("PAIR_TRANSITION_AUROC_RECONSTRUCTION_FAILED")
                expected_net = float(transformed["auroc"]) - float(raw["auroc"])
                if not math.isclose(
                    float(transformed["net_corrected_pair_mass"]),
                    expected_net,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                ):
                    hard_reasons.add("PAIR_CREDIT_AUROC_IDENTITY_FAILED")

    criteria = {
        "input_population_pass": True,
        "model_metric_status_pass": "MODEL_METRIC_STATUS_FAILED" not in hard_reasons,
        "scalar_parity_pass": not any(
            reason.startswith("SCALAR_PARITY") for reason in hard_reasons
        ),
        "formula_reconstruction_pass": "FORMULA_RECONSTRUCTION_FAILED"
        not in hard_reasons,
        "pair_order_identity_pass": not any(
            reason.startswith("PAIR_") or reason.startswith("RAW_TRANSFORMED_PAIR_")
            for reason in hard_reasons
        ),
        "exact_null_oracles_pass": "EXACT_NULL_ORACLE_FAILED" not in hard_reasons,
        "invariant_oracles_pass": not any(
            reason.startswith("INVARIANT_ORACLE") for reason in hard_reasons
        ),
        "non_invariant_witnesses_pass": not any(
            reason.startswith("NON_INVARIANT_WITNESS")
            or reason.startswith("TARGETED_WITNESS")
            or reason.startswith("RAW_TRANSFORMED_WITNESS")
            for reason in hard_reasons
        ),
        "distribution_geometry_source_evidence_pass": not any(
            reason.startswith(prefix)
            for reason in hard_reasons
            for prefix in (
                "GEOMETRY_",
                "SOURCE_",
                "SCORE_SOURCE_",
                "SCORE_COMPONENT_",
                "CACHE_MANIFEST_",
                "RAW_TRANSFORMED_GEOMETRY_",
                "RAW_TRANSFORMED_SOURCE_",
                "RAW_TRANSFORMED_KNN_",
            )
        ),
    }
    return {
        "pass": not hard_reasons,
        "criteria": criteria,
        "reason_codes": sorted(hard_reasons),
    }


def _validate_diagnostic_formula_record(value: Any, *, label: str) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    _require_fields(
        record,
        (
            "status",
            "reason_codes",
            "max_abs_error",
            "max_rel_error",
            "reference_array_sha256",
            "reconstructed_array_sha256",
            "absolute_tolerance",
            "relative_tolerance",
        ),
        label,
    )
    if record["status"] != "PASS" or _require_reason_codes(
        record["reason_codes"], f"{label} reasons"
    ):
        raise ValueError(f"{label} formula reconstruction failed")
    absolute = _finite_float(record["max_abs_error"], f"{label} absolute error")
    relative = _finite_float(record["max_rel_error"], f"{label} relative error")
    if absolute < 0.0 or relative < 0.0:
        raise ValueError(f"{label} formula errors must be nonnegative")
    _require_hex(record["reference_array_sha256"], 64, f"{label} reference SHA256")
    _require_hex(
        record["reconstructed_array_sha256"],
        64,
        f"{label} reconstruction SHA256",
    )
    if (
        _finite_float(record["absolute_tolerance"], f"{label} absolute tolerance")
        != 1.0e-12
        or _finite_float(
            record["relative_tolerance"], f"{label} relative tolerance"
        )
        != 1.0e-10
    ):
        raise ValueError(f"{label} frozen formula tolerance mismatch")
    return record


def _validate_full_array_parity_record(
    value: Any, *, label: str, expected_count: int
) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    _require_fields(
        record,
        (
            "status",
            "count",
            "max_abs_error",
            "max_rel_error",
            "observed_array_sha256",
            "stored_array_sha256",
            "exact_array_sha256_equal",
            "absolute_tolerance",
            "relative_tolerance",
        ),
        label,
    )
    if record["status"] != "PASS":
        raise ValueError(f"{label} full-array parity failed")
    if _require_int(record["count"], f"{label} count") != expected_count:
        raise ValueError(f"{label} full-array parity count mismatch")
    absolute = _finite_float(record["max_abs_error"], f"{label} absolute error")
    relative = _finite_float(record["max_rel_error"], f"{label} relative error")
    if absolute < 0.0 or relative < 0.0:
        raise ValueError(f"{label} parity errors must be nonnegative")
    _require_hex(record["observed_array_sha256"], 64, f"{label} observed SHA256")
    _require_hex(record["stored_array_sha256"], 64, f"{label} stored SHA256")
    if not isinstance(record["exact_array_sha256_equal"], bool):
        raise ValueError(f"{label} exact-SHA diagnostic must be boolean")
    # The worker's PASS is the authoritative elementwise allclose result.
    # Independent thresholds on the two maxima are not equivalent to the
    # frozen np.allclose(stored, observed) per-element condition, whose
    # relative term is evaluated against abs(observed).
    if (
        _finite_float(record["absolute_tolerance"], f"{label} absolute tolerance")
        != 1.0e-12
        or _finite_float(
            record["relative_tolerance"], f"{label} relative tolerance"
        )
        != 1.0e-10
    ):
        raise ValueError(f"{label} frozen parity tolerance mismatch")
    return record


def _validate_diagnostic_component_array(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...],
    nonnegative: bool = False,
) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    _require_fields(record, ("shape", "dtype", "array_sha256", "distribution"), label)
    if _require_shape(record["shape"], f"{label} shape") != expected_shape:
        raise ValueError(f"{label} shape mismatch")
    if record["dtype"] != "float64":
        raise ValueError(f"{label} dtype mismatch")
    array_sha = _require_hex(record["array_sha256"], 64, f"{label} array SHA256")
    count = math.prod(expected_shape)
    distribution = _validate_distribution_summary(
        record["distribution"],
        label=f"{label} distribution",
        expected_count=count,
        nonnegative=nonnegative,
    )
    if distribution["array_sha256"] != array_sha:
        raise ValueError(f"{label} component/distribution digest mismatch")
    return record


def _validate_diagnostic_common_source(
    value: Any,
    *,
    label: str,
    expected_kind: str,
    expected_path: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    checkpoint: Mapping[str, Any],
    allowed_files: Mapping[str, Mapping[str, Any]],
    analysis_id: str,
) -> Mapping[str, Any]:
    reference = _validate_source_reference(
        value,
        label=label,
        expected_kind=expected_kind,
        expected_bundle_id=str(checkpoint["checkpoint_sha256"]),
        expected_allowlist_sha256=str(checkpoint["source_allowlist_sha256"]),
        expected_shape=expected_shape,
        allowed_files=allowed_files,
        expected_logical_path=expected_path,
        analysis_id=analysis_id,
    )
    if reference["dtype"] != expected_dtype:
        raise ValueError(f"{label} dtype mismatch")
    return reference


def _validate_classifier_source_reference(
    value: Any,
    *,
    label: str,
    array_name: str,
    expected_shape: tuple[int, ...],
    checkpoint: Mapping[str, Any],
    allowed_files: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    reference = _require_mapping(value, label)
    required = (
        "source_kind",
        "logical_path",
        "file_sha256",
        "file_size",
        "analysis_array_sha256",
        "analysis_shape",
        "analysis_dtype",
        "bundle_id",
        "source_allowlist_sha256",
    )
    _require_fields(reference, required, label)
    expected_path = f"classifier_{array_name}.npy"
    if (
        reference["source_kind"] != f"selective_reuse_classifier_{array_name}_array"
        or _safe_logical_path(reference["logical_path"], f"{label} path")
        != expected_path
        or _require_shape(reference["analysis_shape"], f"{label} shape")
        != expected_shape
        or reference["analysis_dtype"] != "float64"
        or reference["bundle_id"] != checkpoint["checkpoint_sha256"]
        or reference["source_allowlist_sha256"]
        != checkpoint["source_allowlist_sha256"]
    ):
        raise ValueError(f"{label} classifier source identity mismatch")
    _require_hex(reference["analysis_array_sha256"], 64, f"{label} array SHA256")
    file_sha = _require_hex(reference["file_sha256"], 64, f"{label} file SHA256")
    file_size = _require_int(reference["file_size"], f"{label} file size")
    allowed = _require_mapping(allowed_files.get(expected_path), f"{label} allowlist")
    if file_sha != allowed.get("sha256") or file_size != allowed.get("size"):
        raise ValueError(f"{label} allowlist checksum/size mismatch")
    return reference


def _validate_derived_pure_score_reference(
    value: Any,
    *,
    label: str,
    expected_count: int,
    residual_source: Mapping[str, Any],
    score_array_sha256: str,
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    reference = _require_mapping(value, label)
    required = (
        "source_kind",
        "derivation",
        "component_logical_path",
        "component_file_sha256",
        "component_array_sha256",
        "array_sha256",
        "shape",
        "dtype",
        "bundle_id",
        "source_allowlist_sha256",
    )
    _require_fields(reference, required, label)
    if (
        reference["source_kind"]
        != "derived_negative_of_selective_reuse_diagnostic_array"
        or reference["derivation"] != "score_equals_negative_stored_vim_residual"
        or reference["component_logical_path"] != residual_source["logical_path"]
        or reference["component_file_sha256"] != residual_source["file_sha256"]
        or reference["component_array_sha256"] != residual_source["array_sha256"]
        or _require_hex(reference["array_sha256"], 64, f"{label} array SHA256")
        != score_array_sha256
        or _require_shape(reference["shape"], f"{label} shape")
        != (expected_count,)
        or reference["dtype"] != "float64"
        or reference["bundle_id"] != checkpoint["checkpoint_sha256"]
        or reference["source_allowlist_sha256"]
        != checkpoint["source_allowlist_sha256"]
    ):
        raise ValueError(f"{label} derived-score source binding mismatch")
    return reference


def _validate_diagnostic_parity_group(
    value: Any,
    *,
    label: str,
    num_id: int,
    num_ood: int,
) -> Mapping[str, Mapping[str, Any]]:
    group = _require_mapping(value, label)
    _require_fields(group, ("status", "reason_codes", "id_test", "ood"), label)
    if group["status"] != "PASS" or _require_reason_codes(
        group["reason_codes"], f"{label} reasons"
    ):
        raise ValueError(f"{label} parity group failed")
    return {
        "id_test": _validate_full_array_parity_record(
            group["id_test"], label=f"{label} ID-test", expected_count=num_id
        ),
        "ood": _validate_full_array_parity_record(
            group["ood"], label=f"{label} OOD", expected_count=num_ood
        ),
    }


def _validate_ctm_normalization_status(
    value: Any,
    *,
    label: str,
    query_count: int,
    class_count: int,
) -> Mapping[str, Any]:
    record = _require_mapping(value, label)
    if set(record) != {"status", "reason_codes", "normalization_diagnostics"}:
        raise ValueError(f"{label} normalization-status population mismatch")
    if record["status"] != "success" or _require_reason_codes(
        record["reason_codes"], f"{label} reasons"
    ):
        raise ValueError(f"{label} CTM normalization is not successful")
    diagnostics = _require_mapping(
        record["normalization_diagnostics"], f"{label} diagnostics"
    )
    if set(diagnostics) != {"query", "class_mean"}:
        raise ValueError(f"{label} normalization diagnostic population mismatch")
    for name, expected_count in (("query", query_count), ("class_mean", class_count)):
        item = _require_mapping(diagnostics[name], f"{label} {name}")
        if set(item) != {"epsilon_norm", "exact_zero_count", "near_zero_count"}:
            raise ValueError(f"{label} {name} normalization fields mismatch")
        if (
            _finite_float(item["epsilon_norm"], f"{label} {name} epsilon")
            != EPSILON_NORM
            or _require_int(
                item["exact_zero_count"], f"{label} {name} exact-zero count"
            )
            != 0
            or _require_int(
                item["near_zero_count"], f"{label} {name} near-zero count"
            )
            != 0
        ):
            raise ValueError(f"{label} {name} normalization degeneracy mismatch")
        # The worker does not duplicate sample counts inside norm diagnostics;
        # the expected count is nevertheless explicit at the call site and is
        # bound by the enclosing component shape/population validation.
        if expected_count <= 0:
            raise ValueError(f"{label} {name} population must be positive")
    return record


def _validate_cosine_distribution(
    value: Any, *, label: str, expected_count: int
) -> Mapping[str, Any]:
    distribution = _validate_distribution_summary(
        value, label=label, expected_count=expected_count
    )
    if (
        float(distribution["minimum"]) < -1.0 - 1.0e-12
        or float(distribution["maximum"]) > 1.0 + 1.0e-12
    ):
        raise ValueError(f"{label} must lie in the cosine range [-1, 1]")
    return distribution


def _validate_pure_residual_components(
    value: Any,
    *,
    label: str,
    num_id: int,
    num_ood: int,
    score_distribution: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    components = _require_mapping(value, label)
    if set(components) != {"fit", "id_test", "ood"}:
        raise ValueError(f"{label} Pure Residual component population mismatch")
    fit = _require_mapping(components["fit"], f"{label} fit")
    _require_fields(
        fit,
        (
            "feature_dim",
            "author_dim",
            "residual_subspace_dim",
            "author_dim_rule",
            "origin",
            "covariance_eigenvalues",
            "residual_subspace",
            "alpha",
            "residual_mean",
        ),
        f"{label} fit",
    )
    if (
        _require_int(fit["feature_dim"], f"{label} feature dim") != 640
        or _require_int(fit["author_dim"], f"{label} author dim") != 320
        or _require_int(
            fit["residual_subspace_dim"], f"{label} residual subspace dim"
        )
        != 320
        or fit["author_dim_rule"] != "floor(feature_dim/2)"
    ):
        raise ValueError(f"{label} Pure Residual author-DIM contract mismatch")
    _validate_diagnostic_component_array(
        fit["origin"], label=f"{label} origin", expected_shape=(640,)
    )
    _validate_diagnostic_component_array(
        fit["covariance_eigenvalues"],
        label=f"{label} covariance eigenvalues",
        expected_shape=(640,),
    )
    _validate_diagnostic_component_array(
        fit["residual_subspace"],
        label=f"{label} residual subspace",
        expected_shape=(640, 320),
    )
    _finite_float(fit["alpha"], f"{label} alpha")
    _finite_float(fit["residual_mean"], f"{label} residual mean")
    formula_records: list[Mapping[str, Any]] = []
    for split_name, count in (("id_test", num_id), ("ood", num_ood)):
        split = _require_mapping(components[split_name], f"{label} {split_name}")
        expected_fields = {
            "residual",
            "pure_residual_score",
            "vim_energy",
            "vim_alpha_residual",
            "vim_full_score",
            "pure_residual_formula",
            "full_vim_formula",
        }
        if set(split) != expected_fields:
            raise ValueError(f"{label} {split_name} component population mismatch")
        residual = _validate_diagnostic_component_array(
            split["residual"],
            label=f"{label} {split_name} residual",
            expected_shape=(count,),
            nonnegative=True,
        )
        score = _validate_diagnostic_component_array(
            split["pure_residual_score"],
            label=f"{label} {split_name} pure score",
            expected_shape=(count,),
        )
        for field in ("vim_energy", "vim_alpha_residual", "vim_full_score"):
            _validate_diagnostic_component_array(
                split[field],
                label=f"{label} {split_name} {field}",
                expected_shape=(count,),
            )
        _validate_negative_distance_distribution(
            score["distribution"],
            residual["distribution"],
            label=f"{label} {split_name} negative residual",
        )
        if score["array_sha256"] != score_distribution[split_name]["array_sha256"]:
            raise ValueError(f"{label} {split_name} score/component digest mismatch")
        formula_records.extend(
            (
                _validate_diagnostic_formula_record(
                    split["pure_residual_formula"],
                    label=f"{label} {split_name} Pure Residual formula",
                ),
                _validate_diagnostic_formula_record(
                    split["full_vim_formula"],
                    label=f"{label} {split_name} full ViM formula",
                ),
            )
        )
    return components, formula_records


def _validate_ctm_components(
    value: Any,
    *,
    label: str,
    num_id: int,
    num_ood: int,
    id_train_count: int,
    score_distribution: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    components = _require_mapping(value, label)
    if set(components) != {"fit", "id_test", "ood"}:
        raise ValueError(f"{label} CTM component population mismatch")
    fit = _require_mapping(components["fit"], f"{label} fit")
    _require_fields(
        fit,
        (
            "class_means",
            "class_counts",
            "normalization_status",
            "classwise_id_train_angular_concentration",
        ),
        f"{label} fit",
    )
    _validate_diagnostic_component_array(
        fit["class_means"],
        label=f"{label} class means",
        expected_shape=(10, 640),
    )
    class_counts = _require_list(fit["class_counts"], f"{label} class counts")
    if len(class_counts) != 10:
        raise ValueError(f"{label} class-count population mismatch")
    parsed_counts = [
        _require_int(count, f"{label} class count") for count in class_counts
    ]
    if any(count <= 0 for count in parsed_counts) or sum(parsed_counts) != id_train_count:
        raise ValueError(f"{label} class counts do not cover ID-train")
    _validate_ctm_normalization_status(
        fit["normalization_status"],
        label=f"{label} fit normalization",
        query_count=id_train_count,
        class_count=10,
    )
    concentration = _require_mapping(
        fit["classwise_id_train_angular_concentration"],
        f"{label} angular concentration",
    )
    _require_fields(concentration, ("definition", "all_id_train", "classes"), f"{label} concentration")
    if concentration["definition"] != (
        "cosine_of_each_raw_id_train_feature_to_its_own_raw_before_average_class_mean"
    ):
        raise ValueError(f"{label} angular concentration definition mismatch")
    _validate_cosine_distribution(
        concentration["all_id_train"],
        label=f"{label} all ID-train concentration",
        expected_count=id_train_count,
    )
    classes = _require_list(concentration["classes"], f"{label} class concentrations")
    if len(classes) != 10:
        raise ValueError(f"{label} class concentration population mismatch")
    for class_index, item in enumerate(classes):
        row = _require_mapping(item, f"{label} class concentration {class_index}")
        _require_fields(row, ("class_index", "sample_count", "own_class_cosine"), f"{label} class concentration")
        if (
            _require_int(row["class_index"], f"{label} class index") != class_index
            or _require_int(row["sample_count"], f"{label} class sample count")
            != parsed_counts[class_index]
        ):
            raise ValueError(f"{label} class concentration identity mismatch")
        _validate_cosine_distribution(
            row["own_class_cosine"],
            label=f"{label} class {class_index} concentration",
            expected_count=parsed_counts[class_index],
        )
    formula_records: list[Mapping[str, Any]] = []
    for split_name, count in (("id_test", num_id), ("ood", num_ood)):
        split = _require_mapping(components[split_name], f"{label} {split_name}")
        if set(split) != {
            "matched_cosine",
            "matched_class",
            "prototype_cosine_matrix",
            "normalization_status",
            "ctm_formula",
        }:
            raise ValueError(f"{label} {split_name} CTM component population mismatch")
        matched = _validate_diagnostic_component_array(
            split["matched_cosine"],
            label=f"{label} {split_name} matched cosine",
            expected_shape=(count,),
        )
        _validate_cosine_distribution(
            matched["distribution"],
            label=f"{label} {split_name} matched cosine distribution",
            expected_count=count,
        )
        if matched["array_sha256"] != score_distribution[split_name]["array_sha256"]:
            raise ValueError(f"{label} {split_name} score/component digest mismatch")
        matched_class = _require_mapping(
            split["matched_class"], f"{label} {split_name} matched class"
        )
        _require_fields(
            matched_class,
            ("count", "dtype", "array_sha256", "class_counts"),
            f"{label} matched class",
        )
        if (
            _require_int(matched_class["count"], f"{label} matched count") != count
            or matched_class["dtype"] != "int64"
        ):
            raise ValueError(f"{label} matched-class population mismatch")
        _require_hex(matched_class["array_sha256"], 64, f"{label} matched-class SHA256")
        matched_counts = _require_list(
            matched_class["class_counts"], f"{label} matched class counts"
        )
        if len(matched_counts) != 10 or sum(
            _require_int(item, f"{label} matched class count")
            for item in matched_counts
        ) != count:
            raise ValueError(f"{label} matched-class counts mismatch")
        matrix = _validate_diagnostic_component_array(
            split["prototype_cosine_matrix"],
            label=f"{label} {split_name} prototype matrix",
            expected_shape=(count, 10),
        )
        _validate_cosine_distribution(
            matrix["distribution"],
            label=f"{label} {split_name} prototype cosine distribution",
            expected_count=count * 10,
        )
        _validate_ctm_normalization_status(
            split["normalization_status"],
            label=f"{label} {split_name} normalization",
            query_count=count,
            class_count=10,
        )
        formula_records.append(
            _validate_diagnostic_formula_record(
                split["ctm_formula"], label=f"{label} {split_name} CTM formula"
            )
        )
    return components, formula_records


def _validate_diagnostic_scalar_metrics(
    value: Any, *, row: Mapping[str, Any], label: str
) -> Mapping[str, float]:
    metrics = _require_mapping(value, label)
    if set(metrics) != set(SCALAR_METRIC_FIELDS):
        raise ValueError(f"{label} metric population mismatch")
    parsed: dict[str, float] = {}
    for field in SCALAR_METRIC_FIELDS:
        observed = _finite_float(metrics[field], f"{label} {field}")
        if observed != float(row[field]):
            raise ValueError(f"{label} metric/row mismatch: {field}")
        parsed[field] = observed
    return parsed


def _validate_ctm_scalar_parity(
    value: Any,
    *,
    row: Mapping[str, Any],
    label: str,
    authoritative_baseline: Mapping[str, float] | None,
) -> None:
    scalar = _require_mapping(value, label)
    if set(scalar) != {
        "status",
        "reason_codes",
        "metrics",
        "baseline_metrics",
        "absolute_tolerance",
        "relative_tolerance",
    }:
        raise ValueError(f"{label} CTM scalar-parity schema mismatch")
    if scalar["status"] != "PASS" or _require_reason_codes(
        scalar["reason_codes"], f"{label} reasons"
    ):
        raise ValueError(f"{label} CTM scalar parity failed")
    metrics = _validate_diagnostic_scalar_metrics(
        scalar["metrics"], row=row, label=f"{label} metrics"
    )
    baseline_raw = _require_mapping(
        scalar["baseline_metrics"], f"{label} baseline metrics"
    )
    if set(baseline_raw) != set(SCALAR_METRIC_FIELDS):
        raise ValueError(f"{label} baseline metric population mismatch")
    baseline = {
        field: _finite_float(
            baseline_raw[field], f"{label} baseline metric {field}"
        )
        for field in SCALAR_METRIC_FIELDS
    }
    absolute_tolerance = _finite_float(
        scalar["absolute_tolerance"], f"{label} absolute tolerance"
    )
    relative_tolerance = _finite_float(
        scalar["relative_tolerance"], f"{label} relative tolerance"
    )
    if absolute_tolerance != 1.0e-12 or relative_tolerance != 1.0e-10:
        raise ValueError(f"{label} frozen scalar tolerance mismatch")
    failed = [
        field
        for field in SCALAR_METRIC_FIELDS
        if abs(metrics[field] - baseline[field])
        > absolute_tolerance + relative_tolerance * abs(baseline[field])
    ]
    if failed:
        raise ValueError(f"{label} CTM scalar parity mismatch: {sorted(failed)}")
    if authoritative_baseline is not None:
        if set(authoritative_baseline) != set(SCALAR_METRIC_FIELDS) or any(
            baseline[field] != float(authoritative_baseline[field])
            for field in SCALAR_METRIC_FIELDS
        ):
            raise ValueError(
                f"{label} embedded baseline differs from authoritative v1.2 baseline"
            )


def _validate_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_id: str,
    checkpoints: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authoritative_ctm_baseline: Mapping[
        tuple[str, str], Mapping[str, float]
    ]
    | None,
) -> None:
    expected_diagnostics = {
        "pure_residual_channel": "detector/vim_residual_author_dim",
        "ctm_angular_channel": "detector/ctm_prototype_cosine",
    }
    diagnostic_specs = {
        str(item["diagnostic_id"]): _require_mapping(item, "diagnostic spec")
        for item in _require_list(registry.get("diagnostic_only"), "diagnostic-only specs")
    }
    if not set(expected_diagnostics).issubset(diagnostic_specs):
        raise ValueError("required Stage-2 diagnostic registry population mismatch")
    for diagnostic_id, detector in expected_diagnostics.items():
        spec = diagnostic_specs[diagnostic_id]
        if (
            spec.get("selection_status") != "diagnostic_only"
            or spec.get("targeted_transform") is not None
            or spec.get("detector") != detector
        ):
            raise ValueError(f"diagnostic registry contract mismatch: {diagnostic_id}")
    boundary = _require_mapping(registry.get("selection_boundary"), "selection boundary")
    if boundary.get("diagnostic_can_enter_selection") is not False:
        raise ValueError("diagnostic-only rows are not excluded from candidate selection")

    partition = _require_mapping(policy["ood_partition"], "OOD partition")
    datasets = list(partition["dataset_order"])
    groups = {
        dataset: "near" if dataset in partition["near"] else "far"
        for dataset in datasets
    }
    inventory = _load_json(DEFAULT_CHECKPOINT_INVENTORY, "checkpoint inventory")
    dataset_policy = _require_mapping(inventory["dataset_policy"], "dataset policy")
    id_splits = _require_mapping(dataset_policy["id_splits"], "ID splits")
    ood_splits = _require_mapping(dataset_policy["ood_splits"], "OOD splits")
    num_id = int(_require_mapping(id_splits["id_test"], "ID-test split")["count"])
    id_train_count = int(_require_mapping(id_splits["id_train"], "ID-train split")["count"])
    num_ood = {
        dataset: int(_require_mapping(ood_splits[dataset], f"OOD split {dataset}")["count"])
        for dataset in datasets
    }
    expected_count = len(checkpoints) * len(datasets) * len(expected_diagnostics)
    if len(rows) != expected_count:
        raise ValueError(f"diagnostic metric population must contain exactly {expected_count} rows")
    checkpoint_by_sha = {str(item["checkpoint_sha256"]): item for item in checkpoints}
    reuse = _load_json(DEFAULT_REUSE_MANIFEST, "Stage-2 reuse manifest")
    reuse_by_id = {
        str(item["bundle_id"]): _require_mapping(item, "reuse bundle")
        for item in _require_list(reuse["bundles"], "reuse bundles")
    }
    runtime_sha = _require_hex(
        _require_mapping(
            manifest["numeric_runtime_fingerprint"], "numeric runtime fingerprint"
        )["canonical_lock_sha256"],
        64,
        "numeric runtime fingerprint SHA256",
    )
    required = (
        "schema_version",
        "analysis_id",
        "diagnostic_id",
        "selection_status",
        "candidate_eligibility",
        "ranking_use",
        "config_hash",
        "optimizer_family",
        "role_labels",
        "training_seed",
        "checkpoint_sha256",
        "checkpoint_role",
        "ood_dataset",
        "ood_group",
        "detector",
        "num_id",
        "num_ood",
        "auroc",
        "fpr95",
        "fpr95_threshold",
        "achieved_id_tpr",
        "score_sha256",
        "score_distribution",
        "formula_reconstruction_max_abs_error",
        "formula_reconstruction_max_rel_error",
        "formula_reconstruction_pass",
        "stored_artifact_parity",
        "scalar_parity",
        "score_components",
        "source_references",
        "source_allowlist_sha256",
        "numeric_runtime_fingerprint_sha256",
        "status",
        "reason_codes",
    )
    seen: set[tuple[str, str, str]] = set()
    duplicate_evidence: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        label = f"diagnostic metric row {index}"
        _require_fields(row, required, label)
        if row["schema_version"] != DIAGNOSTIC_METRIC_SCHEMA or row["analysis_id"] != analysis_id:
            raise ValueError(f"{label} schema/analysis identity mismatch")
        diagnostic_id = str(row["diagnostic_id"])
        if diagnostic_id not in expected_diagnostics:
            raise ValueError(f"{label} diagnostic identity mismatch")
        if (
            row["selection_status"] != "diagnostic_only"
            or row["candidate_eligibility"] is not False
            or row["ranking_use"] != "forbidden"
            or row["detector"] != expected_diagnostics[diagnostic_id]
        ):
            raise ValueError(f"{label} diagnostic-selection isolation failed")
        checkpoint_sha = _require_hex(
            row["checkpoint_sha256"], 64, f"{label} checkpoint SHA256"
        )
        if checkpoint_sha not in checkpoint_by_sha:
            raise ValueError(f"{label} checkpoint identity mismatch")
        checkpoint = checkpoint_by_sha[checkpoint_sha]
        for field in (
            "config_hash",
            "optimizer_family",
            "role_labels",
            "training_seed",
            "checkpoint_role",
        ):
            if row[field] != checkpoint[field]:
                raise ValueError(f"{label} checkpoint binding mismatch: {field}")
        dataset = str(row["ood_dataset"])
        if dataset not in groups or row["ood_group"] != groups[dataset]:
            raise ValueError(f"{label} OOD dataset/group mismatch")
        identity = (diagnostic_id, checkpoint_sha, dataset)
        if identity in seen:
            raise ValueError(f"duplicate diagnostic metric identity: {identity}")
        seen.add(identity)
        if (
            _require_int(row["num_id"], f"{label} num_id") != num_id
            or _require_int(row["num_ood"], f"{label} num_ood") != num_ood[dataset]
        ):
            raise ValueError(f"{label} sample population mismatch")
        if row["source_allowlist_sha256"] != checkpoint["source_allowlist_sha256"]:
            raise ValueError(f"{label} source allowlist mismatch")
        if row["numeric_runtime_fingerprint_sha256"] != runtime_sha:
            raise ValueError(f"{label} numeric runtime fingerprint mismatch")
        _unit_float(row["auroc"], f"{label} AUROC")
        _unit_float(row["fpr95"], f"{label} FPR95")
        _finite_float(row["fpr95_threshold"], f"{label} threshold")
        _unit_float(row["achieved_id_tpr"], f"{label} achieved ID TPR")
        _require_hex(row["score_sha256"], 64, f"{label} score SHA256")
        score_distribution = _validate_score_distribution(
            row["score_distribution"],
            label=f"{label} score distribution",
            num_id=num_id,
            num_ood=num_ood[dataset],
        )
        if row["formula_reconstruction_pass"] is not True:
            raise ValueError(f"{label} formula reconstruction failed")
        formula_abs = _finite_float(
            row["formula_reconstruction_max_abs_error"], f"{label} formula abs"
        )
        formula_rel = _finite_float(
            row["formula_reconstruction_max_rel_error"], f"{label} formula rel"
        )
        if formula_abs < 0.0 or formula_rel < 0.0:
            raise ValueError(f"{label} formula errors must be nonnegative")
        if row["status"] != "PASS" or _require_reason_codes(
            row["reason_codes"], f"{label} reasons"
        ):
            raise ValueError(f"{label} diagnostic status failed")

        parity = _require_mapping(row["stored_artifact_parity"], f"{label} stored parity")
        _require_fields(parity, ("status", "reason_codes"), f"{label} stored parity")
        if parity["status"] != "PASS" or _require_reason_codes(
            parity["reason_codes"], f"{label} stored parity reasons"
        ):
            raise ValueError(f"{label} stored artifact parity failed")
        scalar = _require_mapping(row["scalar_parity"], f"{label} scalar parity")

        bundle = reuse_by_id[checkpoint_sha]
        allowed_files = {
            str(item["logical_path"]): _require_mapping(item, "allowlist file")
            for item in _require_list(
                _require_mapping(bundle["remote_integrity"], "remote integrity")["files"],
                "allowlist files",
            )
        }
        sources = _require_mapping(row["source_references"], f"{label} sources")
        if sources.get("source_allowlist_sha256") != checkpoint["source_allowlist_sha256"]:
            raise ValueError(f"{label} source-reference allowlist mismatch")
        feature_sources = _require_mapping(sources.get("feature_arrays"), f"{label} feature sources")
        if set(feature_sources) != {"id_train", "id_test", "ood"}:
            raise ValueError(f"{label} feature source population mismatch")
        feature_paths = {
            "id_train": "cifar10/train/features.npy",
            "id_test": "cifar10/test/features.npy",
            "ood": f"{dataset}/test/features.npy",
        }
        feature_counts = {"id_train": id_train_count, "id_test": num_id, "ood": num_ood[dataset]}
        for split_name, source in feature_sources.items():
            validated = _validate_diagnostic_common_source(
                source,
                label=f"{label} feature {split_name}",
                expected_kind="selective_reuse_feature_array",
                expected_path=feature_paths[split_name],
                expected_shape=(feature_counts[split_name], 640),
                expected_dtype="float32",
                checkpoint=checkpoint,
                allowed_files=allowed_files,
                analysis_id=analysis_id,
            )
            duplicate_key = (checkpoint_sha, f"feature:{split_name if split_name != 'ood' else dataset}", "source")
            if duplicate_evidence.setdefault(duplicate_key, validated) != validated:
                raise ValueError(f"{label} duplicate feature source mismatch")

        if diagnostic_id == "pure_residual_channel":
            if set(scalar) != {"status", "reason_codes", "metrics"}:
                raise ValueError(f"{label} Pure Residual scalar-parity schema mismatch")
            _validate_diagnostic_scalar_metrics(
                scalar["metrics"], row=row, label=f"{label} scalar metrics"
            )
            channel_contract = _require_mapping(
                row.get("channel_contract"), f"{label} channel contract"
            )
            if channel_contract != {
                "pure_residual_uses_energy": False,
                "pure_residual_uses_alpha": False,
                "score_orientation": "negative_residual_id_like",
                "author_dim_rule": "floor(feature_dim/2)",
            }:
                raise ValueError(f"{label} Pure Residual channel contract mismatch")
            if scalar["status"] != "NOT_APPLICABLE" or _require_reason_codes(
                scalar["reason_codes"], f"{label} scalar reasons"
            ) != ["V2_ONLY_PURE_RESIDUAL_HAS_NO_V1_2_SCALAR_BASELINE"]:
                raise ValueError(f"{label} Pure Residual scalar-parity boundary mismatch")
            expected_source_keys = {
                "source_allowlist_sha256",
                "feature_arrays",
                "logit_arrays",
                "classifier_arrays",
                "stored_full_vim_score_arrays",
                "stored_vim_residual_arrays",
                "derived_score_arrays",
            }
            if set(sources) != expected_source_keys:
                raise ValueError(f"{label} Pure Residual source population mismatch")
            components, formulas = _validate_pure_residual_components(
                row["score_components"],
                label=f"{label} components",
                num_id=num_id,
                num_ood=num_ood[dataset],
                score_distribution=score_distribution,
            )
            logits = _require_mapping(sources["logit_arrays"], f"{label} logits")
            if set(logits) != {"id_train", "id_test", "ood"}:
                raise ValueError(f"{label} logit source population mismatch")
            logit_paths = {
                "id_train": "cifar10/train/logits.npy",
                "id_test": "cifar10/test/logits.npy",
                "ood": f"{dataset}/test/logits.npy",
            }
            for split_name, source in logits.items():
                _validate_diagnostic_common_source(
                    source,
                    label=f"{label} logit {split_name}",
                    expected_kind="selective_reuse_logit_array",
                    expected_path=logit_paths[split_name],
                    expected_shape=(feature_counts[split_name], 10),
                    expected_dtype="float32",
                    checkpoint=checkpoint,
                    allowed_files=allowed_files,
                    analysis_id=analysis_id,
                )
            classifier = _require_mapping(sources["classifier_arrays"], f"{label} classifier")
            if set(classifier) != {"weight", "bias"}:
                raise ValueError(f"{label} classifier source population mismatch")
            _validate_classifier_source_reference(
                classifier["weight"], label=f"{label} classifier weight", array_name="weight", expected_shape=(10, 640), checkpoint=checkpoint, allowed_files=allowed_files
            )
            _validate_classifier_source_reference(
                classifier["bias"], label=f"{label} classifier bias", array_name="bias", expected_shape=(10,), checkpoint=checkpoint, allowed_files=allowed_files
            )
            full_sources = _require_mapping(sources["stored_full_vim_score_arrays"], f"{label} full ViM sources")
            residual_sources = _require_mapping(sources["stored_vim_residual_arrays"], f"{label} residual sources")
            derived_sources = _require_mapping(sources["derived_score_arrays"], f"{label} derived sources")
            if any(set(group) != {"id_test", "ood"} for group in (full_sources, residual_sources, derived_sources)):
                raise ValueError(f"{label} stored/derived split population mismatch")
            full_parity = _validate_diagnostic_parity_group(
                parity.get("full_vim_score"), label=f"{label} full ViM parity", num_id=num_id, num_ood=num_ood[dataset]
            )
            residual_parity = _validate_diagnostic_parity_group(
                parity.get("vim_residual"), label=f"{label} residual parity", num_id=num_id, num_ood=num_ood[dataset]
            )
            if set(parity) != {"status", "reason_codes", "full_vim_score", "vim_residual"}:
                raise ValueError(f"{label} Pure Residual parity population mismatch")
            for alias, split_name, count in (("id_test", "id_test", num_id), ("ood", dataset, num_ood[dataset])):
                full_source = _validate_diagnostic_common_source(
                    full_sources[alias], label=f"{label} full ViM source {alias}", expected_kind="selective_reuse_score_array", expected_path=_detector_score_logical_path(split_name, "detector/vim_author_dim"), expected_shape=(count,), expected_dtype="float64", checkpoint=checkpoint, allowed_files=allowed_files, analysis_id=analysis_id
                )
                residual_path = f"metrics/arrays/scores/{split_name}/diagnostics/vim/residual.npy"
                residual_source = _validate_diagnostic_common_source(
                    residual_sources[alias], label=f"{label} residual source {alias}", expected_kind="selective_reuse_diagnostic_array", expected_path=residual_path, expected_shape=(count,), expected_dtype="float64", checkpoint=checkpoint, allowed_files=allowed_files, analysis_id=analysis_id
                )
                component_split = components[alias]
                if (
                    component_split["vim_full_score"]["array_sha256"] != full_source["array_sha256"]
                    or component_split["residual"]["array_sha256"] != residual_source["array_sha256"]
                    or full_parity[alias]["observed_array_sha256"] != full_source["array_sha256"]
                    or full_parity[alias]["stored_array_sha256"] != full_source["stored_array_sha256"]
                    or residual_parity[alias]["observed_array_sha256"] != residual_source["array_sha256"]
                    or residual_parity[alias]["stored_array_sha256"] != residual_source["stored_array_sha256"]
                ):
                    raise ValueError(f"{label} Pure Residual component/source/parity mismatch")
                _validate_derived_pure_score_reference(
                    derived_sources[alias], label=f"{label} derived score {alias}", expected_count=count, residual_source=residual_source, score_array_sha256=component_split["pure_residual_score"]["array_sha256"], checkpoint=checkpoint
                )
        else:
            if "channel_contract" in row:
                raise ValueError(f"{label} CTM must not declare a Pure Residual channel contract")
            authoritative = (
                None
                if authoritative_ctm_baseline is None
                else authoritative_ctm_baseline.get((checkpoint_sha, dataset))
            )
            if authoritative_ctm_baseline is not None and authoritative is None:
                raise ValueError(f"{label} authoritative CTM baseline is absent")
            _validate_ctm_scalar_parity(
                scalar,
                row=row,
                label=f"{label} scalar parity",
                authoritative_baseline=authoritative,
            )
            if set(sources) != {
                "source_allowlist_sha256",
                "feature_arrays",
                "class_label_arrays",
                "stored_score_arrays",
            }:
                raise ValueError(f"{label} CTM source population mismatch")
            components, formulas = _validate_ctm_components(
                row["score_components"], label=f"{label} components", num_id=num_id, num_ood=num_ood[dataset], id_train_count=id_train_count, score_distribution=score_distribution
            )
            labels = _require_mapping(sources["class_label_arrays"], f"{label} class labels")
            if set(labels) != {"id_train"}:
                raise ValueError(f"{label} class-label source population mismatch")
            _validate_diagnostic_common_source(
                labels["id_train"], label=f"{label} ID-train labels", expected_kind="selective_reuse_class_label_array", expected_path="cifar10/train/class_labels.npy", expected_shape=(id_train_count,), expected_dtype="int64", checkpoint=checkpoint, allowed_files=allowed_files, analysis_id=analysis_id
            )
            stored_sources = _require_mapping(sources["stored_score_arrays"], f"{label} stored scores")
            if set(stored_sources) != {"id_test", "ood"}:
                raise ValueError(f"{label} stored-score source population mismatch")
            ctm_parity = _validate_diagnostic_parity_group(
                parity.get("ctm_score"), label=f"{label} CTM parity", num_id=num_id, num_ood=num_ood[dataset]
            )
            if set(parity) != {"status", "reason_codes", "ctm_score"}:
                raise ValueError(f"{label} CTM parity population mismatch")
            for alias, split_name, count in (("id_test", "id_test", num_id), ("ood", dataset, num_ood[dataset])):
                source = _validate_diagnostic_common_source(
                    stored_sources[alias], label=f"{label} CTM score source {alias}", expected_kind="selective_reuse_score_array", expected_path=_detector_score_logical_path(split_name, "detector/ctm_prototype_cosine"), expected_shape=(count,), expected_dtype="float64", checkpoint=checkpoint, allowed_files=allowed_files, analysis_id=analysis_id
                )
                if (
                    components[alias]["matched_cosine"]["array_sha256"] != source["array_sha256"]
                    or ctm_parity[alias]["observed_array_sha256"] != source["array_sha256"]
                    or ctm_parity[alias]["stored_array_sha256"] != source["stored_array_sha256"]
                ):
                    raise ValueError(f"{label} CTM component/source/parity mismatch")
        nested_abs = max(float(record["max_abs_error"]) for record in formulas)
        nested_rel = max(float(record["max_rel_error"]) for record in formulas)
        if not _same_float(formula_abs, nested_abs) or not _same_float(formula_rel, nested_rel):
            raise ValueError(f"{label} formula summary/nested evidence mismatch")
        for component_name in ("fit", "id_test"):
            duplicate_key = (checkpoint_sha, diagnostic_id, f"component:{component_name}")
            component = components[component_name]
            if duplicate_evidence.setdefault(duplicate_key, component) != component:
                raise ValueError(f"{label} duplicate diagnostic component mismatch")

    expected_keys = {
        (diagnostic_id, str(checkpoint["checkpoint_sha256"]), dataset)
        for diagnostic_id in expected_diagnostics
        for checkpoint in checkpoints
        for dataset in datasets
    }
    if seen != expected_keys:
        raise ValueError("diagnostic metric identities do not cover the frozen population")


def load_inputs(
    *,
    selection_manifest_path: str | Path,
    model_metrics_path: str | Path,
    diagnostic_metrics_path: str | Path,
    candidate_registry_path: str | Path = DEFAULT_CANDIDATE_REGISTRY,
    gate_policy_path: str | Path = DEFAULT_GATE_POLICY,
    input_checksums_path: str | Path | None = None,
    validate_git_sources: bool = True,
    reducer_git_sha: str | None = None,
    reducer_code_sources: Sequence[Mapping[str, Any]] | None = None,
) -> InputContext:
    selection_path = Path(selection_manifest_path)
    metrics_path = Path(model_metrics_path)
    diagnostics_path = Path(diagnostic_metrics_path)
    registry_path = Path(candidate_registry_path)
    policy_path = Path(gate_policy_path)
    checksums_path = (
        Path(input_checksums_path)
        if input_checksums_path is not None
        else selection_path.parent / "checksums.sha256"
    )
    if validate_git_sources:
        observed_reducer_git_sha, observed_reducer_sources = (
            validate_committed_reducer_sources(registry_path, policy_path)
        )
        if reducer_git_sha is not None and reducer_git_sha != observed_reducer_git_sha:
            raise ValueError("requested reducer Git SHA differs from HEAD")
        reducer_git_sha = observed_reducer_git_sha
        reducer_code_sources = observed_reducer_sources
        source_validation_mode = "committed_head"
    else:
        reducer_git_sha, reducer_code_sources = _validate_test_source_bypass(
            reducer_git_sha, reducer_code_sources
        )
        source_validation_mode = "test_only_explicit_bypass"
    selection_sha, metric_sha, diagnostic_sha = _validate_input_checksums(
        selection_path, metrics_path, diagnostics_path, checksums_path
    )
    registry_sha = _sha256_file(registry_path)
    policy_sha = _sha256_file(policy_path)
    registry = _load_json(registry_path, "candidate registry")
    policy = _load_json(policy_path, "gate policy")
    if policy.get("policy_id") != "fixed_readout_stage2_gate_v1":
        raise ValueError("gate policy identity mismatch")
    authority = _require_mapping(policy.get("authorities"), "gate authorities")
    registry_authority = _require_mapping(
        authority.get("candidate_registry"), "candidate-registry authority"
    )
    if registry_authority.get("sha256") != registry_sha:
        raise ValueError("gate policy candidate-registry authority mismatch")
    candidates = _load_candidates(registry)
    rows = _load_jsonl(metrics_path)
    diagnostic_rows = _load_jsonl(
        diagnostics_path, label="diagnostic metrics"
    )
    manifest = _load_json(selection_path, "selection manifest")
    analysis_id, checkpoints = _validate_manifest(
        manifest,
        model_metrics_path=metrics_path,
        model_metrics_sha256=metric_sha,
        model_metric_row_count=len(rows),
        diagnostic_metrics_path=diagnostics_path,
        diagnostic_metrics_sha256=diagnostic_sha,
        diagnostic_metric_row_count=len(diagnostic_rows),
        candidate_registry_sha256=registry_sha,
        gate_policy_sha256=policy_sha,
        candidates=candidates,
        policy=policy,
    )
    _validate_authoritative_checkpoint_population(manifest, checkpoints, policy)
    evidence_provenance_validation_mode = _validate_evidence_provenance(
        manifest,
        expected_git_sha=str(reducer_git_sha),
        validate_git_sources=validate_git_sources,
    )
    authoritative_ctm_baseline = (
        _load_authoritative_ctm_baseline(
            manifest, checkpoints=checkpoints, policy=policy
        )
        if validate_git_sources
        else None
    )
    ctm_baseline_validation_mode = (
        "authoritative_v1_2_per_checkpoint_records"
        if authoritative_ctm_baseline is not None
        else "test_only_embedded_scalar_parity"
    )
    hard_validity = _validate_rows(
        rows,
        analysis_id=analysis_id,
        checkpoints=checkpoints,
        candidates=candidates,
        policy=policy,
    )
    _validate_diagnostic_rows(
        diagnostic_rows,
        analysis_id=analysis_id,
        checkpoints=checkpoints,
        registry=registry,
        policy=policy,
        manifest=manifest,
        authoritative_ctm_baseline=authoritative_ctm_baseline,
    )
    return InputContext(
        analysis_id=analysis_id,
        selection_manifest_sha256=selection_sha,
        model_metrics_sha256=metric_sha,
        diagnostic_metrics_sha256=diagnostic_sha,
        candidate_registry_sha256=registry_sha,
        gate_policy_sha256=policy_sha,
        reducer_git_sha=str(reducer_git_sha),
        reducer_code_sources=tuple(reducer_code_sources),
        source_validation_mode=source_validation_mode,
        evidence_provenance_validation_mode=evidence_provenance_validation_mode,
        ctm_baseline_validation_mode=ctm_baseline_validation_mode,
        evidence_git_sha=str(manifest["analysis_git_sha"]),
        checkpoints=checkpoints,
        rows=rows,
        diagnostic_rows=diagnostic_rows,
        candidates=candidates,
        policy=policy,
        hard_validity=hard_validity,
    )


def _index_rows(context: InputContext) -> Mapping[tuple[str, str, int, str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, int, str, str], Mapping[str, Any]] = {}
    for row in context.rows:
        key = (
            str(row["candidate_id"]),
            str(row["config_hash"]),
            int(row["training_seed"]),
            str(row["ood_dataset"]),
            str(row["state"]),
        )
        indexed[key] = row
    return indexed


def _scalar_distribution_summary(values: Sequence[float]) -> Mapping[str, Any]:
    finite = [_finite_float(value, "diagnostic summary value") for value in values]
    if not finite:
        raise ValueError("diagnostic scalar summary requires a non-empty population")
    mean = _mean(finite)
    variance = _mean([(value - mean) ** 2 for value in finite])
    return {
        "summary_schema": DISTRIBUTION_SUMMARY_SCHEMA,
        "count": len(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "mean": mean,
        "std_ddof0": math.sqrt(variance),
        "quantiles": {
            key: _linear_quantile(finite, probability)
            for key, probability in FIXED_QUANTILE_PROBABILITIES.items()
        },
        "array_sha256": _sha256_bytes(
            json.dumps(
                finite, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ),
        "array_sha256_encoding": "canonical_json_ordered_float_population",
    }


def _diagnostic_summaries(context: InputContext) -> list[Mapping[str, Any]]:
    partition = _require_mapping(context.policy["ood_partition"], "OOD partition")
    dataset_order = list(partition["dataset_order"])
    group_by_dataset = {
        dataset: "near" if dataset in partition["near"] else "far"
        for dataset in dataset_order
    }
    diagnostic_order = ("ctm_angular_channel", "pure_residual_channel")
    rows: list[Mapping[str, Any]] = []
    for diagnostic_id in diagnostic_order:
        for dataset in dataset_order:
            population = sorted(
                (
                    row
                    for row in context.diagnostic_rows
                    if row["diagnostic_id"] == diagnostic_id
                    and row["ood_dataset"] == dataset
                ),
                key=lambda row: str(row["checkpoint_sha256"]),
            )
            if len(population) != len(context.checkpoints):
                raise ValueError("validated diagnostic summary population mismatch")
            detector_ids = {str(row["detector"]) for row in population}
            if len(detector_ids) != 1:
                raise ValueError("validated diagnostic detector identity mismatch")
            scalar_statuses = sorted({str(row["scalar_parity"]["status"]) for row in population})
            rows.append(
                {
                    "schema_version": DIAGNOSTIC_SUMMARY_SCHEMA,
                    "analysis_id": context.analysis_id,
                    "diagnostic_metrics_sha256": context.diagnostic_metrics_sha256,
                    "diagnostic_id": diagnostic_id,
                    "selection_status": "diagnostic_only",
                    "candidate_eligibility": False,
                    "ranking_use": "forbidden",
                    "detector": next(iter(detector_ids)),
                    "ood_dataset": dataset,
                    "ood_group": group_by_dataset[dataset],
                    "checkpoint_count": len(population),
                    "seed_set": sorted({int(row["training_seed"]) for row in population}),
                    "auroc_distribution": _scalar_distribution_summary(
                        [float(row["auroc"]) for row in population]
                    ),
                    "fpr95_distribution": _scalar_distribution_summary(
                        [float(row["fpr95"]) for row in population]
                    ),
                    "scalar_parity_statuses": scalar_statuses,
                    "formula_and_stored_parity_all_pass": all(
                        row["formula_reconstruction_pass"] is True
                        and row["stored_artifact_parity"]["status"] == "PASS"
                        for row in population
                    ),
                    "status": "PASS",
                    "reason_codes": [],
                }
            )
    return rows


def _config_means(
    indexed: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    *,
    candidate_id: str,
    dataset: str,
    state: str,
    configs: Sequence[str],
    seeds: Sequence[int],
    field: str,
) -> Mapping[str, float]:
    return {
        config_hash: _mean(
            [
                _finite_float(
                    indexed[(candidate_id, config_hash, seed, dataset, state)][field],
                    f"{candidate_id}/{config_hash}/{seed}/{dataset}/{state}/{field}",
                )
                for seed in seeds
            ]
        )
        for config_hash in configs
    }


def _null_config_means(
    indexed: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    *,
    candidate_id: str,
    dataset: str,
    configs: Sequence[str],
    seeds: Sequence[int],
    endpoint: str,
) -> Mapping[str, float]:
    field = "raw_auroc" if endpoint == "raw" else "transformed_auroc"
    return {
        config_hash: _mean(
            [
                _finite_float(
                    _require_mapping(
                        indexed[
                            (candidate_id, config_hash, seed, dataset, "transformed")
                        ]["exact_null_oracle"],
                        "exact null oracle",
                    )[field],
                    f"null {field}",
                )
                for seed in seeds
            ]
        )
        for config_hash in configs
    }


def _null_attenuation_by_dataset(
    indexed: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    *,
    candidate_id: str,
    datasets: Sequence[str],
    configs: Sequence[str],
    seeds: Sequence[int],
) -> Mapping[str, float]:
    output: dict[str, float] = {}
    for dataset in datasets:
        raw_gap = _pairwise_absolute_median(
            _null_config_means(
                indexed,
                candidate_id=candidate_id,
                dataset=dataset,
                configs=configs,
                seeds=seeds,
                endpoint="raw",
            )
        )
        transformed_gap = _pairwise_absolute_median(
            _null_config_means(
                indexed,
                candidate_id=candidate_id,
                dataset=dataset,
                configs=configs,
                seeds=seeds,
                endpoint="transformed",
            )
        )
        output[dataset] = abs(raw_gap - transformed_gap)
    return output


def _dataset_summary(
    context: InputContext,
    indexed: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    *,
    candidate: CandidateSpec,
    dataset: str,
    configs: Sequence[str],
    seeds: Sequence[int],
    null_q90: float,
    frozen_margins: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_means = _config_means(
        indexed,
        candidate_id=candidate.candidate_id,
        dataset=dataset,
        state="raw",
        configs=configs,
        seeds=seeds,
        field="auroc",
    )
    transformed_means = _config_means(
        indexed,
        candidate_id=candidate.candidate_id,
        dataset=dataset,
        state="transformed",
        configs=configs,
        seeds=seeds,
        field="auroc",
    )
    raw_gap = _pairwise_absolute_median(raw_means)
    transformed_gap = _pairwise_absolute_median(transformed_means)
    attenuation = raw_gap - transformed_gap
    relative = attenuation / raw_gap if raw_gap > 0.0 else None
    seed_noise_values = [
        abs(
            float(indexed[(candidate.candidate_id, config_hash, 1, dataset, "raw")]["auroc"])
            - float(
                indexed[(candidate.candidate_id, config_hash, 2, dataset, "raw")][
                    "auroc"
                ]
            )
        )
        for config_hash in configs
    ]
    seed_noise_q90 = _linear_quantile(seed_noise_values, 0.9)
    if frozen_margins is None:
        gap_margin = max(0.01, seed_noise_q90)
        attenuation_margin = max(0.005, 0.25 * raw_gap, null_q90)
    else:
        gap_margin = float(frozen_margins["practical_gap_margin"])
        attenuation_margin = float(frozen_margins["practical_attenuation_margin"])
        seed_noise_q90 = float(frozen_margins["seed_noise_q90"])
    selected_rows = [
        indexed[(candidate.candidate_id, config_hash, seed, dataset, state)]
        for config_hash in configs
        for seed in seeds
        for state in ("raw", "transformed")
    ]
    null_drifts = [
        float(_require_mapping(row["exact_null_oracle"], "null oracle")["auroc_abs_drift"])
        for row in selected_rows
    ]
    null_rank = [
        float(
            _require_mapping(row["exact_null_oracle"], "null oracle")[
                "rank_disagreement_fraction"
            ]
        )
        for row in selected_rows
    ]
    formula_pass = all(row["formula_reconstruction_pass"] is True for row in selected_rows)
    null_pass = all(
        _require_mapping(row["exact_null_oracle"], "null oracle").get("status")
        == "PASS"
        for row in selected_rows
    )
    gap_pass = raw_gap > gap_margin
    attenuation_pass = attenuation > attenuation_margin
    relative_pass = relative is not None and relative >= float(
        context.policy["practical_margins"]["minimum_relative_attenuation"]
    )
    reverse_absent = attenuation > -attenuation_margin
    dataset_pass = bool(
        formula_pass
        and null_pass
        and gap_pass
        and attenuation_pass
        and relative_pass
        and reverse_absent
    )
    ood_group = (
        "near"
        if dataset in context.policy["ood_partition"]["near"]
        else "far"
    )
    fpr_raw = _pairwise_absolute_median(
        _config_means(
            indexed,
            candidate_id=candidate.candidate_id,
            dataset=dataset,
            state="raw",
            configs=configs,
            seeds=seeds,
            field="fpr95",
        )
    )
    fpr_transformed = _pairwise_absolute_median(
        _config_means(
            indexed,
            candidate_id=candidate.candidate_id,
            dataset=dataset,
            state="transformed",
            configs=configs,
            seeds=seeds,
            field="fpr95",
        )
    )
    return {
        "schema_version": DATASET_SUMMARY_SCHEMA,
        "candidate_id": candidate.candidate_id,
        "ood_dataset": dataset,
        "ood_group": ood_group,
        "selection_seed_set": list(seeds),
        "config_count": len(configs),
        "config_pair_count": len(configs) * (len(configs) - 1) // 2,
        "raw_gap": raw_gap,
        "transformed_gap": transformed_gap,
        "absolute_attenuation": attenuation,
        "relative_attenuation": relative,
        "seed_noise_q90": seed_noise_q90,
        "practical_gap_margin": gap_margin,
        "practical_attenuation_margin": attenuation_margin,
        "null_attenuation_q90": null_q90,
        "null_auroc_drift_max": max(null_drifts),
        "null_rank_disagreement_max": max(null_rank),
        "formula_reconstruction_pass": formula_pass,
        "gap_pass": gap_pass,
        "attenuation_pass": attenuation_pass,
        "null_pass": null_pass,
        "relative_attenuation_pass": relative_pass,
        "practical_reverse_absent": reverse_absent,
        "dataset_pass": dataset_pass,
        "selective_score": attenuation / attenuation_margin,
        "fpr95_raw_gap": fpr_raw,
        "fpr95_transformed_gap": fpr_transformed,
        "fpr95_absolute_attenuation": fpr_raw - fpr_transformed,
        "fpr95_secondary_margin": float(
            context.policy["practical_margins"]["fpr95_secondary_absolute_margin"]
        ),
    }


def _summaries_for_scope(
    context: InputContext,
    indexed: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    *,
    configs: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int] = (1, 2),
    frozen_margins: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for candidate in context.candidates:
        null_values = _null_attenuation_by_dataset(
            indexed,
            candidate_id=candidate.candidate_id,
            datasets=datasets,
            configs=configs,
            seeds=(1, 2),
        )
        null_q90 = _linear_quantile(list(null_values.values()), 0.9)
        for dataset in datasets:
            output[(candidate.candidate_id, dataset)] = _dataset_summary(
                context,
                indexed,
                candidate=candidate,
                dataset=dataset,
                configs=configs,
                seeds=seeds,
                null_q90=null_q90,
                frozen_margins=(
                    frozen_margins[(candidate.candidate_id, dataset)]
                    if frozen_margins is not None
                    else None
                ),
            )
    return output


def _rank_candidates(
    context: InputContext,
    summaries: Mapping[tuple[str, str], Mapping[str, Any]],
    datasets: Sequence[str],
) -> Mapping[str, Any]:
    support_policy = context.policy["dataset_support"]
    near = [dataset for dataset in datasets if dataset in context.policy["ood_partition"]["near"]]
    far = [dataset for dataset in datasets if dataset in context.policy["ood_partition"]["far"]]
    trace: list[dict[str, Any]] = []
    for candidate in context.candidates:
        candidate_rows = [summaries[(candidate.candidate_id, dataset)] for dataset in datasets]
        near_rows = [summaries[(candidate.candidate_id, dataset)] for dataset in near]
        far_rows = [summaries[(candidate.candidate_id, dataset)] for dataset in far]
        total_pass = sum(bool(row["dataset_pass"]) for row in candidate_rows)
        near_pass = sum(bool(row["dataset_pass"]) for row in near_rows)
        far_pass = sum(bool(row["dataset_pass"]) for row in far_rows)
        near_median = _median([float(row["selective_score"]) for row in near_rows])
        far_median = _median([float(row["selective_score"]) for row in far_rows])
        reverse_absent = all(bool(row["practical_reverse_absent"]) for row in candidate_rows)
        support_pass = bool(
            total_pass >= int(support_policy["minimum_total_dataset_support"])
            and near_pass >= int(support_policy["minimum_near_dataset_support"])
            and far_pass >= int(support_policy["minimum_far_dataset_support"])
            and near_median > 0.0
            and far_median > 0.0
            and reverse_absent
        )
        trace.append(
            {
                "candidate_id": candidate.candidate_id,
                "priority": candidate.priority,
                "group_balanced_score": min(near_median, far_median),
                "near_median_selective_score": near_median,
                "far_median_selective_score": far_median,
                "dataset_pass_count": total_pass,
                "near_dataset_pass_count": near_pass,
                "far_dataset_pass_count": far_pass,
                "practical_reverse_absent": reverse_absent,
                "full_data_practical_and_support_pass": support_pass,
            }
        )
    ranked = sorted(
        (item for item in trace if item["full_data_practical_and_support_pass"]),
        key=lambda item: (
            -float(item["group_balanced_score"]),
            int(item["priority"]),
            str(item["candidate_id"]),
        ),
    )
    selected = str(ranked[0]["candidate_id"]) if ranked else None
    return {
        "selected_candidate_id": selected,
        "selection_score": (
            float(ranked[0]["group_balanced_score"]) if ranked else None
        ),
        "eligible_candidate_ids": [str(item["candidate_id"]) for item in ranked],
        "tie_break_trace": sorted(
            trace,
            key=lambda item: (
                -float(item["group_balanced_score"]),
                int(item["priority"]),
                str(item["candidate_id"]),
            ),
        ),
    }


def _lodo_folds(
    context: InputContext,
    indexed: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    *,
    configs: Sequence[str],
    datasets: Sequence[str],
    full_selected: str | None,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for heldout in datasets:
        training_datasets = [dataset for dataset in datasets if dataset != heldout]
        training_summaries = _summaries_for_scope(
            context,
            indexed,
            configs=configs,
            datasets=training_datasets,
        )
        ranking = _rank_candidates(context, training_summaries, training_datasets)
        if full_selected is not None:
            selected_spec = next(
                candidate
                for candidate in context.candidates
                if candidate.candidate_id == full_selected
            )
            training_null_q90 = float(
                training_summaries[(full_selected, training_datasets[0])][
                    "null_attenuation_q90"
                ]
            )
            # The held-out dataset supplies its own effect and seed-noise
            # terms, but the fold's null margin is frozen from the five
            # training datasets.  Using ``full_summaries`` here would leak the
            # held-out null into its own threshold.
            heldout_summary = _dataset_summary(
                context,
                indexed,
                candidate=selected_spec,
                dataset=heldout,
                configs=configs,
                seeds=(1, 2),
                null_q90=training_null_q90,
            )
        else:
            training_null_q90 = None
            heldout_summary = None
        heldout_attenuation = (
            float(heldout_summary["absolute_attenuation"])
            if heldout_summary is not None
            else None
        )
        heldout_score = (
            heldout_attenuation / float(heldout_summary["practical_attenuation_margin"])
            if heldout_summary is not None
            else None
        )
        heldout_pass = bool(
            full_selected is not None
            and ranking["selected_candidate_id"] == full_selected
            and heldout_summary is not None
            and heldout_summary["dataset_pass"] is True
        )
        folds.append(
            {
                "schema_version": CV_FOLD_SCHEMA,
                "fold_scheme": "lodo",
                "fold_id": f"lodo:{heldout}",
                "heldout_ood_dataset": heldout,
                "heldout_config_hash": None,
                "training_ood_datasets": training_datasets,
                "training_config_hashes": list(configs),
                "selection_seed_set": [1, 2],
                "eligible_candidate_ids": ranking["eligible_candidate_ids"],
                "selected_candidate_id": ranking["selected_candidate_id"],
                "selection_score": ranking["selection_score"],
                "tie_break_trace": ranking["tie_break_trace"],
                "training_summaries_sha256": _sha256_bytes(
                    _canonical_json_bytes(
                        [
                            training_summaries[key]
                            for key in sorted(training_summaries)
                        ]
                    )
                ),
                "training_null_attenuation_q90": training_null_q90,
                "heldout_absolute_attenuation": heldout_attenuation,
                "heldout_practical_attenuation_margin": (
                    float(heldout_summary["practical_attenuation_margin"])
                    if heldout_summary is not None
                    else None
                ),
                "heldout_selective_score": heldout_score,
                "heldout_pass": heldout_pass,
            }
        )
    return folds


def _heldout_config_attenuation(
    indexed: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    *,
    candidate_id: str,
    dataset: str,
    heldout_config: str,
    training_configs: Sequence[str],
) -> float:
    heldout_raw = _mean(
        [
            float(indexed[(candidate_id, heldout_config, seed, dataset, "raw")]["auroc"])
            for seed in (1, 2)
        ]
    )
    heldout_transformed = _mean(
        [
            float(
                indexed[
                    (candidate_id, heldout_config, seed, dataset, "transformed")
                ]["auroc"]
            )
            for seed in (1, 2)
        ]
    )
    training_raw = _config_means(
        indexed,
        candidate_id=candidate_id,
        dataset=dataset,
        state="raw",
        configs=training_configs,
        seeds=(1, 2),
        field="auroc",
    )
    training_transformed = _config_means(
        indexed,
        candidate_id=candidate_id,
        dataset=dataset,
        state="transformed",
        configs=training_configs,
        seeds=(1, 2),
        field="auroc",
    )
    raw_gap = _median([abs(heldout_raw - value) for value in training_raw.values()])
    transformed_gap = _median(
        [abs(heldout_transformed - value) for value in training_transformed.values()]
    )
    return raw_gap - transformed_gap


def _loco_folds(
    context: InputContext,
    indexed: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    *,
    configs: Sequence[str],
    datasets: Sequence[str],
    full_selected: str | None,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    near = list(context.policy["ood_partition"]["near"])
    far = list(context.policy["ood_partition"]["far"])
    for heldout in configs:
        training_configs = [config_hash for config_hash in configs if config_hash != heldout]
        training_summaries = _summaries_for_scope(
            context,
            indexed,
            configs=training_configs,
            datasets=datasets,
        )
        ranking = _rank_candidates(context, training_summaries, datasets)
        attenuations: dict[str, float] = {}
        ratios: dict[str, float] = {}
        if full_selected is not None:
            for dataset in datasets:
                attenuation = _heldout_config_attenuation(
                    indexed,
                    candidate_id=full_selected,
                    dataset=dataset,
                    heldout_config=heldout,
                    training_configs=training_configs,
                )
                attenuations[dataset] = attenuation
                ratios[dataset] = attenuation / float(
                    training_summaries[(full_selected, dataset)][
                        "practical_attenuation_margin"
                    ]
                )
            selective_score = min(
                _median([ratios[dataset] for dataset in near]),
                _median([ratios[dataset] for dataset in far]),
            )
            reverse_absent = all(
                attenuations[dataset]
                > -float(
                    training_summaries[(full_selected, dataset)][
                        "practical_attenuation_margin"
                    ]
                )
                for dataset in datasets
            )
        else:
            selective_score = None
            reverse_absent = False
        heldout_pass = bool(
            full_selected is not None
            and ranking["selected_candidate_id"] == full_selected
            and selective_score is not None
            and selective_score > 0.0
            and reverse_absent
        )
        folds.append(
            {
                "schema_version": CV_FOLD_SCHEMA,
                "fold_scheme": "loco",
                "fold_id": f"loco:{heldout}",
                "heldout_ood_dataset": None,
                "heldout_config_hash": heldout,
                "training_ood_datasets": list(datasets),
                "training_config_hashes": training_configs,
                "selection_seed_set": [1, 2],
                "eligible_candidate_ids": ranking["eligible_candidate_ids"],
                "selected_candidate_id": ranking["selected_candidate_id"],
                "selection_score": ranking["selection_score"],
                "tie_break_trace": ranking["tie_break_trace"],
                "training_summaries_sha256": _sha256_bytes(
                    _canonical_json_bytes(
                        [
                            training_summaries[key]
                            for key in sorted(training_summaries)
                        ]
                    )
                ),
                "heldout_absolute_attenuation": attenuations,
                "heldout_selective_score": selective_score,
                "heldout_pass": heldout_pass,
            }
        )
    return folds


def _legacy_contrasts(
    context: InputContext,
    indexed: Mapping[tuple[str, str, int, str, str], Mapping[str, Any]],
    *,
    summaries: Mapping[tuple[str, str], Mapping[str, Any]],
    datasets: Sequence[str],
) -> list[dict[str, Any]]:
    available_configs = {str(checkpoint["config_hash"]) for checkpoint in context.checkpoints}
    id_by_config_seed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in context.rows:
        key = (str(row["config_hash"]), int(row["training_seed"]))
        metrics = _require_mapping(row["id_metrics"], "ID metrics")
        if key in id_by_config_seed and id_by_config_seed[key] != metrics:
            raise ValueError("ID metrics differ across duplicated checkpoint rows")
        id_by_config_seed[key] = metrics

    role_entries: list[tuple[str, str, str, str]] = []
    legacy = context.policy["legacy_adam_adamw_contrasts"]
    for role in legacy["primary_discovery_roles"]:
        role_entries.append(
            (
                str(role["role_code"]),
                str(role["adam_config_hash"]),
                str(role["adamw_config_hash"]),
                "primary_discovery",
            )
        )
    for role in legacy["sensitivity_only"]:
        role_entries.append(
            (
                str(role["role_code"]),
                str(role["adam_config_hash"]),
                str(role["adamw_config_hash"]),
                "sensitivity_only",
            )
        )
    output: list[dict[str, Any]] = []
    for role_code, adam, adamw, role_scope in role_entries:
        if adam not in available_configs or adamw not in available_configs:
            continue
        for candidate in context.candidates:
            for seed in (0, 1, 2):
                adam_id = id_by_config_seed[(adam, seed)]
                adamw_id = id_by_config_seed[(adamw, seed)]
                for dataset in datasets:
                    raw_gap = float(
                        indexed[(candidate.candidate_id, adamw, seed, dataset, "raw")][
                            "auroc"
                        ]
                    ) - float(
                        indexed[(candidate.candidate_id, adam, seed, dataset, "raw")][
                            "auroc"
                        ]
                    )
                    transformed_gap = float(
                        indexed[
                            (
                                candidate.candidate_id,
                                adamw,
                                seed,
                                dataset,
                                "transformed",
                            )
                        ]["auroc"]
                    ) - float(
                        indexed[
                            (
                                candidate.candidate_id,
                                adam,
                                seed,
                                dataset,
                                "transformed",
                            )
                        ]["auroc"]
                    )
                    signed_attenuation = raw_gap - transformed_gap
                    magnitude_attenuation = abs(raw_gap) - abs(transformed_gap)
                    margin = float(
                        summaries[(candidate.candidate_id, dataset)][
                            "practical_attenuation_margin"
                        ]
                    )
                    overshoot = bool(
                        raw_gap * transformed_gap < 0.0
                        and abs(transformed_gap) > margin
                    )
                    raw_fpr_gap = float(
                        indexed[(candidate.candidate_id, adamw, seed, dataset, "raw")][
                            "fpr95"
                        ]
                    ) - float(
                        indexed[(candidate.candidate_id, adam, seed, dataset, "raw")][
                            "fpr95"
                        ]
                    )
                    transformed_fpr_gap = float(
                        indexed[
                            (
                                candidate.candidate_id,
                                adamw,
                                seed,
                                dataset,
                                "transformed",
                            )
                        ]["fpr95"]
                    ) - float(
                        indexed[
                            (
                                candidate.candidate_id,
                                adam,
                                seed,
                                dataset,
                                "transformed",
                            )
                        ]["fpr95"]
                    )
                    output.append(
                        {
                            "schema_version": CONTRAST_SCHEMA,
                            "candidate_id": candidate.candidate_id,
                            "contrast_type": "legacy_adamw_minus_adam",
                            "contrast_scope": role_scope,
                            "causal_interpretation": False,
                            "role_code": role_code,
                            "training_seed": seed,
                            "ood_dataset": dataset,
                            "detector": candidate.raw_detector,
                            "transformed_detector": candidate.transformed_detector,
                            "raw_gap": raw_gap,
                            "transformed_gap": transformed_gap,
                            "signed_attenuation": signed_attenuation,
                            "magnitude_attenuation": magnitude_attenuation,
                            "relative_attenuation": (
                                magnitude_attenuation / abs(raw_gap)
                                if raw_gap != 0.0
                                else None
                            ),
                            "overshoot": overshoot,
                            "id_accuracy_gap": float(adamw_id["accuracy"])
                            - float(adam_id["accuracy"]),
                            "id_nll_gap": float(adamw_id["nll"])
                            - float(adam_id["nll"]),
                            "id_ece_gap": float(adamw_id["ece"])
                            - float(adam_id["ece"]),
                            "fpr95_raw_gap": raw_fpr_gap,
                            "fpr95_transformed_gap": transformed_fpr_gap,
                            "fpr95_signed_attenuation": raw_fpr_gap
                            - transformed_fpr_gap,
                            "fpr95_magnitude_attenuation": abs(raw_fpr_gap)
                            - abs(transformed_fpr_gap),
                            "status": "PASS",
                            "reason_codes": [],
                        }
                    )
    return output


def reduce_gate(context: InputContext) -> Mapping[str, Any]:
    indexed = _index_rows(context)
    configs = sorted({str(checkpoint["config_hash"]) for checkpoint in context.checkpoints})
    datasets = list(context.policy["ood_partition"]["dataset_order"])
    combined = _summaries_for_scope(
        context, indexed, configs=configs, datasets=datasets, seeds=(1, 2)
    )
    full_ranking = _rank_candidates(context, combined, datasets)
    full_selected = full_ranking["selected_candidate_id"]

    all_summaries: list[Mapping[str, Any]] = []
    seed_rankings: dict[tuple[int, ...], Mapping[str, Any]] = {}
    for view in ((1,), (2,), (1, 2)):
        summaries = (
            combined
            if view == (1, 2)
            else _summaries_for_scope(
                context,
                indexed,
                configs=configs,
                datasets=datasets,
                seeds=view,
                frozen_margins=combined,
            )
        )
        seed_rankings[view] = _rank_candidates(context, summaries, datasets)
        all_summaries.extend(
            summaries[(candidate.candidate_id, dataset)]
            for candidate in context.candidates
            for dataset in datasets
        )

    lodo = _lodo_folds(
        context,
        indexed,
        configs=configs,
        datasets=datasets,
        full_selected=full_selected,
    )
    loco = _loco_folds(
        context,
        indexed,
        configs=configs,
        datasets=datasets,
        full_selected=full_selected,
    )
    cv_rows = lodo + loco
    lodo_count = sum(row["heldout_pass"] is True for row in lodo)
    loco_count = sum(row["heldout_pass"] is True for row in loco)
    seed1_selected = seed_rankings[(1,)]["selected_candidate_id"]
    seed2_selected = seed_rankings[(2,)]["selected_candidate_id"]
    seed_stability = bool(
        full_selected is not None
        and seed1_selected == full_selected
        and seed2_selected == full_selected
        and seed_rankings[(1, 2)]["selected_candidate_id"] == full_selected
    )
    lodo_pass = lodo_count >= int(
        context.policy["cross_validation"]["lodo"][
            "minimum_same_candidate_and_heldout_pass_count"
        ]
    )
    loco_pass = loco_count >= int(
        context.policy["cross_validation"]["loco"][
            "minimum_same_candidate_and_heldout_pass_count"
        ]
    )
    stable = seed_stability and lodo_pass and loco_pass
    hard_pass = bool(context.hard_validity["pass"])
    if not hard_pass:
        status = "FAILED"
        selected_candidate = None
    elif full_selected is None:
        status = "NO_GO"
        selected_candidate = None
    elif not stable:
        status = "INCONCLUSIVE"
        selected_candidate = None
    else:
        status = "PASS"
        selected_candidate = full_selected

    if full_selected is None:
        near_pass_count = 0
        far_pass_count = 0
    else:
        near_pass_count = sum(
            bool(combined[(full_selected, dataset)]["dataset_pass"])
            for dataset in context.policy["ood_partition"]["near"]
        )
        far_pass_count = sum(
            bool(combined[(full_selected, dataset)]["dataset_pass"])
            for dataset in context.policy["ood_partition"]["far"]
        )

    limitations = [
        "Stage 2 is protected discovery evidence, not confirmation or causality.",
        "Seed 0 is excluded from selection and margins and is retained only for descriptive legacy contrasts.",
        "OOD datasets are environments; no raw samples or datasets are pooled.",
        "This reducer does not generate preconfirmation_freeze.json.",
    ]
    if status == "FAILED":
        limitations.append("A hard validity failure prevents a scientific go/no-go decision.")
    elif status == "NO_GO":
        limitations.append("No frozen candidate passed the full-data practical and dataset-support rule.")
    elif status == "INCONCLUSIVE":
        limitations.append("A full-data candidate existed but the frozen seed/LODO/LOCO stability rule failed.")
    if context.source_validation_mode != "committed_head":
        limitations.append(
            "TEST-ONLY source-validation bypass was used; this output cannot authorize a scientific launch."
        )

    decision = {
        "schema_version": GATE_DECISION_SCHEMA,
        "analysis_id": context.analysis_id,
        "status": status,
        "selected_candidate_id": selected_candidate,
        "full_data_candidate_id": full_selected,
        "scientific_launch_allowed": bool(
            status == "PASS"
            and context.source_validation_mode == "committed_head"
            and context.evidence_provenance_validation_mode
            == "committed_head_and_frozen_v1_2_sources"
            and context.ctm_baseline_validation_mode
            == "authoritative_v1_2_per_checkpoint_records"
        ),
        "criteria": {
            "hard_validity": context.hard_validity,
            "full_data_practical_and_dataset_support_pass": full_selected is not None,
            "seed_stability_pass": seed_stability,
            "lodo_stability_pass": lodo_pass,
            "loco_stability_pass": loco_pass,
            "decision_precedence": list(
                context.policy["decision_semantics"]["decision_precedence"]
            ),
        },
        "lodo_selection_count": lodo_count,
        "loco_selection_count": loco_count,
        "seed1_selected_candidate": seed1_selected,
        "seed2_selected_candidate": seed2_selected,
        "combined_selected_candidate": seed_rankings[(1, 2)][
            "selected_candidate_id"
        ],
        "near_dataset_pass_count": near_pass_count,
        "far_dataset_pass_count": far_pass_count,
        "candidate_ranking": full_ranking["tie_break_trace"],
        "selection_score": full_ranking["selection_score"],
        "provenance": {
            "selection_manifest_sha256": context.selection_manifest_sha256,
            "model_metrics_sha256": context.model_metrics_sha256,
            "diagnostic_metrics_sha256": context.diagnostic_metrics_sha256,
            "candidate_registry_sha256": context.candidate_registry_sha256,
            "gate_policy_sha256": context.gate_policy_sha256,
            "evidence_git_sha": context.evidence_git_sha,
            "reducer_git_sha": context.reducer_git_sha,
            "reducer_code_sources": list(context.reducer_code_sources),
            "source_validation_mode": context.source_validation_mode,
            "evidence_provenance_validation_mode": (
                context.evidence_provenance_validation_mode
            ),
            "ctm_baseline_validation_mode": context.ctm_baseline_validation_mode,
            "selection_seed_set": [1, 2],
            "seed_zero_used_for_selection": False,
            "config_count": len(configs),
            "config_pair_count": len(configs) * (len(configs) - 1) // 2,
            "ood_datasets": datasets,
        },
        "limitations": limitations,
    }
    contrasts = _legacy_contrasts(
        context, indexed, summaries=combined, datasets=datasets
    )
    return {
        "contrast_metrics.jsonl": contrasts,
        "dataset_summaries.jsonl": all_summaries,
        "cv_folds.jsonl": cv_rows,
        "diagnostic_summaries.jsonl": _diagnostic_summaries(context),
        "gate_decision.json": decision,
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_output_destination(
    output_directory: str | Path,
    *,
    input_paths: Sequence[str | Path] = (),
) -> Path:
    """Fail closed on unsafe output topology before evidence inputs are read."""
    destination = Path(os.path.abspath(os.fspath(output_directory)))
    if os.path.lexists(destination):
        raise FileExistsError(f"output directory already exists: {destination}")

    parent = destination.parent
    if not os.path.lexists(parent):
        raise FileNotFoundError(f"output parent must already exist: {parent}")
    current = parent
    while True:
        if current.is_symlink():
            raise ValueError(
                f"output parent chain must not contain a symlink: {current}"
            )
        if current.parent == current:
            break
        current = current.parent
    if not parent.is_dir():
        raise ValueError(f"output parent must be a real directory: {parent}")

    resolved_destination = destination.resolve(strict=False)
    repository_root = REPOSITORY_ROOT.resolve()
    if (
        resolved_destination == repository_root
        or repository_root in resolved_destination.parents
    ):
        raise ValueError("reducer output must be outside the Git worktree")

    for raw_input_path in input_paths:
        input_tree = Path(raw_input_path).resolve(strict=False).parent
        if _paths_overlap(resolved_destination, input_tree):
            raise ValueError(
                "reducer output and evidence input trees must be disjoint: "
                f"{resolved_destination} vs {input_tree}"
            )
    return resolved_destination


def write_outputs_no_overwrite(
    artifacts: Mapping[str, Any], output_directory: str | Path
) -> Mapping[str, str]:
    destination = validate_output_destination(output_directory)
    if set(artifacts) != set(OUTPUT_FILENAMES):
        raise ValueError("reducer output artifact set mismatch")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    digests: dict[str, str] = {}
    try:
        for filename in OUTPUT_FILENAMES:
            payload = artifacts[filename]
            if filename.endswith(".jsonl"):
                if not isinstance(payload, list):
                    raise ValueError(f"{filename} payload must be a list")
                data = _canonical_jsonl_bytes(payload)
            else:
                data = _canonical_json_bytes(payload)
            (temporary / filename).write_bytes(data)
            digests[filename] = _sha256_bytes(data)
        checksum_data = "".join(
            f"{digests[filename]}  {filename}\n" for filename in OUTPUT_FILENAMES
        ).encode("utf-8")
        (temporary / "checksums.sha256").write_bytes(checksum_data)
        os.mkdir(destination)
        linked: list[Path] = []
        try:
            for filename in (*OUTPUT_FILENAMES, "checksums.sha256"):
                target = destination / filename
                os.link(temporary / filename, target)
                linked.append(target)
        except BaseException:
            for target in reversed(linked):
                target.unlink(missing_ok=True)
            destination.rmdir()
            raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return digests


def _optional_file_sha256(path: Path) -> str | None:
    try:
        return _sha256_file(path) if path.is_file() else None
    except OSError:
        return None


def _preflight_failure_reason(error: BaseException) -> str:
    message = str(error).lower()
    if "checksum" in message or "sha256" in message:
        return "ARTIFACT_CHECKSUM_FAILED"
    if "missing" in message or "absent" in message or "no such file" in message:
        return "ARTIFACT_FILE_MISSING"
    if "source differs from head" in message or "git sha" in message:
        return "REDUCER_SOURCE_VALIDATION_FAILED"
    if "schema" in message:
        return "ARTIFACT_SCHEMA_FAILED"
    if (
        "population" in message
        or "identity" in message
        or "binding" in message
        or "must bind" in message
        or "checkpoint" in message
    ):
        return "ARTIFACT_IDENTITY_FAILED"
    return "ARTIFACT_PREFLIGHT_FAILED"


def _failed_preflight_artifacts(
    *,
    error: BaseException,
    selection_manifest_path: Path,
    model_metrics_path: Path,
    diagnostic_metrics_path: Path,
    candidate_registry_path: Path,
    gate_policy_path: Path,
    reducer_git_sha: str | None,
    reducer_code_sources: Sequence[Mapping[str, Any]] | None,
    validate_git_sources: bool,
) -> Mapping[str, Any]:
    try:
        observed_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        observed_head = None
    actual = {
        "selection_manifest_sha256": _optional_file_sha256(selection_manifest_path),
        "model_metrics_sha256": _optional_file_sha256(model_metrics_path),
        "diagnostic_metrics_sha256": _optional_file_sha256(
            diagnostic_metrics_path
        ),
        "candidate_registry_sha256": _optional_file_sha256(candidate_registry_path),
        "gate_policy_sha256": _optional_file_sha256(gate_policy_path),
        "reducer_script_sha256": _optional_file_sha256(Path(__file__).resolve()),
    }
    input_analysis_id: str | None = None
    try:
        manifest = _load_json(selection_manifest_path, "selection manifest")
        candidate = manifest.get("analysis_id")
        if isinstance(candidate, str) and len(candidate) == 64 and all(
            character in "0123456789abcdef" for character in candidate
        ):
            input_analysis_id = candidate
    except (ValueError, OSError):
        pass
    reason_code = _preflight_failure_reason(error)
    failure_identity = {
        "input_analysis_id": input_analysis_id,
        "reason_code": reason_code,
        "selection_manifest_sha256": actual["selection_manifest_sha256"],
        "model_metrics_sha256": actual["model_metrics_sha256"],
        "diagnostic_metrics_sha256": actual["diagnostic_metrics_sha256"],
        "candidate_registry_sha256": actual["candidate_registry_sha256"],
        "gate_policy_sha256": actual["gate_policy_sha256"],
    }
    failure_analysis_id = _sha256_bytes(_canonical_json_bytes(failure_identity))
    source_mode = (
        "committed_head_validation_attempted"
        if validate_git_sources
        else "test_only_explicit_bypass"
    )
    try:
        decision_precedence = list(
            _require_mapping(
                _load_json(gate_policy_path, "gate policy").get(
                    "decision_semantics"
                ),
                "decision semantics",
            ).get("decision_precedence", [])
        )
    except (ValueError, OSError):
        decision_precedence = []
    decision = {
        "schema_version": GATE_DECISION_SCHEMA,
        "analysis_id": failure_analysis_id,
        "input_analysis_id": input_analysis_id,
        "status": "FAILED",
        "selected_candidate_id": None,
        "full_data_candidate_id": None,
        "scientific_launch_allowed": False,
        "criteria": {
            "hard_validity": {
                "pass": False,
                "criteria": {"artifact_preflight_pass": False},
                "reason_codes": [reason_code],
            },
            "full_data_practical_and_dataset_support_pass": False,
            "seed_stability_pass": False,
            "lodo_stability_pass": False,
            "loco_stability_pass": False,
            "decision_precedence": decision_precedence,
        },
        "lodo_selection_count": 0,
        "loco_selection_count": 0,
        "seed1_selected_candidate": None,
        "seed2_selected_candidate": None,
        "combined_selected_candidate": None,
        "near_dataset_pass_count": 0,
        "far_dataset_pass_count": 0,
        "candidate_ranking": [],
        "selection_score": None,
        "provenance": {
            **actual,
            "reducer_git_sha": reducer_git_sha or observed_head,
            "reducer_code_sources": list(reducer_code_sources or ()),
            "source_validation_mode": source_mode,
            "preflight_error_type": type(error).__name__,
            "preflight_error": str(error),
            "preflight_trusted": False,
            "seed_zero_used_for_selection": False,
        },
        "limitations": [
            "Artifact preflight failed before any scientific estimand was reduced.",
            "Empty contrast, dataset-summary, and cross-validation outputs are intentional.",
            "This FAILED artifact cannot authorize a scientific launch or preconfirmation freeze.",
        ],
    }
    return {
        "contrast_metrics.jsonl": [],
        "dataset_summaries.jsonl": [],
        "cv_folds.jsonl": [],
        "diagnostic_summaries.jsonl": [],
        "gate_decision.json": decision,
    }


def run_reducer(
    *,
    selection_manifest_path: str | Path,
    model_metrics_path: str | Path,
    diagnostic_metrics_path: str | Path,
    output_directory: str | Path,
    candidate_registry_path: str | Path = DEFAULT_CANDIDATE_REGISTRY,
    gate_policy_path: str | Path = DEFAULT_GATE_POLICY,
    input_checksums_path: str | Path | None = None,
    validate_git_sources: bool = True,
    reducer_git_sha: str | None = None,
    reducer_code_sources: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    selection_path = Path(selection_manifest_path)
    metrics_path = Path(model_metrics_path)
    diagnostics_path = Path(diagnostic_metrics_path)
    registry_path = Path(candidate_registry_path)
    policy_path = Path(gate_policy_path)
    checksums_path = (
        Path(input_checksums_path)
        if input_checksums_path is not None
        else selection_path.parent / "checksums.sha256"
    )
    safe_output_directory = validate_output_destination(
        output_directory,
        input_paths=(selection_path, metrics_path, diagnostics_path, checksums_path),
    )
    try:
        context = load_inputs(
            selection_manifest_path=selection_path,
            model_metrics_path=metrics_path,
            diagnostic_metrics_path=diagnostics_path,
            candidate_registry_path=registry_path,
            gate_policy_path=policy_path,
            input_checksums_path=input_checksums_path,
            validate_git_sources=validate_git_sources,
            reducer_git_sha=reducer_git_sha,
            reducer_code_sources=reducer_code_sources,
        )
    except (ValueError, FileNotFoundError) as error:
        artifacts = _failed_preflight_artifacts(
            error=error,
            selection_manifest_path=selection_path,
            model_metrics_path=metrics_path,
            diagnostic_metrics_path=diagnostics_path,
            candidate_registry_path=registry_path,
            gate_policy_path=policy_path,
            reducer_git_sha=reducer_git_sha,
            reducer_code_sources=reducer_code_sources,
            validate_git_sources=validate_git_sources,
        )
    else:
        artifacts = reduce_gate(context)
    digests = write_outputs_no_overwrite(artifacts, safe_output_directory)
    return {
        "status": artifacts["gate_decision.json"]["status"],
        "selected_candidate_id": artifacts["gate_decision.json"][
            "selected_candidate_id"
        ],
        "digests": digests,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--model-metrics", type=Path, required=True)
    parser.add_argument("--diagnostic-metrics", type=Path, required=True)
    parser.add_argument("--input-checksums", type=Path)
    parser.add_argument(
        "--candidate-registry", type=Path, default=DEFAULT_CANDIDATE_REGISTRY
    )
    parser.add_argument("--gate-policy", type=Path, default=DEFAULT_GATE_POLICY)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_reducer(
        selection_manifest_path=args.selection_manifest,
        model_metrics_path=args.model_metrics,
        diagnostic_metrics_path=args.diagnostic_metrics,
        output_directory=args.output_dir,
        candidate_registry_path=args.candidate_registry,
        gate_policy_path=args.gate_policy,
        input_checksums_path=args.input_checksums,
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0 if result["status"] in {"PASS", "NO_GO", "INCONCLUSIVE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
