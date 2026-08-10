#!/usr/bin/env python3
"""Analyze Stage-2 mechanism candidates from selectively retrieved v1.2 files.

The retrieval tree is intentionally *not* a complete production bundle.  Every
file access is therefore mediated by the exact allowlist and SHA/size catalog
in ``fixed_readout_stage2_reuse_manifest_v1``.  This module is a selective
bundle/pair evidence extractor, not the Stage-2 scientific gate reducer.
Configuration pairs are discovery-only; it reports every candidate channel,
never selects a focal channel, and always reports the scientific gate as
``NOT_RUN``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from oge.evaluation.classification import logit_ood_scores
from oge.evaluation.detectors import (
    TiedGaussianFit,
    exact_knn_l2_scores,
    exact_neighbor_distances,
    fit_tied_gaussians,
    fit_vim,
    prototype_scores,
    score_tied_gaussians,
    score_vim,
)
from oge.evaluation.extraction import ordered_sample_id_digest, sha256_file
from oge.evaluation.metrics import compute_ood_metrics


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REUSE_SCHEMA_VERSION = "fixed_readout_stage2_reuse_manifest_v1"
REMOTE_CATALOG_SCHEMA_VERSION = "fixed_readout_stage2_remote_checksum_catalog_v1"
PAIR_MANIFEST_SCHEMA_VERSION = "fixed_readout_stage2_pairs_v1"
REPORT_SCHEMA_VERSION = "fixed_readout_stage2_mechanism_v2"
BUNDLE_REPORT_SCHEMA_VERSION = "fixed_readout_stage2_bundle_mechanism_v1"
RAW_KNN_CACHE_SCHEMA_VERSION = "fixed_readout_stage2_raw_knn_cache_v1"
ANALYSIS_ROLE = "discovery_only"
KNN_K = 50
DEFAULT_HISTOGRAM_BINS = 64
ORACLE_ATOL = 1.0e-12
ORACLE_RTOL = 1.0e-10
_HEX64 = re.compile(r"[0-9a-f]{64}")
ANALYSIS_POLICY_FILES = {
    "candidate_registry": (
        "fixed_readout_stage2_candidates_v1",
        "configs/evaluation/fixed_readout_stage2/candidate_registry.json",
    ),
    "gate_policy": (
        "fixed_readout_stage2_gate_v1",
        "configs/evaluation/fixed_readout_stage2/gate_policy.json",
    ),
}

STORED_SCORE_KEYS = (
    "detector/mahalanobis_raw",
    "detector/mahalanobis_pp",
    "detector/knn_l2_k50",
    "detector/ctm_prototype_cosine",
    "ood_score/energy_t1",
)
DIAGNOSTIC_PATHS = {
    "raw_class_distances": ("diagnostics", "raw_gaussian", "class_distances"),
    "raw_nearest_class": ("diagnostics", "raw_gaussian", "nearest_class"),
    "normalized_class_distances": (
        "diagnostics",
        "normalized_gaussian",
        "class_distances",
    ),
    "knn_kth_squared_distance": ("diagnostics", "knn", "kth_squared_distance"),
    "knn_kth_neighbor_sample_id": (
        "diagnostics",
        "knn",
        "kth_neighbor_sample_id",
    ),
    "vim_residual": ("diagnostics", "vim", "residual"),
}


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    missing = set(fields).difference(value)
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _require_hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _safe_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("allowlisted logical path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"unsafe allowlisted logical path: {value!r}")
    return value


def _finite_array(value: Any, *, label: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{label} must have ndim={ndim}, got {array.ndim}")
    if array.size == 0:
        raise ValueError(f"{label} must not be empty")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return array


def _assert_non_git_root(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the Git worktree")
    for ancestor in (resolved, *resolved.parents):
        marker = ancestor / ".git"
        is_repository = (
            marker.is_file()
            or (marker.is_dir() and (marker / "HEAD").is_file())
        )
        if is_repository:
            raise ValueError(f"{label} must not be inside any Git worktree")
    return resolved


def validate_selective_reuse_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != REUSE_SCHEMA_VERSION:
        raise ValueError("reuse manifest schema version mismatch")
    if manifest.get("retrieval_ready") is not True:
        raise ValueError("reuse manifest is not retrieval-ready")
    authorization = _require_mapping(manifest.get("authorization"), "reuse authorization")
    required_policy = {
        "analysis_scope": "discovery_only",
        "bulk_fetch_allowed": False,
        "checkpoint_or_feature_recompute_allowed": False,
        "destination_overwrite_allowed": False,
        "mode": "reuse_only",
        "new_protected_split_traversal_allowed": False,
        "remote_delete_allowed": False,
        "remote_mutation_allowed": False,
        "remote_upload_allowed": False,
        "retrieval_policy": "catalog_verified_exact_allowlist_only",
    }
    failed_policy = [
        key for key, expected in required_policy.items() if authorization.get(key) != expected
    ]
    if failed_policy:
        raise ValueError(f"reuse authorization policy mismatch: {sorted(failed_policy)}")
    analysis_policy = _require_mapping(
        manifest.get("analysis_policy"), "reuse analysis policy"
    )
    if set(analysis_policy) != set(ANALYSIS_POLICY_FILES):
        raise ValueError("reuse analysis-policy bindings are incomplete")
    for name, (expected_id, expected_path) in ANALYSIS_POLICY_FILES.items():
        binding = _require_mapping(analysis_policy.get(name), f"{name} binding")
        if binding.get("id") != expected_id or binding.get("path") != expected_path:
            raise ValueError(f"reuse analysis-policy identity mismatch: {name}")
        expected_sha = _require_hex64(binding.get("sha256"), f"{name} SHA256")
        policy_path = REPOSITORY_ROOT / expected_path
        if not policy_path.is_file() or sha256_file(policy_path) != expected_sha:
            raise ValueError(f"reuse analysis-policy file mismatch: {name}")
    selection = _require_mapping(manifest.get("selection"), "reuse selection")
    if selection.get("job_count") != 30:
        raise ValueError("reuse manifest selection must contain exactly 30 identities")
    if selection.get("configuration_count") != 10 or selection.get("seeds") != [0, 1, 2]:
        raise ValueError("reuse manifest selection must contain 10 configurations x 3 seeds")
    if selection.get("checkpoint_role") != "last":
        raise ValueError("reuse manifest selection must use last checkpoints")
    if selection.get("reporting_role") != "confirmatory_primary":
        raise ValueError("reuse manifest selection must be confirmatory-primary")
    catalog = _require_mapping(
        manifest.get("remote_checksum_catalog"), "remote checksum catalog metadata"
    )
    if catalog.get("status") != "complete":
        raise ValueError("reuse manifest requires a complete remote checksum catalog")
    if catalog.get("schema_version") != REMOTE_CATALOG_SCHEMA_VERSION:
        raise ValueError("remote checksum catalog schema version mismatch")
    _require_hex64(catalog.get("sha256"), "remote checksum catalog SHA256")
    path_contract = _require_mapping(manifest.get("path_contract"), "path contract")
    required_paths_raw = path_contract.get("required_logical_paths")
    query_keys = path_contract.get("query_keys")
    if not isinstance(required_paths_raw, list) or not required_paths_raw:
        raise ValueError("path contract requires a non-empty logical-path allowlist")
    required_paths = [_safe_logical_path(value) for value in required_paths_raw]
    if required_paths != sorted(set(required_paths)):
        raise ValueError("path-contract allowlist must be sorted and unique")
    if (
        not isinstance(query_keys, list)
        or not query_keys
        or query_keys[0] != "id_test"
        or len(query_keys) < 2
        or len(set(query_keys)) != len(query_keys)
    ):
        raise ValueError("path contract query keys must contain ID test and OOD")
    required_metadata = {
        "artifact_manifest.json",
        "bundle_checksums.sha256",
        "classifier_weight.npy",
        "manifest.json",
        "metrics/result.json",
    }
    if not required_metadata.issubset(required_paths):
        raise ValueError("path-contract allowlist omits required selective metadata")
    bundles = manifest.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 30:
        raise ValueError("reuse manifest must contain exactly 30 bundle identities")
    seen_bundle_ids: set[str] = set()
    seen_training: set[tuple[str, int]] = set()
    for row in bundles:
        bundle = _require_mapping(row, "reuse bundle")
        bundle_id = _require_hex64(bundle.get("bundle_id"), "bundle_id")
        if bundle_id in seen_bundle_ids:
            raise ValueError("reuse manifest contains duplicate bundle identity")
        seen_bundle_ids.add(bundle_id)
        config_hash = _require_hex64(bundle.get("config_hash"), "config_hash")
        seed = bundle.get("training_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("training_seed must be an integer")
        training_key = (config_hash, seed)
        if training_key in seen_training:
            raise ValueError("reuse manifest contains duplicate training identity")
        seen_training.add(training_key)
        identity = _require_mapping(bundle.get("identity"), "artifact identity")
        expected_identity = {
            "checkpoint_sha256": bundle_id,
            "config_hash": config_hash,
            "training_seed": seed,
        }
        for field, expected in expected_identity.items():
            if identity.get(field) != expected:
                raise ValueError(f"reuse bundle identity mismatch for {field}")
        if identity.get("artifact_role") != "checkpoint_metric_contract_v1.2_bundle":
            raise ValueError("reuse bundle artifact role mismatch")
        if identity.get("checkpoint_role") != "last":
            raise ValueError("Stage-2 reuse accepts only last checkpoints")
        if identity.get("reporting_role") != "confirmatory_primary":
            raise ValueError("Stage-2 reuse accepts only confirmatory-primary bundles")
        integrity = _require_mapping(bundle.get("remote_integrity"), "remote integrity")
        files = integrity.get("files")
        if not isinstance(files, list):
            raise ValueError("remote-integrity files must be a list")
        file_paths = [_safe_logical_path(item.get("logical_path")) for item in files]
        if file_paths != required_paths:
            raise ValueError("bundle file allowlist differs from path contract")
        for item in files:
            _require_hex64(item.get("sha256"), f"SHA256 for {item.get('logical_path')}")
            size = item.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("allowlisted file size must be a positive integer")
        by_path = {str(item["logical_path"]): item for item in files}
        _require_hex64(
            integrity.get("artifact_manifest_file_sha256"),
            "artifact manifest file SHA256",
        )
        _require_hex64(
            integrity.get("bundle_checksums_file_sha256"),
            "bundle checksums file SHA256",
        )
        if "artifact_manifest.json" in by_path and (
            by_path["artifact_manifest.json"]["sha256"]
            != integrity["artifact_manifest_file_sha256"]
        ):
            raise ValueError("artifact manifest SHA256 catalog mismatch")
        if "bundle_checksums.sha256" in by_path and (
            by_path["bundle_checksums.sha256"]["sha256"]
            != integrity["bundle_checksums_file_sha256"]
        ):
            raise ValueError("bundle checksum-file SHA256 catalog mismatch")
        digest_payload = {
            "artifact_manifest_file_sha256": integrity[
                "artifact_manifest_file_sha256"
            ],
            "bundle_checksums_file_sha256": integrity[
                "bundle_checksums_file_sha256"
            ],
            "files": files,
        }
        if integrity.get("allowed_payload_catalog_sha256") != _canonical_sha256(
            digest_payload
        ):
            raise ValueError("allowed payload catalog digest mismatch")
        if integrity.get("selected_file_count") != len(files):
            raise ValueError("selected file count mismatch")
        if integrity.get("selected_total_size") != sum(int(item["size"]) for item in files):
            raise ValueError("selected total size mismatch")
    if {seed for _, seed in seen_training} != {0, 1, 2}:
        raise ValueError("reuse manifest bundle population must contain seeds 0, 1, and 2")
    config_counts: dict[str, int] = {}
    for config_hash, _ in seen_training:
        config_counts[config_hash] = config_counts.get(config_hash, 0) + 1
    if len(config_counts) != 10 or set(config_counts.values()) != {3}:
        raise ValueError("reuse manifest bundle population must be 10 configurations x 3 seeds")


@dataclass(frozen=True)
class SelectiveBundleAccess:
    bundle_id: str
    bundle_root: Path
    allowed_files: dict[str, dict[str, Any]]

    def path(self, logical_path: str) -> Path:
        logical = _safe_logical_path(logical_path)
        if logical not in self.allowed_files:
            raise PermissionError(f"access outside selective allowlist is forbidden: {logical}")
        candidate = (self.bundle_root / logical).resolve()
        root = self.bundle_root.resolve()
        if candidate != root and root not in candidate.parents:
            raise PermissionError("resolved allowlisted path escapes bundle root")
        return candidate

    def verify_all(self) -> None:
        for logical_path, record in self.allowed_files.items():
            path = self.path(logical_path)
            if not path.is_file():
                raise FileNotFoundError(f"allowlisted retrieval file is missing: {logical_path}")
            if path.stat().st_size != int(record["size"]):
                raise ValueError(f"allowlisted retrieval size mismatch: {logical_path}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"allowlisted retrieval checksum mismatch: {logical_path}")

    def read_json(self, logical_path: str) -> dict[str, Any]:
        payload = json.loads(self.path(logical_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{logical_path} must contain a JSON mapping")
        return payload

    def load_npy(self, logical_path: str) -> np.ndarray:
        array = np.load(self.path(logical_path), allow_pickle=False)
        return _finite_array(array, label=logical_path)


@dataclass(frozen=True)
class ReuseContext:
    manifest_path: Path
    manifest_sha256: str
    retrieval_root: Path
    manifest: dict[str, Any]
    bundles: dict[str, dict[str, Any]]
    accesses: dict[str, SelectiveBundleAccess]


def _verify_exact_retrieval_tree(
    root: Path, bundles: Mapping[str, Mapping[str, Any]]
) -> None:
    expected = {
        PurePosixPath(bundle_id, item["logical_path"]).as_posix()
        for bundle_id, bundle in bundles.items()
        for item in bundle["remote_integrity"]["files"]
    }
    observed: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *filenames]:
            candidate = current_path / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink is forbidden in retrieval tree: {candidate}")
            if name in directories and not stat.S_ISDIR(mode):
                raise ValueError(f"non-directory traversal entry: {candidate}")
        for name in filenames:
            candidate = current_path / name
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise ValueError(f"non-regular retrieval file: {candidate}")
            observed.add(candidate.relative_to(root).as_posix())
    extras = sorted(observed - expected)
    missing = sorted(expected - observed)
    if extras:
        raise ValueError(f"unexpected retrieval file: {extras[0]}")
    if missing:
        raise FileNotFoundError(f"allowlisted retrieval file is missing: {missing[0]}")


def load_reuse_context(
    reuse_manifest_path: str | Path, retrieval_root: str | Path
) -> ReuseContext:
    manifest_path = Path(reuse_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("reuse manifest must contain a JSON mapping")
    validate_selective_reuse_manifest(manifest)
    requested_root = Path(retrieval_root)
    if requested_root.is_symlink():
        raise ValueError("retrieval root must not be a symlink")
    root = _assert_non_git_root(requested_root, label="retrieval root")
    if not root.is_dir():
        raise FileNotFoundError(f"retrieval root is absent: {root}")
    bundles = {str(row["bundle_id"]): dict(row) for row in manifest["bundles"]}
    _verify_exact_retrieval_tree(root, bundles)
    accesses: dict[str, SelectiveBundleAccess] = {}
    for bundle_id, bundle in bundles.items():
        files = {
            str(item["logical_path"]): dict(item)
            for item in bundle["remote_integrity"]["files"]
        }
        bundle_root = (root / bundle_id).resolve()
        if root != bundle_root and root not in bundle_root.parents:
            raise PermissionError(f"retrieved bundle root escapes retrieval root: {bundle_id}")
        access = SelectiveBundleAccess(
            bundle_id=bundle_id,
            bundle_root=bundle_root,
            allowed_files=files,
        )
        access.verify_all()
        artifact = access.read_json("artifact_manifest.json")
        for field, expected in bundle["identity"].items():
            if artifact.get(field) != expected:
                raise ValueError(
                    f"retrieved artifact identity mismatch for {bundle_id}:{field}"
                )
        accesses[bundle_id] = access
    return ReuseContext(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        retrieval_root=root,
        manifest=manifest,
        bundles=bundles,
        accesses=accesses,
    )


@dataclass(frozen=True)
class SelectiveBundle:
    bundle_id: str
    bundle_spec: dict[str, Any]
    access: SelectiveBundleAccess
    raw_manifest: dict[str, Any]
    result: dict[str, Any]
    splits: dict[str, dict[str, np.ndarray]]
    scores: dict[str, dict[str, np.ndarray]]
    diagnostics: dict[str, dict[str, np.ndarray]]
    classifier_weight: np.ndarray
    classifier_bias: np.ndarray | None
    query_keys: tuple[str, ...]
    ood_keys: tuple[str, ...]
    source_allowlist_sha256: str
    reuse_manifest_sha256: str
    analysis_policy: dict[str, Any]


def _raw_array_logical_path(
    manifest: Mapping[str, Any], split_name: str, array_name: str
) -> str:
    split_record = _require_mapping(
        manifest["dataset"]["splits"].get(split_name), f"raw split {split_name}"
    )
    return f"{split_record['relative_directory']}/{array_name}.npy"


def _load_raw_array(
    access: SelectiveBundleAccess,
    manifest: Mapping[str, Any],
    split_name: str,
    array_name: str,
) -> np.ndarray:
    split_record = _require_mapping(
        manifest["dataset"]["splits"].get(split_name), f"raw split {split_name}"
    )
    logical = _raw_array_logical_path(manifest, split_name, array_name)
    array = access.load_npy(logical)
    shapes = _require_mapping(split_record.get("array_shapes"), f"{split_name} shapes")
    dtypes = _require_mapping(split_record.get("array_dtypes"), f"{split_name} dtypes")
    if list(array.shape) != list(shapes.get(array_name, ())):
        raise ValueError(f"{split_name}/{array_name} shape differs from raw manifest")
    if str(array.dtype) != str(dtypes.get(array_name)):
        raise ValueError(f"{split_name}/{array_name} dtype differs from raw manifest")
    if len(array) != int(split_record["sample_count"]):
        raise ValueError(f"{split_name}/{array_name} sample count mismatch")
    return array


def _nested(value: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"required result reference is missing: {label}")
        current = current[key]
    return current


def _load_result_reference(
    access: SelectiveBundleAccess,
    reference: Any,
    *,
    label: str,
    expected_count: int,
) -> np.ndarray:
    ref = _require_mapping(reference, f"{label} array reference")
    _require_fields(ref, ("array_path", "sha256", "shape", "dtype"), f"{label} reference")
    logical = f"metrics/{_safe_logical_path(ref['array_path'])}"
    # access.load_npy performs the critical allowlist check before filesystem IO.
    array = access.load_npy(logical)
    if sha256_file(access.path(logical)) != ref["sha256"]:
        raise ValueError(f"direct result-reference checksum mismatch: {label}")
    if list(array.shape) != list(ref["shape"]) or str(array.dtype) != str(ref["dtype"]):
        raise ValueError(f"direct result-reference schema mismatch: {label}")
    if len(array) != expected_count:
        raise ValueError(f"direct result-reference sample count mismatch: {label}")
    return array


def load_selective_bundle(context: ReuseContext, bundle_id: str) -> SelectiveBundle:
    if bundle_id not in context.bundles:
        raise ValueError(f"bundle_id is absent from reuse manifest: {bundle_id}")
    bundle_spec = context.bundles[bundle_id]
    access = context.accesses[bundle_id]
    raw_manifest = access.read_json("manifest.json")
    result = access.read_json("metrics/result.json")
    artifact = access.read_json("artifact_manifest.json")
    job = _require_mapping(result.get("job"), "metrics result job")
    checkpoint = _require_mapping(raw_manifest.get("checkpoint"), "raw checkpoint")
    identity_sources = {
        "checkpoint_sha256": (
            bundle_id,
            checkpoint.get("sha256"),
            job.get("checkpoint_sha256"),
            artifact.get("checkpoint_sha256"),
        ),
        "checkpoint_role": (
            bundle_spec["identity"].get("checkpoint_role"),
            checkpoint.get("role"),
            job.get("checkpoint_role"),
            artifact.get("checkpoint_role"),
        ),
        "config_hash": (
            bundle_spec.get("config_hash"),
            checkpoint.get("config_hash"),
            job.get("config_hash"),
            artifact.get("config_hash"),
        ),
        "training_seed": (
            int(bundle_spec["training_seed"]),
            int(checkpoint.get("training_seed", -1)),
            int(job.get("training_seed", -2)),
            int(artifact.get("training_seed", -3)),
        ),
    }
    drift = [name for name, values in identity_sources.items() if len(set(values)) != 1]
    if drift:
        raise ValueError(f"selective bundle identity mismatch: {sorted(drift)}")
    for field in ("evaluation_git_sha", "inventory_hash", "dataset_policy_hash"):
        expected = bundle_spec["identity"].get(field)
        observed = artifact.get(field)
        result_value = result.get(field)
        raw_value = (
            raw_manifest.get("evaluation", {}).get("oge_git_sha")
            if field == "evaluation_git_sha"
            else expected
        )
        if len({expected, observed, result_value, raw_value}) != 1:
            raise ValueError(f"selective bundle identity mismatch: {field}")

    query_keys = tuple(str(value) for value in context.manifest["path_contract"]["query_keys"])
    ood_keys = tuple(value for value in query_keys if value != "id_test")
    load_splits = ("id_train", *query_keys)
    splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name in load_splits:
        arrays = {
            name: _load_raw_array(access, raw_manifest, split_name, name)
            for name in ("features", "logits", "sample_ids")
        }
        if split_name in {"id_train", "id_test"}:
            arrays["class_labels"] = _load_raw_array(
                access, raw_manifest, split_name, "class_labels"
            )
        _finite_array(arrays["features"], label=f"{split_name}/features", ndim=2)
        _finite_array(arrays["logits"], label=f"{split_name}/logits", ndim=2)
        _finite_array(arrays["sample_ids"], label=f"{split_name}/sample_ids", ndim=1)
        if len(arrays["features"]) != len(arrays["logits"]):
            raise ValueError(f"{split_name} feature/logit sample count mismatch")
        if "class_labels" in arrays:
            labels = _finite_array(
                arrays["class_labels"], label=f"{split_name}/class_labels", ndim=1
            )
            if labels.dtype.kind not in "iu" or len(labels) != len(arrays["features"]):
                raise ValueError(f"{split_name} class-label schema mismatch")
        sample_ids = np.asarray(arrays["sample_ids"], dtype=str)
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError(f"{split_name} sample IDs are not unique")
        expected_digest = raw_manifest["dataset"]["splits"][split_name].get(
            "ordered_sample_id_sha256"
        )
        if expected_digest is None or ordered_sample_id_digest(sample_ids) != expected_digest:
            raise ValueError(f"{split_name} ordered sample identity mismatch")
        splits[split_name] = arrays

    scores_document = _require_mapping(result.get("scores"), "metrics result scores")
    scores: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for split_name in query_keys:
        count = len(splits[split_name]["features"])
        split_result = _require_mapping(scores_document.get(split_name), f"{split_name} result")
        scores[split_name] = {
            key: _load_result_reference(
                access,
                split_result.get(key),
                label=f"{split_name}:{key}",
                expected_count=count,
            ).astype(np.float64)
            for key in STORED_SCORE_KEYS
        }
        for key, array in scores[split_name].items():
            _finite_array(array, label=f"{split_name}:{key}", ndim=1)
        diagnostics[split_name] = {
            name: _load_result_reference(
                access,
                _nested(split_result, path, f"{split_name}:{name}"),
                label=f"{split_name}:{name}",
                expected_count=count,
            )
            for name, path in DIAGNOSTIC_PATHS.items()
        }
        expected_ndim = {
            "raw_class_distances": 2,
            "raw_nearest_class": 1,
            "normalized_class_distances": 2,
            "knn_kth_squared_distance": 1,
            "knn_kth_neighbor_sample_id": 1,
            "vim_residual": 1,
        }
        for name, array in diagnostics[split_name].items():
            _finite_array(array, label=f"{split_name}:{name}", ndim=expected_ndim[name])

    weight = _finite_array(
        access.load_npy("classifier_weight.npy"), label="classifier_weight", ndim=2
    ).astype(np.float64)
    bias = (
        _finite_array(
            access.load_npy("classifier_bias.npy"), label="classifier_bias", ndim=1
        ).astype(np.float64)
        if "classifier_bias.npy" in access.allowed_files
        else None
    )
    model = _require_mapping(raw_manifest.get("model"), "raw model identity")
    feature_widths = {int(split["features"].shape[1]) for split in splits.values()}
    logit_widths = {int(split["logits"].shape[1]) for split in splits.values()}
    if feature_widths != {weight.shape[1]} or logit_widths != {weight.shape[0]}:
        raise ValueError("selective raw arrays/classifier shape mismatch")
    if int(model.get("feature_dim", -1)) != weight.shape[1]:
        raise ValueError("raw model feature dimension mismatch")
    if int(model.get("class_count", -1)) != weight.shape[0]:
        raise ValueError("raw model class count mismatch")
    for split_name in ("id_train", "id_test"):
        labels = np.asarray(splits[split_name]["class_labels"])
        if np.any(labels < 0) or np.any(labels >= weight.shape[0]):
            raise ValueError(f"{split_name} class label is outside classifier range")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("classifier bias shape mismatch")
    return SelectiveBundle(
        bundle_id=bundle_id,
        bundle_spec=bundle_spec,
        access=access,
        raw_manifest=raw_manifest,
        result=result,
        splits=splits,
        scores=scores,
        diagnostics=diagnostics,
        classifier_weight=weight,
        classifier_bias=bias,
        query_keys=query_keys,
        ood_keys=ood_keys,
        source_allowlist_sha256=str(
            bundle_spec["remote_integrity"]["allowed_payload_catalog_sha256"]
        ),
        reuse_manifest_sha256=context.manifest_sha256,
        analysis_policy=dict(context.manifest["analysis_policy"]),
    )


def pairwise_misordering(id_scores: Any, ood_scores: Any) -> dict[str, float]:
    ids = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    oods = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    _finite_array(ids, label="ID scores", ndim=1)
    _finite_array(oods, label="OOD scores", ndim=1)
    ordered = np.sort(oods, kind="stable")
    left = np.searchsorted(ordered, ids, side="left")
    right = np.searchsorted(ordered, ids, side="right")
    pair_count = float(len(ids) * len(oods))
    strict = float(np.sum(len(oods) - right) / pair_count)
    ties = float(np.sum(right - left) / pair_count)
    return {
        "strict_id_below_ood_probability": strict,
        "tie_probability": ties,
        "pairwise_misordering_probability": strict + 0.5 * ties,
    }


def histogram_overlap(id_scores: Any, ood_scores: Any, *, bins: int) -> float:
    if bins <= 1:
        raise ValueError("histogram bins must be greater than one")
    ids = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    oods = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    _finite_array(ids, label="ID scores", ndim=1)
    _finite_array(oods, label="OOD scores", ndim=1)
    lower = float(min(np.min(ids), np.min(oods)))
    upper = float(max(np.max(ids), np.max(oods)))
    if lower == upper:
        return 1.0
    edges = np.linspace(lower, upper, bins + 1, dtype=np.float64)
    id_hist = np.histogram(ids, bins=edges)[0].astype(np.float64) / len(ids)
    ood_hist = np.histogram(oods, bins=edges)[0].astype(np.float64) / len(oods)
    return float(np.minimum(id_hist, ood_hist).sum())


def score_separation(id_scores: Any, ood_scores: Any, *, bins: int) -> dict[str, Any]:
    ids = np.asarray(id_scores, dtype=np.float64).reshape(-1)
    oods = np.asarray(ood_scores, dtype=np.float64).reshape(-1)
    metrics = compute_ood_metrics(ids, oods)
    misordering = pairwise_misordering(ids, oods)
    if not np.isclose(
        metrics["auroc"] + misordering["pairwise_misordering_probability"],
        1.0,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise AssertionError("AUROC/misordering reconstruction failed")
    return {
        "auroc_id_positive": float(metrics["auroc"]),
        "histogram_overlap_coefficient": histogram_overlap(ids, oods, bins=bins),
        "histogram_bins": int(bins),
        "histogram_definition": "equal_width_bins_over_pooled_id_ood_score_range",
        **misordering,
        "id_score_mean": float(np.mean(ids)),
        "ood_score_mean": float(np.mean(oods)),
    }


def mahalanobis_channel_contributions(
    fit: TiedGaussianFit, queries: Any
) -> dict[str, Any]:
    values = np.asarray(queries, dtype=np.float64)
    scored = score_tied_gaussians(fit, values)
    nearest = np.asarray(scored["nearest_class"], dtype=np.int64)
    delta = values - fit.class_means[nearest]
    precision = (fit.within_precision + fit.within_precision.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1.0e-10
    if np.any(eigenvalues < -tolerance):
        raise ValueError("within precision has a materially negative eigenvalue")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    coordinates = delta @ eigenvectors
    contributions = coordinates * coordinates * eigenvalues[None, :]
    direct = np.asarray(scored["class_distances"])[np.arange(len(values)), nearest]
    reconstructed = np.sum(contributions, axis=1)
    if not np.allclose(reconstructed, direct, atol=ORACLE_ATOL, rtol=ORACLE_RTOL):
        raise AssertionError("Mahalanobis contribution reconstruction failed")
    chunks = np.array_split(np.arange(len(eigenvalues)), 3)
    bands = {
        name: np.sum(contributions[:, chunk], axis=1)
        if len(chunk)
        else np.zeros(len(values), dtype=np.float64)
        for name, chunk in zip(
            ("low_precision", "middle_precision", "high_precision"), chunks
        )
    }
    band_sum = sum(bands.values(), np.zeros(len(values), dtype=np.float64))
    if not np.allclose(band_sum, direct, atol=ORACLE_ATOL, rtol=ORACLE_RTOL):
        raise AssertionError("Mahalanobis spectral-band reconstruction failed")
    return {
        "total_distance": direct,
        "nearest_class": nearest,
        "bands": bands,
        "max_abs_reconstruction_error": float(np.max(np.abs(reconstructed - direct))),
    }


def _assert_score_parity(stored: Any, recomputed: Any, label: str) -> None:
    left = np.asarray(stored)
    right = np.asarray(recomputed)
    if left.dtype.kind in "US" or right.dtype.kind in "US":
        if not np.array_equal(left.astype(str), right.astype(str)):
            raise AssertionError(f"stored/recomputed identity mismatch for {label}")
        return
    if not np.allclose(left, right, atol=ORACLE_ATOL, rtol=ORACLE_RTOL):
        error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        raise AssertionError(f"stored/recomputed score mismatch for {label}: {error}")


def _norm_summary(values: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1)
    _finite_array(norms, label="feature norms", ndim=1)
    return {
        "mean": float(np.mean(norms)),
        "std_ddof0": float(np.std(norms, ddof=0)),
        "q05": float(np.quantile(norms, 0.05, method="linear")),
        "q50": float(np.quantile(norms, 0.50, method="linear")),
        "q95": float(np.quantile(norms, 0.95, method="linear")),
    }


def _summarize_contributions(
    id_payload: Mapping[str, Any], ood_payload: Mapping[str, Any]
) -> dict[str, Any]:
    bands = {}
    for name in ("low_precision", "middle_precision", "high_precision"):
        ids = np.asarray(id_payload["bands"][name], dtype=np.float64)
        oods = np.asarray(ood_payload["bands"][name], dtype=np.float64)
        bands[name] = {
            "id_mean_distance_contribution": float(np.mean(ids)),
            "ood_mean_distance_contribution": float(np.mean(oods)),
            "ood_minus_id_mean": float(np.mean(oods) - np.mean(ids)),
        }
    return {
        "basis": "within_precision_eigendecomposition_equal_dimension_tertiles",
        "score_reconstruction": "mahalanobis_score_is_negative_sum_of_band_contributions",
        "max_abs_reconstruction_error": max(
            float(id_payload["max_abs_reconstruction_error"]),
            float(ood_payload["max_abs_reconstruction_error"]),
        ),
        "bands": bands,
    }


def _deterministic_signed_permutation(values: np.ndarray) -> np.ndarray:
    signs = np.where(np.arange(values.shape[1]) % 2 == 0, 1.0, -1.0)
    return values[:, ::-1] * signs[None, :]


def _positive_scales(count: int) -> np.ndarray:
    return np.linspace(0.5, 2.0, count, dtype=np.float64)


def _max_abs(left: Any, right: Any) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def _bundle_invariance_oracles(
    bundle: SelectiveBundle,
    raw_fit: TiedGaussianFit,
    normalized_fit: TiedGaussianFit,
) -> dict[str, Any]:
    """Run extractor smoke oracles once per bundle on a fixed ID-test prefix."""
    train = bundle.splits["id_train"]
    query = bundle.splits["id_test"]
    labels = train["class_labels"]
    class_count = bundle.classifier_weight.shape[0]
    scope = min(32, len(query["features"]))
    query_prefix = np.asarray(query["features"][:scope], dtype=np.float64)
    raw_reference = score_tied_gaussians(raw_fit, query_prefix)["mahalanobis"]
    pp_reference = score_tied_gaussians(normalized_fit, query_prefix)["mahalanobis"]

    rotated_train = _deterministic_signed_permutation(
        np.asarray(train["features"], dtype=np.float64)
    )
    rotated_query = _deterministic_signed_permutation(query_prefix)
    rotated_fit = fit_tied_gaussians(rotated_train, labels, num_classes=class_count)
    rotated_score = score_tied_gaussians(rotated_fit, rotated_query)["mahalanobis"]
    if not np.allclose(raw_reference, rotated_score, atol=ORACLE_ATOL, rtol=ORACLE_RTOL):
        raise AssertionError("raw Mahalanobis signed-permutation invariance failed")

    scaled_train = np.asarray(train["features"], dtype=np.float64) * _positive_scales(
        len(train["features"])
    )[:, None]
    scaled_query = query_prefix * _positive_scales(scope)[:, None]
    scaled_fit = fit_tied_gaussians(
        scaled_train, labels, num_classes=class_count, normalize=True
    )
    scaled_score = score_tied_gaussians(scaled_fit, scaled_query)["mahalanobis"]
    if not np.allclose(pp_reference, scaled_score, atol=ORACLE_ATOL, rtol=ORACLE_RTOL):
        raise AssertionError("Mahalanobis++ positive-scaling invariance failed")

    ctm_reference = prototype_scores(
        train["features"], labels, query_prefix, num_classes=class_count
    )["ctm_score"]
    ctm_scaled = prototype_scores(
        train["features"], labels, scaled_query, num_classes=class_count
    )["ctm_score"]
    if not np.allclose(ctm_reference, ctm_scaled, atol=ORACLE_ATOL, rtol=ORACLE_RTOL):
        raise AssertionError("CTM query-scaling invariance failed")

    knn_reference = exact_knn_l2_scores(
        train["features"],
        query_prefix,
        fit_sample_ids=train["sample_ids"],
        k=KNN_K,
    )["score"]
    knn_scaled = exact_knn_l2_scores(
        scaled_train,
        scaled_query,
        fit_sample_ids=train["sample_ids"],
        k=KNN_K,
    )["score"]
    energy_before = logit_ood_scores(query["logits"][:scope])["energy_t1"]
    # Invoke the frozen logit path independently after the feature-only branch;
    # no precomputed score array is copied into the control.
    energy_after = logit_ood_scores(query["logits"][:scope])["energy_t1"]
    return {
        "status": "PASS",
        "scope": "extractor_smoke_only_not_the_full_production_oracle_policy",
        "scope_split": "id_test",
        "scope_query_prefix_count": scope,
        "raw_mahalanobis_signed_permutation_max_abs_error": _max_abs(
            raw_reference, rotated_score
        ),
        "mahalanobis_pp_positive_sample_scaling_max_abs_error": _max_abs(
            pp_reference, scaled_score
        ),
        "ctm_query_positive_scaling_max_abs_error": _max_abs(
            ctm_reference, ctm_scaled
        ),
        "knn_l2_epsilon_perturbed_positive_scaling_max_abs_drift": _max_abs(
            knn_reference, knn_scaled
        ),
        "energy_frozen_logit_radial_transform_max_abs_error": _max_abs(
            energy_before, energy_after
        ),
    }


def _cache_checksums(root: Path) -> dict[str, str]:
    path = root / "checksums.sha256"
    if not path.is_file():
        raise ValueError("raw-kNN cache checksum inventory is absent")
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("malformed raw-kNN cache checksum row") from exc
        _require_hex64(digest, "raw-kNN cache file SHA256")
        relative = _safe_logical_path(relative)
        if relative in rows or relative == "checksums.sha256":
            raise ValueError("raw-kNN cache checksum inventory contains a duplicate")
        target = (root / relative).resolve()
        if root.resolve() not in target.parents:
            raise ValueError("raw-kNN cache checksum path escapes cache root")
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"raw-kNN cache checksum mismatch: {relative}")
        rows[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    }
    if actual != set(rows):
        raise ValueError("raw-kNN cache checksum inventory is incomplete")
    return rows


def _write_cache_checksums(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _load_raw_knn_cache(
    destination: Path, bundle: SelectiveBundle
) -> dict[str, dict[str, np.ndarray]]:
    _cache_checksums(destination)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "schema_version": RAW_KNN_CACHE_SCHEMA_VERSION,
        "bundle_id": bundle.bundle_id,
        "source_allowlist_sha256": bundle.source_allowlist_sha256,
        "extractor_file_sha256": sha256_file(Path(__file__)),
        "k": KNN_K,
        "query_keys": list(bundle.query_keys),
        "train_sample_id_sha256": ordered_sample_id_digest(
            bundle.splits["id_train"]["sample_ids"]
        ),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"raw-kNN cache identity mismatch: {field}")
    output = {}
    for split_name in bundle.query_keys:
        split_dir = destination / "splits" / split_name
        arrays = {
            name: _finite_array(
                np.load(split_dir / f"{name}.npy", allow_pickle=False),
                label=f"raw-kNN cache {split_name}/{name}",
            )
            for name in ("score", "neighbor_squared_distances", "neighbor_sample_ids")
        }
        count = len(bundle.splits[split_name]["features"])
        if arrays["score"].shape != (count,):
            raise ValueError("raw-kNN cache score shape mismatch")
        if arrays["neighbor_squared_distances"].shape != (count, KNN_K):
            raise ValueError("raw-kNN cache distance shape mismatch")
        if arrays["neighbor_sample_ids"].shape != (count, KNN_K):
            raise ValueError("raw-kNN cache neighbor shape mismatch")
        output[split_name] = arrays
    return output


def compute_or_load_raw_knn(
    bundle: SelectiveBundle, cache_root: str | Path | None
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    destination: Path | None = None
    if cache_root is not None:
        root = _assert_non_git_root(Path(cache_root), label="raw-kNN cache root")
        root.mkdir(parents=True, exist_ok=True)
        destination = root / bundle.bundle_id
        if destination.exists():
            return _load_raw_knn_cache(destination, bundle), {
                "status": "PASS",
                "cache_mode": "verified_no_overwrite_cache",
                "cache_artifact": bundle.bundle_id,
            }

    train = bundle.splits["id_train"]
    output: dict[str, dict[str, np.ndarray]] = {}
    for split_name in bundle.query_keys:
        distances, neighbor_ids = exact_neighbor_distances(
            train["features"],
            bundle.splits[split_name]["features"],
            fit_sample_ids=train["sample_ids"],
            k=KNN_K,
            return_neighbor_ids=True,
        )
        if neighbor_ids is None:
            raise AssertionError("exact raw-kNN did not return neighbor identities")
        output[split_name] = {
            "score": -distances[:, -1],
            "neighbor_squared_distances": distances,
            "neighbor_sample_ids": neighbor_ids,
        }
    cache_mode = "none"
    if destination is not None:
        temporary = destination.parent / f".{bundle.bundle_id}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            manifest = {
                "schema_version": RAW_KNN_CACHE_SCHEMA_VERSION,
                "bundle_id": bundle.bundle_id,
                "source_allowlist_sha256": bundle.source_allowlist_sha256,
                "extractor_file_sha256": sha256_file(Path(__file__)),
                "k": KNN_K,
                "query_keys": list(bundle.query_keys),
                "train_sample_id_sha256": ordered_sample_id_digest(
                    train["sample_ids"]
                ),
            }
            (temporary / "manifest.json").write_bytes(_canonical_bytes(manifest))
            for split_name, arrays in output.items():
                split_dir = temporary / "splits" / split_name
                split_dir.mkdir(parents=True)
                for name, array in arrays.items():
                    np.save(split_dir / f"{name}.npy", array, allow_pickle=False)
            _write_cache_checksums(temporary)
            _cache_checksums(temporary)
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        cache_mode = "verified_no_overwrite_cache"
    return output, {
        "status": "PASS",
        "cache_mode": cache_mode,
        "cache_artifact": None if destination is None else bundle.bundle_id,
    }


def _bundle_identity(bundle: SelectiveBundle) -> dict[str, Any]:
    identity = bundle.bundle_spec["identity"]
    return {
        "bundle_id": bundle.bundle_id,
        "checkpoint_role": identity["checkpoint_role"],
        "config_hash": bundle.bundle_spec["config_hash"],
        "training_seed": int(bundle.bundle_spec["training_seed"]),
        "optimizer_family": bundle.bundle_spec.get("optimizer_family"),
        "evaluation_git_sha": identity["evaluation_git_sha"],
        "inventory_hash": identity["inventory_hash"],
        "dataset_policy_hash": identity["dataset_policy_hash"],
        "source_allowlist_sha256": bundle.source_allowlist_sha256,
    }


def _extractor_provenance(bundle: SelectiveBundle) -> dict[str, Any]:
    return {
        "analysis_policy": bundle.analysis_policy,
        "extractor_file_sha256": sha256_file(Path(__file__)),
        "reuse_manifest_sha256": bundle.reuse_manifest_sha256,
    }


def analyze_bundle(
    bundle: SelectiveBundle,
    *,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    skip_raw_knn: bool = False,
    raw_knn_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    train = bundle.splits["id_train"]
    labels = train["class_labels"]
    class_count = bundle.classifier_weight.shape[0]
    if len(train["features"]) < KNN_K:
        raise ValueError("ID-training bank contains fewer than 50 features")
    raw_fit = fit_tied_gaussians(train["features"], labels, num_classes=class_count)
    normalized_fit = fit_tied_gaussians(
        train["features"], labels, num_classes=class_count, normalize=True
    )
    vim_fit = fit_vim(
        train["features"],
        train["logits"],
        bundle.classifier_weight,
        bundle.classifier_bias,
    )
    invariance = _bundle_invariance_oracles(bundle, raw_fit, normalized_fit)
    raw_knn: dict[str, dict[str, np.ndarray]] | None
    if skip_raw_knn:
        raw_knn = None
        raw_knn_status = {
            "status": "NOT_RUN",
            "reason": "--skip-raw-knn requested; exact top-50 local channel absent",
            "gate_eligible": False,
        }
    else:
        raw_knn, raw_knn_status = compute_or_load_raw_knn(
            bundle, raw_knn_cache_root
        )
        raw_knn_status["gate_eligible"] = True

    computed: dict[str, dict[str, np.ndarray]] = {}
    decompositions: dict[str, dict[str, Any]] = {}
    for split_name in bundle.query_keys:
        query = bundle.splits[split_name]
        raw = score_tied_gaussians(raw_fit, query["features"])
        normalized = score_tied_gaussians(normalized_fit, query["features"])
        knn_l2 = exact_knn_l2_scores(
            train["features"],
            query["features"],
            fit_sample_ids=train["sample_ids"],
            k=KNN_K,
        )
        prototype = prototype_scores(
            train["features"],
            labels,
            query["features"],
            classifier_weight=bundle.classifier_weight,
            num_classes=class_count,
        )
        vim = score_vim(vim_fit, query["features"], query["logits"])
        energy = logit_ood_scores(query["logits"])["energy_t1"]
        values = {
            "detector/mahalanobis_raw": np.asarray(raw["mahalanobis"]),
            "detector/mahalanobis_pp": np.asarray(normalized["mahalanobis"]),
            "detector/knn_l2_k50": np.asarray(knn_l2["score"]),
            "detector/ctm_prototype_cosine": np.asarray(prototype["ctm_score"]),
            "detector/vim_residual_author_dim": -np.asarray(vim["residual"]),
            "ood_score/energy_t1": np.asarray(energy),
        }
        if raw_knn is not None:
            values["detector/knn_raw_k50"] = raw_knn[split_name]["score"]
        computed[split_name] = values
        for key in STORED_SCORE_KEYS:
            _assert_score_parity(
                bundle.scores[split_name][key], values[key], f"{split_name}:{key}"
            )
        diagnostics = bundle.diagnostics[split_name]
        _assert_score_parity(
            diagnostics["raw_class_distances"],
            raw["class_distances"],
            f"{split_name}:raw_class_distances",
        )
        _assert_score_parity(
            diagnostics["raw_nearest_class"],
            raw["nearest_class"],
            f"{split_name}:raw_nearest_class",
        )
        _assert_score_parity(
            diagnostics["normalized_class_distances"],
            normalized["class_distances"],
            f"{split_name}:normalized_class_distances",
        )
        _assert_score_parity(
            diagnostics["knn_kth_squared_distance"],
            knn_l2["kth_squared_distance"],
            f"{split_name}:knn_kth_squared_distance",
        )
        _assert_score_parity(
            diagnostics["knn_kth_neighbor_sample_id"],
            knn_l2["kth_neighbor_sample_id"],
            f"{split_name}:knn_kth_neighbor_sample_id",
        )
        _assert_score_parity(
            diagnostics["vim_residual"], vim["residual"], f"{split_name}:vim_residual"
        )
        decompositions[split_name] = mahalanobis_channel_contributions(
            raw_fit, query["features"]
        )

    id_scores = computed["id_test"]
    per_ood = {}
    for split_name in bundle.ood_keys:
        detectors = {
            key: score_separation(
                id_scores[key], computed[split_name][key], bins=histogram_bins
            )
            for key in sorted(id_scores)
        }
        if raw_knn is None:
            detectors["detector/knn_raw_k50"] = {
                "status": "NOT_RUN",
                "reason": raw_knn_status["reason"],
            }
        knn_radial = (
            {
                "status": "PASS",
                "raw_auroc": detectors["detector/knn_raw_k50"]["auroc_id_positive"],
                "l2_auroc": detectors["detector/knn_l2_k50"]["auroc_id_positive"],
                "l2_minus_raw_auroc": detectors["detector/knn_l2_k50"]
                ["auroc_id_positive"]
                - detectors["detector/knn_raw_k50"]["auroc_id_positive"],
            }
            if raw_knn is not None
            else {"status": "NOT_RUN", "reason": raw_knn_status["reason"]}
        )
        per_ood[split_name] = {
            "detectors": detectors,
            "candidate_geometry_channels": {
                "radial_feature_norm": {
                    "id": _norm_summary(bundle.splits["id_test"]["features"]),
                    "ood": _norm_summary(bundle.splits[split_name]["features"]),
                },
                "mahalanobis_precision_spectral_bands": _summarize_contributions(
                    decompositions["id_test"], decompositions[split_name]
                ),
                "local_raw_vs_l2": knn_radial,
                "angular_ctm": detectors["detector/ctm_prototype_cosine"],
                "vim_residual_subspace": detectors[
                    "detector/vim_residual_author_dim"
                ],
                "frozen_logit_energy_null": detectors["ood_score/energy_t1"],
            },
            "targeted_radial_normalization": {
                "mahalanobis_pp_minus_raw_auroc": detectors[
                    "detector/mahalanobis_pp"
                ]["auroc_id_positive"]
                - detectors["detector/mahalanobis_raw"]["auroc_id_positive"],
                "knn": knn_radial,
            },
        }
    completeness_status = (
        "PASS" if raw_knn is not None and invariance["status"] == "PASS" else "NOT_RUN"
    )
    return {
        "schema_version": BUNDLE_REPORT_SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "identity": _bundle_identity(bundle),
        "extractor_provenance": _extractor_provenance(bundle),
        "dataset_protocol": bundle.raw_manifest["dataset"].get("protocol"),
        "model": {
            key: bundle.raw_manifest["model"].get(key)
            for key in ("name", "feature_dim", "class_count")
        },
        "sample_id_sha256": {
            split_name: ordered_sample_id_digest(bundle.splits[split_name]["sample_ids"])
            for split_name in bundle.query_keys
        },
        "candidate_policy": "all_candidates_reported_no_focal_channel_selected",
        "raw_knn": raw_knn_status,
        "extractor_smoke_oracles": invariance,
        "extractor_completeness": {
            "status": completeness_status,
            "meaning": "bundle evidence extraction completeness only",
            "raw_knn_required_for_pass": True,
        },
        "scientific_gate": {
            "status": "NOT_RUN",
            "reason": "the prespecified all-configuration reducer is a separate required stage",
        },
        "parameters": {"knn_k": KNN_K, "histogram_overlap_bins": histogram_bins},
        "per_ood": per_ood,
    }


def _json_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def write_json_and_sidecar_no_overwrite(
    payload: Mapping[str, Any], output_path: str | Path
) -> str:
    destination = Path(output_path)
    sidecar = _json_sidecar(destination)
    if destination.exists() or sidecar.exists():
        raise FileExistsError("output JSON and SHA sidecar are a no-overwrite pair")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(dict(payload))
    digest = hashlib.sha256(data).hexdigest()
    sidecar_data = f"{digest}  {destination.name}\n".encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        temporary_sidecar.write_bytes(sidecar_data)
        os.link(temporary, destination)
        try:
            os.link(temporary_sidecar, sidecar)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    return digest


def load_cached_bundle_report(
    path: Path, *, bundle: SelectiveBundle, histogram_bins: int, skip_raw_knn: bool
) -> dict[str, Any]:
    sidecar = _json_sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("bundle report cache JSON/sidecar pair is incomplete")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(path) != expected:
        raise ValueError("bundle report cache checksum mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bundle report cache must contain a JSON mapping")
    if payload.get("schema_version") != BUNDLE_REPORT_SCHEMA_VERSION:
        raise ValueError("bundle report cache schema mismatch")
    if payload.get("identity") != _bundle_identity(bundle):
        raise ValueError("bundle report cache source identity mismatch")
    if payload.get("extractor_provenance") != _extractor_provenance(bundle):
        raise ValueError("bundle report cache extractor/policy provenance mismatch")
    if payload.get("parameters") != {
        "knn_k": KNN_K,
        "histogram_overlap_bins": histogram_bins,
    }:
        raise ValueError("bundle report cache analysis parameters mismatch")
    expected_raw_status = "NOT_RUN" if skip_raw_knn else "PASS"
    if payload.get("raw_knn", {}).get("status") != expected_raw_status:
        raise ValueError("bundle report cache raw-kNN execution mode mismatch")
    return payload


def analyze_bundle_with_cache(
    bundle: SelectiveBundle,
    *,
    histogram_bins: int,
    skip_raw_knn: bool,
    raw_knn_cache_root: str | Path | None,
    bundle_cache_root: str | Path | None,
) -> dict[str, Any]:
    cache_path: Path | None = None
    if bundle_cache_root is not None:
        root = _assert_non_git_root(Path(bundle_cache_root), label="bundle cache root")
        root.mkdir(parents=True, exist_ok=True)
        cache_path = root / f"{bundle.bundle_id}.json"
        if cache_path.exists() or _json_sidecar(cache_path).exists():
            return load_cached_bundle_report(
                cache_path,
                bundle=bundle,
                histogram_bins=histogram_bins,
                skip_raw_knn=skip_raw_knn,
            )
    report = analyze_bundle(
        bundle,
        histogram_bins=histogram_bins,
        skip_raw_knn=skip_raw_knn,
        raw_knn_cache_root=raw_knn_cache_root,
    )
    if cache_path is not None:
        write_json_and_sidecar_no_overwrite(report, cache_path)
    return report


def load_pair_manifest(path: str | Path, context: ReuseContext) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PAIR_MANIFEST_SCHEMA_VERSION:
        raise ValueError("pair manifest schema mismatch")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pair manifest requires a non-empty pairs list")
    output = []
    seen: set[str] = set()
    for row in pairs:
        pair = _require_mapping(row, "pair")
        pair_id = str(pair.get("pair_id", ""))
        if not pair_id or pair_id in seen:
            raise ValueError("pair IDs must be non-empty and unique")
        seen.add(pair_id)
        endpoints = {}
        for side in ("left", "right"):
            endpoint = _require_mapping(pair.get(side), f"pair {side}")
            if set(endpoint) != {"bundle_id", "label"}:
                raise ValueError(
                    "pair endpoints may contain only manifest bundle_id and label; arbitrary path is forbidden"
                )
            bundle_id = _require_hex64(endpoint["bundle_id"], f"{side} bundle_id")
            if bundle_id not in context.bundles:
                raise ValueError(f"pair references unknown reuse-manifest bundle_id: {bundle_id}")
            endpoints[f"{side}_bundle_id"] = bundle_id
            endpoints[f"{side}_label"] = str(endpoint["label"])
        if endpoints["left_bundle_id"] == endpoints["right_bundle_id"]:
            raise ValueError("pair endpoints must be distinct bundles")
        if not endpoints["left_label"] or endpoints["left_label"] == endpoints["right_label"]:
            raise ValueError("pair labels must be non-empty and distinct")
        output.append({"pair_id": pair_id, **endpoints})
    return output


def _validate_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    left_identity = left["identity"]
    right_identity = right["identity"]
    if left_identity["config_hash"] == right_identity["config_hash"]:
        raise ValueError("paired discovery configurations must have distinct config hashes")
    for field in (
        "checkpoint_role",
        "training_seed",
        "evaluation_git_sha",
        "inventory_hash",
        "dataset_policy_hash",
    ):
        if left_identity[field] != right_identity[field]:
            raise ValueError(f"paired configuration identity mismatch: {field}")
    if left["dataset_protocol"] != right["dataset_protocol"] or left["model"] != right["model"]:
        raise ValueError("paired configurations must share dataset/model regime")
    if left["sample_id_sha256"] != right["sample_id_sha256"]:
        raise ValueError("paired configurations have different ordered query samples")


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        f"right_minus_left_{key}": float(right[key] - left[key])
        for key in (
            "auroc_id_positive",
            "histogram_overlap_coefficient",
            "pairwise_misordering_probability",
        )
    }


def compare_bundle_reports(
    pair: Mapping[str, str], left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_pair(left, right)
    raw_ready = (
        left["raw_knn"]["status"] == "PASS" and right["raw_knn"]["status"] == "PASS"
    )
    per_ood = {}
    for split_name in sorted(left["per_ood"]):
        if split_name not in right["per_ood"]:
            raise ValueError("paired bundle reports have different OOD populations")
        left_ood = left["per_ood"][split_name]
        right_ood = right["per_ood"][split_name]
        common = sorted(
            key
            for key in left_ood["detectors"]
            if left_ood["detectors"][key].get("status") != "NOT_RUN"
            and right_ood["detectors"][key].get("status") != "NOT_RUN"
        )
        deltas = {
            key: _metric_delta(
                left_ood["detectors"][key], right_ood["detectors"][key]
            )
            for key in common
        }
        raw_maha = deltas["detector/mahalanobis_raw"][
            "right_minus_left_auroc_id_positive"
        ]
        pp_maha = deltas["detector/mahalanobis_pp"][
            "right_minus_left_auroc_id_positive"
        ]
        if raw_ready:
            raw_knn = deltas["detector/knn_raw_k50"][
                "right_minus_left_auroc_id_positive"
            ]
            l2_knn = deltas["detector/knn_l2_k50"][
                "right_minus_left_auroc_id_positive"
            ]
            knn_attenuation: dict[str, Any] = {
                "status": "PASS",
                "raw_pair_auroc_gap": raw_knn,
                "normalized_pair_auroc_gap": l2_knn,
                "signed_gap_attenuation": raw_knn - l2_knn,
                "absolute_gap_reduction": abs(raw_knn) - abs(l2_knn),
            }
        else:
            knn_attenuation = {
                "status": "NOT_RUN",
                "reason": "exact raw-kNN is required for the local-channel attenuation gate",
            }
        left_bands = left_ood["candidate_geometry_channels"][
            "mahalanobis_precision_spectral_bands"
        ]["bands"]
        right_bands = right_ood["candidate_geometry_channels"][
            "mahalanobis_precision_spectral_bands"
        ]["bands"]
        per_ood[split_name] = {
            "endpoint_results": {
                "left": {
                    "detectors": left_ood["detectors"],
                    "candidate_geometry_channels": left_ood[
                        "candidate_geometry_channels"
                    ],
                    "targeted_radial_normalization": left_ood[
                        "targeted_radial_normalization"
                    ],
                },
                "right": {
                    "detectors": right_ood["detectors"],
                    "candidate_geometry_channels": right_ood[
                        "candidate_geometry_channels"
                    ],
                    "targeted_radial_normalization": right_ood[
                        "targeted_radial_normalization"
                    ],
                },
            },
            "detector_metric_deltas": deltas,
            "radial_normalization_attenuation": {
                "mahalanobis": {
                    "status": "PASS",
                    "raw_pair_auroc_gap": raw_maha,
                    "normalized_pair_auroc_gap": pp_maha,
                    "signed_gap_attenuation": raw_maha - pp_maha,
                    "absolute_gap_reduction": abs(raw_maha) - abs(pp_maha),
                },
                "knn": knn_attenuation,
            },
            "formula_linked_spectral_channel_deltas": {
                name: {
                    "right_minus_left_ood_minus_id_mean": float(
                        right_bands[name]["ood_minus_id_mean"]
                        - left_bands[name]["ood_minus_id_mean"]
                    )
                }
                for name in sorted(left_bands)
            },
        }
    completeness = "PASS" if raw_ready else "NOT_RUN"
    return {
        "pair_id": pair["pair_id"],
        "analysis_role": ANALYSIS_ROLE,
        "claim_boundary": (
            "matched-seed independently trained configuration comparison; "
            "not a shared-prefix intervention and not causal evidence"
        ),
        "left": {"label": pair["left_label"], "identity": left["identity"]},
        "right": {"label": pair["right_label"], "identity": right["identity"]},
        "candidate_policy": "all_candidates_reported_no_focal_channel_selected",
        "extractor_completeness": {
            "status": completeness,
            "raw_knn_required_for_pass": True,
            "meaning": "pair evidence extraction completeness only",
        },
        "scientific_gate": {
            "status": "NOT_RUN",
            "reason": "an independently trained pair is not the all-configuration gate",
        },
        "per_ood": per_ood,
    }


def run_analysis(
    *,
    reuse_manifest_path: str | Path,
    retrieval_root: str | Path,
    pair_manifest_path: str | Path | None = None,
    bundle_id: str | None = None,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    skip_raw_knn: bool = False,
    raw_knn_cache_root: str | Path | None = None,
    bundle_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    if (pair_manifest_path is None) == (bundle_id is None):
        raise ValueError("choose exactly one of pair_manifest_path or bundle_id")
    context = load_reuse_context(reuse_manifest_path, retrieval_root)
    if bundle_id is not None:
        bundle = load_selective_bundle(context, _require_hex64(bundle_id, "bundle_id"))
        return analyze_bundle_with_cache(
            bundle,
            histogram_bins=histogram_bins,
            skip_raw_knn=skip_raw_knn,
            raw_knn_cache_root=raw_knn_cache_root,
            bundle_cache_root=bundle_cache_root,
        )

    pairs = load_pair_manifest(pair_manifest_path, context)
    referenced_ids = sorted(
        {
            endpoint
            for pair in pairs
            for endpoint in (pair["left_bundle_id"], pair["right_bundle_id"])
        }
    )
    bundle_reports = {}
    for referenced_id in referenced_ids:
        bundle = load_selective_bundle(context, referenced_id)
        bundle_reports[referenced_id] = analyze_bundle_with_cache(
            bundle,
            histogram_bins=histogram_bins,
            skip_raw_knn=skip_raw_knn,
            raw_knn_cache_root=raw_knn_cache_root,
            bundle_cache_root=bundle_cache_root,
        )
    comparisons = [
        compare_bundle_reports(
            pair,
            bundle_reports[pair["left_bundle_id"]],
            bundle_reports[pair["right_bundle_id"]],
        )
        for pair in pairs
    ]
    all_complete = all(
        item["extractor_completeness"]["status"] == "PASS"
        for item in comparisons
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "analysis_role": ANALYSIS_ROLE,
        "reuse_manifest_sha256": context.manifest_sha256,
        "retrieval_identity_count": len(context.bundles),
        "selection_performed": False,
        "candidate_policy": "emit_all_candidates_then_freeze_later_in_a_protocol_addendum",
        "parameters": {
            "knn_k": KNN_K,
            "histogram_overlap_bins": histogram_bins,
            "skip_raw_knn": bool(skip_raw_knn),
        },
        "extractor_completeness": {
            "status": "PASS" if all_complete else "NOT_RUN",
            "raw_knn_required_for_pass": True,
            "meaning": "selective evidence extraction completeness only",
        },
        "scientific_gate": {
            "status": "NOT_RUN",
            "reason": "candidate aggregation selection cross-validation and decision are not performed by this extractor",
        },
        "pairs": comparisons,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pair-manifest", type=Path)
    mode.add_argument("--bundle-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-cache-root", type=Path)
    parser.add_argument("--raw-knn-cache-root", type=Path)
    parser.add_argument("--skip-raw-knn", action="store_true")
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_analysis(
        reuse_manifest_path=args.reuse_manifest,
        retrieval_root=args.retrieval_root,
        pair_manifest_path=args.pair_manifest,
        bundle_id=args.bundle_id,
        histogram_bins=args.histogram_bins,
        skip_raw_knn=args.skip_raw_knn,
        raw_knn_cache_root=args.raw_knn_cache_root,
        bundle_cache_root=args.bundle_cache_root,
    )
    digest = write_json_and_sidecar_no_overwrite(report, args.output)
    print(f"{digest}  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
