#!/usr/bin/env python3
"""Build or verify the fixed-readout Stage 2 reuse-only manifest.

This module is intentionally network-free.  It selects the frozen Metric
Contract v1.2 ``last`` population and consumes a separately collected remote
checksum catalog.  Without that catalog, retrieval readiness validation fails
and no manifest is materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "fixed_readout_stage2_reuse_manifest_v1"
CATALOG_SCHEMA_VERSION = "fixed_readout_stage2_remote_checksum_catalog_v1"
SOURCE_INVENTORY_RELATIVE_PATH = (
    "configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json"
)
DEFAULT_INVENTORY = ROOT / SOURCE_INVENTORY_RELATIVE_PATH
DEFAULT_OUTPUT = (
    ROOT / "configs/evaluation/fixed_readout_stage2/reuse_manifest.json"
)
DEFAULT_CANDIDATE_REGISTRY = (
    ROOT / "configs/evaluation/fixed_readout_stage2/candidate_registry.json"
)
DEFAULT_GATE_POLICY = ROOT / "configs/evaluation/fixed_readout_stage2/gate_policy.json"
EXPECTED_INVENTORY_FILE_SHA256 = (
    "39c89b30aecbea0984cafd1ac75cf8f8676808ee12b4728fb4fb21e845f6e7a1"
)
EXPECTED_INVENTORY_HASH = (
    "35a1a66f78dfd6255c4d804586ee6863969be78d21acec5cd0ccf00476ddaa5c"
)
EXPECTED_DATASET_POLICY_HASH = (
    "f80984d28ef9a7550633fb3aea514f783f3db1fa655a6d3f49850b10857e6afc"
)
EXPECTED_EVALUATION_GIT_SHA = "c38b09694be88aa74de0741b39e9d3ba0d6ff61a"
SOURCE_PROTECTED_AUTHORIZATION = "issue-57-protected-metric-v1.2"
HF_ROOT = "hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract_v1.2"
DESTINATION_ROOT = (
    "/home/contra333/2026여름방학실험코드/"
    "fixed_readout_stage2_reuse/fixed_readout_stage2_reuse_manifest_v1"
)
PARITY_BASELINE_ROOT = (
    "/home/contra333/2026여름방학실험코드/issue57_metric_results/"
    f"{EXPECTED_EVALUATION_GIT_SHA}/aggregate"
)
PARITY_BASELINE_REMOTE_ROOT = (
    f"{HF_ROOT}/_aggregates/{EXPECTED_INVENTORY_HASH}/"
    f"{EXPECTED_EVALUATION_GIT_SHA}"
)
PARITY_BASELINE_FILES = {
    "aggregate.json": "d9e51acedadaf532e5cd37fe06d745f962bc036d61234ce1c819de0d855e4cac",
    "artifact_manifest.json": "2fd313f751d56059b2ed7054bc62389b167dd5c17a7c61800f29c5b46b477a24",
    "detector_rank_concordance.jsonl": "3f22c203f41bbb5e2dc90d0fda1653b504a6e8dbfc8af23d89f39f06f2234be1",
    "per_checkpoint_records.jsonl": "c845ba8734d03ba170a48da121fec7a4fd0ccbbebb5e54c3442bc32156ad85cc",
    "seed_aggregates.jsonl": "8d0891051fc719455364602652932a8cadd2308552ec266963cf9e74b54197d2",
}
PARITY_BASELINE_FILE_SIZES = {
    "aggregate.json": 33_594,
    "artifact_manifest.json": 786,
    "detector_rank_concordance.jsonl": 30_210,
    "per_checkpoint_records.jsonl": 104_646_790,
    "seed_aggregates.jsonl": 54_097_229,
}

QUERY_KEYS = (
    "id_test",
    "cifar100",
    "tin",
    "mnist",
    "svhn",
    "texture",
    "places365",
)
RAW_ROOTS = {
    "id_train": "cifar10/train",
    "id_test": "cifar10/test",
    "cifar100": "cifar100/test",
    "tin": "tin/test",
    "mnist": "mnist/test",
    "svhn": "svhn/test",
    "texture": "texture/test",
    "places365": "places365/test",
}

METADATA_PATHS = (
    "JOB_COMPLETE.json",
    "REMOTE_COMPLETE.json",
    "artifact_manifest.json",
    "bundle_checksums.sha256",
    "checksums.sha256",
    "classifier_bias.npy",
    "classifier_weight.npy",
    "manifest.json",
    "metrics/METRICS_COMPLETE.json",
    "metrics/checksums.sha256",
    "metrics/result.json",
)
DIRECT_SCORE_FILENAMES = (
    "detector__ctm_prototype_cosine.npy",
    "detector__knn_l2_k50.npy",
    "detector__mahalanobis_pp.npy",
    "detector__mahalanobis_raw.npy",
    "detector__relative_mahalanobis_pp.npy",
    "detector__relative_mahalanobis_raw.npy",
    "detector__vim_author_dim.npy",
    "ood_score__energy_t1.npy",
    "ood_score__msp.npy",
)
DIAGNOSTIC_SCORE_PATHS = (
    "diagnostics/knn/kth_neighbor_sample_id.npy",
    "diagnostics/knn/kth_squared_distance.npy",
    "diagnostics/normalized_gaussian/class_distances.npy",
    "diagnostics/normalized_gaussian/global_distances.npy",
    "diagnostics/normalized_gaussian/marginal_mahalanobis.npy",
    "diagnostics/normalized_gaussian/nearest_class.npy",
    "diagnostics/raw_gaussian/class_distances.npy",
    "diagnostics/raw_gaussian/global_distances.npy",
    "diagnostics/raw_gaussian/nearest_class.npy",
    "diagnostics/vim/alpha_residual.npy",
    "diagnostics/vim/energy.npy",
    "diagnostics/vim/residual.npy",
)

AUTHORIZATION = {
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
    "source_artifact_authorization": SOURCE_PROTECTED_AUTHORIZATION,
}

DESTINATION = {
    "create_automatically": False,
    "existing_file_policy": "verified_exact_resume_only",
    "kind": "absolute_non_git_directory",
    "mismatched_or_unexpected_existing_files": "reject",
    "must_be_outside_git_worktree": True,
    "root": DESTINATION_ROOT,
    "verified_existing_files": "skip_without_overwrite",
}

PARITY_BASELINE = {
    "files": [
        {
            "local_path": f"{PARITY_BASELINE_ROOT}/{name}",
            "logical_path": name,
            "remote_uri": f"{PARITY_BASELINE_REMOTE_ROOT}/{name}",
            "sha256": digest,
            "size": PARITY_BASELINE_FILE_SIZES[name],
        }
        for name, digest in sorted(PARITY_BASELINE_FILES.items())
    ],
    "metric_direction": "id_like_higher_is_id",
    "required_metrics": ["auroc", "fpr95"],
    "role": "completed_metric_contract_v1.2_scalar_parity_baseline",
}

_HEX64 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_hex64(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _require_safe_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("catalog logical_path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"unsafe catalog logical_path: {value!r}")
    return value


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"input must be inside the repository: {path}") from exc


def _required_logical_paths() -> list[str]:
    paths = set(METADATA_PATHS)
    for split_key, physical_root in RAW_ROOTS.items():
        paths.add(f"{physical_root}/features.npy")
        paths.add(f"{physical_root}/logits.npy")
        paths.add(f"{physical_root}/sample_ids.npy")
        if split_key in {"id_train", "id_test"}:
            paths.add(f"{physical_root}/class_labels.npy")
    for query_key in QUERY_KEYS:
        score_root = f"metrics/arrays/scores/{query_key}"
        for filename in DIRECT_SCORE_FILENAMES:
            paths.add(f"{score_root}/{filename}")
        for relative in DIAGNOSTIC_SCORE_PATHS:
            paths.add(f"{score_root}/{relative}")
    return sorted(paths)


def _path_contract() -> dict[str, Any]:
    return {
        "exact_metadata_paths": sorted(METADATA_PATHS),
        "patterns": [
            {
                "bindings": {"physical_root": sorted(RAW_ROOTS.values())},
                "name": "raw_feature",
                "template": "{physical_root}/features.npy",
            },
            {
                "bindings": {"physical_root": sorted(RAW_ROOTS.values())},
                "name": "frozen_logit",
                "template": "{physical_root}/logits.npy",
            },
            {
                "bindings": {"physical_root": sorted(RAW_ROOTS.values())},
                "name": "sample_identity",
                "template": "{physical_root}/sample_ids.npy",
            },
            {
                "bindings": {
                    "physical_root": [RAW_ROOTS["id_train"], RAW_ROOTS["id_test"]]
                },
                "name": "id_class_label",
                "template": "{physical_root}/class_labels.npy",
            },
            {
                "bindings": {
                    "filename": sorted(DIRECT_SCORE_FILENAMES),
                    "query_key": list(QUERY_KEYS),
                },
                "name": "approved_detector_score",
                "template": "metrics/arrays/scores/{query_key}/{filename}",
            },
            {
                "bindings": {
                    "diagnostic_path": sorted(DIAGNOSTIC_SCORE_PATHS),
                    "query_key": list(QUERY_KEYS),
                },
                "name": "score_formula_diagnostic",
                "template": (
                    "metrics/arrays/scores/{query_key}/{diagnostic_path}"
                ),
            },
        ],
        "query_keys": list(QUERY_KEYS),
        "required_logical_paths": _required_logical_paths(),
    }


def _analysis_policy_bindings(repository_root: Path) -> dict[str, Any]:
    bindings = {}
    for name, path, expected_id in (
        (
            "candidate_registry",
            DEFAULT_CANDIDATE_REGISTRY,
            "fixed_readout_stage2_candidates_v1",
        ),
        ("gate_policy", DEFAULT_GATE_POLICY, "fixed_readout_stage2_gate_v1"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed_id = payload.get("registry_id", payload.get("policy_id"))
        if observed_id != expected_id:
            raise ValueError(f"Stage-2 {name} identity mismatch")
        bindings[name] = {
            "id": expected_id,
            "path": _repository_relative(path, repository_root),
            "sha256": sha256_file(path),
        }
    return bindings


def _load_frozen_inventory(
    inventory_path: Path, repository_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed_sha256 = sha256_file(inventory_path)
    if observed_sha256 != EXPECTED_INVENTORY_FILE_SHA256:
        raise ValueError("frozen source inventory file SHA-256 mismatch")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("inventory_hash") != EXPECTED_INVENTORY_HASH:
        raise ValueError("frozen source inventory internal hash mismatch")
    if inventory.get("dataset_policy_hash") != EXPECTED_DATASET_POLICY_HASH:
        raise ValueError("frozen source inventory dataset-policy hash mismatch")
    sidecar = inventory_path.with_name(f"{inventory_path.name}.sha256")
    if sidecar.is_file() and sidecar.read_text(encoding="utf-8").split()[0] != observed_sha256:
        raise ValueError("frozen source inventory SHA-256 sidecar mismatch")
    source = {
        "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
        "inventory_hash": EXPECTED_INVENTORY_HASH,
        "path": _repository_relative(inventory_path, repository_root),
        "sha256": observed_sha256,
    }
    return inventory, source


def _select_jobs(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        job
        for job in inventory.get("checkpoint_jobs", [])
        if job.get("checkpoint_role") == "last"
        and job.get("reporting_role") == "confirmatory_primary"
    ]
    jobs.sort(
        key=lambda job: (
            str(job["optimizer_family"]),
            str(job["config_hash"]),
            int(job["training_seed"]),
            str(job["checkpoint_sha256"]),
        )
    )
    if len(jobs) != 30:
        raise ValueError("frozen inventory must yield exactly 30 primary last jobs")
    checkpoint_shas = [
        _require_hex64(job.get("checkpoint_sha256"), "checkpoint_sha256")
        for job in jobs
    ]
    if len(set(checkpoint_shas)) != 30:
        raise ValueError("selected primary last checkpoint SHA-256 values must be unique")
    if {int(job["training_seed"]) for job in jobs} != {0, 1, 2}:
        raise ValueError("selected primary last population must contain seeds 0, 1, and 2")
    if len({str(job["config_hash"]) for job in jobs}) != 10:
        raise ValueError("selected primary last population must contain 10 configurations")
    for job in jobs:
        expected_uri = f"{HF_ROOT}/{job['checkpoint_sha256']}"
        if job.get("hf_output_uri") != expected_uri:
            raise ValueError("selected job HF output URI differs from its checkpoint identity")
    return [deepcopy(job) for job in jobs]


def _expected_artifact_identity(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_role": "checkpoint_metric_contract_v1.2_bundle",
        "checkpoint_role": "last",
        "checkpoint_sha256": job["checkpoint_sha256"],
        "config_hash": job["config_hash"],
        "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
        "evaluation_git_sha": EXPECTED_EVALUATION_GIT_SHA,
        "inventory_hash": EXPECTED_INVENTORY_HASH,
        "protected_authorization": SOURCE_PROTECTED_AUTHORIZATION,
        "reporting_role": "confirmatory_primary",
        "status": "LOCAL_COMPLETE",
        "training_seed": int(job["training_seed"]),
    }


def _load_catalog(
    catalog_path: Path,
    selected_jobs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    catalog_sha256 = sha256_file(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("remote checksum catalog schema version mismatch")
    if catalog.get("hf_root") != HF_ROOT:
        raise ValueError("remote checksum catalog HF root mismatch")
    if catalog.get("source_inventory_sha256") != EXPECTED_INVENTORY_FILE_SHA256:
        raise ValueError("remote checksum catalog source inventory mismatch")
    rows = catalog.get("bundles")
    if not isinstance(rows, list):
        raise ValueError("remote checksum catalog bundles must be a list")
    by_sha: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("remote checksum catalog bundle must be an object")
        checkpoint_sha = _require_hex64(
            row.get("checkpoint_sha256"), "catalog checkpoint_sha256"
        )
        if checkpoint_sha in by_sha:
            raise ValueError("remote checksum catalog has duplicate checkpoint bundle")
        by_sha[checkpoint_sha] = row
    expected_shas = {str(job["checkpoint_sha256"]) for job in selected_jobs}
    if set(by_sha) != expected_shas:
        raise ValueError("remote checksum catalog must contain exactly the 30 selected bundles")
    return (
        {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "sha256": catalog_sha256,
            "status": "complete",
        },
        by_sha,
    )


def _catalog_integrity_for_job(
    row: Mapping[str, Any],
    job: Mapping[str, Any],
    required_paths: Sequence[str],
) -> dict[str, Any]:
    if row.get("hf_uri") != job["hf_output_uri"]:
        raise ValueError("remote checksum catalog bundle HF URI mismatch")
    expected_identity = _expected_artifact_identity(job)
    if row.get("artifact_identity") != expected_identity:
        raise ValueError("remote checksum catalog artifact identity mismatch")
    artifact_manifest_sha256 = _require_hex64(
        row.get("artifact_manifest_file_sha256"),
        "artifact_manifest_file_sha256",
    )
    bundle_checksums_sha256 = _require_hex64(
        row.get("bundle_checksums_file_sha256"),
        "bundle_checksums_file_sha256",
    )
    file_rows = row.get("files")
    if not isinstance(file_rows, list):
        raise ValueError("remote checksum catalog files must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for file_row in file_rows:
        if not isinstance(file_row, dict):
            raise ValueError("remote checksum catalog file must be an object")
        logical_path = _require_safe_logical_path(file_row.get("logical_path"))
        if logical_path in by_path:
            raise ValueError("remote checksum catalog has a duplicate logical path")
        digest = _require_hex64(file_row.get("sha256"), f"sha256 for {logical_path}")
        size = file_row.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"size for {logical_path} must be a positive integer")
        by_path[logical_path] = {
            "logical_path": logical_path,
            "sha256": digest,
            "size": size,
        }
    missing = sorted(set(required_paths) - set(by_path))
    if missing:
        raise ValueError(
            "remote checksum catalog is missing required allowlisted files: "
            + ", ".join(missing[:3])
        )
    if by_path["artifact_manifest.json"]["sha256"] != artifact_manifest_sha256:
        raise ValueError("artifact manifest SHA-256 disagrees with catalog file row")
    if by_path["bundle_checksums.sha256"]["sha256"] != bundle_checksums_sha256:
        raise ValueError("bundle checksum-file SHA-256 disagrees with catalog file row")
    selected_files = [by_path[path] for path in sorted(required_paths)]
    digest_payload = {
        "artifact_manifest_file_sha256": artifact_manifest_sha256,
        "bundle_checksums_file_sha256": bundle_checksums_sha256,
        "files": selected_files,
    }
    return {
        **digest_payload,
        "allowed_payload_catalog_sha256": _canonical_sha256(digest_payload),
        "catalog_file_count": len(file_rows),
        "selected_file_count": len(selected_files),
        "selected_total_size": sum(item["size"] for item in selected_files),
    }


def build_reuse_manifest(
    *,
    inventory_path: str | Path = DEFAULT_INVENTORY,
    checksum_catalog_path: str | Path | None = None,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    repository = Path(repository_root)
    inventory, source_inventory = _load_frozen_inventory(
        Path(inventory_path), repository
    )
    selected_jobs = _select_jobs(inventory)
    path_contract = _path_contract()
    required_paths = path_contract["required_logical_paths"]
    catalog_metadata: dict[str, Any]
    catalog_by_sha: dict[str, dict[str, Any]] | None
    if checksum_catalog_path is None:
        catalog_metadata = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "sha256": None,
            "status": "missing",
        }
        catalog_by_sha = None
    else:
        catalog_metadata, catalog_by_sha = _load_catalog(
            Path(checksum_catalog_path), selected_jobs
        )

    bundles = []
    for job in selected_jobs:
        checkpoint_sha = str(job["checkpoint_sha256"])
        remote_integrity = None
        if catalog_by_sha is not None:
            remote_integrity = _catalog_integrity_for_job(
                catalog_by_sha[checkpoint_sha], job, required_paths
            )
        bundles.append(
            {
                "assigned_host": job["assigned_host"],
                "bundle_id": checkpoint_sha,
                "config_hash": job["config_hash"],
                "hf_uri": job["hf_output_uri"],
                "identity": _expected_artifact_identity(job),
                "job_id": job["job_id"],
                "labels": list(job["labels"]),
                "optimizer_family": job["optimizer_family"],
                "remote_integrity": remote_integrity,
                "training_identity_id": job["training_identity_id"],
                "training_seed": int(job["training_seed"]),
            }
        )

    return {
        "analysis_policy": _analysis_policy_bindings(repository),
        "authorization": deepcopy(AUTHORIZATION),
        "bundles": bundles,
        "destination": deepcopy(DESTINATION),
        "hf_root": HF_ROOT,
        "path_contract": path_contract,
        "parity_baseline": deepcopy(PARITY_BASELINE),
        "purpose": (
            "Reuse existing Metric Contract v1.2 artifacts for the fixed-readout "
            "Stage 2 mechanism discovery gate."
        ),
        "remote_checksum_catalog": catalog_metadata,
        "retrieval_ready": catalog_by_sha is not None,
        "schema_version": SCHEMA_VERSION,
        "selection": {
            "checkpoint_role": "last",
            "configuration_count": 10,
            "job_count": 30,
            "reporting_role": "confirmatory_primary",
            "seeds": [0, 1, 2],
        },
        "source_inventory": source_inventory,
    }


def validate_reuse_manifest(
    manifest: Mapping[str, Any], *, require_remote_catalog: bool = True
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("reuse manifest schema version mismatch")
    if manifest.get("hf_root") != HF_ROOT:
        raise ValueError("reuse manifest HF root mismatch")
    if manifest.get("authorization") != AUTHORIZATION:
        raise ValueError("reuse manifest authorization policy mismatch")
    if manifest.get("analysis_policy") != _analysis_policy_bindings(ROOT):
        raise ValueError("reuse manifest Stage-2 analysis policy mismatch")
    if manifest.get("destination") != DESTINATION:
        raise ValueError("reuse manifest destination policy mismatch")
    if manifest.get("path_contract") != _path_contract():
        raise ValueError("reuse manifest logical path contract mismatch")
    if manifest.get("parity_baseline") != PARITY_BASELINE:
        raise ValueError("reuse manifest parity baseline mismatch")
    if manifest.get("source_inventory") != {
        "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
        "inventory_hash": EXPECTED_INVENTORY_HASH,
        "path": SOURCE_INVENTORY_RELATIVE_PATH,
        "sha256": EXPECTED_INVENTORY_FILE_SHA256,
    }:
        raise ValueError("reuse manifest source inventory identity mismatch")
    if manifest.get("selection") != {
        "checkpoint_role": "last",
        "configuration_count": 10,
        "job_count": 30,
        "reporting_role": "confirmatory_primary",
        "seeds": [0, 1, 2],
    }:
        raise ValueError("reuse manifest selection contract mismatch")

    catalog = manifest.get("remote_checksum_catalog")
    if not isinstance(catalog, Mapping):
        raise ValueError("reuse manifest remote checksum catalog metadata is absent")
    catalog_complete = catalog.get("status") == "complete"
    if require_remote_catalog and not catalog_complete:
        raise ValueError(
            "remote checksum metadata catalog is required; unsafe bulk fetch is forbidden"
        )
    if manifest.get("retrieval_ready") is not catalog_complete:
        raise ValueError("reuse manifest retrieval readiness/catalog status mismatch")
    if catalog_complete:
        if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
            raise ValueError("reuse manifest catalog schema mismatch")
        _require_hex64(catalog.get("sha256"), "remote checksum catalog sha256")

    bundles = manifest.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 30:
        raise ValueError("reuse manifest must contain exactly 30 bundles")
    frozen_inventory, _ = _load_frozen_inventory(DEFAULT_INVENTORY, ROOT)
    expected_jobs = _select_jobs(frozen_inventory)
    expected_checkpoint_order = [job["checkpoint_sha256"] for job in expected_jobs]
    if [bundle.get("bundle_id") for bundle in bundles] != expected_checkpoint_order:
        raise ValueError("reuse manifest bundle order/population differs from frozen selection")
    required_paths = _required_logical_paths()
    checkpoint_shas = set()
    identities = set()
    for bundle, expected_job in zip(bundles, expected_jobs, strict=True):
        if not isinstance(bundle, Mapping):
            raise ValueError("reuse manifest bundle must be an object")
        checkpoint_sha = _require_hex64(bundle.get("bundle_id"), "bundle_id")
        if checkpoint_sha in checkpoint_shas:
            raise ValueError("reuse manifest contains a duplicate bundle")
        checkpoint_shas.add(checkpoint_sha)
        if bundle.get("hf_uri") != f"{HF_ROOT}/{checkpoint_sha}":
            raise ValueError("reuse manifest bundle HF URI mismatch")
        expected_bundle_fields = {
            "assigned_host": expected_job["assigned_host"],
            "bundle_id": checkpoint_sha,
            "config_hash": expected_job["config_hash"],
            "hf_uri": expected_job["hf_output_uri"],
            "job_id": expected_job["job_id"],
            "labels": list(expected_job["labels"]),
            "optimizer_family": expected_job["optimizer_family"],
            "training_identity_id": expected_job["training_identity_id"],
            "training_seed": int(expected_job["training_seed"]),
        }
        for field, expected_value in expected_bundle_fields.items():
            if bundle.get(field) != expected_value:
                raise ValueError(f"reuse manifest bundle {field} differs from frozen job")
        identity = bundle.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("reuse manifest artifact identity is absent")
        expected_identity = {
            "artifact_role": "checkpoint_metric_contract_v1.2_bundle",
            "checkpoint_role": "last",
            "checkpoint_sha256": checkpoint_sha,
            "config_hash": bundle.get("config_hash"),
            "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
            "evaluation_git_sha": EXPECTED_EVALUATION_GIT_SHA,
            "inventory_hash": EXPECTED_INVENTORY_HASH,
            "protected_authorization": SOURCE_PROTECTED_AUTHORIZATION,
            "reporting_role": "confirmatory_primary",
            "status": "LOCAL_COMPLETE",
            "training_seed": bundle.get("training_seed"),
        }
        if identity != expected_identity:
            raise ValueError("reuse manifest bundle artifact identity mismatch")
        identity_key = (bundle.get("config_hash"), bundle.get("training_seed"))
        if identity_key in identities:
            raise ValueError("reuse manifest contains a duplicate training identity")
        identities.add(identity_key)
        integrity = bundle.get("remote_integrity")
        if catalog_complete:
            if not isinstance(integrity, Mapping):
                raise ValueError("catalog-complete bundle lacks remote integrity")
            artifact_sha = _require_hex64(
                integrity.get("artifact_manifest_file_sha256"),
                "artifact_manifest_file_sha256",
            )
            checksum_sha = _require_hex64(
                integrity.get("bundle_checksums_file_sha256"),
                "bundle_checksums_file_sha256",
            )
            files = integrity.get("files")
            if not isinstance(files, list):
                raise ValueError("bundle remote integrity files must be a list")
            if [item.get("logical_path") for item in files] != required_paths:
                raise ValueError("bundle remote integrity path allowlist mismatch")
            for item in files:
                _require_hex64(item.get("sha256"), "allowlisted file sha256")
                size = item.get("size")
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise ValueError("allowlisted file size must be a positive integer")
            by_path = {item["logical_path"]: item for item in files}
            if by_path["artifact_manifest.json"]["sha256"] != artifact_sha:
                raise ValueError("artifact manifest SHA-256 mismatch in bundle integrity")
            if by_path["bundle_checksums.sha256"]["sha256"] != checksum_sha:
                raise ValueError("bundle checksum SHA-256 mismatch in bundle integrity")
            digest_payload = {
                "artifact_manifest_file_sha256": artifact_sha,
                "bundle_checksums_file_sha256": checksum_sha,
                "files": files,
            }
            if integrity.get("allowed_payload_catalog_sha256") != _canonical_sha256(
                digest_payload
            ):
                raise ValueError("allowed payload catalog digest mismatch")
            if integrity.get("selected_file_count") != len(required_paths):
                raise ValueError("selected file count mismatch")
            if integrity.get("selected_total_size") != sum(
                int(item["size"]) for item in files
            ):
                raise ValueError("selected total size mismatch")
        elif integrity is not None:
            raise ValueError("catalog-missing bundle must not claim remote integrity")
    if len(identities) != 30:
        raise ValueError("reuse manifest training identities must be unique")


def render_reuse_manifest(manifest: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(dict(manifest))


def materialize_reuse_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> str:
    validate_reuse_manifest(manifest, require_remote_catalog=True)
    output = Path(output_path)
    payload = render_reuse_manifest(manifest)
    with output.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def verify_materialized_reuse_manifest(
    manifest: Mapping[str, Any], output_path: str | Path
) -> str:
    validate_reuse_manifest(manifest, require_remote_catalog=True)
    output = Path(output_path)
    expected = render_reuse_manifest(manifest)
    if not output.is_file():
        raise FileNotFoundError(f"reuse manifest is absent: {output}")
    observed = output.read_bytes()
    if observed != expected:
        raise ValueError("materialized reuse manifest is not byte-identical")
    return hashlib.sha256(observed).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the fixed-readout Stage 2 reuse manifest."
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--checksum-catalog", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_reuse_manifest(
        inventory_path=args.inventory,
        checksum_catalog_path=args.checksum_catalog,
    )
    if args.check:
        digest = verify_materialized_reuse_manifest(manifest, args.output)
    else:
        digest = materialize_reuse_manifest(manifest, args.output)
    print(f"{digest}  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
