#!/usr/bin/env python3
"""Collect a verified Stage 2 remote checksum catalog from local evidence.

The collector is deliberately network-free.  It combines a metadata-only
download of the selected remote bundles with a saved ``hf buckets sync`` plan:
the downloaded metadata supplies authoritative checksums, while the plan
supplies remote sizes for payload files that are intentionally not downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from scripts.build_fixed_readout_stage2_reuse_manifest import (
    CATALOG_SCHEMA_VERSION,
    DEFAULT_INVENTORY,
    EXPECTED_DATASET_POLICY_HASH,
    EXPECTED_EVALUATION_GIT_SHA,
    EXPECTED_INVENTORY_FILE_SHA256,
    EXPECTED_INVENTORY_HASH,
    HF_ROOT,
    ROOT,
    SOURCE_PROTECTED_AUTHORIZATION,
    _expected_artifact_identity,
    _load_frozen_inventory,
    _required_logical_paths,
    _select_jobs,
    sha256_file,
)


DEFINITION_VERSION = "metric_contract_v1.2"
METADATA_LOGICAL_PATHS = (
    "JOB_COMPLETE.json",
    "REMOTE_COMPLETE.json",
    "artifact_manifest.json",
    "bundle_checksums.sha256",
    "manifest.json",
    "metrics/METRICS_COMPLETE.json",
    "metrics/result.json",
)
UNCHECKSUMMED_REMOTE_PATHS = frozenset(
    {"REMOTE_COMPLETE.json", "bundle_checksums.sha256"}
)

_HEX64 = re.compile(r"[0-9a-f]{64}")
_CHECKSUM_ROW = re.compile(r"([0-9a-f]{64})  (.+)")


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


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_safe_posix_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty normalized POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"unsafe or non-normalized {field}: {value!r}")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _metadata_path(bundle_root: Path, logical_path: str) -> Path:
    relative = PurePosixPath(logical_path)
    path = bundle_root.joinpath(*relative.parts)
    if not path.is_file():
        raise ValueError(f"downloaded metadata is missing: {logical_path}")
    if path.is_symlink():
        raise ValueError(f"downloaded metadata must not be a symlink: {logical_path}")
    try:
        path.resolve().relative_to(bundle_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"downloaded metadata escapes its checkpoint bundle: {logical_path}"
        ) from exc
    return path


def _parse_checksum_file(path: Path, *, label: str) -> dict[str, str]:
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not rows:
        raise ValueError(f"{label} must not be empty")
    parsed: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=1):
        match = _CHECKSUM_ROW.fullmatch(row)
        if match is None:
            raise ValueError(f"malformed {label} row {line_number}")
        digest, logical_path = match.groups()
        logical_path = _require_safe_posix_path(
            logical_path, f"{label} logical path"
        )
        if logical_path in parsed:
            raise ValueError(f"duplicate path in {label}: {logical_path}")
        parsed[logical_path] = digest
    return parsed


def _load_sync_plan(
    path: Path,
    *,
    checkpoint_shas: Sequence[str],
    required_paths: Sequence[str],
) -> dict[tuple[str, str], int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read HF sync plan: {path}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("HF sync plan must be non-empty JSONL without blank rows")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed HF sync plan row {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"HF sync plan row {line_number} must be an object")
        rows.append(row)

    header = rows[0]
    if header.get("type") != "header":
        raise ValueError("HF sync plan first row must be its header")
    if header.get("source") != HF_ROOT:
        raise ValueError("HF sync plan source does not match the frozen HF root")
    if any(row.get("type") == "header" for row in rows[1:]):
        raise ValueError("HF sync plan must contain exactly one header")
    summary = header.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("HF sync plan header lacks a summary")
    uploads = _require_nonnegative_int(summary.get("uploads"), "plan uploads")
    deletes = _require_nonnegative_int(summary.get("deletes"), "plan deletes")
    if uploads or deletes:
        raise ValueError("HF sync plan must not upload or delete remote data")

    selected_shas = set(checkpoint_shas)
    required = set(required_paths)
    sizes: dict[tuple[str, str], int] = {}
    remote_paths: set[str] = set()
    actions = {"download": 0, "skip": 0}
    download_size = 0
    for line_number, row in enumerate(rows[1:], start=2):
        if row.get("type") != "operation":
            raise ValueError(f"HF sync plan row {line_number} is not an operation")
        action = row.get("action")
        if action not in actions:
            raise ValueError(
                f"HF sync plan row {line_number} has a non-download action"
            )
        actions[action] += 1
        remote_path = _require_safe_posix_path(
            row.get("path"), "HF sync plan remote path"
        )
        if remote_path in remote_paths:
            raise ValueError(f"HF sync plan contains a duplicate remote path: {remote_path}")
        remote_paths.add(remote_path)
        size = _require_positive_int(
            row.get("size"), f"HF sync plan size for {remote_path}"
        )
        if action == "download":
            download_size += size
        parts = PurePosixPath(remote_path).parts
        if len(parts) >= 2 and parts[0] in selected_shas:
            checkpoint_sha = parts[0]
            logical_path = PurePosixPath(*parts[1:]).as_posix()
            if logical_path in required:
                sizes[(checkpoint_sha, logical_path)] = size

    for field, action in (("downloads", "download"), ("skips", "skip")):
        observed = _require_nonnegative_int(summary.get(field), f"plan {field}")
        if observed != actions[action]:
            raise ValueError(f"HF sync plan {field} summary does not match operations")
    total_size = _require_nonnegative_int(
        summary.get("total_size"), "plan total_size"
    )
    if total_size != download_size:
        raise ValueError("HF sync plan total_size does not match download operations")

    expected = {
        (checkpoint_sha, logical_path)
        for checkpoint_sha in checkpoint_shas
        for logical_path in required_paths
    }
    if set(sizes) != expected:
        missing = sorted(expected - set(sizes))
        raise ValueError(
            "HF sync plan misses selected allowlisted paths: "
            + f"missing={missing[:3]}"
        )
    return sizes


def _expect_fields(
    observed: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    mismatched = [key for key, value in expected.items() if observed.get(key) != value]
    if mismatched:
        raise ValueError(f"{label} identity mismatch: {sorted(mismatched)}")


def _validate_artifact_identity(
    *,
    job: Mapping[str, Any],
    paths: Mapping[str, Path],
    bundle_checksums: Mapping[str, str],
) -> dict[str, Any]:
    artifact = _load_json_object(paths["artifact_manifest.json"], "artifact manifest")
    _expect_fields(artifact, _expected_artifact_identity(job), "artifact manifest")

    completion = _load_json_object(paths["JOB_COMPLETE.json"], "job completion")
    _expect_fields(
        completion,
        {
            "authorization": SOURCE_PROTECTED_AUTHORIZATION,
            "checkpoint_role": "last",
            "checkpoint_sha256": job["checkpoint_sha256"],
            "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
            "definition_version": DEFINITION_VERSION,
            "evaluation_git_sha": EXPECTED_EVALUATION_GIT_SHA,
            "inventory_hash": EXPECTED_INVENTORY_HASH,
            "job_id": job["job_id"],
            "status": "LOCAL_COMPLETE",
        },
        "job completion",
    )
    _require_positive_int(
        completion.get("metric_file_count"), "job completion metric_file_count"
    )
    _require_positive_int(
        completion.get("scalar_record_count"), "job completion scalar_record_count"
    )

    raw = _load_json_object(paths["manifest.json"], "raw feature manifest")
    _expect_fields(
        raw,
        {
            "artifact_role": "raw_checkpoint_feature_cache",
            "definition_version": DEFINITION_VERSION,
        },
        "raw feature manifest",
    )
    checkpoint = raw.get("checkpoint")
    evaluation = raw.get("evaluation")
    if not isinstance(checkpoint, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("raw feature manifest lacks checkpoint/evaluation identity")
    _expect_fields(
        checkpoint,
        {
            "config_hash": job["config_hash"],
            "role": "last",
            "sha256": job["checkpoint_sha256"],
            "training_seed": int(job["training_seed"]),
        },
        "raw feature checkpoint",
    )
    _expect_fields(
        evaluation,
        {
            "git_dirty": False,
            "oge_git_sha": EXPECTED_EVALUATION_GIT_SHA,
            "protected_split_authorization": SOURCE_PROTECTED_AUTHORIZATION,
            "smoke_only": False,
        },
        "raw feature evaluation",
    )

    result = _load_json_object(paths["metrics/result.json"], "metric result")
    _expect_fields(
        result,
        {
            "authorization": SOURCE_PROTECTED_AUTHORIZATION,
            "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
            "definition_version": DEFINITION_VERSION,
            "evaluation_git_sha": EXPECTED_EVALUATION_GIT_SHA,
            "inventory_hash": EXPECTED_INVENTORY_HASH,
            "protected_result": True,
            "smoke_only": False,
        },
        "metric result",
    )
    result_job = result.get("job")
    if not isinstance(result_job, Mapping):
        raise ValueError("metric result lacks its frozen job identity")
    _expect_fields(
        result_job,
        {
            "checkpoint_role": "last",
            "checkpoint_sha256": job["checkpoint_sha256"],
            "config_hash": job["config_hash"],
            "job_id": job["job_id"],
            "reporting_role": "confirmatory_primary",
            "training_seed": int(job["training_seed"]),
        },
        "metric result job",
    )

    metrics_completion = _load_json_object(
        paths["metrics/METRICS_COMPLETE.json"], "metric completion"
    )
    _expect_fields(
        metrics_completion,
        {
            "checkpoint_sha256": job["checkpoint_sha256"],
            "status": "METRICS_COMPLETE",
        },
        "metric completion",
    )
    _require_positive_int(
        metrics_completion.get("scalar_record_count"),
        "metric completion scalar_record_count",
    )

    artifact_sha = sha256_file(paths["artifact_manifest.json"])
    raw_sha = sha256_file(paths["manifest.json"])
    result_sha = sha256_file(paths["metrics/result.json"])
    if artifact.get("raw_manifest_sha256") != raw_sha:
        raise ValueError("artifact manifest raw-manifest SHA-256 mismatch")
    if artifact.get("metric_result_sha256") != result_sha:
        raise ValueError("artifact manifest metric-result SHA-256 mismatch")
    scalar_sha = _require_hex64(
        artifact.get("scalar_records_sha256"),
        "artifact manifest scalar_records_sha256",
    )
    if bundle_checksums.get("metrics/scalar_records.jsonl") != scalar_sha:
        raise ValueError("artifact manifest scalar-record SHA-256 mismatch")

    remote = _load_json_object(paths["REMOTE_COMPLETE.json"], "remote completion")
    _expect_fields(
        remote,
        {
            "artifact_manifest_sha256": artifact_sha,
            "destination": job["hf_output_uri"],
            "status": "REMOTE_VERIFIED",
        },
        "remote completion",
    )
    _require_positive_int(remote.get("file_count"), "remote completion file_count")
    _require_positive_int(remote.get("total_size"), "remote completion total_size")
    return artifact


def _collect_bundle(
    *,
    job: Mapping[str, Any],
    metadata_root: Path,
    required_paths: Sequence[str],
    plan_sizes: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    checkpoint_sha = str(job["checkpoint_sha256"])
    bundle_root = metadata_root / checkpoint_sha
    if not bundle_root.is_dir() or bundle_root.is_symlink():
        raise ValueError(f"metadata bundle directory is missing or unsafe: {checkpoint_sha}")
    paths = {
        logical_path: _metadata_path(bundle_root, logical_path)
        for logical_path in METADATA_LOGICAL_PATHS
    }
    bundle_checksums = _parse_checksum_file(
        paths["bundle_checksums.sha256"], label="bundle checksum manifest"
    )
    if set(bundle_checksums).intersection(UNCHECKSUMMED_REMOTE_PATHS):
        raise ValueError(
            "bundle checksum manifest must not enumerate itself or remote completion"
        )
    expected_checksummed = set(required_paths) - UNCHECKSUMMED_REMOTE_PATHS
    missing = sorted(expected_checksummed - set(bundle_checksums))
    if missing:
        raise ValueError(
            "bundle checksum manifest misses required allowlisted files: "
            + ", ".join(missing[:3])
        )

    artifact = _validate_artifact_identity(
        job=job,
        paths=paths,
        bundle_checksums=bundle_checksums,
    )

    actual_digests: dict[str, str] = {}
    for logical_path, path in paths.items():
        actual_size = path.stat().st_size
        expected_size = plan_sizes[(checkpoint_sha, logical_path)]
        if actual_size != expected_size:
            raise ValueError(
                f"downloaded metadata size disagrees with plan: {logical_path}"
            )
        actual_digest = sha256_file(path)
        actual_digests[logical_path] = actual_digest
        if logical_path not in UNCHECKSUMMED_REMOTE_PATHS:
            if bundle_checksums.get(logical_path) != actual_digest:
                raise ValueError(
                    "downloaded metadata SHA-256 disagrees with bundle checksum: "
                    f"{logical_path}"
                )

    files = []
    for logical_path in required_paths:
        digest = (
            actual_digests[logical_path]
            if logical_path in UNCHECKSUMMED_REMOTE_PATHS
            else bundle_checksums[logical_path]
        )
        files.append(
            {
                "logical_path": logical_path,
                "sha256": _require_hex64(digest, f"SHA-256 for {logical_path}"),
                "size": plan_sizes[(checkpoint_sha, logical_path)],
            }
        )
    return {
        "artifact_identity": {
            key: artifact[key] for key in _expected_artifact_identity(job)
        },
        "artifact_manifest_file_sha256": actual_digests[
            "artifact_manifest.json"
        ],
        "bundle_checksums_file_sha256": actual_digests[
            "bundle_checksums.sha256"
        ],
        "checkpoint_sha256": checkpoint_sha,
        "files": files,
        "hf_uri": job["hf_output_uri"],
    }


def _resolve_bundle_container(
    metadata_root: Path, checkpoint_shas: Sequence[str]
) -> Path:
    candidates = (
        metadata_root,
        metadata_root / "evaluations" / DEFINITION_VERSION,
        metadata_root / DEFINITION_VERSION,
    )
    matching: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_dir() and all(
            (candidate / checkpoint_sha).is_dir()
            for checkpoint_sha in checkpoint_shas
        ):
            matching.append(candidate)
    if len(matching) != 1:
        raise ValueError(
            "metadata root must resolve uniquely to the directory containing "
            "all 30 selected checkpoint bundles"
        )
    return matching[0]


def build_remote_catalog(
    *,
    metadata_root: str | Path,
    sync_plan_path: str | Path,
    inventory_path: str | Path = DEFAULT_INVENTORY,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    inventory, _ = _load_frozen_inventory(
        Path(inventory_path), Path(repository_root)
    )
    selected_jobs = _select_jobs(inventory)
    required_paths = _required_logical_paths()
    checkpoint_shas = [str(job["checkpoint_sha256"]) for job in selected_jobs]
    plan_path = Path(sync_plan_path)
    plan_sizes = _load_sync_plan(
        plan_path,
        checkpoint_shas=checkpoint_shas,
        required_paths=required_paths,
    )
    root = Path(metadata_root)
    if not root.is_dir():
        raise ValueError(f"metadata-only sync root is not a directory: {root}")
    bundle_container = _resolve_bundle_container(root, checkpoint_shas)
    bundles = [
        _collect_bundle(
            job=job,
            metadata_root=bundle_container,
            required_paths=required_paths,
            plan_sizes=plan_sizes,
        )
        for job in selected_jobs
    ]
    metadata_observations = [
        {
            "checkpoint_sha256": bundle["checkpoint_sha256"],
            **file_row,
        }
        for bundle in bundles
        for file_row in bundle["files"]
        if file_row["logical_path"] in METADATA_LOGICAL_PATHS
    ]
    return {
        "bundles": bundles,
        "hf_root": HF_ROOT,
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source_evidence": {
            "downloaded_metadata_catalog_sha256": _canonical_sha256(
                metadata_observations
            ),
            "downloaded_metadata_file_count": len(metadata_observations),
            "hf_sync_plan_sha256": sha256_file(plan_path),
            "hf_sync_plan_size": plan_path.stat().st_size,
        },
        "source_inventory_sha256": EXPECTED_INVENTORY_FILE_SHA256,
    }


def validate_remote_catalog(catalog: Mapping[str, Any]) -> None:
    if catalog.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("remote catalog schema version mismatch")
    if catalog.get("hf_root") != HF_ROOT:
        raise ValueError("remote catalog HF root mismatch")
    if catalog.get("source_inventory_sha256") != EXPECTED_INVENTORY_FILE_SHA256:
        raise ValueError("remote catalog source inventory mismatch")
    inventory, _ = _load_frozen_inventory(DEFAULT_INVENTORY, ROOT)
    jobs = _select_jobs(inventory)
    rows = catalog.get("bundles")
    if not isinstance(rows, list) or len(rows) != len(jobs):
        raise ValueError("remote catalog must contain exactly 30 bundles")
    required_paths = _required_logical_paths()
    metadata_observations = []
    for row, job in zip(rows, jobs, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("remote catalog bundle must be an object")
        checkpoint_sha = str(job["checkpoint_sha256"])
        if row.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("remote catalog bundle order/identity mismatch")
        if row.get("hf_uri") != job["hf_output_uri"]:
            raise ValueError("remote catalog bundle HF URI mismatch")
        if row.get("artifact_identity") != _expected_artifact_identity(job):
            raise ValueError("remote catalog artifact identity mismatch")
        artifact_sha = _require_hex64(
            row.get("artifact_manifest_file_sha256"),
            "artifact_manifest_file_sha256",
        )
        checksums_sha = _require_hex64(
            row.get("bundle_checksums_file_sha256"),
            "bundle_checksums_file_sha256",
        )
        files = row.get("files")
        if not isinstance(files, list):
            raise ValueError("remote catalog bundle files must be a list")
        if [item.get("logical_path") for item in files if isinstance(item, Mapping)] != list(
            required_paths
        ) or len(files) != len(required_paths):
            raise ValueError("remote catalog files differ from the dynamic allowlist")
        by_path: dict[str, Mapping[str, Any]] = {}
        for item in files:
            if not isinstance(item, Mapping):
                raise ValueError("remote catalog file row must be an object")
            logical_path = _require_safe_posix_path(
                item.get("logical_path"), "remote catalog logical path"
            )
            if logical_path in by_path:
                raise ValueError("remote catalog contains a duplicate logical path")
            _require_hex64(item.get("sha256"), f"SHA-256 for {logical_path}")
            _require_positive_int(item.get("size"), f"size for {logical_path}")
            by_path[logical_path] = item
            if logical_path in METADATA_LOGICAL_PATHS:
                metadata_observations.append(
                    {
                        "checkpoint_sha256": checkpoint_sha,
                        "logical_path": logical_path,
                        "sha256": item["sha256"],
                        "size": item["size"],
                    }
                )
        if by_path["artifact_manifest.json"].get("sha256") != artifact_sha:
            raise ValueError("remote catalog artifact-manifest digest mismatch")
        if by_path["bundle_checksums.sha256"].get("sha256") != checksums_sha:
            raise ValueError("remote catalog bundle-checksum digest mismatch")
    evidence = catalog.get("source_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("remote catalog source evidence is absent")
    _require_hex64(
        evidence.get("hf_sync_plan_sha256"), "source HF sync plan SHA-256"
    )
    _require_positive_int(evidence.get("hf_sync_plan_size"), "source HF sync plan size")
    expected_metadata_count = len(jobs) * len(METADATA_LOGICAL_PATHS)
    if evidence.get("downloaded_metadata_file_count") != expected_metadata_count:
        raise ValueError("remote catalog downloaded metadata count mismatch")
    if evidence.get("downloaded_metadata_catalog_sha256") != _canonical_sha256(
        metadata_observations
    ):
        raise ValueError("remote catalog downloaded metadata digest mismatch")


def render_remote_catalog(catalog: Mapping[str, Any]) -> bytes:
    validate_remote_catalog(catalog)
    return _canonical_bytes(dict(catalog))


def materialize_remote_catalog(
    catalog: Mapping[str, Any], output_path: str | Path
) -> str:
    payload = render_remote_catalog(catalog)
    output = Path(output_path)
    with output.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def verify_materialized_remote_catalog(
    catalog: Mapping[str, Any], output_path: str | Path
) -> str:
    expected = render_remote_catalog(catalog)
    output = Path(output_path)
    if not output.is_file():
        raise FileNotFoundError(f"remote catalog is absent: {output}")
    observed = output.read_bytes()
    if observed != expected:
        raise ValueError("materialized remote catalog is not byte-identical")
    return hashlib.sha256(observed).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the fixed-readout Stage 2 checksum catalog from a "
            "metadata-only local sync and a saved HF sync plan."
        )
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--sync-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    catalog = build_remote_catalog(
        inventory_path=args.inventory,
        metadata_root=args.metadata_root,
        sync_plan_path=args.sync_plan,
    )
    if args.check:
        digest = verify_materialized_remote_catalog(catalog, args.output)
    else:
        digest = materialize_remote_catalog(catalog, args.output)
    print(f"{digest}  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
