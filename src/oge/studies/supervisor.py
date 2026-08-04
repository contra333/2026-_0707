"""Detached follow-up execution, integrity, and Hugging Face upload helpers."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from oge.training import repository_git_state

from .artifacts import atomic_write_json, sha256_file
from .execution import host_trial_ids, load_followup_execution_plan
from .schemas import validate_trial_record

PROTECTED_TRIAL_FIELDS = {
    "id_test",
    "ood",
    "ood_metrics",
    "geometry",
    "geometry_nc",
    "neural_collapse",
    "nc_metrics",
    "detector",
    "detector_metrics",
}
DEFERRED_ID_TEST = {
    "status": "deferred",
    "metrics_available": False,
    "artifacts_created": False,
}


class SupervisorBlockedError(RuntimeError):
    """A safety or identity gate that requires human intervention."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_supervisor_state(
    state_path: str | Path,
    state: Mapping[str, Any],
    status: str,
    **updates: Any,
) -> dict[str, Any]:
    value = dict(state)
    value.update(updates)
    value["status"] = status
    value["updated_at"] = utc_now()
    atomic_write_json(state_path, value)
    return value


def verify_clean_git(
    repository_root: str | Path,
    expected_git_sha: str,
) -> None:
    sha, dirty = repository_git_state(repository_root)
    if sha != expected_git_sha:
        raise SupervisorBlockedError(
            f"repository Git SHA {sha!r} differs from {expected_git_sha!r}"
        )
    if dirty:
        raise SupervisorBlockedError("repository checkout is not clean")


def _run_checked(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        check=True,
        capture_output=True,
        text=True,
    )


def verify_hf_preflight(
    hf_cli: str | Path,
    *,
    account: str,
    bucket: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_checked,
) -> None:
    executable = Path(hf_cli)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SupervisorBlockedError(f"HF CLI is not executable: {executable}")
    whoami = run([str(executable), "auth", "whoami"])
    if f"user: {account}" not in whoami.stdout:
        raise SupervisorBlockedError(
            f"HF CLI is not authenticated as the required account {account!r}"
        )
    run([str(executable), "buckets", "info", bucket])


def _gpu_inventory(
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_checked,
) -> dict[str, str]:
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    inventory: dict[str, str] = {}
    for line in result.stdout.splitlines():
        index, uuid = [value.strip() for value in line.split(",", maxsplit=1)]
        inventory[uuid] = index
    return inventory


def busy_gpu_uuids(
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_checked,
) -> set[str]:
    result = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def wait_for_idle_gpus(
    requested_gpu_uuids: Sequence[str],
    *,
    state_path: str | Path,
    state: Mapping[str, Any],
    poll_seconds: float = 60.0,
    timeout_hours: float = 24.0,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_checked,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not requested_gpu_uuids or len(requested_gpu_uuids) != len(
        set(requested_gpu_uuids)
    ):
        raise SupervisorBlockedError("GPU UUIDs must be a non-empty unique list")
    inventory = _gpu_inventory(run)
    unknown = set(requested_gpu_uuids).difference(inventory)
    if unknown:
        raise SupervisorBlockedError(f"requested GPU UUIDs are absent: {sorted(unknown)}")
    deadline = datetime.now(timezone.utc) + timedelta(hours=timeout_hours)
    value = dict(state)
    while True:
        busy = sorted(set(requested_gpu_uuids).intersection(busy_gpu_uuids(run)))
        if not busy:
            return update_supervisor_state(
                state_path,
                value,
                "PREFLIGHT_PASSED",
                gpu_uuids=list(requested_gpu_uuids),
                gpu_indices=[inventory[uuid] for uuid in requested_gpu_uuids],
            )
        value = update_supervisor_state(
            state_path,
            value,
            "WAITING_FOR_GPUS",
            busy_gpu_uuids=busy,
        )
        if datetime.now(timezone.utc) >= deadline:
            raise SupervisorBlockedError(
                f"assigned GPUs remained busy for {timeout_hours} hours"
            )
        sleep(poll_seconds)


def _verify_checksum_sidecar(path: Path) -> None:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing SHA-256 sidecar for {path}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) < 2 or fields[0] != sha256_file(path) or fields[-1] != path.name:
        raise ValueError(f"SHA-256 sidecar mismatch for {path}")


def _verify_metric(metric: Any, *, label: str, expected_epoch: int | None = None) -> None:
    if not isinstance(metric, Mapping):
        raise ValueError(f"{label} is missing")
    epoch = int(metric["epoch"])
    accuracy = float(metric["accuracy"])
    nll = float(metric["nll"])
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError(f"{label} epoch mismatch")
    if not 0.0 <= accuracy <= 1.0 or not math.isfinite(nll):
        raise ValueError(f"{label} contains a non-finite or out-of-range metric")


def verify_followup_study_artifacts(
    study_root: str | Path,
    *,
    execution_plan: Mapping[str, Any],
    followup_plan: Mapping[str, Any],
    host_id: str,
    expected_git_sha: str,
) -> dict[str, Any]:
    root = Path(study_root)
    expected_ids = host_trial_ids(execution_plan, host_id)
    scheduled = {
        str(row["trial_id"]): row for row in followup_plan["scheduled"]
    }
    for filename in ("study_manifest.json", "study_summary.json", "ordered_plan.json"):
        path = root / filename
        if not path.is_file():
            raise ValueError(f"study artifact is missing {filename}")
        _verify_checksum_sidecar(path)
    study = json.loads((root / "study_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "study_summary.json").read_text(encoding="utf-8"))
    if study.get("status") != "completed" or summary.get("status") != "completed":
        raise ValueError("follow-up study is not completed")
    assignment = study.get("execution_assignment")
    if not isinstance(assignment, Mapping):
        raise ValueError("study is missing its execution assignment")
    if (
        assignment.get("execution_id") != execution_plan["execution_id"]
        or assignment.get("host_id") != host_id
        or assignment.get("assigned_trial_ids") != expected_ids
    ):
        raise ValueError("study execution assignment differs from the frozen shard")
    if study.get("git") != {"sha": expected_git_sha, "dirty": False}:
        raise ValueError("study Git identity differs from production")
    accounting = study.get("terminal_slot_accounting")
    if not isinstance(accounting, Mapping) or any(
        int(accounting.get(key, -1)) != len(expected_ids)
        for key in ("assigned", "planned", "terminal", "completed")
    ):
        raise ValueError("study terminal accounting does not match the host shard")
    if int(accounting.get("failed_pending_classification", -1)) != 0:
        raise ValueError("study contains failed trials")

    trial_paths = sorted((root / "trials").glob("*/trial.json"))
    if {path.parent.name for path in trial_paths} != set(expected_ids):
        raise ValueError("materialized trial IDs differ from the frozen host shard")
    trial_checksums: dict[str, str] = {}
    for path in trial_paths:
        _verify_checksum_sidecar(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_trial_record(record)
        trial_id = str(record["trial_id"])
        expected = scheduled[trial_id]
        if (
            record["phase"] != "followup"
            or record["status"] != "completed"
            or int(record["completed_epoch"]) != 200
            or record["git_sha"] != expected_git_sha
            or record["config_hash"] != expected["config_hash"]
            or int(record["training_seed"]) != int(expected["training_seed"])
        ):
            raise ValueError(f"trial {trial_id!r} differs from the frozen identity")
        protected = PROTECTED_TRIAL_FIELDS.intersection(record)
        if protected:
            raise ValueError(
                f"trial {trial_id!r} contains protected fields: {sorted(protected)}"
            )
        _verify_metric(record["best_validation"], label=f"{trial_id} best validation")
        _verify_metric(
            record["final_validation"],
            label=f"{trial_id} final validation",
            expected_epoch=200,
        )
        for checkpoint in ("last", "best_val"):
            reference = record["checkpoints"][checkpoint]
            checkpoint_path = Path(str(reference["path"]))
            if not checkpoint_path.is_file():
                raise ValueError(f"trial {trial_id!r} is missing {checkpoint}")
            if sha256_file(checkpoint_path) != reference["sha256"]:
                raise ValueError(f"trial {trial_id!r} {checkpoint} checksum mismatch")
        summary_reference = record["artifact_references"]["summary"]
        run_summary_path = Path(str(summary_reference["path"]))
        if (
            not run_summary_path.is_file()
            or sha256_file(run_summary_path) != summary_reference["sha256"]
        ):
            raise ValueError(f"trial {trial_id!r} run summary checksum mismatch")
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        if run_summary.get("id_test") != DEFERRED_ID_TEST:
            raise ValueError(f"trial {trial_id!r} did not defer ID test")
        if any(
            key in run_summary
            for key in (
                "final_id_test",
                "best_val_id_test",
                "ood",
                "geometry",
                "geometry_nc",
                "detector_metrics",
            )
        ):
            raise ValueError(f"trial {trial_id!r} contains protected results")
        evaluation_dir = run_summary_path.parent / "evaluation"
        if evaluation_dir.is_dir() and any(evaluation_dir.iterdir()):
            raise ValueError(f"trial {trial_id!r} contains evaluation artifacts")
        trial_checksums[trial_id] = sha256_file(path)

    report = {
        "schema_version": "1.0",
        "execution_id": execution_plan["execution_id"],
        "host_id": host_id,
        "production_git_sha": expected_git_sha,
        "assigned_trial_ids": expected_ids,
        "assigned_trial_count": len(expected_ids),
        "status": "completed_and_verified",
        "protected_evaluation": "deferred",
        "trial_json_sha256": trial_checksums,
        "verified_at": utc_now(),
    }
    atomic_write_json(root / "artifact_validation.json", report)
    return report


def build_artifact_manifest(study_root: str | Path) -> dict[str, Any]:
    root = Path(study_root)
    excluded = {"artifact_manifest.json", "artifact_manifest.json.sha256"}
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "file_count": len(entries),
        "total_size": sum(int(entry["size"]) for entry in entries),
        "files": entries,
    }
    atomic_write_json(root / "artifact_manifest.json", manifest)
    return manifest


def parse_hf_sync_dry_run(text: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or rows[0].get("type") != "header":
        raise ValueError("HF dry run did not return a JSONL header")
    summary = rows[0].get("summary")
    if not isinstance(summary, Mapping) or int(summary.get("deletes", -1)) != 0:
        raise ValueError("HF dry run does not prove zero deletes")
    invalid = [
        row
        for row in rows[1:]
        if row.get("type") == "operation"
        and row.get("action") not in ("upload", "skip")
    ]
    if invalid:
        raise ValueError("HF dry run contains a non-upload operation")
    return {"header": rows[0], "operations": rows[1:]}


def _remote_prefix(destination: str, bucket: str) -> str:
    marker = f"hf://buckets/{bucket}/"
    if not destination.startswith(marker):
        raise ValueError("HF destination is outside the frozen bucket")
    return destination[len(marker) :].strip("/")


def _remote_file_map(
    hf_cli: str | Path,
    destination: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_checked,
) -> dict[str, dict[str, Any]]:
    result = run(
        [str(hf_cli), "buckets", "list", destination, "--json", "-R"]
    )
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    if not isinstance(rows, list):
        raise ValueError("HF remote listing is not a JSON array")
    return {
        str(row["path"]): dict(row)
        for row in rows
        if row.get("type") == "file"
    }


def upload_artifact_tree(
    study_root: str | Path,
    *,
    hf_cli: str | Path,
    bucket: str,
    destination: str,
    control_root: str | Path,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_checked,
) -> dict[str, Any]:
    root = Path(study_root)
    control = Path(control_root)
    control.mkdir(parents=True, exist_ok=True)
    prefix = _remote_prefix(destination, bucket)
    completion_remote_path = f"{prefix}/REMOTE_COMPLETE.json"
    local_files = {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }
    remote_before = _remote_file_map(hf_cli, destination, run=run)
    normalized_before = {
        path[len(prefix) + 1 :]: int(row["size"])
        for path, row in remote_before.items()
        if path.startswith(f"{prefix}/")
    }
    if "REMOTE_COMPLETE.json" in normalized_before:
        raise SupervisorBlockedError("HF destination already has a completion marker")
    unexpected = set(normalized_before).difference(local_files)
    mismatched = {
        path
        for path in set(normalized_before).intersection(local_files)
        if normalized_before[path] != local_files[path]
    }
    if unexpected or mismatched:
        raise SupervisorBlockedError(
            "HF destination contains conflicting files: "
            f"unexpected={sorted(unexpected)}, size_mismatch={sorted(mismatched)}"
        )

    dry_run = run(
        [
            str(hf_cli),
            "buckets",
            "sync",
            str(root),
            destination,
            "--dry-run",
            "--no-delete",
            "--ignore-existing",
            "--json",
        ]
    )
    (control / "upload_dry_run.jsonl").write_text(
        dry_run.stdout,
        encoding="utf-8",
    )
    parsed = parse_hf_sync_dry_run(dry_run.stdout)
    plan_path = control / "upload_plan.jsonl"
    run(
        [
            str(hf_cli),
            "buckets",
            "sync",
            str(root),
            destination,
            "--plan",
            str(plan_path),
            "--no-delete",
            "--ignore-existing",
        ]
    )
    parse_hf_sync_dry_run(plan_path.read_text(encoding="utf-8"))
    run([str(hf_cli), "buckets", "sync", "--apply", str(plan_path)])

    remote_after = _remote_file_map(hf_cli, destination, run=run)
    normalized_after = {
        path[len(prefix) + 1 :]: int(row["size"])
        for path, row in remote_after.items()
        if path.startswith(f"{prefix}/")
    }
    if normalized_after != local_files:
        raise ValueError("HF path/size listing differs from the local artifact tree")
    manifest_path = root / "artifact_manifest.json"
    marker = {
        "schema_version": "1.0",
        "status": "REMOTE_VERIFIED",
        "destination": destination,
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "file_count": len(local_files),
        "total_size": sum(local_files.values()),
        "verified_at": utc_now(),
    }
    marker_path = control / "REMOTE_COMPLETE.json"
    atomic_write_json(marker_path, marker)
    run(
        [
            str(hf_cli),
            "buckets",
            "cp",
            str(marker_path),
            f"{destination}/REMOTE_COMPLETE.json",
        ]
    )
    final_remote = _remote_file_map(hf_cli, destination, run=run)
    if completion_remote_path not in final_remote:
        raise ValueError("HF completion marker is absent after upload")
    evidence = {
        "destination": destination,
        "dry_run_summary": parsed["header"]["summary"],
        "file_count": len(local_files),
        "total_size": sum(local_files.values()),
        "artifact_manifest_sha256": marker["artifact_manifest_sha256"],
        "remote_files": [
            {
                "path": path,
                "size": int(row["size"]),
                "xet_hash": row.get("xet_hash"),
            }
            for path, row in sorted(final_remote.items())
        ],
        "status": "REMOTE_VERIFIED",
    }
    atomic_write_json(control / "upload_evidence.json", evidence)
    return evidence


def production_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment
