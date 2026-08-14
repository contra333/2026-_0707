"""Terminal validation and upload for one Task F source-training host shard."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from oge.evaluation.task_f_fresh import EXPECTED_SPECIFICATION_SHA256
from oge.evaluation.task_f_fresh_orchestration import (
    HOST_COUNTS,
    HOSTS,
    SOURCE_EXECUTION_ID,
    SOURCE_TRAINING_SHA,
)
from oge.studies.artifacts import sha256_file
from oge.studies.supervisor import SupervisorBlockedError, verify_clean_git


SNAPSHOT_EPOCHS = (0, 1, 10, 30, 60, 61, 120, 121, 160, 161, 200)
AUDIT_STEPS = (1, 352, 3520, 10560, 21120, 21121, 42240, 42241, 56320, 56321, 70400)
EXPORT_GATE_WAITING = "WAITING"
EXPORT_GATE_COMPLETE = "COMPLETE"
EXPORT_GATE_FAILED = "FAILED"


def classify_export_states(states: Sequence[str | None]) -> tuple[str, str | None]:
    """Classify watcher states without treating an in-progress export as failure."""

    if not states:
        raise ValueError("at least one Task F export state is required")
    failures: list[str] = []
    all_complete = True
    for index, state in enumerate(states):
        if state is None:
            all_complete = False
            continue
        normalized = state.strip()
        if normalized.startswith("COMPLETE"):
            continue
        all_complete = False
        if normalized.startswith("RUNNING"):
            continue
        if normalized.startswith("FAILED"):
            failures.append(f"gpu{index}={normalized}")
        else:
            failures.append(f"gpu{index}=INVALID({normalized})")
    if failures:
        return EXPORT_GATE_FAILED, "; ".join(failures)
    if all_complete:
        return EXPORT_GATE_COMPLETE, None
    return EXPORT_GATE_WAITING, None


def read_export_states(control_root: str | Path, concurrency: int) -> list[str | None]:
    if concurrency <= 0:
        raise ValueError("Task F finalizer concurrency must be positive")
    control = Path(control_root)
    states: list[str | None] = []
    for gpu_index in range(concurrency):
        path = control / f"export_gpu{gpu_index}.status"
        states.append(path.read_text(encoding="utf-8").strip() if path.is_file() else None)
    return states


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def wait_for_export_completion(
    *,
    control_root: str | Path,
    concurrency: int,
    poll_seconds: float = 60.0,
    timeout_hours: float = 72.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> list[str]:
    """Wait for every export watcher, failing only on explicit/invalid failure."""

    if poll_seconds < 0 or timeout_hours <= 0:
        raise ValueError("invalid Task F finalizer wait interval")
    control = Path(control_root)
    status_path = control / "host_finalizer.status"
    log_path = control / "host_finalizer.log"
    deadline = monotonic() + timeout_hours * 3600.0
    previous: tuple[str | None, ...] | None = None
    while monotonic() < deadline:
        states = read_export_states(control, concurrency)
        decision, detail = classify_export_states(states)
        snapshot = tuple(states)
        if snapshot != previous:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                    f"EXPORT_GATE decision={decision} states={states}\n"
                )
            previous = snapshot
        if decision == EXPORT_GATE_FAILED:
            _atomic_write_text(status_path, f"FAILED export_state={states}\n")
            raise SupervisorBlockedError(detail or "Task F export watcher failed")
        if decision == EXPORT_GATE_COMPLETE:
            return [str(state) for state in states]
        _atomic_write_text(status_path, f"WAITING export_state={states}\n")
        sleep(poll_seconds)
    _atomic_write_text(status_path, "FAILED export_wait_timeout\n")
    raise SupervisorBlockedError("Task F export completion wait timed out")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON mapping")
    return value


def finalize_task_f_source_host(
    *,
    repository_root: str | Path,
    expected_finalizer_git_sha: str,
    host_id: str,
    source_worktree: str | Path,
    output_root: str | Path,
    hf_cli: str | Path,
    poll_seconds: float = 60.0,
    timeout_hours: float = 72.0,
) -> dict[str, Any]:
    """Wait, validate the frozen source shard, upload it, and publish last marker."""

    if host_id not in HOSTS:
        raise ValueError(f"unsupported Task F host: {host_id}")
    repository = Path(repository_root).resolve()
    source = Path(source_worktree).resolve()
    output = Path(output_root).resolve()
    verify_clean_git(repository, expected_finalizer_git_sha)
    verify_clean_git(source, SOURCE_TRAINING_SHA)
    if not output.is_dir():
        raise SupervisorBlockedError(f"Task F source output root is absent: {output}")
    control = output / "control"
    launch = source / "artifacts" / SOURCE_EXECUTION_ID / "launch_bundle"
    matrix = _load_json(launch / "run_matrix.json")
    assignment = _load_json(launch / "host_assignment.json")
    export_plan = _load_json(launch / "export_jobs.json")
    gpu_queues = _load_json(launch / "gpu_queues.json")
    queues = gpu_queues.get("hosts", {}).get(host_id)
    if not isinstance(queues, Mapping) or not queues:
        raise SupervisorBlockedError("Task F host GPU queue is absent")
    concurrency = len(queues)
    expected = HOST_COUNTS[host_id]
    status_path = control / "host_finalizer.status"
    log_path = control / "host_finalizer.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")

    def fail(message: str) -> None:
        _atomic_write_text(status_path, f"FAILED {message}\n")
        log(f"FAILED {message}")
        raise SupervisorBlockedError(message)

    log(
        "FINALIZER_START_V2 "
        f"host={host_id} finalizer_sha={expected_finalizer_git_sha}"
    )
    wait_for_export_completion(
        control_root=control,
        concurrency=concurrency,
        poll_seconds=poll_seconds,
        timeout_hours=timeout_hours,
    )
    _atomic_write_text(status_path, "VALIDATING\n")
    log("VALIDATION_START")

    from oge.feature_export import verify_task_f_artifact

    groups = set(assignment["host_assignments"][host_id]["sibling_group_ids"])
    run_rows = [row for row in matrix["runs"] if row["sibling_group_id"] in groups]
    run_ids = [str(row["run_id"]) for row in run_rows]
    if len(run_ids) != expected["runs"] or len(run_ids) != len(set(run_ids)):
        fail("assigned_run_count")
    planned_jobs = [job for job in export_plan["jobs"] if job["host_id"] == host_id]
    if len(planned_jobs) != expected["source_exports"]:
        fail("assigned_export_count")
    expected_export_keys = {
        (
            job["run_id"],
            job["checkpoint_epoch"],
            job["checkpoint_role"],
            job["depth_tap"],
            job["dataset_split"],
        )
        for job in planned_jobs
    }
    validation_runs: list[dict[str, Any]] = []
    actual_export_keys: set[tuple[Any, ...]] = set()
    group_witness: dict[str, tuple[str, str]] = {}
    for row in run_rows:
        run_id = str(row["run_id"])
        run_directory = output / "runs" / run_id
        summary = _load_json(run_directory / "summary.json")
        if (
            summary.get("status") != "completed"
            or summary.get("completed_epoch") != 200
            or summary.get("global_step") != 70400
        ):
            fail(f"summary run={run_id}")
        if summary.get("id_test", {}).get("status") != "deferred":
            fail(f"id_test run={run_id}")
        history = [
            json.loads(line)
            for line in (run_directory / "history.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if [item["epoch"] for item in history] != list(range(1, 201)) or [
            item["global_step"] for item in history
        ] != [352 * epoch for epoch in range(1, 201)]:
            fail(f"history run={run_id}")
        metadata = _load_json(run_directory / "run_metadata.json")
        if (
            metadata.get("oge_git_sha") != SOURCE_TRAINING_SHA
            or metadata.get("id_test_evaluation") != "deferred"
        ):
            fail(f"metadata run={run_id}")
        provenance = metadata["paired_control_provenance"]
        if provenance.get("execution_only") is not False:
            fail(f"execution_only run={run_id}")
        environment = _load_json(run_directory / "environment.json")["executions"][-1]
        if environment.get("actual_visible_gpu_uuid") != environment.get(
            "expected_physical_gpu_uuid"
        ):
            fail(f"gpu_uuid run={run_id}")
        checkpoints = run_directory / "checkpoints"
        if not (checkpoints / "last.pt").is_file() or not (
            checkpoints / "best_val.pt"
        ).is_file():
            fail(f"checkpoint run={run_id}")
        snapshot_names = sorted(
            path.name for path in (checkpoints / "snapshots").glob("epoch_*.pt")
        )
        if snapshot_names != [f"epoch_{epoch:04d}.pt" for epoch in SNAPSHOT_EPOCHS]:
            fail(f"snapshots run={run_id}")
        telemetry = [
            json.loads(line)
            for line in (run_directory / "update_telemetry.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        if sorted({item["global_step"] for item in telemetry}) != list(AUDIT_STEPS):
            fail(f"telemetry run={run_id}")
        manifests = sorted((output / "exports" / run_id).glob("**/manifest.json"))
        run_jobs = [job for job in planned_jobs if job["run_id"] == run_id]
        if len(manifests) != len(run_jobs):
            fail(
                f"export_count run={run_id} actual={len(manifests)} "
                f"expected={len(run_jobs)}"
            )
        initialization_hashes: set[str] = set()
        data_stream_hashes: set[str] = set()
        output_ids: list[str] = []
        last_export_hashes: set[str] = set()
        for manifest_path in manifests:
            verified = verify_task_f_artifact(manifest_path.parent)
            manifest = verified["manifest"]
            if manifest["specification_sha256"] != EXPECTED_SPECIFICATION_SHA256:
                fail(f"spec run={run_id}")
            if (
                manifest["oge_git_sha"] != SOURCE_TRAINING_SHA
                or manifest["execution_only"] is not False
            ):
                fail(f"export_provenance run={run_id}")
            if (
                manifest["runtime"]["device_type"] != "cuda"
                or manifest["dataset_split"] != "id_train"
            ):
                fail(f"export_runtime run={run_id}")
            actual_export_keys.add(
                (
                    run_id,
                    manifest["checkpoint_epoch"],
                    manifest["checkpoint_role"],
                    manifest["depth_tap"],
                    manifest["dataset_split"],
                )
            )
            initialization_hashes.add(manifest["initialization_sha256"])
            data_stream_hashes.add(manifest["data_stream_sha256"])
            output_ids.append(manifest["output_identity_sha256"])
            if (
                manifest["checkpoint_epoch"] == 200
                and manifest["checkpoint_role"] == "last"
            ):
                last_export_hashes.add(manifest["checkpoint_sha256"])
        if (
            len(initialization_hashes) != 1
            or len(data_stream_hashes) != 1
            or len(last_export_hashes) != 1
        ):
            fail(f"witness run={run_id}")
        initialization_sha256 = next(iter(initialization_hashes))
        data_stream_sha256 = next(iter(data_stream_hashes))
        last_sha256 = sha256_file(checkpoints / "last.pt")
        if (
            initialization_sha256 != provenance["initialization_sha256"]
            or last_export_hashes != {last_sha256}
        ):
            fail(f"identity run={run_id}")
        witness = (initialization_sha256, data_stream_sha256)
        sibling_group_id = str(row["sibling_group_id"])
        prior = group_witness.setdefault(sibling_group_id, witness)
        if prior != witness:
            fail(f"sibling_witness group={sibling_group_id}")
        validation_runs.append(
            {
                "run_id": run_id,
                "sibling_group_id": sibling_group_id,
                "training_seed": row["training_seed"],
                "gpu_uuid": environment["actual_visible_gpu_uuid"],
                "initialization_sha256": initialization_sha256,
                "data_stream_sha256": data_stream_sha256,
                "last_pt_sha256": last_sha256,
                "last_pt_bytes": (checkpoints / "last.pt").stat().st_size,
                "export_count": len(manifests),
                "export_output_identity_sha256": sorted(output_ids),
            }
        )
    if actual_export_keys != expected_export_keys:
        fail("export_job_coverage")

    validation = {
        "status": "PASS",
        "host_id": host_id,
        "execution_sha": SOURCE_TRAINING_SHA,
        "finalizer_git_sha": expected_finalizer_git_sha,
        "specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "run_count": len(validation_runs),
        "export_count": len(actual_export_keys),
        "audit_steps": list(AUDIT_STEPS),
        "snapshot_epochs": list(SNAPSHOT_EPOCHS),
        "runs": validation_runs,
    }
    validation_path = output / "host_validation.json"
    temporary = validation_path.with_name(f".{validation_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, validation_path)
    validation_sha256 = sha256_file(validation_path)
    log(
        f"VALIDATION_PASS runs={len(validation_runs)} "
        f"exports={len(actual_export_keys)} sha={validation_sha256}"
    )

    _atomic_write_text(status_path, "UPLOADING\n")
    remote = (
        f"hf://buckets/contra333/ICLR_RUN/servers/{host_id}/{SOURCE_EXECUTION_ID}"
    )
    listing = subprocess.run(
        [str(hf_cli), "buckets", "list", remote, "-R", "--format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if listing.returncode != 0:
        fail("remote_preflight")
    if json.loads(listing.stdout):
        fail("remote_prefix_not_empty")
    plans = control / "upload_plans"
    plans.mkdir(exist_ok=True)
    upload_log = control / "upload.log"

    def sync_directory(source_directory: Path, destination: str, name: str) -> None:
        dry_run_path = plans / f"{name}.dryrun.jsonl"
        with dry_run_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                [
                    str(hf_cli),
                    "buckets",
                    "sync",
                    str(source_directory),
                    destination,
                    "--dry-run",
                    "--no-delete",
                    "--format",
                    "json",
                ],
                check=False,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            fail(f"upload_dryrun name={name} rc={result.returncode}")
        with upload_log.open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                [
                    str(hf_cli),
                    "buckets",
                    "sync",
                    str(source_directory),
                    destination,
                    "--no-delete",
                    "--format",
                    "json",
                ],
                check=False,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            fail(f"upload name={name} rc={result.returncode}")

    for run_id in run_ids:
        sync_directory(
            output / "runs" / run_id,
            f"{remote}/{run_id}/training",
            f"{run_id}.training",
        )
        sync_directory(
            output / "exports" / run_id,
            f"{remote}/{run_id}/exports",
            f"{run_id}.exports",
        )
    sync_directory(output / "logs", f"{remote}/logs", "host.logs")
    sync_directory(launch, f"{remote}/launch_bundle", "launch_bundle")
    metadata_directory = output / "upload_metadata"
    metadata_directory.mkdir(exist_ok=True)
    shutil.copy2(validation_path, metadata_directory / "host_validation.json")
    sync_directory(metadata_directory, f"{remote}/metadata", "metadata")

    listing = subprocess.run(
        [str(hf_cli), "buckets", "list", remote, "-R", "--format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if listing.returncode != 0:
        fail("remote_listing")
    rows = json.loads(listing.stdout)
    paths = {row["path"] for row in rows}
    prefix = f"servers/{host_id}/{SOURCE_EXECUTION_ID}"
    for run_id in run_ids:
        if f"{prefix}/{run_id}/training/checkpoints/last.pt" not in paths:
            fail(f"remote_last run={run_id}")
        if not any(
            path.startswith(f"{prefix}/{run_id}/exports/")
            and path.endswith("/manifest.json")
            for path in paths
        ):
            fail(f"remote_exports run={run_id}")
    (metadata_directory / "remote_listing.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sync_directory(metadata_directory, f"{remote}/metadata", "metadata_final")
    marker = {
        "status": "REMOTE_VERIFIED",
        "host_id": host_id,
        "execution_sha": SOURCE_TRAINING_SHA,
        "finalizer_git_sha": expected_finalizer_git_sha,
        "run_count": len(run_ids),
        "export_count": len(actual_export_keys),
        "validation_sha256": validation_sha256,
        "remote_file_count_before_marker": len(rows),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    marker_directory = output / "remote_marker"
    marker_directory.mkdir(exist_ok=True)
    (marker_directory / "REMOTE_COMPLETE.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sync_directory(marker_directory, remote, "remote_marker")
    final_listing = subprocess.run(
        [str(hf_cli), "buckets", "list", remote, "-R", "--format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if final_listing.returncode != 0 or f"{prefix}/REMOTE_COMPLETE.json" not in {
        row["path"] for row in json.loads(final_listing.stdout)
    }:
        fail("remote_marker_verify")
    _atomic_write_text(status_path, "COMPLETE REMOTE_VERIFIED\n")
    log(f"FINALIZER_COMPLETE remote_files={len(json.loads(final_listing.stdout))}")
    return marker
