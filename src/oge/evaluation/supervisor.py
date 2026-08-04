"""Source-local host supervision for protected metric evaluation jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from oge.studies.artifacts import atomic_write_json
from oge.studies.supervisor import (
    SupervisorBlockedError,
    build_artifact_manifest,
    production_environment,
    upload_artifact_tree,
    verify_clean_git,
    verify_hf_preflight,
    wait_for_idle_gpus,
)

from .planning import validate_execution_manifest
from .production import (
    PROTECTED_AUTHORIZATION,
    inventory_job,
    load_frozen_inventory,
    validate_dataset_policy,
    validate_job_checkpoint,
    verify_production_bundle,
)


def load_execution_manifest(
    path: str | Path, inventory: Mapping[str, Any]
) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation execution manifest must be a mapping")
    validate_execution_manifest(payload, inventory)
    return payload


def parse_source_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("source roots must use execution_id:stage=/absolute/path")
        key, raw_path = value.split("=", 1)
        if ":" not in key or not raw_path:
            raise ValueError("source roots must use execution_id:stage=/absolute/path")
        if key in roots:
            raise ValueError(f"duplicate source-root key {key!r}")
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError("source roots must be absolute")
        roots[key] = path
    return roots


def resolve_checkpoint_path(job: Mapping[str, Any], source_roots: Mapping[str, Path]) -> Path:
    source = job["source"]
    key = f"{source['source_execution_id']}:{source['source_stage']}"
    if key not in source_roots:
        raise ValueError(f"source-root mapping is missing {key!r}")
    return source_roots[key] / source["artifact_relative_path"]


def host_jobs(
    inventory: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    host_id: str,
) -> list[dict[str, Any]]:
    if host_id not in execution["hosts"]:
        raise ValueError(f"host {host_id!r} is absent from the execution manifest")
    return [
        inventory_job(inventory, job_id=job_id, host_id=host_id)
        for job_id in execution["hosts"][host_id]["checkpoint_job_ids"]
    ]


def _next_attempt_root(state_root: Path, job_id: str) -> Path:
    job_root = state_root / "jobs" / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    attempt = len(list(job_root.glob("attempt-*"))) + 1
    attempt_root = job_root / f"attempt-{attempt:03d}"
    attempt_root.mkdir()
    return attempt_root


def _completed_upload_evidence(control_root: Path, destination: str) -> dict[str, Any] | None:
    marker_path = control_root / "REMOTE_COMPLETE.json"
    evidence_path = control_root / "upload_evidence.json"
    if not marker_path.is_file() or not evidence_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if marker.get("status") != "REMOTE_VERIFIED" or marker.get("destination") != destination:
        raise ValueError("preserved upload marker identity mismatch")
    if evidence.get("status") != "REMOTE_VERIFIED" or evidence.get("destination") != destination:
        raise ValueError("preserved upload evidence identity mismatch")
    return evidence


def execute_host_jobs(
    *,
    repository_root: str | Path,
    inventory_path: str | Path,
    execution_path: str | Path,
    host_id: str,
    expected_git_sha: str,
    source_roots: Mapping[str, Path],
    data_root: str | Path,
    artifact_root: str | Path,
    state_root: str | Path,
    gpu_uuids: Sequence[str],
    hf_cli: str | Path,
    dataset_config_path: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    blas_threads: int = 4,
    hf_bucket: str = "contra333/ICLR_RUN",
    hf_account: str = "contra333",
    minimum_free_gb: float = 100.0,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    artifact = Path(artifact_root)
    state_directory = Path(state_root)
    state_directory.mkdir(parents=True, exist_ok=True)
    state_path = state_directory / "state.json"
    inventory = load_frozen_inventory(inventory_path)
    execution = load_execution_manifest(execution_path, inventory)
    jobs = host_jobs(inventory, execution, host_id=host_id)
    expected_concurrency = int(execution["hosts"][host_id]["concurrency"])
    if len(gpu_uuids) != expected_concurrency or len(set(gpu_uuids)) != len(gpu_uuids):
        raise SupervisorBlockedError(
            f"host {host_id} requires exactly {expected_concurrency} unique GPU UUIDs"
        )
    if blas_threads <= 0:
        raise ValueError("blas_threads must be positive")
    verify_clean_git(repository, expected_git_sha)
    if not Path(data_root).is_dir():
        raise SupervisorBlockedError(f"data root is absent: {data_root}")
    artifact.mkdir(parents=True, exist_ok=True)
    free_bytes = os.statvfs(artifact).f_bavail * os.statvfs(artifact).f_frsize
    if free_bytes < minimum_free_gb * 1024**3:
        raise SupervisorBlockedError(
            f"artifact filesystem has less than {minimum_free_gb} GiB free"
        )
    verify_hf_preflight(hf_cli, account=hf_account, bucket=hf_bucket)
    state: dict[str, Any] = {
        "schema_version": "1.0",
        "host_id": host_id,
        "expected_git_sha": expected_git_sha,
        "inventory_hash": inventory["inventory_hash"],
        "dataset_policy_hash": inventory["dataset_policy_hash"],
        "assigned_job_ids": [job["job_id"] for job in jobs],
        "assigned_job_count": len(jobs),
        "job_states": {},
        "status": "PREFLIGHT",
    }
    atomic_write_json(state_path, state)
    for job in jobs:
        checkpoint = resolve_checkpoint_path(job, source_roots)
        if not checkpoint.is_file():
            raise SupervisorBlockedError(f"source checkpoint is absent: {checkpoint}")
        validate_job_checkpoint(job, checkpoint_path=checkpoint)
        state["job_states"][job["job_id"]] = {
            "status": "CHECKPOINT_VERIFIED",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": job["checkpoint_sha256"],
        }
    validate_dataset_policy(
        inventory,
        dataset_config_path=dataset_config_path,
        data_root=data_root,
    )
    atomic_write_json(state_path, state)
    state = wait_for_idle_gpus(
        gpu_uuids,
        state_path=state_path,
        state=state,
    )
    state.setdefault("job_states", {})
    lock = threading.Lock()

    def update_job(job_id: str, **updates: Any) -> None:
        with lock:
            state["job_states"].setdefault(job_id, {}).update(updates)
            state["status"] = "RUNNING"
            atomic_write_json(state_path, state)

    def execute_queue(gpu_uuid: str, queue: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for job in queue:
            job_id = str(job["job_id"])
            checkpoint = resolve_checkpoint_path(job, source_roots)
            bundle = artifact / str(job["checkpoint_sha256"])
            control = state_directory / "uploads" / str(job["checkpoint_sha256"])
            destination = str(job["hf_output_uri"])
            preserved = _completed_upload_evidence(control, destination)
            if preserved is not None:
                verify_production_bundle(bundle)
                result = {"job_id": job_id, "status": "REMOTE_VERIFIED", "resumed": True}
                update_job(job_id, **result)
                results.append(result)
                continue
            attempt_root = _next_attempt_root(state_directory, job_id)
            update_job(
                job_id,
                status="RUNNING",
                gpu_uuid=gpu_uuid,
                attempt_root=str(attempt_root),
            )
            command = [
                sys.executable,
                str(repository / "scripts/run_metric_evaluation_job.py"),
                "--job-id",
                job_id,
                "--host-id",
                host_id,
                "--checkpoint",
                str(checkpoint),
                "--data-root",
                str(data_root),
                "--artifact-root",
                str(artifact),
                "--device",
                "cuda:0",
                "--batch-size",
                str(batch_size),
                "--num-workers",
                str(num_workers),
                "--protected-authorization",
                PROTECTED_AUTHORIZATION,
                "--expected-evaluation-git-sha",
                expected_git_sha,
                "--inventory",
                str(inventory_path),
                "--dataset-config",
                str(dataset_config_path),
            ]
            environment = production_environment()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
            environment["OMP_NUM_THREADS"] = str(blas_threads)
            environment["MKL_NUM_THREADS"] = str(blas_threads)
            environment["OPENBLAS_NUM_THREADS"] = str(blas_threads)
            with (attempt_root / "stdout.log").open("w", encoding="utf-8") as stdout, (
                attempt_root / "stderr.log"
            ).open("w", encoding="utf-8") as stderr:
                completed = run(
                    command,
                    cwd=repository,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                )
            atomic_write_json(
                attempt_root / "attempt.json",
                {
                    "schema_version": "1.0",
                    "job_id": job_id,
                    "gpu_uuid": gpu_uuid,
                    "command": command,
                    "returncode": int(completed.returncode),
                },
            )
            if completed.returncode != 0:
                result = {
                    "job_id": job_id,
                    "status": "FAILED",
                    "returncode": int(completed.returncode),
                    "attempt_root": str(attempt_root),
                }
                update_job(job_id, **result)
                results.append(result)
                continue
            verification = verify_production_bundle(bundle)
            update_job(job_id, status="LOCAL_VERIFIED", local_file_count=len(verification["verified_files"]))
            delay = 10.0
            evidence: dict[str, Any] | None = None
            for upload_attempt in range(1, 7):
                try:
                    evidence = upload_artifact_tree(
                        bundle,
                        hf_cli=hf_cli,
                        bucket=hf_bucket,
                        destination=destination,
                        control_root=control,
                    )
                except subprocess.CalledProcessError as exc:
                    if upload_attempt == 6:
                        result = {
                            "job_id": job_id,
                            "status": "FAILED",
                            "reason": f"upload failed after retries: {exc}",
                        }
                        update_job(job_id, **result)
                        results.append(result)
                        break
                    time.sleep(delay)
                    delay = min(delay * 2.0, 300.0)
                else:
                    break
            if evidence is not None:
                result = {
                    "job_id": job_id,
                    "status": "REMOTE_VERIFIED",
                    "resumed": False,
                    "file_count": evidence["file_count"],
                    "total_size": evidence["total_size"],
                    "destination": destination,
                }
                update_job(job_id, **result)
                results.append(result)
        return results

    queues = [jobs[index:: len(gpu_uuids)] for index in range(len(gpu_uuids))]
    with ThreadPoolExecutor(max_workers=len(gpu_uuids)) as pool:
        futures = [
            pool.submit(execute_queue, gpu_uuid, queue)
            for gpu_uuid, queue in zip(gpu_uuids, queues)
        ]
        results = [item for future in futures for item in future.result()]
    by_job = {result["job_id"]: result for result in results}
    if set(by_job) != {job["job_id"] for job in jobs}:
        raise ValueError("host supervisor did not return every assigned job")
    failed = sorted(job_id for job_id, result in by_job.items() if result["status"] != "REMOTE_VERIFIED")
    state["status"] = "REMOTE_VERIFIED" if not failed else "FAILED"
    state["terminal_job_count"] = len(by_job)
    state["remote_verified_job_count"] = len(by_job) - len(failed)
    state["failed_job_ids"] = failed
    atomic_write_json(state_path, state)
    ledger = {
        "schema_version": "1.0",
        "status": state["status"],
        "host_id": host_id,
        "evaluation_git_sha": expected_git_sha,
        "inventory_hash": inventory["inventory_hash"],
        "dataset_policy_hash": inventory["dataset_policy_hash"],
        "assigned_job_ids": [job["job_id"] for job in jobs],
        "terminal_job_count": len(by_job),
        "remote_verified_job_count": len(by_job) - len(failed),
        "failed_job_ids": failed,
        "jobs": [by_job[job["job_id"]] for job in jobs],
    }
    if not failed:
        operational = state_directory / "operational_bundle"
        if not operational.exists():
            operational.mkdir()
            atomic_write_json(operational / "host_ledger.json", ledger)
            for job in jobs:
                checkpoint_sha = str(job["checkpoint_sha256"])
                job_root = operational / "jobs" / checkpoint_sha
                job_root.mkdir(parents=True)
                shutil.copy2(
                    artifact / checkpoint_sha / "JOB_COMPLETE.json",
                    job_root / "JOB_COMPLETE.json",
                )
                shutil.copy2(
                    artifact / checkpoint_sha / "metrics/scalar_records.jsonl",
                    job_root / "scalar_records.jsonl",
                )
                shutil.copy2(
                    state_directory / "uploads" / checkpoint_sha / "upload_evidence.json",
                    job_root / "upload_evidence.json",
                )
            build_artifact_manifest(operational)
        operational_destination = (
            f"hf://buckets/{hf_bucket}/evaluations/metric_contract_v1.2/"
            f"_operations/{expected_git_sha}/{host_id}"
        )
        operational_control = state_directory / "operational_upload"
        operational_evidence = _completed_upload_evidence(
            operational_control, operational_destination
        )
        if operational_evidence is None:
            operational_evidence = upload_artifact_tree(
                operational,
                hf_cli=hf_cli,
                bucket=hf_bucket,
                destination=operational_destination,
                control_root=operational_control,
            )
        ledger["operational_destination"] = operational_destination
        ledger["operational_upload_status"] = operational_evidence["status"]
    atomic_write_json(state_directory / "HOST_COMPLETE.json", ledger)
    return ledger
