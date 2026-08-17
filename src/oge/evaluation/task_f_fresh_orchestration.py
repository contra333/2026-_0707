"""Multi-host production adapter for Task F fresh ID-only evaluation.

This adapter binds the already-frozen 1,320 logical records to the source
training shard, preserves the original run/host/GPU placement, and builds an
immutable staged graph for feature export, bridge, geometry, and sibling
alignment.  It contains no protected split, OOD, detector, or training path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from oge.feature_export import verify_task_f_artifact
from oge.studies.artifacts import atomic_write_json, sha256_file
from oge.studies.hashing import canonical_sha256
from oge.studies.staged_pipeline import (
    PIPELINE_SCHEMA_VERSION,
    AtomicStageLedger,
    execute_resource_queues,
    validate_staged_manifest,
)
from oge.studies.supervisor import (
    SupervisorBlockedError,
    build_artifact_manifest,
    production_environment,
    upload_artifact_tree,
    verify_clean_git,
    verify_hf_preflight,
    wait_for_idle_gpus,
)

from .task_f_fresh import (
    EXPECTED_SPECIFICATION_SHA256,
    PLAN_SCHEMA_VERSION,
    sha256_file as fresh_sha256_file,
)


SOURCE_TRAINING_SHA = "9eb3c1fa56d880ea5220badac7bc71ba75786d22"
SOURCE_EXECUTION_ID = "task_f_full_9eb3c1fa"
HOSTS = ("curie", "lise", "precision_medicine")
HOST_COUNTS = {
    "curie": {"runs": 20, "source_exports": 124, "supplemental_exports": 404, "geometry": 264, "cpu_workers": 4},
    "lise": {"runs": 11, "source_exports": 79, "supplemental_exports": 233, "geometry": 156, "cpu_workers": 2},
    "precision_medicine": {"runs": 19, "source_exports": 107, "supplemental_exports": 373, "geometry": 240, "cpu_workers": 4},
}
PAIR_DIRECTIONS = (
    ("coupled_minus_decoupled", "coupled", "decoupled"),
    ("coupled_minus_zero", "coupled", "zero"),
    ("decoupled_minus_zero", "decoupled", "zero"),
)
PROTECTED_TOKENS = (
    "id_test",
    "protected_ood",
    "cifar100",
    "tinyimagenet",
    "mnist",
    "svhn",
    "places365",
    "texture",
)


def _reject_protected(value: Any, label: str) -> None:
    normalized = str(value).lower().replace("-", "_")
    if any(token in normalized for token in PROTECTED_TOKENS):
        raise ValueError(f"{label} contains a protected reference")


def _record_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    checkpoint_role = str(value["checkpoint_role"])
    return (
        str(value["run_id"]),
        checkpoint_role,
        (
            None
            if checkpoint_role == "best_val" or value.get("checkpoint_epoch") is None
            else int(value["checkpoint_epoch"])
        ),
        str(value["depth_tap"]),
        str(value["dataset_split"]),
    )


def _export_job_id(record_id: str) -> str:
    return f"export::{record_id}"


def _bridge_job_id(record_id: str) -> str:
    return f"bridge::{record_id}"


def _geometry_context(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(record["run_id"]),
        str(record["checkpoint_role"]),
        record.get("checkpoint_epoch"),
        str(record["depth_tap"]),
    )


def _normalized_role(record: Mapping[str, Any]) -> str:
    role = str(record["sibling_role"])
    if role == "zero":
        return "zero"
    if role == "alpha_1" or role.endswith("_coupled"):
        return "coupled"
    if role == "alpha_0" or role.endswith("_decoupled"):
        return "decoupled"
    if role == "alpha_0_5":
        return "alpha_0_5"
    raise ValueError(f"unrecognized Task F sibling role {role!r}")


def build_task_f_pipeline_manifest(
    *,
    evaluation_plan: Mapping[str, Any],
    observed_export_jobs: Sequence[Mapping[str, Any]],
    host_assignment: Mapping[str, Any],
    gpu_queues: Mapping[str, Any],
    evaluation_git_sha: str,
) -> dict[str, Any]:
    """Bind the frozen Task F plan to the original three-host GPU placement."""

    if evaluation_plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Task F fresh evaluation plan")
    if evaluation_plan.get("task_f_specification_sha256") != EXPECTED_SPECIFICATION_SHA256:
        raise ValueError("Task F specification identity changed")
    records = evaluation_plan.get("records")
    if not isinstance(records, list) or len(records) != 1320:
        raise ValueError("Task F pipeline requires exactly 1,320 logical records")
    if len(evaluation_git_sha) != 40:
        raise ValueError("evaluation Git SHA must be a full commit identity")
    if host_assignment.get("execution_sha") != SOURCE_TRAINING_SHA:
        raise ValueError("host assignment source training SHA mismatch")
    if gpu_queues.get("execution_sha") != SOURCE_TRAINING_SHA:
        raise ValueError("GPU queue source training SHA mismatch")
    if set(host_assignment.get("host_assignments", {})) != set(HOSTS):
        raise ValueError("Task F host assignment differs from the frozen hosts")

    run_location: dict[str, dict[str, Any]] = {}
    for host_id, queues in gpu_queues.get("hosts", {}).items():
        if host_id not in HOSTS or not isinstance(queues, Mapping):
            raise ValueError("GPU queues contain an unexpected host")
        for raw_index, queue in queues.items():
            index = int(raw_index)
            gpu_uuid = str(queue["gpu_uuid"])
            for run_id in queue["run_ids"]:
                location = {
                    "host_id": host_id,
                    "gpu_index": index,
                    "gpu_uuid": gpu_uuid,
                }
                if str(run_id) in run_location:
                    raise ValueError(f"run {run_id!r} has duplicate GPU placement")
                run_location[str(run_id)] = location
    if len(run_location) != 50:
        raise ValueError("Task F GPU queues must place exactly 50 runs")

    expected_by_key = {_record_key(record): record for record in records}
    if len(expected_by_key) != 1320:
        raise ValueError("Task F logical plan contains duplicate records")
    observed_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for observed in observed_export_jobs:
        _reject_protected(observed.get("dataset_split"), "observed export split")
        key = _record_key(observed)
        if key in observed_by_key:
            raise ValueError("observed export manifest contains duplicates")
        if key not in expected_by_key:
            raise ValueError("observed export is outside the Task F logical plan")
        location = run_location[str(observed["run_id"])]
        for field in ("host_id", "gpu_index", "gpu_uuid"):
            if observed.get(field) != location[field]:
                raise ValueError(f"observed export {field} differs from frozen placement")
        observed_by_key[key] = observed
    if len(observed_by_key) != 310:
        raise ValueError("source export manifest must contain exactly 310 jobs")

    jobs: list[dict[str, Any]] = []
    record_by_id = {str(record["record_id"]): record for record in records}
    for record in records:
        _reject_protected(record["dataset_split"], "Task F record split")
        location = run_location[str(record["run_id"])]
        materialization = "source" if _record_key(record) in observed_by_key else "supplemental"
        export_id = _export_job_id(str(record["record_id"]))
        jobs.append(
            {
                "job_id": export_id,
                "stage": "feature_export",
                "resource": f"gpu:{location['gpu_uuid']}",
                "dependencies": [],
                "host_id": location["host_id"],
                "gpu_index": location["gpu_index"],
                "gpu_uuid": location["gpu_uuid"],
                "materialization": materialization,
                "record": dict(record),
            }
        )
        jobs.append(
            {
                "job_id": _bridge_job_id(str(record["record_id"])),
                "stage": "bridge",
                "resource": f"cpu:{location['host_id']}",
                "dependencies": [export_id],
                "host_id": location["host_id"],
                "record": dict(record),
            }
        )

    paired_records: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    zero_records: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    train_records: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    validation_records: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for record in records:
        context = _geometry_context(record)
        if record["dataset_split"] == "id_train":
            train_records[context] = record
            role = _normalized_role(record)
            paired_records[
                (
                    record["cell_id"],
                    record["training_seed"],
                    record["checkpoint_role"],
                    record.get("checkpoint_epoch"),
                    record["depth_tap"],
                )
            ][role] = record
            if role == "zero":
                zero_key = (
                    record["sibling_group_id"],
                    record["training_seed"],
                    record["checkpoint_role"],
                    record.get("checkpoint_epoch"),
                    record["depth_tap"],
                )
                if zero_key in zero_records:
                    raise ValueError("Task F zero reference is duplicated")
                zero_records[zero_key] = record
        elif record["dataset_split"] == "id_validation":
            validation_records[context] = record
    if set(train_records) != set(validation_records) or len(train_records) != 660:
        raise ValueError("Task F geometry requires 660 train/validation record pairs")
    geometry_by_context: dict[tuple[Any, ...], str] = {}
    cpu_counter: Counter[str] = Counter()
    for context in sorted(train_records, key=str):
        train = train_records[context]
        validation = validation_records[context]
        host_id = run_location[str(train["run_id"])]["host_id"]
        slot = cpu_counter[host_id] % int(HOST_COUNTS[host_id]["cpu_workers"])
        cpu_counter[host_id] += 1
        geometry_id = "geometry::" + "::".join(map(str, context))
        geometry_by_context[context] = geometry_id
        jobs.append(
            {
                "job_id": geometry_id,
                "stage": "geometry",
                "resource": f"cpu:{host_id}:{slot}",
                "dependencies": [
                    _bridge_job_id(str(train["record_id"])),
                    _bridge_job_id(str(validation["record_id"])),
                ],
                "host_id": host_id,
                "train_record_id": train["record_id"],
                "validation_record_id": validation["record_id"],
            }
        )

    alignment_counts: Counter[str] = Counter()
    for pair_context, roles in sorted(paired_records.items(), key=lambda item: str(item[0])):
        for direction, left_role, right_role in PAIR_DIRECTIONS:
            left_train = roles.get(left_role)
            right_train = roles.get(right_role)
            if right_role == "zero" and left_train is not None:
                right_train = zero_records.get(
                    (
                        left_train["sibling_group_id"],
                        left_train["training_seed"],
                        left_train["checkpoint_role"],
                        left_train.get("checkpoint_epoch"),
                        left_train["depth_tap"],
                    )
                )
            if left_train is None or right_train is None:
                raise ValueError(f"Task F alignment context lacks {direction}: {pair_context}")
            left_validation = validation_records[_geometry_context(left_train)]
            right_validation = validation_records[_geometry_context(right_train)]
            host_id = run_location[str(left_train["run_id"])]["host_id"]
            if run_location[str(right_train["run_id"])]["host_id"] != host_id:
                raise ValueError("Task F sibling alignment crosses hosts")
            slot = alignment_counts[host_id] % int(HOST_COUNTS[host_id]["cpu_workers"])
            alignment_counts[host_id] += 1
            alignment_id = "alignment::" + "::".join(map(str, (*pair_context, direction)))
            bridge_ids = [
                _bridge_job_id(str(record["record_id"]))
                for record in (
                    left_train,
                    left_validation,
                    right_train,
                    right_validation,
                )
            ]
            jobs.append(
                {
                    "job_id": alignment_id,
                    "stage": "alignment",
                    "resource": f"cpu:{host_id}:{slot}",
                    "dependencies": [
                        *bridge_ids,
                        geometry_by_context[_geometry_context(left_train)],
                        geometry_by_context[_geometry_context(right_train)],
                    ],
                    "host_id": host_id,
                    "pair_direction": direction,
                    "left_train_record_id": left_train["record_id"],
                    "left_validation_record_id": left_validation["record_id"],
                    "right_train_record_id": right_train["record_id"],
                    "right_validation_record_id": right_validation["record_id"],
                }
            )

    supplemental_counts = Counter(
        job["host_id"]
        for job in jobs
        if job["stage"] == "feature_export" and job["materialization"] == "supplemental"
    )
    geometry_counts = Counter(job["host_id"] for job in jobs if job["stage"] == "geometry")
    expected_supplemental = {
        host: int(values["supplemental_exports"]) for host, values in HOST_COUNTS.items()
    }
    expected_geometry = {host: int(values["geometry"]) for host, values in HOST_COUNTS.items()}
    if dict(supplemental_counts) != expected_supplemental:
        raise ValueError("Task F supplemental host counts changed")
    if dict(geometry_counts) != expected_geometry:
        raise ValueError("Task F geometry host counts changed")
    if sum(alignment_counts.values()) != 657:
        raise ValueError("Task F sibling alignment coverage changed")
    jobs_by_id = {job["job_id"]: job for job in jobs}
    sentinel_by_host: dict[str, dict[str, Any]] = {}
    for host_id in HOSTS:
        candidates = [
            job
            for job in jobs
            if job["stage"] == "geometry"
            and job["host_id"] == host_id
            and record_by_id[str(job["train_record_id"])]["checkpoint_epoch"] == 10
            and record_by_id[str(job["train_record_id"])]["depth_tap"] == "penultimate"
        ]
        if not candidates:
            raise ValueError(f"Task F host {host_id} lacks a production sentinel")
        sentinel = sorted(candidates, key=lambda job: job["job_id"])[0]
        bridge_ids = list(sentinel["dependencies"])
        export_ids = [jobs_by_id[bridge_id]["dependencies"][0] for bridge_id in bridge_ids]
        supplemental_ids = [
            export_id
            for export_id in export_ids
            if jobs_by_id[export_id]["materialization"] == "supplemental"
        ]
        if len(supplemental_ids) != 1:
            raise ValueError("Task F production sentinel must add one validation export")
        sentinel_by_host[host_id] = {
            "export_job_ids": supplemental_ids,
            "bridge_job_ids": bridge_ids,
            "geometry_job_id": sentinel["job_id"],
            "validation": "checksum_shape_bridge_geometry_reconstruction",
        }
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "adapter": "task_f_fresh_id_multihost_v1",
        "identity": {
            "source_execution_id": SOURCE_EXECUTION_ID,
            "source_training_sha": SOURCE_TRAINING_SHA,
            "evaluation_git_sha": evaluation_git_sha,
            "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
            "evaluation_plan_sha256": canonical_sha256(evaluation_plan),
            "host_assignment_sha256": canonical_sha256(host_assignment),
            "gpu_queues_sha256": canonical_sha256(gpu_queues),
        },
        "counts": {
            "research_runs": 50,
            "logical_exports": 1320,
            "source_exports": 310,
            "supplemental_exports": 1010,
            "bridges": 1320,
            "geometry": 660,
            "alignments": 657,
            "supplemental_by_host": dict(sorted(supplemental_counts.items())),
            "geometry_by_host": dict(sorted(geometry_counts.items())),
            "alignment_by_host": dict(sorted(alignment_counts.items())),
        },
        "production_sentinel_by_host": sentinel_by_host,
        "records": [dict(record) for record in records],
        "jobs": jobs,
    }
    return validate_task_f_pipeline_manifest(manifest)


def validate_task_f_pipeline_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = validate_staged_manifest(value)
    identity = manifest["identity"]
    if identity.get("source_training_sha") != SOURCE_TRAINING_SHA:
        raise ValueError("Task F pipeline source SHA mismatch")
    if identity.get("task_f_specification_sha256") != EXPECTED_SPECIFICATION_SHA256:
        raise ValueError("Task F pipeline specification mismatch")
    counts = manifest.get("counts", {})
    expected = {
        "research_runs": 50,
        "logical_exports": 1320,
        "source_exports": 310,
        "supplemental_exports": 1010,
        "bridges": 1320,
        "geometry": 660,
        "alignments": 657,
    }
    if any(int(counts.get(key, -1)) != count for key, count in expected.items()):
        raise ValueError("Task F pipeline counts differ from the frozen contract")
    if set(counts.get("supplemental_by_host", {})) != set(HOSTS):
        raise ValueError("Task F pipeline host coverage changed")
    sentinels = manifest.get("production_sentinel_by_host")
    if not isinstance(sentinels, Mapping) or set(sentinels) != set(HOSTS):
        raise ValueError("Task F pipeline production sentinels changed")
    for job in manifest["jobs"]:
        _reject_protected(job, "Task F pipeline job")
        if job.get("host_id") not in HOSTS:
            raise ValueError("Task F pipeline job has an unexpected host")
    return manifest


def validate_source_gate_documents(documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Validate all three pre-existing training/export terminal witnesses."""

    if set(documents) != set(HOSTS):
        raise SupervisorBlockedError("global source gate requires all three hosts")
    total_runs = 0
    total_exports = 0
    hosts: dict[str, Any] = {}
    for host_id in HOSTS:
        document = documents[host_id]
        marker = document.get("marker")
        validation = document.get("validation")
        if not isinstance(marker, Mapping) or not isinstance(validation, Mapping):
            raise SupervisorBlockedError(f"source gate documents are incomplete for {host_id}")
        if marker.get("status") != "REMOTE_VERIFIED" or marker.get("host_id") != host_id:
            raise SupervisorBlockedError(f"source completion marker is invalid for {host_id}")
        if validation.get("status") != "PASS" or validation.get("host_id") != host_id:
            raise SupervisorBlockedError(f"source validation is invalid for {host_id}")
        if marker.get("execution_sha") != SOURCE_TRAINING_SHA or validation.get("execution_sha") != SOURCE_TRAINING_SHA:
            raise SupervisorBlockedError(f"source training SHA mismatch for {host_id}")
        if validation.get("specification_sha256") != EXPECTED_SPECIFICATION_SHA256:
            raise SupervisorBlockedError(f"Task F specification mismatch for {host_id}")
        expected = HOST_COUNTS[host_id]
        run_count = int(validation.get("run_count", -1))
        export_count = int(validation.get("export_count", -1))
        if run_count != expected["runs"] or export_count != expected["source_exports"]:
            raise SupervisorBlockedError(f"source accounting mismatch for {host_id}")
        if int(marker.get("run_count", -1)) != run_count or int(
            marker.get("export_count", -1)
        ) != export_count:
            raise SupervisorBlockedError(f"source marker accounting mismatch for {host_id}")
        total_runs += run_count
        total_exports += export_count
        hosts[host_id] = {
            "run_count": run_count,
            "export_count": export_count,
            "validation_sha256": fresh_sha256_file_from_payload(validation),
        }
    if total_runs != 50 or total_exports != 310:
        raise SupervisorBlockedError("global source accounting is not 50 runs / 310 exports")
    return {
        "status": "PASS",
        "source_training_sha": SOURCE_TRAINING_SHA,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "run_count": total_runs,
        "export_count": total_exports,
        "hosts": hosts,
    }


def validate_local_source_gate(
    *,
    source_root: str | Path,
    host_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one host's completed local source witness before evaluation compute."""

    if host_id not in HOSTS:
        raise SupervisorBlockedError(f"unsupported Task F host: {host_id}")
    source = Path(source_root)
    validation_path = source / "host_validation.json"
    status_path = source / "control" / "host_finalizer.status"
    if not validation_path.is_file() or not status_path.is_file():
        raise SupervisorBlockedError("local Task F source validation witness is absent")
    finalizer_status = status_path.read_text(encoding="utf-8").strip()
    if finalizer_status != "UPLOADING" and not finalizer_status.startswith(
        "COMPLETE REMOTE_VERIFIED"
    ):
        raise SupervisorBlockedError(
            f"local Task F source finalizer is not upload-ready: {finalizer_status}"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not isinstance(validation, Mapping):
        raise SupervisorBlockedError("local Task F source validation is not a mapping")
    expected = HOST_COUNTS[host_id]
    if (
        validation.get("status") != "PASS"
        or validation.get("host_id") != host_id
        or validation.get("execution_sha") != SOURCE_TRAINING_SHA
        or validation.get("specification_sha256") != EXPECTED_SPECIFICATION_SHA256
        or int(validation.get("run_count", -1)) != expected["runs"]
        or int(validation.get("export_count", -1)) != expected["source_exports"]
    ):
        raise SupervisorBlockedError("local Task F source validation identity mismatch")
    validated_manifest = validate_task_f_pipeline_manifest(manifest)
    host_jobs = [job for job in validated_manifest["jobs"] if job["host_id"] == host_id]
    expected_run_ids = {
        str(job["record"]["run_id"])
        for job in host_jobs
        if job["stage"] == "feature_export"
    }
    expected_source_exports = {
        _record_key(job["record"])
        for job in host_jobs
        if job["stage"] == "feature_export" and job["materialization"] == "source"
    }
    rows = validation.get("runs")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise SupervisorBlockedError("local Task F source run witnesses are invalid")
    actual_run_ids = [str(row.get("run_id")) for row in rows]
    if len(actual_run_ids) != len(set(actual_run_ids)) or set(actual_run_ids) != expected_run_ids:
        raise SupervisorBlockedError("local Task F source run coverage mismatch")
    if sum(int(row.get("export_count", -1)) for row in rows) != len(
        expected_source_exports
    ):
        raise SupervisorBlockedError("local Task F source export accounting mismatch")
    for row in rows:
        run_id = str(row["run_id"])
        checkpoint = source / "runs" / run_id / "checkpoints" / "last.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size != int(
            row.get("last_pt_bytes", -1)
        ):
            raise SupervisorBlockedError(
                f"local Task F source checkpoint witness mismatch: {run_id}"
            )
    return {
        "status": "PASS",
        "host_id": host_id,
        "source_training_sha": SOURCE_TRAINING_SHA,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "run_count": expected["runs"],
        "export_count": expected["source_exports"],
        "finalizer_status_at_start": finalizer_status,
        "validation_sha256": canonical_sha256(validation),
        "validation_file_sha256": sha256_file(validation_path),
        "pipeline_manifest_sha256": validated_manifest["manifest_sha256"],
    }


def validate_remote_source_gate_matches_local(
    *,
    local_gate: Mapping[str, Any],
    remote_gate: Mapping[str, Any],
) -> None:
    """Require the eventual remote witness to match the local compute witness."""

    host_id = str(local_gate.get("host_id"))
    if (
        local_gate.get("status") != "PASS"
        or remote_gate.get("status") != "PASS"
        or remote_gate.get("source_training_sha") != SOURCE_TRAINING_SHA
        or remote_gate.get("task_f_specification_sha256")
        != EXPECTED_SPECIFICATION_SHA256
    ):
        raise SupervisorBlockedError("Task F local/remote source gate identity mismatch")
    remote_host = remote_gate.get("hosts", {}).get(host_id)
    if not isinstance(remote_host, Mapping) or remote_host.get(
        "validation_sha256"
    ) != local_gate.get("validation_sha256"):
        raise SupervisorBlockedError(
            f"Task F remote source witness differs from local compute witness: {host_id}"
        )


def fresh_sha256_file_from_payload(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def load_source_gate_directory(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    documents = {}
    for host_id in HOSTS:
        validation_path = root / host_id / "host_validation.json"
        marker_path = root / host_id / "REMOTE_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("validation_sha256") != sha256_file(validation_path):
            raise SupervisorBlockedError(
                f"source validation checksum differs from marker for {host_id}"
            )
        documents[host_id] = {
            "marker": marker,
            "validation": json.loads(validation_path.read_text(encoding="utf-8")),
        }
    return validate_source_gate_documents(documents)


def refresh_source_gate_from_hf(
    *,
    hf_cli: str | Path,
    destination: str | Path,
    bucket: str = "contra333/ICLR_RUN",
    command_timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Download three small source witnesses and publish them only after validation."""

    root = Path(destination)
    temporary = root.with_name(f".{root.name}.refresh-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"source-gate refresh path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for host_id in HOSTS:
            host_root = temporary / host_id
            host_root.mkdir()
            remote = f"hf://buckets/{bucket}/servers/{host_id}/{SOURCE_EXECUTION_ID}"
            for remote_relative, local_name in (
                ("REMOTE_COMPLETE.json", "REMOTE_COMPLETE.json"),
                ("metadata/host_validation.json", "host_validation.json"),
            ):
                subprocess.run(
                    [str(hf_cli), "buckets", "cp", f"{remote}/{remote_relative}", str(host_root / local_name)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=command_timeout_seconds,
                )
        result = load_source_gate_directory(temporary)
        if root.exists():
            preserved = load_source_gate_directory(root)
            if preserved != result:
                raise ValueError("preserved source-gate evidence differs from HF")
        else:
            os.replace(temporary, root)
        return result
    finally:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)


def wait_for_global_source_gate(
    *,
    hf_cli: str | Path,
    gate_root: str | Path,
    poll_seconds: float = 60.0,
    timeout_hours: float = 72.0,
    command_timeout_seconds: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_hours * 3600.0
    last_error = "not checked"
    while time.monotonic() < deadline:
        try:
            return refresh_source_gate_from_hf(
                hf_cli=hf_cli,
                destination=gate_root,
                command_timeout_seconds=command_timeout_seconds,
            )
        except (
            OSError,
            ValueError,
            KeyError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            SupervisorBlockedError,
        ) as exc:
            last_error = str(exc)
            sleep(poll_seconds)
    raise SupervisorBlockedError(f"global source gate timed out: {last_error}")


def index_feature_artifacts(roots: Sequence[str | Path]) -> dict[tuple[Any, ...], Path]:
    """Checksum-verify every visible feature bundle and reject duplicate identities."""

    result: dict[tuple[Any, ...], Path] = {}
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("*/manifest.json")):
            verified = verify_task_f_artifact(manifest_path.parent)
            manifest = verified["manifest"]
            _reject_protected(manifest["dataset_split"], "feature artifact split")
            key = _record_key(manifest)
            if key in result:
                raise ValueError(f"duplicate Task F feature artifact: {key}")
            result[key] = manifest_path.parent
    return result


def _job_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(job["job_id"]): job for job in manifest["jobs"]}


def _run_json_worker(
    *,
    repository_root: Path,
    python: str | Path,
    action: str,
    spec_path: Path,
    blas_threads: int,
) -> dict[str, Any]:
    environment = production_environment()
    environment["PYTHONPATH"] = str(repository_root / "src")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = str(blas_threads)
    completed = subprocess.run(
        [str(python), str(repository_root / "scripts/supervise_task_f_fresh_id.py"), action, "--job-spec", str(spec_path)],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{action} worker failed: {completed.stderr[-2000:]}")
    output = json.loads(completed.stdout.strip().splitlines()[-1])
    if output.get("status") != "PASS":
        raise RuntimeError(f"{action} worker returned non-PASS")
    return output


def load_completed_upload_evidence(
    control_root: str | Path, destination: str
) -> dict[str, Any] | None:
    control = Path(control_root)
    marker_path = control / "REMOTE_COMPLETE.json"
    evidence_path = control / "upload_evidence.json"
    if not marker_path.is_file() or not evidence_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if marker.get("status") != "REMOTE_VERIFIED" or marker.get("destination") != destination:
        raise ValueError("preserved remote completion marker identity mismatch")
    if evidence.get("status") != "REMOTE_VERIFIED" or evidence.get("destination") != destination:
        raise ValueError("preserved remote upload evidence identity mismatch")
    return evidence


def execute_task_f_host(
    *,
    repository_root: str | Path,
    manifest_path: str | Path,
    host_id: str,
    expected_evaluation_git_sha: str,
    source_root: str | Path,
    data_root: str | Path,
    id_train_input: str | Path,
    id_validation_input: str | Path,
    dataset_config_path: str | Path,
    artifact_root: str | Path,
    state_root: str | Path,
    python: str | Path,
    hf_cli: str | Path,
    batch_size: int = 512,
    blas_threads: int = 4,
    minimum_free_gb: float = 100.0,
) -> dict[str, Any]:
    """Compute one source-local host shard without publishing a remote terminal."""

    if host_id not in HOSTS:
        raise ValueError("unsupported Task F host")
    repository = Path(repository_root).resolve()
    manifest = validate_task_f_pipeline_manifest(
        json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    )
    if manifest["identity"]["evaluation_git_sha"] != expected_evaluation_git_sha:
        raise SupervisorBlockedError("evaluation execution SHA differs from manifest")
    verify_clean_git(repository, expected_evaluation_git_sha)
    source = Path(source_root)
    if not source.is_dir():
        raise SupervisorBlockedError(f"source training root is absent: {source}")
    for path in (data_root, id_train_input, id_validation_input, dataset_config_path):
        if not Path(path).exists():
            raise SupervisorBlockedError(f"required Task F input is absent: {path}")
        _reject_protected(path, "Task F host input path")
    artifact = Path(artifact_root)
    state = Path(state_root)
    artifact.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    free_bytes = os.statvfs(artifact).f_bavail * os.statvfs(artifact).f_frsize
    host_pipeline_jobs = [
        job for job in manifest["jobs"] if job["host_id"] == host_id
    ]
    sample_counts = {"id_train": 45000, "id_validation": 5000}
    tap_widths = {"stage1": 160, "stage2": 320, "stage3": 640, "penultimate": 640}
    supplemental_feature_bytes = sum(
        sample_counts[job["record"]["dataset_split"]]
        * tap_widths[job["record"]["depth_tap"]]
        * 4
        + 1024**2
        for job in host_pipeline_jobs
        if job["stage"] == "feature_export"
        and job["materialization"] == "supplemental"
    )
    static_analysis_upper_bytes = (
        sum(job["stage"] == "geometry" for job in host_pipeline_jobs) * 96 * 1024**2
        + sum(job["stage"] == "alignment" for job in host_pipeline_jobs) * 24 * 1024**2
        + sum(job["stage"] == "bridge" for job in host_pipeline_jobs) * 4 * 1024**2
    )
    conservative_required_bytes = max(
        int(minimum_free_gb * 1024**3),
        int(1.25 * (supplemental_feature_bytes + static_analysis_upper_bytes)),
    )
    if free_bytes < conservative_required_bytes:
        raise SupervisorBlockedError("Task F evaluation filesystem lacks minimum free disk")
    atomic_write_json(
        state / "DISK_PREFLIGHT.json",
        {
            "status": "PASS",
            "host_id": host_id,
            "actual_free_bytes": free_bytes,
            "supplemental_feature_planning_bytes": supplemental_feature_bytes,
            "analysis_static_upper_planning_bytes": static_analysis_upper_bytes,
            "safety_factor": 1.25,
            "minimum_free_gb": minimum_free_gb,
            "conservative_required_bytes": conservative_required_bytes,
            "performance_benchmark_used": False,
        },
    )
    verify_hf_preflight(hf_cli, account="contra333", bucket="contra333/ICLR_RUN")
    host_jobs = host_pipeline_jobs
    host_manifest = {
        **manifest,
        "jobs": host_jobs,
        "manifest_sha256": manifest["manifest_sha256"],
    }
    # The host ledger retains the full manifest identity but only its local job set.
    ledger_manifest = dict(host_manifest)
    ledger_manifest.pop("manifest_sha256", None)
    ledger_manifest["parent_manifest_sha256"] = manifest["manifest_sha256"]
    ledger_manifest = validate_staged_manifest(ledger_manifest)
    ledger = AtomicStageLedger(state / "ledger.json", ledger_manifest)
    jobs_by_id = _job_by_id(ledger_manifest)

    source_exports = source / "exports"
    supplemental_exports = artifact / "exports"
    feature_index = index_feature_artifacts((source_exports, supplemental_exports))
    feature_paths: dict[str, Path] = {}
    export_jobs = [job for job in host_jobs if job["stage"] == "feature_export"]
    for job in export_jobs:
        record = job["record"]
        key = _record_key(record)
        if key in feature_index:
            path = feature_index[key]
            feature_paths[job["job_id"]] = path
            if ledger.job_status(job["job_id"]) != "PASS":
                ledger.start(job["job_id"], resumed=True)
                ledger.pass_job(job["job_id"], {"artifact_path": str(path), "resumed": True})
        elif job["materialization"] == "source":
            raise SupervisorBlockedError(f"verified source feature is absent: {record['record_id']}")

    supplemental = [
        job
        for job in export_jobs
        if job["materialization"] == "supplemental" and ledger.job_status(job["job_id"]) != "PASS"
    ]
    gpu_uuids = sorted({str(job["gpu_uuid"]) for job in supplemental})
    if gpu_uuids:
        wait_for_idle_gpus(
            gpu_uuids,
            state_path=state / "gpu_wait_state.json",
            state={"host_id": host_id, "status": "PREFLIGHT"},
            timeout_hours=72.0,
        )

    records = {str(record["record_id"]): record for record in manifest["records"]}
    checkpoint_root = source / "runs"
    id_inputs = {"id_train": Path(id_train_input), "id_validation": Path(id_validation_input)}

    def export_worker(job: Mapping[str, Any]) -> Mapping[str, Any]:
        record = job["record"]
        checkpoint = checkpoint_root / record["run_id"] / record["checkpoint_relative_path"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Task F checkpoint is absent: {checkpoint}")
        environment = production_environment()
        environment.update(
            {
                "PYTHONPATH": str(repository / "src"),
                "CUDA_VISIBLE_DEVICES": str(job["gpu_index"]),
                "OGE_PHYSICAL_GPU_UUID": str(job["gpu_uuid"]),
            }
        )
        command = [
            str(python),
            str(repository / "scripts/export_task_f_id_features.py"),
            "--checkpoint",
            str(checkpoint),
            "--input-npz",
            str(id_inputs[str(record["dataset_split"])]),
            "--artifact-root",
            str(supplemental_exports),
            "--dataset-split",
            str(record["dataset_split"]),
            "--depth-tap",
            str(record["depth_tap"]),
            "--device",
            "cuda:0",
            "--batch-size",
            str(batch_size),
            "--progress-every-batches",
            "25",
        ]
        log = state / "logs" / f"{job['job_id'].replace('/', '_')}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Task F feature export failed rc={completed.returncode}")
        refreshed = index_feature_artifacts((supplemental_exports,))
        path = refreshed.get(_record_key(record))
        if path is None:
            raise ValueError("completed Task F export is absent from verified index")
        feature_paths[job["job_id"]] = path
        return {"artifact_path": str(path), "resumed": False, "gpu_uuid": job["gpu_uuid"]}

    cpu_workers = int(HOST_COUNTS[host_id]["cpu_workers"])
    cpu_jobs = [
        job for job in host_jobs if job["stage"] in {"bridge", "geometry"}
    ]
    worker_specs = state / "worker_specs"
    worker_specs.mkdir(exist_ok=True)
    cpu_pool = ThreadPoolExecutor(max_workers=cpu_workers)

    def cpu_worker(job: Mapping[str, Any]) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "job_id": job["job_id"],
            "stage": job["stage"],
            "dataset_config_path": str(dataset_config_path),
            "bridge_root": str(artifact / "bridges"),
            "geometry_root": str(artifact / "geometry"),
            "chunk_size": 2048,
        }
        if job["stage"] == "bridge":
            record = job["record"]
            export_id = job["dependencies"][0]
            result = ledger.snapshot()["jobs"][export_id]["result"]
            spec.update(
                {
                    "record": record,
                    "checkpoint_path": str(checkpoint_root / record["run_id"] / record["checkpoint_relative_path"]),
                    "feature_artifact_path": result["artifact_path"],
                }
            )
            action = "_bridge-worker"
        else:
            train_bridge, validation_bridge = [
                ledger.snapshot()["jobs"][dependency]["result"]["artifact_path"]
                for dependency in job["dependencies"]
            ]
            train_record = records[str(job["train_record_id"])]
            validation_record = records[str(job["validation_record_id"])]
            spec.update(
                {
                    "train_binding": {
                        "record_id": train_record["record_id"],
                        "bridge_path": train_bridge,
                        "feature_artifact_path": ledger.snapshot()["jobs"][_export_job_id(str(train_record["record_id"]))]["result"]["artifact_path"],
                    },
                    "validation_binding": {
                        "record_id": validation_record["record_id"],
                        "bridge_path": validation_bridge,
                        "feature_artifact_path": ledger.snapshot()["jobs"][_export_job_id(str(validation_record["record_id"]))]["result"]["artifact_path"],
                    },
                }
            )
            action = "_geometry-worker"
        spec_path = worker_specs / f"{canonical_sha256(spec)}.json"
        atomic_write_json(spec_path, spec)
        return _run_json_worker(
            repository_root=repository,
            python=python,
            action=action,
            spec_path=spec_path,
            blas_threads=blas_threads,
        )

    def execute_one_cpu(job: Mapping[str, Any]) -> dict[str, Any]:
        job_id = str(job["job_id"])
        if ledger.job_status(job_id) == "PASS":
            result = ledger.snapshot()["jobs"][job_id]["result"]
            return dict(result)
        if not ledger.dependencies_pass(job):
            raise RuntimeError(f"Task F sentinel dependency is not PASS: {job_id}")
        ledger.start(job_id, resource=job["resource"], production_sentinel=True)
        try:
            result = cpu_worker(job)
        except BaseException as exc:
            ledger.fail_job(job_id, exc)
            raise
        ledger.pass_job(job_id, result)
        return result

    sentinel = manifest["production_sentinel_by_host"][host_id]
    sentinel_exports = [jobs_by_id[job_id] for job_id in sentinel["export_job_ids"]]
    execute_resource_queues(jobs=sentinel_exports, ledger=ledger, worker=export_worker)
    for job_id in sentinel["bridge_job_ids"]:
        execute_one_cpu(jobs_by_id[job_id])
    sentinel_geometry = execute_one_cpu(jobs_by_id[sentinel["geometry_job_id"]])
    atomic_write_json(
        state / "PRODUCTION_SENTINEL_PASS.json",
        {
            "status": "PASS",
            "host_id": host_id,
            "validation": sentinel["validation"],
            "export_job_ids": sentinel["export_job_ids"],
            "bridge_job_ids": sentinel["bridge_job_ids"],
            "geometry_job_id": sentinel["geometry_job_id"],
            "geometry_artifact_path": sentinel_geometry["artifact_path"],
        },
    )

    gpu_future: Future[Any] | None = None
    gpu_pool = ThreadPoolExecutor(max_workers=1)
    if supplemental:
        gpu_future = gpu_pool.submit(
            execute_resource_queues,
            jobs=supplemental,
            ledger=ledger,
            worker=export_worker,
        )

    pending = {job["job_id"]: job for job in cpu_jobs if ledger.job_status(job["job_id"]) != "PASS"}
    running: dict[Future[dict[str, Any]], Mapping[str, Any]] = {}

    while pending or running:
        progressed = False
        ready = [
            job for job in pending.values() if ledger.dependencies_pass(job)
        ]
        ready.sort(key=lambda job: (0 if job["stage"] == "bridge" else 1, job["job_id"]))
        while ready and len(running) < cpu_workers:
            job = ready.pop(0)
            if ledger.start(job["job_id"], resource=job["resource"]):
                future = cpu_pool.submit(cpu_worker, job)
                running[future] = job
            pending.pop(job["job_id"])
            progressed = True
        for future, job in list(running.items()):
            if not future.done():
                continue
            try:
                result = future.result()
            except BaseException as exc:
                ledger.fail_job(job["job_id"], exc)
                raise
            else:
                ledger.pass_job(job["job_id"], result)
            running.pop(future)
            progressed = True
        if not progressed:
            if gpu_future is not None and gpu_future.done():
                gpu_future.result()
                gpu_future = None
            time.sleep(1.0)
    if gpu_future is not None:
        gpu_future.result()
    gpu_pool.shutdown()
    cpu_pool.shutdown()

    alignment_jobs = [job for job in host_jobs if job["stage"] == "alignment"]

    def alignment_worker(job: Mapping[str, Any]) -> Mapping[str, Any]:
        spec = {
            "job_id": job["job_id"],
            "stage": "alignment",
            "pair_direction": job["pair_direction"],
            "alignment_root": str(artifact / "alignments"),
            "chunk_size": 2048,
        }
        for label in (
            "left_train",
            "left_validation",
            "right_train",
            "right_validation",
        ):
            record = records[str(job[f"{label}_record_id"])]
            spec[f"{label}_binding"] = {
                "record_id": record["record_id"],
                "bridge_path": ledger.snapshot()["jobs"][_bridge_job_id(str(record["record_id"]))]["result"]["artifact_path"],
                "feature_artifact_path": ledger.snapshot()["jobs"][_export_job_id(str(record["record_id"]))]["result"]["artifact_path"],
            }
        spec_path = worker_specs / f"{canonical_sha256(spec)}.json"
        atomic_write_json(spec_path, spec)
        return _run_json_worker(
            repository_root=repository,
            python=python,
            action="_alignment-worker",
            spec_path=spec_path,
            blas_threads=blas_threads,
        )

    execute_resource_queues(jobs=alignment_jobs, ledger=ledger, worker=alignment_worker)
    snapshot = ledger.snapshot()
    if snapshot["status"] != "PASS":
        raise RuntimeError("Task F host pipeline did not reach PASS")
    stage_counts = Counter(
        jobs_by_id[job_id]["stage"]
        for job_id, record in snapshot["jobs"].items()
        if record["status"] == "PASS"
    )
    summary = {
        "schema_version": "task_f_fresh_id_host_terminal_v1",
        "status": "PASS",
        "host_id": host_id,
        "source_training_sha": SOURCE_TRAINING_SHA,
        "evaluation_git_sha": expected_evaluation_git_sha,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "parent_manifest_sha256": manifest["manifest_sha256"],
        "stage_counts": dict(sorted(stage_counts.items())),
        "protected_data_access": False,
    }
    operational = artifact / "operational_bundle"
    if operational.exists():
        existing = json.loads((operational / "HOST_COMPLETE.json").read_text(encoding="utf-8"))
        if existing != summary:
            raise FileExistsError("preserved Task F host terminal differs")
    else:
        operational.mkdir(parents=True)
        atomic_write_json(operational / "HOST_COMPLETE.json", summary)
        seed_records = []
        from oge.analysis.task_f_fresh_id import geometry_seed_records

        geometry_paths = [
            Path(record["result"]["artifact_path"])
            for job_id, record in snapshot["jobs"].items()
            if jobs_by_id[job_id]["stage"] == "geometry" and record["status"] == "PASS"
        ]
        seed_records = geometry_seed_records(geometry_paths)
        atomic_write_json(
            operational / "seed_records.json",
            {
                "schema_version": "task_f_fresh_id_seed_records_v1",
                "record_count": len(seed_records),
                "records": seed_records,
            },
        )
        atomic_write_json(
            operational / "alignment_inventory.json",
            {
                "schema_version": "task_f_fresh_id_alignment_inventory_v1",
                "alignment_count": int(stage_counts["alignment"]),
                "artifacts": sorted(
                    [
                        {
                            "output_identity_sha256": Path(record["result"]["artifact_path"]).name,
                            "manifest_sha256": sha256_file(
                                Path(record["result"]["artifact_path"]) / "manifest.json"
                            ),
                        }
                        for job_id, record in snapshot["jobs"].items()
                        if jobs_by_id[job_id]["stage"] == "alignment"
                        and record["status"] == "PASS"
                    ],
                    key=lambda row: row["output_identity_sha256"],
                ),
            },
        )
        build_artifact_manifest(operational)
    atomic_write_json(state / "HOST_COMPUTE_COMPLETE.json", summary)
    return summary


def publish_task_f_host_result(
    *,
    artifact_root: str | Path,
    state_root: str | Path,
    host_id: str,
    expected_evaluation_git_sha: str,
    hf_cli: str | Path,
    remote_source_gate: Mapping[str, Any],
    local_source_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a computed host shard only after the global source remote gate."""

    if host_id not in HOSTS:
        raise ValueError("unsupported Task F host")
    if (
        remote_source_gate.get("status") != "PASS"
        or remote_source_gate.get("source_training_sha") != SOURCE_TRAINING_SHA
        or remote_source_gate.get("task_f_specification_sha256")
        != EXPECTED_SPECIFICATION_SHA256
        or int(remote_source_gate.get("run_count", -1)) != 50
        or int(remote_source_gate.get("export_count", -1)) != 310
        or set(remote_source_gate.get("hosts", {})) != set(HOSTS)
    ):
        raise SupervisorBlockedError(
            "Task F host publication requires the exact global source remote gate"
        )
    if local_source_gate is not None:
        validate_remote_source_gate_matches_local(
            local_gate=local_source_gate,
            remote_gate=remote_source_gate,
        )
    artifact = Path(artifact_root)
    state = Path(state_root)
    operational = artifact / "operational_bundle"
    compute_path = state / "HOST_COMPUTE_COMPLETE.json"
    terminal_path = operational / "HOST_COMPLETE.json"
    if not compute_path.is_file() or not terminal_path.is_file():
        raise SupervisorBlockedError("Task F host compute terminal is absent")
    compute = json.loads(compute_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if compute != terminal:
        raise SupervisorBlockedError("Task F host compute terminal differs from bundle")
    if (
        terminal.get("status") != "PASS"
        or terminal.get("host_id") != host_id
        or terminal.get("source_training_sha") != SOURCE_TRAINING_SHA
        or terminal.get("evaluation_git_sha") != expected_evaluation_git_sha
        or terminal.get("task_f_specification_sha256")
        != EXPECTED_SPECIFICATION_SHA256
    ):
        raise SupervisorBlockedError("Task F host compute identity mismatch")
    destination = (
        f"hf://buckets/contra333/ICLR_RUN/evaluations/task_f_fresh_id_v1/"
        f"{expected_evaluation_git_sha}/{host_id}"
    )
    upload_control = state / "operational_upload"
    evidence = load_completed_upload_evidence(upload_control, destination)
    if evidence is None:
        evidence = upload_artifact_tree(
            operational,
            hf_cli=hf_cli,
            bucket="contra333/ICLR_RUN",
            destination=destination,
            control_root=upload_control,
        )
    result = {**terminal, "remote": evidence}
    atomic_write_json(state / "HOST_COMPLETE.json", result)
    return result


def collect_task_f_host_summaries(
    *,
    host_roots: Mapping[str, str | Path],
    evaluation_plan: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Verify three small host shards and run deterministic central aggregation."""

    if set(host_roots) != set(HOSTS):
        raise ValueError("Task F collection requires exactly three host roots")
    all_records: list[dict[str, Any]] = []
    host_records: dict[str, Any] = {}
    total_alignments = 0
    for host_id in HOSTS:
        root = Path(host_roots[host_id])
        artifact_manifest_path = root / "artifact_manifest.json"
        artifact_manifest = json.loads(
            artifact_manifest_path.read_text(encoding="utf-8")
        )
        sidecar = root / "artifact_manifest.json.sha256"
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if len(fields) < 2 or fields[0] != sha256_file(artifact_manifest_path):
            raise ValueError(f"host {host_id} artifact manifest sidecar mismatch")
        expected_files = {
            str(row["path"]): (int(row["size"]), str(row["sha256"]))
            for row in artifact_manifest.get("files", ())
        }
        if len(expected_files) != int(artifact_manifest.get("file_count", -1)):
            raise ValueError(f"host {host_id} artifact manifest count mismatch")
        if sum(size for size, _ in expected_files.values()) != int(
            artifact_manifest.get("total_size", -1)
        ):
            raise ValueError(f"host {host_id} artifact manifest size mismatch")
        for relative, (size, digest) in expected_files.items():
            path = root / relative
            if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
                raise ValueError(f"host {host_id} artifact checksum mismatch: {relative}")
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        expected_actual = set(expected_files) | {
            "artifact_manifest.json",
            "artifact_manifest.json.sha256",
            "REMOTE_COMPLETE.json",
        }
        if actual_files != expected_actual:
            raise ValueError(f"host {host_id} downloaded file inventory mismatch")
        marker = json.loads((root / "REMOTE_COMPLETE.json").read_text(encoding="utf-8"))
        terminal = json.loads((root / "HOST_COMPLETE.json").read_text(encoding="utf-8"))
        records = json.loads((root / "seed_records.json").read_text(encoding="utf-8"))
        alignments = json.loads((root / "alignment_inventory.json").read_text(encoding="utf-8"))
        if marker.get("status") != "REMOTE_VERIFIED":
            raise ValueError(f"host {host_id} lacks REMOTE_VERIFIED marker")
        if marker.get("artifact_manifest_sha256") != sha256_file(
            artifact_manifest_path
        ):
            raise ValueError(f"host {host_id} remote marker manifest mismatch")
        if terminal.get("status") != "PASS" or terminal.get("host_id") != host_id:
            raise ValueError(f"host {host_id} terminal identity mismatch")
        if terminal.get("source_training_sha") != SOURCE_TRAINING_SHA:
            raise ValueError(f"host {host_id} source SHA mismatch")
        if terminal.get("task_f_specification_sha256") != EXPECTED_SPECIFICATION_SHA256:
            raise ValueError(f"host {host_id} specification mismatch")
        expected_geometry = int(HOST_COUNTS[host_id]["geometry"])
        if int(terminal["stage_counts"].get("geometry", -1)) != expected_geometry:
            raise ValueError(f"host {host_id} geometry coverage mismatch")
        if int(alignments.get("alignment_count", -1)) != int(
            terminal["stage_counts"].get("alignment", -2)
        ):
            raise ValueError(f"host {host_id} alignment coverage mismatch")
        alignment_rows = alignments.get("artifacts")
        if not isinstance(alignment_rows, list) or len(alignment_rows) != int(
            alignments["alignment_count"]
        ):
            raise ValueError(f"host {host_id} alignment inventory count mismatch")
        alignment_ids = [row.get("output_identity_sha256") for row in alignment_rows]
        if len(alignment_ids) != len(set(alignment_ids)) or any(
            not isinstance(row.get("manifest_sha256"), str)
            or len(row["manifest_sha256"]) != 64
            for row in alignment_rows
        ):
            raise ValueError(f"host {host_id} alignment inventory identity mismatch")
        if int(records.get("record_count", -1)) != len(records.get("records", ())):
            raise ValueError(f"host {host_id} seed-record count mismatch")
        all_records.extend(dict(record) for record in records["records"])
        total_alignments += int(alignments["alignment_count"])
        host_records[host_id] = {
            "seed_record_count": int(records["record_count"]),
            "alignment_count": int(alignments["alignment_count"]),
            "terminal_sha256": sha256_file(root / "HOST_COMPLETE.json"),
            "seed_records_sha256": sha256_file(root / "seed_records.json"),
            "alignment_inventory_sha256": sha256_file(root / "alignment_inventory.json"),
        }
    if len(all_records) != 1920:
        raise ValueError("Task F central collection requires 1,920 seed records")
    if total_alignments != 657:
        raise ValueError("Task F central collection requires 657 alignments")
    from oge.analysis.task_f_fresh_id import (
        aggregate_paired_records,
        build_aggregation_contract,
        write_aggregation_artifacts,
    )

    contract = build_aggregation_contract(evaluation_plan)
    aggregation = aggregate_paired_records(
        records=all_records,
        expected_seeds_by_cell=contract["expected_seeds_by_cell"],
        expected_contexts=contract["contexts"],
        protected_id_test_available=False,
    )
    if aggregation["status"] != "PASS":
        raise ValueError(f"Task F paired aggregation is not PASS: {aggregation['status']}")
    destination = Path(output_directory)
    aggregation_root = write_aggregation_artifacts(
        payload=aggregation,
        output_directory=destination / "aggregation",
    )
    terminal = {
        "schema_version": "task_f_fresh_id_central_terminal_v1",
        "status": "PASS",
        "source_training_sha": SOURCE_TRAINING_SHA,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "host_count": 3,
        "logical_export_count": 1320,
        "geometry_count": 660,
        "seed_record_count": len(all_records),
        "alignment_count": total_alignments,
        "protected_data_access": False,
        "id_equivalence": "PENDING_PROTECTED_ID_TEST",
        "hosts": host_records,
        "aggregation_sha256": aggregation["aggregate_sha256"],
        "aggregation_root": str(aggregation_root),
    }
    atomic_write_json(destination / "CENTRAL_COMPLETE.json", terminal)
    return terminal
