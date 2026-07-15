"""Independent-subprocess scheduling with one physical GPU per trial."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, create_preserved_attempt_directory
from .protocol import PROTOCOL_VERSION
from .schemas import ATTEMPT_SCHEMA_VERSION, validate_attempt_record, validate_trial_record

DEFAULT_MAX_CONCURRENCY = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PhysicalGPU:
    parent_visible_index: int
    uuid: str
    model: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PhysicalGPU":
        return cls(
            parent_visible_index=int(value["parent_visible_index"]),
            uuid=str(value["uuid"]),
            model=str(value["model"]),
        )


def validate_gpu_inventory(gpus: Sequence[PhysicalGPU]) -> None:
    if not gpus:
        raise ValueError("at least one physical GPU is required")
    if len({gpu.uuid for gpu in gpus}) != len(gpus):
        raise ValueError("duplicate physical GPU UUID assignment is forbidden")
    if len({gpu.parent_visible_index for gpu in gpus}) != len(gpus):
        raise ValueError("duplicate parent-visible GPU index is forbidden")
    if any(not gpu.uuid or not gpu.model for gpu in gpus):
        raise ValueError("GPU UUID and model/class are required")


def effective_concurrency(
    gpus: Sequence[PhysicalGPU],
    requested: int = DEFAULT_MAX_CONCURRENCY,
) -> int:
    validate_gpu_inventory(gpus)
    if requested <= 0:
        raise ValueError("concurrency cap must be positive")
    return min(requested, DEFAULT_MAX_CONCURRENCY, len(gpus))


def child_gpu_environment(gpu: PhysicalGPU, base: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu.parent_visible_index)
    environment["OGE_PHYSICAL_GPU_UUID"] = gpu.uuid
    environment["OGE_CHILD_DEVICE"] = "cuda:0"
    return environment


def ordered_trial_plan(
    trials: Sequence[Mapping[str, Any]],
    gpus: Sequence[PhysicalGPU],
    *,
    concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> list[dict[str, Any]]:
    slots = list(gpus[: effective_concurrency(gpus, concurrency)])
    plan = []
    for index, trial in enumerate(trials):
        validate_trial_record(trial)
        gpu = slots[index % len(slots)]
        plan.append(
            {
                "order": index,
                "trial_id": trial["trial_id"],
                "config_hash": trial["config_hash"],
                "training_seed": trial["training_seed"],
                "physical_gpu_uuid": gpu.uuid,
                "gpu_model": gpu.model,
                "parent_visible_index": gpu.parent_visible_index,
                "cuda_visible_devices": str(gpu.parent_visible_index),
                "child_local_device": "cuda:0",
            }
        )
    return plan


def _default_spawn(command: Sequence[str], *, env: Mapping[str, str], cwd: Path) -> int:
    console_path = Path(env["OGE_ATTEMPT_CONSOLE"])
    with console_path.open("w", encoding="utf-8") as console:
        return subprocess.run(
            list(command),
            env=dict(env),
            cwd=cwd,
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def run_independent_trials(
    *,
    study_id: str,
    trials: Sequence[Mapping[str, Any]],
    gpus: Sequence[PhysicalGPU],
    artifact_root: str | Path,
    repository_root: str | Path,
    command_builder: Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Sequence[str]],
    concurrency: int = DEFAULT_MAX_CONCURRENCY,
    dry_run: bool = False,
    spawn: Callable[..., int] = _default_spawn,
    attempt_templates: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one sequential queue per selected GPU; no GPU can overlap itself."""
    plan = ordered_trial_plan(trials, gpus, concurrency=concurrency)
    root = Path(artifact_root)
    plan_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "study_id": study_id,
        "dry_run": dry_run,
        "default_max_concurrency": DEFAULT_MAX_CONCURRENCY,
        "effective_concurrency": effective_concurrency(gpus, concurrency),
        "ordered_trials": plan,
    }
    atomic_write_json(root / study_id / "ordered_plan.json", plan_payload)
    if dry_run:
        return plan_payload

    trial_by_id = {str(trial["trial_id"]): trial for trial in trials}
    selected_gpus = list(gpus[: plan_payload["effective_concurrency"]])
    queue_by_uuid = {gpu.uuid: [] for gpu in selected_gpus}
    gpu_by_uuid = {gpu.uuid: gpu for gpu in selected_gpus}
    for item in plan:
        queue_by_uuid[item["physical_gpu_uuid"]].append(item)

    def run_queue(gpu_uuid: str) -> list[dict[str, Any]]:
        gpu = gpu_by_uuid[gpu_uuid]
        results = []
        for item in queue_by_uuid[gpu_uuid]:
            trial = trial_by_id[item["trial_id"]]
            template = None if attempt_templates is None else attempt_templates.get(str(trial["trial_id"]))
            attempt_number = int(template["attempt_number"]) if template is not None else 1
            attempt_id = (
                str(template["attempt_id"])
                if template is not None
                else f"{trial['trial_id']}-attempt-{attempt_number:03d}"
            )
            trial_dir = root / study_id / "trials" / str(trial["trial_id"])
            trial_path = trial_dir / "trial.json"
            if trial_path.is_file():
                if json.loads(trial_path.read_text(encoding="utf-8")) != trial:
                    raise ValueError("existing trial record differs from retry identity")
            else:
                atomic_write_json(trial_path, trial)
            attempt_dir = create_preserved_attempt_directory(
                root, study_id, str(trial["trial_id"]), attempt_id
            )
            checkpoint_decision = (
                copy.deepcopy(template["checkpoint_decision"])
                if template is not None
                else {"action": "fresh", "checkpoint": None}
            )
            command = list(command_builder(trial, attempt_dir, checkpoint_decision))
            attempt = {
                "schema_version": ATTEMPT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "trial_id": trial["trial_id"],
                "origin": str(template["origin"]) if template is not None else "fresh",
                "prior_attempt_ids": list(template["prior_attempt_ids"]) if template is not None else [],
                "identity": {
                    "config_hash": trial["config_hash"],
                    "study_id": trial["study_id"],
                    "assigned_slot": trial["assigned_slot"],
                    "training_seed": trial["training_seed"],
                    "phase": trial["phase"],
                    "git_sha": trial["git_sha"],
                    "dataset_membership_hashes": copy.deepcopy(trial["dataset_membership_hashes"]),
                    "provenance_hash": trial["provenance_hash"],
                },
                "gpu": {
                    "physical_uuid": gpu.uuid,
                    "model": gpu.model,
                    "parent_visible_index": gpu.parent_visible_index,
                    "cuda_visible_devices": str(gpu.parent_visible_index),
                    "child_local_index": 0,
                    "child_local_device": "cuda:0",
                    "concurrent_study_trial_cap": plan_payload["effective_concurrency"],
                },
                "command": command,
                "environment": {
                    "CUDA_VISIBLE_DEVICES": str(gpu.parent_visible_index),
                    "OGE_PHYSICAL_GPU_UUID": gpu.uuid,
                    "OGE_CHILD_DEVICE": "cuda:0",
                },
                "started_at": _utc_now(),
                "finished_at": None,
                "elapsed_seconds": None,
                "exit_status": None,
                "checkpoint_decision": checkpoint_decision,
                "logs": {"console": str(attempt_dir / "console.log")},
                "output_paths": {"attempt_dir": str(attempt_dir), "run": str(attempt_dir / "run")},
                "status": "running",
                "failure": None,
            }
            validate_attempt_record(attempt)
            atomic_write_json(attempt_dir / "attempt.json", attempt)
            started = time.perf_counter()
            child_environment = child_gpu_environment(gpu)
            child_environment["OGE_ATTEMPT_CONSOLE"] = str(attempt_dir / "console.log")
            exit_status = int(spawn(command, env=child_environment, cwd=Path(repository_root)))
            attempt["elapsed_seconds"] = time.perf_counter() - started
            attempt["finished_at"] = _utc_now()
            attempt["exit_status"] = exit_status
            attempt["status"] = "completed" if exit_status == 0 else "failed_unclassified"
            if exit_status != 0:
                attempt["failure"] = {
                    "class": "unclassified",
                    "reason_code": "requires_classification_before_retry",
                }
            validate_attempt_record(attempt)
            atomic_write_json(attempt_dir / "attempt.json", attempt)
            persisted_trial = json.loads(trial_path.read_text(encoding="utf-8"))
            persisted_trial["attempt_ids"].append(attempt_id)
            persisted_trial["status"] = (
                "completed" if exit_status == 0 else "failed_pending_classification"
            )
            persisted_trial["failure"] = attempt["failure"]
            persisted_trial["artifact_references"]["latest_attempt"] = str(attempt_dir)
            validate_trial_record(persisted_trial)
            atomic_write_json(trial_path, persisted_trial)
            results.append(attempt)
        return results

    with ThreadPoolExecutor(max_workers=len(selected_gpus)) as executor:
        futures = [executor.submit(run_queue, gpu.uuid) for gpu in selected_gpus]
        attempts = [attempt for future in futures for attempt in future.result()]
    attempts.sort(key=lambda item: plan.index(next(row for row in plan if row["trial_id"] == item["trial_id"])))
    terminal_accounting = {
        "planned": len(plan),
        "completed": sum(attempt["status"] == "completed" for attempt in attempts),
        "failed_pending_classification": sum(
            attempt["status"] == "failed_unclassified" for attempt in attempts
        ),
    }
    gpu_hours_per_optimizer: dict[str, float] = {}
    for attempt in attempts:
        family = str(trial_by_id[str(attempt["trial_id"])]["optimizer_family"])
        gpu_hours_per_optimizer[family] = gpu_hours_per_optimizer.get(family, 0.0) + float(
            attempt["elapsed_seconds"]
        ) / 3600.0
    summary = {
        **plan_payload,
        "attempts": attempts,
        "terminal_slot_accounting": terminal_accounting,
        "gpu_hours": {
            "total": sum(gpu_hours_per_optimizer.values()),
            "per_optimizer": gpu_hours_per_optimizer,
        },
    }
    atomic_write_json(root / study_id / "study_summary.json", summary)
    return summary


def prepare_retry_run_directory(
    checkpoint_decision: Mapping[str, Any],
    attempt_dir: str | Path,
) -> Path | None:
    """Copy a prior run into a new attempt boundary and return its local last.pt."""
    if checkpoint_decision.get("action") != "resume_epoch_boundary":
        return None
    source_checkpoint = Path(str(checkpoint_decision["checkpoint"]))
    if source_checkpoint.name != "last.pt" or source_checkpoint.parent.name != "checkpoints":
        raise ValueError("retry resume source must be a run/checkpoints/last.pt")
    source_run = source_checkpoint.parent.parent
    destination_run = Path(attempt_dir) / "run"
    if destination_run.exists():
        raise ValueError("new retry attempt run directory must not already exist")
    shutil.copytree(source_run, destination_run)
    destination_checkpoint = destination_run / "checkpoints/last.pt"
    if not destination_checkpoint.is_file():
        raise ValueError("copied retry run is missing checkpoints/last.pt")
    return destination_checkpoint


def discover_nvidia_gpus() -> list[PhysicalGPU]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus = []
    for line in result.stdout.splitlines():
        index, uuid, model = (part.strip() for part in line.split(",", maxsplit=2))
        gpus.append(PhysicalGPU(int(index), uuid, model))
    validate_gpu_inventory(gpus)
    return gpus
