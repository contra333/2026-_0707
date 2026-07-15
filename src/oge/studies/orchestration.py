"""Independent-subprocess scheduling with one physical GPU per trial."""

from __future__ import annotations

import copy
import json
import math
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

from oge.training import load_torch_artifact

from .artifacts import atomic_write_json, create_preserved_attempt_directory, sha256_file
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


def _validated_metrics(value: Any, *, label: str) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    required = {"epoch", "accuracy", "nll"}
    if not required.issubset(value):
        raise ValueError(f"{label} is missing required metrics")
    epoch = int(value["epoch"])
    accuracy = float(value["accuracy"])
    nll = float(value["nll"])
    if epoch < 1 or not math.isfinite(accuracy) or not math.isfinite(nll):
        raise ValueError(f"{label} contains an invalid epoch or non-finite metric")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"{label} accuracy must be in [0, 1]")
    return {"epoch": epoch, "accuracy": accuracy, "nll": nll}


def collect_completed_attempt_result(
    attempt_dir: str | Path,
    trial: Mapping[str, Any],
    expected_gpu: PhysicalGPU,
) -> dict[str, Any]:
    """Validate a successful child run before publishing completed study state."""
    run_dir = Path(attempt_dir) / "run"
    paths = {
        "summary": run_dir / "summary.json",
        "history": run_dir / "history.jsonl",
        "resolved_config": run_dir / "resolved_config.yaml",
        "run_metadata": run_dir / "run_metadata.json",
        "environment": run_dir / "environment.json",
        "last": run_dir / "checkpoints/last.pt",
        "best_val": run_dir / "checkpoints/best_val.pt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"completed child run is missing artifacts: {missing}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError("child summary is not completed")
    completed_epoch = int(summary["completed_epoch"])
    expected_epoch = int(trial["scientific_config"]["training"]["max_epochs"])
    if completed_epoch != expected_epoch:
        raise ValueError("child completed_epoch does not match the assigned scientific config")
    best_validation = _validated_metrics(summary.get("best_validation"), label="best_validation")
    final_validation = _validated_metrics(summary.get("final_validation"), label="final_validation")
    if final_validation["epoch"] != completed_epoch:
        raise ValueError("final validation epoch does not match completed_epoch")
    if summary.get("id_test") != {
        "status": "deferred",
        "metrics_available": False,
        "artifacts_created": False,
    }:
        raise ValueError("study child summary must record deferred ID test")
    if "final_id_test" in summary or "best_val_id_test" in summary:
        raise ValueError("study child summary contains forbidden ID-test metrics")
    artifact_paths = summary.get("artifact_paths")
    if not isinstance(artifact_paths, Mapping) or any(
        key in artifact_paths for key in ("final_id_test", "best_val_id_test")
    ):
        raise ValueError("study child summary contains forbidden ID-test artifacts")
    if (run_dir / "evaluation/final_id_test.json").exists() or (
        run_dir / "evaluation/best_val_id_test.json"
    ).exists():
        raise ValueError("study child materialized forbidden ID-test artifacts")

    metadata = json.loads(paths["run_metadata"].read_text(encoding="utf-8"))
    if metadata.get("oge_git_sha") != trial["git_sha"] or metadata.get("git_dirty") is not False:
        raise ValueError("child run Git identity differs from the trial")
    if metadata.get("id_test_evaluation") != "deferred":
        raise ValueError("child run metadata does not record deferred ID test")
    environment = json.loads(paths["environment"].read_text(encoding="utf-8"))
    executions = environment.get("executions")
    if not isinstance(executions, list) or not executions:
        raise ValueError("child environment is missing execution metadata")
    execution = executions[-1]
    if execution.get("device") != "cuda:0" or int(execution.get("cuda_device_count", 0)) != 1:
        raise ValueError("child did not run as cuda:0 with exactly one visible GPU")
    if execution.get("cuda_visible_devices") != str(expected_gpu.parent_visible_index):
        raise ValueError("child CUDA_VISIBLE_DEVICES differs from the assigned physical GPU")
    if execution.get("expected_physical_gpu_uuid") != expected_gpu.uuid:
        raise ValueError("child expected physical GPU UUID differs from the attempt assignment")
    if execution.get("actual_visible_gpu_uuid") != expected_gpu.uuid:
        raise ValueError("child-visible GPU UUID differs from the attempt assignment")

    last = load_torch_artifact(paths["last"], map_location="cpu")
    best = load_torch_artifact(paths["best_val"], map_location="cpu")
    if last.get("checkpoint_type") != "last" or best.get("checkpoint_type") != "best_val":
        raise ValueError("child checkpoint roles are invalid")
    if int(last.get("completed_epoch", -1)) != completed_epoch:
        raise ValueError("last.pt completed_epoch differs from the summary")
    if int(best.get("completed_epoch", -1)) != best_validation["epoch"]:
        raise ValueError("best_val.pt epoch differs from the summary")
    run_id = summary.get("run_id")
    if not run_id or any(item.get("run_id") != run_id for item in (last, best)):
        raise ValueError("child summary and checkpoint run IDs differ")
    if any(item.get("oge_git_sha") != trial["git_sha"] for item in (last, best)):
        raise ValueError("child checkpoint Git identity differs from the trial")
    if last.get("best_validation") != summary["best_validation"]:
        raise ValueError("last.pt best-validation record differs from the summary")

    checkpoints = {
        role: {
            "path": str(paths[role]),
            "sha256": sha256_file(paths[role]),
            "completed_epoch": int(checkpoint["completed_epoch"]),
        }
        for role, checkpoint in (("last", last), ("best_val", best))
    }
    artifact_references = {
        name: {"path": str(paths[name]), "sha256": sha256_file(paths[name])}
        for name in ("summary", "history", "resolved_config", "run_metadata", "environment")
    }
    return {
        "completed_epoch": completed_epoch,
        "best_validation": best_validation,
        "final_validation": final_validation,
        "id_test": copy.deepcopy(summary["id_test"]),
        "checkpoints": checkpoints,
        "artifact_references": artifact_references,
    }


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
                    "OGE_ATTEMPT_DIR": str(attempt_dir),
                },
                "started_at": _utc_now(),
                "finished_at": None,
                "elapsed_seconds": None,
                "exit_status": None,
                "checkpoint_decision": checkpoint_decision,
                "logs": {"console": str(attempt_dir / "console.log")},
                "output_paths": {"attempt_dir": str(attempt_dir), "run": str(attempt_dir / "run")},
                "result": None,
                "status": "running",
                "failure": None,
            }
            validate_attempt_record(attempt)
            atomic_write_json(attempt_dir / "attempt.json", attempt)
            started = time.perf_counter()
            child_environment = child_gpu_environment(gpu)
            child_environment["OGE_ATTEMPT_CONSOLE"] = str(attempt_dir / "console.log")
            child_environment["OGE_ATTEMPT_DIR"] = str(attempt_dir)
            exit_status = int(spawn(command, env=child_environment, cwd=Path(repository_root)))
            attempt["elapsed_seconds"] = time.perf_counter() - started
            attempt["finished_at"] = _utc_now()
            attempt["exit_status"] = exit_status
            if exit_status == 0:
                try:
                    attempt["result"] = collect_completed_attempt_result(
                        attempt_dir,
                        trial,
                        gpu,
                    )
                except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    attempt["status"] = "failed_unclassified"
                    attempt["failure"] = {
                        "class": "implementation",
                        "reason_code": "child_artifact_validation_failed",
                        "detail": str(exc),
                    }
                else:
                    attempt["status"] = "completed"
            else:
                attempt["status"] = "failed_unclassified"
                attempt["failure"] = {
                    "class": "unclassified",
                    "reason_code": "requires_classification_before_retry",
                }
            validate_attempt_record(attempt)
            atomic_write_json(attempt_dir / "attempt.json", attempt)
            persisted_trial = json.loads(trial_path.read_text(encoding="utf-8"))
            persisted_trial["attempt_ids"].append(attempt_id)
            persisted_trial["status"] = (
                "completed" if attempt["status"] == "completed" else "failed_pending_classification"
            )
            persisted_trial["failure"] = attempt["failure"]
            persisted_trial["artifact_references"]["latest_attempt"] = str(attempt_dir)
            if attempt["status"] == "completed":
                result = attempt["result"]
                persisted_trial["completed_epoch"] = result["completed_epoch"]
                persisted_trial["best_validation"] = result["best_validation"]
                persisted_trial["final_validation"] = result["final_validation"]
                persisted_trial["checkpoints"] = result["checkpoints"]
                persisted_trial["artifact_references"].update(result["artifact_references"])
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
