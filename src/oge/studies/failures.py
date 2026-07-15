"""Protocol-v1.1 failure classification and retry identity rules."""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from oge.training import load_torch_artifact

from .artifacts import sha256_file
from .protocol import PROTOCOL_VERSION
from .schemas import ATTEMPT_SCHEMA_VERSION, validate_attempt_record, validate_trial_record

SCIENTIFIC_FAILURES = {
    "oom_assigned_config",
    "non_finite_loss_or_model",
    "invalid_scientific_config",
}
INFRASTRUCTURE_FAILURES = {
    "unrelated_gpu_contention",
    "external_interruption",
    "external_data_or_mount",
    "dataset_manifest_mismatch",
    "checkpoint_corruption",
    "gpu_reset",
    "driver_runtime_infrastructure",
}
RESULT_DRIVEN_REASONS = {
    "low_validation_accuracy",
    "unfavorable_result",
    "ranking_changed",
    "selective_replacement",
}


def classify_failure(reason_code: str) -> str:
    if reason_code in SCIENTIFIC_FAILURES:
        return "scientific"
    if reason_code in INFRASTRUCTURE_FAILURES:
        return "infrastructure"
    if reason_code in RESULT_DRIVEN_REASONS:
        raise ValueError("result-driven rerun or selective replacement is forbidden")
    raise ValueError(f"unknown failure reason code: {reason_code!r}")


def checkpoint_resume_decision(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"action": "restart_epoch_0", "checkpoint": None, "sha256": None}
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return {"action": "restart_epoch_0", "checkpoint": str(checkpoint_path), "sha256": None}
    try:
        checkpoint = load_torch_artifact(checkpoint_path, map_location="cpu")
        if checkpoint.get("checkpoint_type") != "last":
            raise ValueError("resume source is not an atomic last.pt")
        completed_epoch = int(checkpoint["completed_epoch"])
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError):
        return {
            "action": "restart_epoch_0",
            "checkpoint": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "integrity": "invalid",
        }
    return {
        "action": "resume_epoch_boundary",
        "checkpoint": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "integrity": "valid",
        "completed_epoch": completed_epoch,
    }


def make_retry_attempt(
    trial: Mapping[str, Any],
    prior_attempts: Sequence[Mapping[str, Any]],
    *,
    reason_code: str,
    checkpoint_path: str | Path | None,
) -> dict[str, Any]:
    validate_trial_record(trial)
    if classify_failure(reason_code) != "infrastructure":
        raise ValueError("only infrastructure failures are retry eligible")
    for attempt in prior_attempts:
        validate_attempt_record(attempt)
        if attempt["trial_id"] != trial["trial_id"]:
            raise ValueError("prior attempt belongs to a different trial")
        identity = attempt["identity"]
        expected = {
            "config_hash": trial["config_hash"],
            "study_id": trial["study_id"],
            "assigned_slot": trial["assigned_slot"],
            "training_seed": trial["training_seed"],
            "phase": trial["phase"],
            "git_sha": trial["git_sha"],
            "dataset_membership_hashes": trial["dataset_membership_hashes"],
            "provenance_hash": trial["provenance_hash"],
        }
        if identity != expected:
            raise ValueError("prior attempt identity differs from the parent trial")
    if not prior_attempts or prior_attempts[-1].get("status") == "completed":
        raise ValueError("infrastructure retry requires a preserved non-success attempt")
    attempt_number = len(prior_attempts) + 1
    attempt_id = f"{trial['trial_id']}-attempt-{attempt_number:03d}"
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "trial_id": trial["trial_id"],
        "origin": "infrastructure_retry",
        "prior_attempt_ids": [attempt["attempt_id"] for attempt in prior_attempts],
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
            "physical_uuid": None,
            "model": None,
            "parent_visible_index": None,
            "cuda_visible_devices": None,
            "child_local_index": 0,
            "child_local_device": "cuda:0",
            "concurrent_study_trial_cap": None,
        },
        "command": [],
        "environment": {},
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": None,
        "exit_status": None,
        "checkpoint_decision": checkpoint_resume_decision(checkpoint_path),
        "logs": {},
        "output_paths": {},
        "result": None,
        "status": "planned",
        "failure": {"class": "infrastructure", "reason_code": reason_code},
    }
