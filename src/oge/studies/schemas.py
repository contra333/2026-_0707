"""Versioned study, trial, attempt, and immutable-freeze schemas."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .hashing import canonical_sha256
from .protocol import PROTOCOL_VERSION

STUDY_SCHEMA_VERSION = "1.0"
TRIAL_SCHEMA_VERSION = "1.0"
ATTEMPT_SCHEMA_VERSION = "1.0"
FREEZE_SCHEMA_VERSION = "1.0"

STUDY_REQUIRED = {
    "schema_version",
    "protocol_version",
    "study_id",
    "search_space_version",
    "search_space_hash",
    "sampler",
    "ordered_table_hashes",
    "optimizer_families",
    "assigned_budget",
    "seed_roles",
    "objective",
    "checkpoint_policy",
    "selection_declarations",
    "git",
    "dataset_membership_hashes",
    "artifact_root",
    "artifact_schema_version",
    "created_at",
    "finished_at",
    "status",
    "gpu_hours",
    "terminal_slot_accounting",
}
TRIAL_REQUIRED = {
    "schema_version",
    "protocol_version",
    "trial_id",
    "study_id",
    "phase",
    "optimizer_family",
    "assigned_slot",
    "scientific_config",
    "config_hash",
    "training_seed",
    "git_sha",
    "dataset_membership_hashes",
    "provenance_hash",
    "status",
    "failure",
    "attempt_ids",
    "completed_epoch",
    "best_validation",
    "final_validation",
    "ranking",
    "checkpoints",
    "artifact_references",
}
ATTEMPT_REQUIRED = {
    "schema_version",
    "protocol_version",
    "attempt_id",
    "attempt_number",
    "trial_id",
    "origin",
    "prior_attempt_ids",
    "identity",
    "gpu",
    "command",
    "environment",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "exit_status",
    "checkpoint_decision",
    "logs",
    "output_paths",
    "result",
    "status",
    "failure",
}


def _require_fields(record: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required.difference(record)
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def validate_study_record(record: Mapping[str, Any]) -> None:
    _require_fields(record, STUDY_REQUIRED, "study record")
    if record["schema_version"] != STUDY_SCHEMA_VERSION:
        raise ValueError("unsupported study schema_version")
    if record["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("study protocol_version mismatch")
    declarations = record["selection_declarations"]
    if not isinstance(declarations, Mapping) or any(
        declarations.get(key) != "forbidden"
        for key in ("id_test", "ood", "geometry_nc", "detector_metrics")
    ):
        raise ValueError("study selection declarations are incomplete")
    if record["assigned_budget"] != {"per_optimizer": 16, "total": 64}:
        raise ValueError("study assigned budget must be 16 per optimizer and 64 total")


def validate_trial_record(record: Mapping[str, Any]) -> None:
    _require_fields(record, TRIAL_REQUIRED, "trial record")
    if record["schema_version"] != TRIAL_SCHEMA_VERSION:
        raise ValueError("unsupported trial schema_version")
    if record["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("trial protocol_version mismatch")
    if not isinstance(record["attempt_ids"], list) or len(record["attempt_ids"]) != len(set(record["attempt_ids"])):
        raise ValueError("trial attempt_ids must be an ordered unique list")
    if record["status"] == "completed":
        if record["completed_epoch"] is None:
            raise ValueError("completed trial is missing completed_epoch")
        if not isinstance(record["best_validation"], Mapping):
            raise ValueError("completed trial is missing best_validation")
        if not isinstance(record["final_validation"], Mapping):
            raise ValueError("completed trial is missing final_validation")
        checkpoints = record["checkpoints"]
        if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"last", "best_val"}:
            raise ValueError("completed trial must record last and best_val checkpoints")


def validate_attempt_record(record: Mapping[str, Any]) -> None:
    _require_fields(record, ATTEMPT_REQUIRED, "attempt record")
    if record["schema_version"] != ATTEMPT_SCHEMA_VERSION:
        raise ValueError("unsupported attempt schema_version")
    if record["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("attempt protocol_version mismatch")
    gpu = record["gpu"]
    if not isinstance(gpu, Mapping) or gpu.get("child_local_device") != "cuda:0":
        raise ValueError("attempt must map its only visible GPU to child cuda:0")
    identity = record["identity"]
    for key in (
        "config_hash",
        "study_id",
        "assigned_slot",
        "training_seed",
        "phase",
        "git_sha",
        "dataset_membership_hashes",
        "provenance_hash",
    ):
        if key not in identity:
            raise ValueError(f"attempt identity is missing {key!r}")
    if "concurrent_study_trial_cap" not in gpu:
        raise ValueError("attempt GPU mapping is missing the concurrency cap")
    if record["status"] == "completed" and not isinstance(record["result"], Mapping):
        raise ValueError("completed attempt is missing validated child results")


def seal_freeze(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "freeze_kind": kind,
        "payload": copy.deepcopy(dict(payload)),
    }
    record["freeze_hash"] = canonical_sha256(record)
    return record


def validate_freeze(record: Mapping[str, Any], *, expected_kind: str | None = None) -> None:
    required = {"schema_version", "protocol_version", "freeze_kind", "payload", "freeze_hash"}
    _require_fields(record, required, "freeze record")
    if record["schema_version"] != FREEZE_SCHEMA_VERSION:
        raise ValueError("unsupported freeze schema_version")
    if record["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("freeze protocol_version mismatch")
    if expected_kind is not None and record["freeze_kind"] != expected_kind:
        raise ValueError(f"expected {expected_kind!r} freeze")
    unhashed = dict(record)
    actual_hash = unhashed.pop("freeze_hash")
    if actual_hash != canonical_sha256(unhashed):
        raise ValueError("immutable freeze hash mismatch")
