"""Deterministic aggregation for the frozen protocol-v1.2 follow-up study."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from .hashing import canonical_sha256
from .schemas import validate_trial_record
from .supervisor import PROTECTED_TRIAL_FIELDS

AGGREGATE_SCHEMA_VERSION = "1.0"
EXPECTED_SEEDS = (0, 1, 2)


def _metric_value(record: Mapping[str, Any], metric: str) -> float:
    value = record.get(metric)
    if not isinstance(value, (int, float)):
        raise ValueError(f"metric {metric!r} must be numeric")
    return float(value)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("sample standard deviation requires at least two values")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
    }


def _validate_completed_record(
    record: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    *,
    expected_phase: str,
    expected_git_sha: str | None,
) -> None:
    validate_trial_record(record)
    if record["status"] != "completed" or int(record["completed_epoch"]) != 200:
        raise ValueError(f"trial {record['trial_id']!r} is not terminal at epoch 200")
    if record["phase"] != expected_phase:
        raise ValueError(f"trial {record['trial_id']!r} has the wrong phase")
    if record["config_hash"] != plan_row["config_hash"]:
        raise ValueError(f"trial {record['trial_id']!r} has the wrong config hash")
    if int(record["training_seed"]) != int(plan_row["training_seed"]):
        raise ValueError(f"trial {record['trial_id']!r} has the wrong training seed")
    if expected_git_sha is not None and record["git_sha"] != expected_git_sha:
        raise ValueError(f"trial {record['trial_id']!r} has the wrong Git SHA")
    if PROTECTED_TRIAL_FIELDS.intersection(record):
        raise ValueError(f"trial {record['trial_id']!r} contains protected evidence")
    for metric_name in ("best_validation", "final_validation"):
        metric = record[metric_name]
        if not isinstance(metric, Mapping):
            raise ValueError(f"trial {record['trial_id']!r} is missing {metric_name}")
        _metric_value(metric, "accuracy")
        _metric_value(metric, "nll")
    if int(record["final_validation"].get("epoch", -1)) != 200:
        raise ValueError(f"trial {record['trial_id']!r} final validation is not epoch 200")
    checkpoints = record["checkpoints"]
    if int(checkpoints["last"].get("completed_epoch", -1)) != 200:
        raise ValueError(f"trial {record['trial_id']!r} last.pt is not epoch 200")


def _condense_record(
    plan_row: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    source: str,
    host_id: str,
) -> dict[str, Any]:
    return {
        "aggregate_trial_id": plan_row["trial_id"],
        "source_trial_id": record["trial_id"],
        "config_hash": plan_row["config_hash"],
        "source_grid_config_hash": plan_row["source_grid_config_hash"],
        "optimizer_family": plan_row["optimizer_family"],
        "optimizer": dict(record["scientific_config"]["optimizer"]),
        "training_seed": int(plan_row["training_seed"]),
        "labels": sorted(plan_row["labels"]),
        "source": source,
        "host_id": host_id,
        "phase": record["phase"],
        "git_sha": record["git_sha"],
        "provenance_hash": record["provenance_hash"],
        "attempt_ids": list(record["attempt_ids"]),
        "completed_epoch": int(record["completed_epoch"]),
        "best_validation": dict(record["best_validation"]),
        "final_validation": dict(record["final_validation"]),
        "checkpoints": {
            name: {
                "completed_epoch": int(reference["completed_epoch"]),
                "sha256": reference["sha256"],
            }
            for name, reference in record["checkpoints"].items()
        },
        "dataset_membership_hashes": dict(record["dataset_membership_hashes"]),
    }


def _host_completion(
    host_id: str,
    state: Mapping[str, Any],
    expected_trial_ids: Sequence[str],
    expected_git_sha: str,
    execution_id: str,
) -> dict[str, Any]:
    if state.get("host_id") != host_id or state.get("status") != "REMOTE_VERIFIED":
        raise ValueError(f"host {host_id!r} is not REMOTE_VERIFIED")
    if state.get("expected_git_sha") != expected_git_sha:
        raise ValueError(f"host {host_id!r} has the wrong production Git SHA")
    verification = state.get("verification")
    if not isinstance(verification, Mapping):
        raise ValueError(f"host {host_id!r} is missing artifact verification")
    if (
        verification.get("status") != "completed_and_verified"
        or verification.get("assigned_trial_ids") != list(expected_trial_ids)
        or int(verification.get("assigned_trial_count", -1)) != len(expected_trial_ids)
    ):
        raise ValueError(f"host {host_id!r} verification differs from its assignment")
    upload = state.get("upload_evidence")
    if not isinstance(upload, Mapping):
        raise ValueError(f"host {host_id!r} is missing upload evidence")
    dry_run = upload.get("dry_run_summary")
    if not isinstance(dry_run, Mapping) or int(dry_run.get("deletes", -1)) != 0:
        raise ValueError(f"host {host_id!r} upload was not a zero-delete sync")
    remote_files = upload.get("remote_files")
    if not isinstance(remote_files, list):
        raise ValueError(f"host {host_id!r} is missing its remote listing")
    markers = [
        item
        for item in remote_files
        if isinstance(item, Mapping)
        and str(item.get("path", "")).endswith("/REMOTE_COMPLETE.json")
    ]
    expected_marker_path = f"servers/{host_id}/{execution_id}/REMOTE_COMPLETE.json"
    if len(markers) != 1 or markers[0].get("path") != expected_marker_path:
        raise ValueError(f"host {host_id!r} must have one REMOTE_COMPLETE marker")
    file_count = int(upload.get("file_count", -1))
    if len(remote_files) != file_count + 1:
        raise ValueError(
            f"host {host_id!r} remote listing must contain the verified files "
            "plus the final REMOTE_COMPLETE marker"
        )
    if int(dry_run.get("uploads", -1)) != file_count:
        raise ValueError(f"host {host_id!r} upload count is inconsistent")
    return {
        "status": state["status"],
        "assigned_trial_count": len(expected_trial_ids),
        "finished_at": state.get("finished_at"),
        "verified_at": verification.get("verified_at"),
        "destination": upload.get("destination"),
        "artifact_manifest_sha256": upload.get("artifact_manifest_sha256"),
        "verified_file_count_before_marker": file_count,
        "remote_listing_file_count": len(remote_files),
        "total_size": int(upload["total_size"]),
        "dry_run_summary": dict(dry_run),
        "remote_complete": dict(markers[0]),
    }


def _config_statistics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["config_hash"]), []).append(record)
    statistics_rows: list[dict[str, Any]] = []
    for config_hash, values in grouped.items():
        ordered = sorted(values, key=lambda item: int(item["training_seed"]))
        seeds = tuple(int(item["training_seed"]) for item in ordered)
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"config {config_hash!r} does not contain seeds 0, 1, and 2")
        labels = {tuple(item["labels"]) for item in ordered}
        optimizers = {str(item["optimizer_family"]) for item in ordered}
        if len(labels) != 1 or len(optimizers) != 1:
            raise ValueError(f"config {config_hash!r} changes identity across seeds")
        per_seed = [
            {
                "training_seed": int(item["training_seed"]),
                "host_id": item["host_id"],
                "accuracy": _metric_value(item["final_validation"], "accuracy"),
                "nll": _metric_value(item["final_validation"], "nll"),
            }
            for item in ordered
        ]
        statistics_rows.append(
            {
                "config_hash": config_hash,
                "optimizer_family": ordered[0]["optimizer_family"],
                "optimizer": dict(ordered[0]["optimizer"]),
                "labels": list(next(iter(labels))),
                "per_seed": per_seed,
                "final_validation": {
                    "accuracy": _summary([row["accuracy"] for row in per_seed]),
                    "nll": _summary([row["nll"] for row in per_seed]),
                },
            }
        )
    return sorted(
        statistics_rows,
        key=lambda item: (str(item["optimizer_family"]), str(item["config_hash"])),
    )


def _pair_statistics(configs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for config in configs:
        for label in config["labels"]:
            if label == "pair:adam_adamw":
                optimizer = config["optimizer"]
                cell = f"lr={optimizer['lr']},weight_decay={optimizer['weight_decay']}"
                groups.setdefault((label, cell), []).append(config)
            elif str(label).startswith("pair:sgd_sgdw:"):
                groups.setdefault((str(label), str(label).rsplit(":", 1)[-1]), []).append(
                    config
                )
    output: list[dict[str, Any]] = []
    for (label, cell), endpoints in sorted(groups.items()):
        if label == "pair:adam_adamw":
            coupled_name, decoupled_name = "adam", "adamw"
        else:
            coupled_name, decoupled_name = "sgd", "sgdw"
        by_family = {str(item["optimizer_family"]): item for item in endpoints}
        if set(by_family) != {coupled_name, decoupled_name}:
            raise ValueError(f"pair {label!r} cell {cell!r} is missing an endpoint")
        coupled = by_family[coupled_name]
        decoupled = by_family[decoupled_name]
        per_seed: list[dict[str, Any]] = []
        for seed in EXPECTED_SEEDS:
            coupled_seed = coupled["per_seed"][seed]
            decoupled_seed = decoupled["per_seed"][seed]
            if (
                int(coupled_seed["training_seed"]) != seed
                or int(decoupled_seed["training_seed"]) != seed
            ):
                raise ValueError(f"pair {label!r} has misaligned seed ordering")
            per_seed.append(
                {
                    "training_seed": seed,
                    "coupled_accuracy": coupled_seed["accuracy"],
                    "decoupled_accuracy": decoupled_seed["accuracy"],
                    "accuracy_difference": (
                        decoupled_seed["accuracy"] - coupled_seed["accuracy"]
                    ),
                    "coupled_nll": coupled_seed["nll"],
                    "decoupled_nll": decoupled_seed["nll"],
                    "nll_difference": decoupled_seed["nll"] - coupled_seed["nll"],
                }
            )
        output.append(
            {
                "label": label,
                "cell": cell,
                "difference_direction": f"{decoupled_name}_minus_{coupled_name}",
                "coupled": {
                    "optimizer_family": coupled_name,
                    "config_hash": coupled["config_hash"],
                },
                "decoupled": {
                    "optimizer_family": decoupled_name,
                    "config_hash": decoupled["config_hash"],
                },
                "per_seed": per_seed,
                "paired_difference": {
                    "accuracy": _summary(
                        [row["accuracy_difference"] for row in per_seed]
                    ),
                    "nll": _summary([row["nll_difference"] for row in per_seed]),
                },
            }
        )
    return output


def build_followup_aggregate(
    *,
    followup_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    grid_records: Sequence[Mapping[str, Any]],
    host_records: Mapping[str, Sequence[Mapping[str, Any]]],
    host_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and combine 12 reused and 30 newly executed trial records."""
    expected_git_shas = {
        state.get("expected_git_sha") for state in host_states.values()
    }
    if len(expected_git_shas) != 1:
        raise ValueError("host states do not agree on one production Git SHA")
    production_git_sha = next(iter(expected_git_shas))
    if not isinstance(production_git_sha, str) or not production_git_sha:
        raise ValueError("host states do not agree on one production Git SHA")
    expected_git_sha = production_git_sha
    grid_by_identity = {
        (str(record["config_hash"]), int(record["training_seed"])): record
        for record in grid_records
    }
    records: list[dict[str, Any]] = []
    for plan_row in followup_plan["reused"]:
        key = (str(plan_row["config_hash"]), int(plan_row["training_seed"]))
        if key not in grid_by_identity:
            raise ValueError(f"reused identity {key!r} is missing from grid results")
        record = grid_by_identity[key]
        _validate_completed_record(
            record,
            plan_row,
            expected_phase="grid",
            expected_git_sha=None,
        )
        records.append(
            _condense_record(
                plan_row,
                record,
                source="seed0_grid_reuse",
                host_id=execution_plan["seed0_reuse_hosts"][plan_row["config_hash"]],
            )
        )

    scheduled = {str(row["trial_id"]): row for row in followup_plan["scheduled"]}
    completion: dict[str, Any] = {}
    for host_id, host in execution_plan["hosts"].items():
        expected_ids = list(host["trial_ids"])
        actual = {str(record["trial_id"]): record for record in host_records[host_id]}
        if set(actual) != set(expected_ids) or len(actual) != len(expected_ids):
            raise ValueError(f"host {host_id!r} trial records differ from its assignment")
        completion[host_id] = _host_completion(
            host_id,
            host_states[host_id],
            expected_ids,
            expected_git_sha,
            str(execution_plan["execution_id"]),
        )
        for trial_id in expected_ids:
            plan_row = scheduled[trial_id]
            record = actual[trial_id]
            _validate_completed_record(
                record,
                plan_row,
                expected_phase="followup",
                expected_git_sha=expected_git_sha,
            )
            records.append(
                _condense_record(
                    plan_row,
                    record,
                    source="followup_execution",
                    host_id=host_id,
                )
            )

    expected_count = len(followup_plan["reused"]) + len(followup_plan["scheduled"])
    identities = {
        (str(record["config_hash"]), int(record["training_seed"]))
        for record in records
    }
    if len(records) != expected_count or len(identities) != expected_count:
        raise ValueError("aggregate contains a duplicate or missing (config_hash, seed)")
    ordered_records = sorted(
        records,
        key=lambda item: (
            str(item["optimizer_family"]),
            str(item["config_hash"]),
            int(item["training_seed"]),
        ),
    )
    configs = _config_statistics(ordered_records)
    payload: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "protocol_version": execution_plan["protocol_version"],
        "execution_id": execution_plan["execution_id"],
        "production_git_sha": expected_git_sha,
        "role_freeze_hash": followup_plan["role_freeze_hash"],
        "followup_plan_hash": followup_plan["plan_hash"],
        "grid_manifest_hash": followup_plan["grid_manifest_hash"],
        "aggregation_policy": {
            "primary_checkpoint": "last.pt",
            "metric_source": "final_validation",
            "checkpoint_epoch": 200,
            "seeds": list(EXPECTED_SEEDS),
            "dispersion": "sample_sd_ddof_1",
            "pair_difference": "decoupled_minus_coupled",
            "protected_evaluation": "deferred",
        },
        "counts": {
            "reused_seed0": len(followup_plan["reused"]),
            "new_followup": len(followup_plan["scheduled"]),
            "aggregate_records": len(ordered_records),
            "unique_config_seed_identities": len(identities),
            "unique_configurations": len(configs),
        },
        "host_completion": completion,
        "records": ordered_records,
        "config_statistics": configs,
        "pair_statistics": _pair_statistics(configs),
        "reporting_caveats": [
            "Seed 0 selected the frozen roles, so role-bundle summaries retain selection bias toward seed 0.",
            "All seed values are retained; no seed was dropped, replaced, or rerun to improve a mean.",
            "n=3 summaries are descriptive and are not strong significance evidence.",
            "Pair controls are diagnostic and do not promote pair-only configurations to primary role evidence.",
        ],
    }
    payload["aggregate_hash"] = canonical_sha256(payload)
    return payload


def validate_followup_aggregate(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != AGGREGATE_SCHEMA_VERSION:
        raise ValueError("unsupported follow-up aggregate schema version")
    unhashed = dict(payload)
    actual_hash = unhashed.pop("aggregate_hash", None)
    if actual_hash != canonical_sha256(unhashed):
        raise ValueError("follow-up aggregate hash mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("follow-up aggregate records must be a list")
    identities = {
        (str(record["config_hash"]), int(record["training_seed"]))
        for record in records
    }
    if len(identities) != len(records):
        raise ValueError("follow-up aggregate contains duplicate identities")
