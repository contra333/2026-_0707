"""Validated multi-host execution assignments for frozen optimizer grids."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .protocol import GRID_TRAINING_SEED, PROTOCOL_VERSION, validate_grid_bundle

EXECUTION_SCHEMA_VERSION = "1.0"


def load_seed0_execution_plan(
    path: str | Path,
    grid_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Load an exact-once host assignment for the frozen seed-0 grid."""
    validate_grid_bundle(grid_bundle)
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("execution plan must be a YAML mapping")
    plan = copy.deepcopy(dict(value))
    required = {
        "schema_version",
        "protocol_version",
        "execution_id",
        "grid_manifest_hash",
        "training_seed",
        "canary_trial_id",
        "hosts",
    }
    missing = required.difference(plan)
    if missing:
        raise ValueError(f"execution plan is missing fields: {sorted(missing)}")
    if plan["schema_version"] != EXECUTION_SCHEMA_VERSION:
        raise ValueError("unsupported execution-plan schema_version")
    if plan["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("execution-plan protocol_version mismatch")
    if plan["grid_manifest_hash"] != grid_bundle["manifest"]["manifest_hash"]:
        raise ValueError("execution-plan grid manifest hash mismatch")
    if int(plan["training_seed"]) != GRID_TRAINING_SEED:
        raise ValueError("execution plan must assign the frozen seed-0 grid")
    if not isinstance(plan["execution_id"], str) or not plan["execution_id"].strip():
        raise ValueError("execution plan requires a non-empty execution_id")
    hosts = plan["hosts"]
    if not isinstance(hosts, Mapping) or not hosts:
        raise ValueError("execution plan requires at least one host assignment")

    rows = [
        row
        for optimizer in grid_bundle["manifest"]["optimizer_order"]
        for row in grid_bundle["tables"][optimizer]["rows"]
    ]
    expected = {str(row["trial_id"]): row for row in rows}
    canary_trial_id = str(plan["canary_trial_id"])
    if canary_trial_id not in expected:
        raise ValueError("execution-plan canary trial is absent from the frozen grid")
    if expected[canary_trial_id]["optimizer_family"] != "sgd":
        raise ValueError("execution-plan canary must be an SGD grid cell")
    assigned: list[str] = []
    for host_id, host_value in hosts.items():
        if not isinstance(host_id, str) or not host_id:
            raise ValueError("execution-plan host IDs must be non-empty strings")
        if not isinstance(host_value, Mapping):
            raise ValueError(f"execution-plan host {host_id!r} must be a mapping")
        trial_ids = host_value.get("trial_ids")
        concurrency = host_value.get("concurrency")
        if not isinstance(trial_ids, list) or not trial_ids:
            raise ValueError(f"execution-plan host {host_id!r} has no trial_ids")
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError(f"execution-plan host {host_id!r} repeats a trial ID")
        if (
            not isinstance(concurrency, int)
            or isinstance(concurrency, bool)
            or concurrency <= 0
        ):
            raise ValueError(f"execution-plan host {host_id!r} has invalid concurrency")
        unknown = set(trial_ids).difference(expected)
        if unknown:
            raise ValueError(
                f"execution-plan host {host_id!r} contains unknown trials: {sorted(unknown)}"
            )
        families = {str(expected[trial_id]["optimizer_family"]) for trial_id in trial_ids}
        if families != set(grid_bundle["manifest"]["optimizer_order"]):
            raise ValueError(
                f"execution-plan host {host_id!r} must include every optimizer family"
            )
        assigned.extend(str(trial_id) for trial_id in trial_ids)

    if len(assigned) != len(set(assigned)):
        raise ValueError("execution plan assigns at least one trial more than once")
    if set(assigned) != set(expected):
        missing_trials = sorted(set(expected).difference(assigned))
        raise ValueError(
            "execution plan must assign all and only the frozen grid trials; "
            f"missing={missing_trials}"
        )
    return plan


def host_trial_ids(plan: Mapping[str, Any], host_id: str) -> list[str]:
    """Return the preserved trial order for one validated host assignment."""
    hosts = plan.get("hosts")
    if not isinstance(hosts, Mapping) or host_id not in hosts:
        raise ValueError(f"host {host_id!r} is absent from the execution plan")
    host = hosts[host_id]
    if not isinstance(host, Mapping) or not isinstance(host.get("trial_ids"), list):
        raise ValueError(f"host {host_id!r} has an invalid execution assignment")
    return [str(trial_id) for trial_id in host["trial_ids"]]


def staged_host_trial_ids(
    plan: Mapping[str, Any],
    host_id: str,
    stage: str,
) -> list[str]:
    """Select the canary or post-canary portion of one host assignment."""
    assigned = host_trial_ids(plan, host_id)
    canary_trial_id = str(plan.get("canary_trial_id"))
    if stage == "canary":
        if canary_trial_id not in assigned:
            raise ValueError(
                f"host {host_id!r} is not assigned the production canary"
            )
        return [canary_trial_id]
    if stage == "remaining":
        return [trial_id for trial_id in assigned if trial_id != canary_trial_id]
    raise ValueError("execution stage must be 'canary' or 'remaining'")
