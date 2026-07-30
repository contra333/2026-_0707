"""Validated multi-host execution assignments for frozen optimizer studies."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .protocol import GRID_TRAINING_SEED, PROTOCOL_VERSION, validate_grid_bundle

EXECUTION_SCHEMA_VERSION = "1.0"
FOLLOWUP_EXECUTION_SCHEMA_VERSION = "1.0"
FOLLOWUP_APPROVED_HOSTS = ("curie", "lise", "precision_medicine")
FOLLOWUP_HOST_COUNTS = {
    "curie": 13,
    "lise": 10,
    "precision_medicine": 7,
}


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


def load_followup_execution_plan(
    path: str | Path,
    followup_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the exact-once host assignment for the frozen follow-up plan."""
    if not isinstance(followup_plan, Mapping):
        raise ValueError("frozen follow-up plan must be a mapping")
    scheduled = followup_plan.get("scheduled")
    reused = followup_plan.get("reused")
    if not isinstance(scheduled, list) or not isinstance(reused, list):
        raise ValueError("frozen follow-up plan must contain scheduled and reused rows")

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("follow-up execution plan must be a YAML mapping")
    plan = copy.deepcopy(dict(value))
    required = {
        "schema_version",
        "protocol_version",
        "execution_id",
        "grid_manifest_hash",
        "role_freeze_hash",
        "followup_plan_hash",
        "approved_hosts",
        "seed0_reuse_hosts",
        "pair_colocations",
        "hf_account",
        "hf_bucket",
        "hf_destination_template",
        "hosts",
    }
    missing = required.difference(plan)
    if missing:
        raise ValueError(
            f"follow-up execution plan is missing fields: {sorted(missing)}"
        )
    if plan["schema_version"] != FOLLOWUP_EXECUTION_SCHEMA_VERSION:
        raise ValueError("unsupported follow-up execution-plan schema_version")
    if plan["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("follow-up execution-plan protocol_version mismatch")
    for field, source_field in (
        ("grid_manifest_hash", "grid_manifest_hash"),
        ("role_freeze_hash", "role_freeze_hash"),
        ("followup_plan_hash", "plan_hash"),
    ):
        if plan[field] != followup_plan.get(source_field):
            raise ValueError(f"follow-up execution-plan {field} mismatch")
    if not isinstance(plan["execution_id"], str) or not plan["execution_id"].strip():
        raise ValueError("follow-up execution plan requires a non-empty execution_id")
    if tuple(plan["approved_hosts"]) != FOLLOWUP_APPROVED_HOSTS:
        raise ValueError("follow-up execution plan approved_hosts mismatch")
    if plan["hf_account"] != "contra333" or plan["hf_bucket"] != "contra333/ICLR_RUN":
        raise ValueError("follow-up execution plan Hugging Face target mismatch")
    template = plan["hf_destination_template"]
    if (
        not isinstance(template, str)
        or template.count("{host_id}") != 1
        or not template.startswith("hf://buckets/contra333/ICLR_RUN/servers/")
    ):
        raise ValueError("follow-up execution plan has an invalid HF destination")

    expected_rows = {str(row["trial_id"]): row for row in scheduled}
    if len(expected_rows) != len(scheduled):
        raise ValueError("frozen follow-up plan repeats a scheduled trial ID")
    hosts = plan["hosts"]
    if not isinstance(hosts, Mapping) or set(hosts) != set(FOLLOWUP_APPROVED_HOSTS):
        raise ValueError("follow-up execution plan must assign the approved hosts")

    assigned_hosts: dict[str, str] = {}
    for host_id in FOLLOWUP_APPROVED_HOSTS:
        host_value = hosts[host_id]
        if not isinstance(host_value, Mapping):
            raise ValueError(f"follow-up host {host_id!r} must be a mapping")
        trial_ids = host_value.get("trial_ids")
        concurrency = host_value.get("concurrency")
        if not isinstance(trial_ids, list) or not trial_ids:
            raise ValueError(f"follow-up host {host_id!r} has no trial_ids")
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError(f"follow-up host {host_id!r} repeats a trial ID")
        if len(trial_ids) != FOLLOWUP_HOST_COUNTS[host_id]:
            raise ValueError(
                f"follow-up host {host_id!r} must have "
                f"{FOLLOWUP_HOST_COUNTS[host_id]} trials"
            )
        expected_concurrency = {"curie": 4, "lise": 2, "precision_medicine": 4}[
            host_id
        ]
        if concurrency != expected_concurrency:
            raise ValueError(
                f"follow-up host {host_id!r} concurrency must be "
                f"{expected_concurrency}"
            )
        unknown = set(trial_ids).difference(expected_rows)
        if unknown:
            raise ValueError(
                f"follow-up host {host_id!r} contains unknown trials: "
                f"{sorted(unknown)}"
            )
        families = {
            str(expected_rows[str(trial_id)]["optimizer_family"])
            for trial_id in trial_ids
        }
        if families != {"sgd", "sgdw", "adam", "adamw"}:
            raise ValueError(
                f"follow-up host {host_id!r} must include every optimizer family"
            )
        for trial_id in trial_ids:
            trial_id = str(trial_id)
            if trial_id in assigned_hosts:
                raise ValueError(
                    "follow-up execution plan assigns at least one trial more than once"
                )
            assigned_hosts[trial_id] = host_id
    if set(assigned_hosts) != set(expected_rows):
        missing_trials = sorted(set(expected_rows).difference(assigned_hosts))
        raise ValueError(
            "follow-up execution plan must assign all scheduled trials; "
            f"missing={missing_trials}"
        )

    seed0_reuse_hosts = plan["seed0_reuse_hosts"]
    reused_by_config = {
        str(row["config_hash"]): row
        for row in reused
        if int(row["training_seed"]) == 0
    }
    if not isinstance(seed0_reuse_hosts, Mapping):
        raise ValueError("seed0_reuse_hosts must be a mapping")
    if set(seed0_reuse_hosts) != set(reused_by_config):
        raise ValueError("seed0_reuse_hosts must cover every reused configuration")
    if any(host not in FOLLOWUP_APPROVED_HOSTS for host in seed0_reuse_hosts.values()):
        raise ValueError("seed0_reuse_hosts contains an unapproved host")

    config_seed_hosts: dict[str, dict[int, str]] = {
        config_hash: {0: str(seed0_reuse_hosts[config_hash])}
        for config_hash in reused_by_config
    }
    for trial_id, host_id in assigned_hosts.items():
        row = expected_rows[trial_id]
        config_hash = str(row["config_hash"])
        seed = int(row["training_seed"])
        seed_hosts = config_seed_hosts.setdefault(config_hash, {})
        if seed in seed_hosts:
            raise ValueError("follow-up execution plan repeats a config/seed identity")
        seed_hosts[seed] = host_id
    if len(config_seed_hosts) != 14:
        raise ValueError("follow-up execution plan must cover 14 configurations")
    for config_hash, seed_hosts in config_seed_hosts.items():
        if set(seed_hosts) != {0, 1, 2}:
            raise ValueError(
                f"configuration {config_hash!r} does not cover seeds 0, 1, and 2"
            )
        if set(seed_hosts.values()) != set(FOLLOWUP_APPROVED_HOSTS):
            raise ValueError(
                f"configuration {config_hash!r} does not rotate across all hosts"
            )

    pair_colocations = plan["pair_colocations"]
    if not isinstance(pair_colocations, list) or len(pair_colocations) != 4:
        raise ValueError("follow-up execution plan must declare four pair colocations")
    pair_ids: set[str] = set()
    for pair in pair_colocations:
        if not isinstance(pair, Mapping):
            raise ValueError("pair colocation entries must be mappings")
        pair_id = pair.get("pair_id")
        left = str(pair.get("left_config_hash"))
        right = str(pair.get("right_config_hash"))
        seeds = pair.get("seeds")
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise ValueError("pair colocation IDs must be unique non-empty strings")
        pair_ids.add(pair_id)
        if left not in config_seed_hosts or right not in config_seed_hosts:
            raise ValueError(f"pair colocation {pair_id!r} references an unknown config")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"pair colocation {pair_id!r} requires seeds")
        for seed in seeds:
            if int(seed) not in (0, 1, 2):
                raise ValueError(f"pair colocation {pair_id!r} has an invalid seed")
            if (
                config_seed_hosts[left][int(seed)]
                != config_seed_hosts[right][int(seed)]
            ):
                raise ValueError(
                    f"pair colocation {pair_id!r} is not on the same host at seed {seed}"
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
