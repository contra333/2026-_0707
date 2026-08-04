"""Deterministic protected-evaluation inventory and launch-plan freezing."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from oge.studies.hashing import canonical_sha256

from .common import DEFINITION_VERSION

INVENTORY_SCHEMA_VERSION = "1.0"
EXECUTION_SCHEMA_VERSION = "1.0"
EVALUATION_PROTOCOL_VERSION = "wrn28_10_cifar10_metric_v1.2"
RUNTIME_MERGE_GIT_SHA = "e22070ed436c772888e08ca1a4cb5a21760812a9"
SOURCE_AGGREGATE_FILE_SHA256 = (
    "1e0dee035ca77afd63c00b6ae9e74b8ed82246f69d95e1a8483f91624baf596b"
)
SOURCE_AGGREGATE_HASH = (
    "a403d23b747ba6b927211dd3a9acc7526fabcbba1daf9b56ad735219191fc856"
)
OPENOOD_SOURCE_COMMIT = "3c35632ee91b54b09d1f085d04f94744cece7d0b"
APPROVED_HOSTS = ("curie", "lise", "precision_medicine")
HOST_CONCURRENCY = {"curie": 4, "lise": 2, "precision_medicine": 4}
COORDINATOR_HOST = "curie"
CHECKPOINT_ROLES = ("last", "best_val")
ROLE_REPORTING = {
    "last": "confirmatory_primary",
    "best_val": "deployment_control",
}
EXPECTED_SOURCE_IDENTITIES = 42
EXPECTED_AUTHORIZED_IDENTITIES = 30
EXPECTED_EXCLUDED_IDENTITIES = 12
EXPECTED_CHECKPOINT_JOBS = 60
EXPECTED_HOST_JOB_COUNT = 20

OOD_SPLIT_POLICY = {
    "cifar100": {
        "dataset_name": "cifar100",
        "group": "near",
        "count": 9000,
        "imglist_sha256": "f907cb13024ba0fd13099954064df8f2f1ab8e3369c27b95ab44f3c5fc87ad16",
    },
    "tin": {
        "dataset_name": "tin",
        "group": "near",
        "count": 7793,
        "imglist_sha256": "a44fd34e3c925d984063d2ad64969ff71e8fa6f5524fbe9da481c8b0c87b27cd",
    },
    "mnist": {
        "dataset_name": "mnist",
        "group": "far",
        "count": 70000,
        "imglist_sha256": "48f8055d7327cbe91bff3ab6b43f9b623c3983355c55c246207907841c2ccb92",
    },
    "svhn": {
        "dataset_name": "svhn",
        "group": "far",
        "count": 26032,
        "imglist_sha256": "f0cb7b825a985af2a6bf803cf51d7ca004e24ee8d8051eb6805bf77c77621c1e",
    },
    "texture": {
        "dataset_name": "texture",
        "group": "far",
        "count": 5640,
        "imglist_sha256": "644fcaec364ee1ee440ff5285e1581d162b957378bb240ac4b7fc26e4f2f48e4",
    },
    "places365": {
        "dataset_name": "places365",
        "group": "far",
        "count": 35195,
        "imglist_sha256": "fa127faa5c1bbc5b567e1cd2871b04b570f045d1676550f441ec5e7c91ab6cc7",
    },
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_sha256_sidecar(path: Path, expected_filename: str) -> str:
    rows = path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one checksum row")
    digest, separator, filename = rows[0].partition("  ")
    if not separator or filename != expected_filename or len(digest) != 64:
        raise ValueError(f"{path} has an invalid checksum row")
    return digest


def _load_verified_aggregate(
    aggregate_path: str | Path,
    aggregate_sha256_path: str | Path,
) -> tuple[dict[str, Any], str]:
    path = Path(aggregate_path)
    sidecar = Path(aggregate_sha256_path)
    expected_file_sha = _read_sha256_sidecar(sidecar, path.name)
    actual_file_sha = _sha256_file(path)
    if expected_file_sha != actual_file_sha:
        raise ValueError("follow-up aggregate file SHA256 sidecar mismatch")
    if actual_file_sha != SOURCE_AGGREGATE_FILE_SHA256:
        raise ValueError("follow-up aggregate file differs from the Issue #49 seal")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("follow-up aggregate must be a JSON mapping")
    unhashed = dict(value)
    actual_internal_hash = unhashed.pop("aggregate_hash", None)
    if actual_internal_hash != canonical_sha256(unhashed):
        raise ValueError("follow-up aggregate internal hash mismatch")
    if actual_internal_hash != SOURCE_AGGREGATE_HASH:
        raise ValueError("follow-up aggregate identity is not the Issue #49 aggregate")
    return value, actual_file_sha


def _load_dataset_policy(dataset_config_path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(dataset_config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("dataset config must be a YAML mapping")
    if config.get("protocol_name") != "oge_cifar10_holdout_v1":
        raise ValueError("evaluation planning requires oge_cifar10_holdout_v1")
    datasets = config.get("datasets")
    membership = config.get("membership_manifest")
    if not isinstance(datasets, Mapping) or not isinstance(membership, Mapping):
        raise ValueError("dataset config is missing datasets or membership manifest")

    id_expected = {
        "id_train": (45000, "54fae08d35de5be7239b8fc35413531335a71858288f00c5583de44d759bfbc6"),
        "id_validation": (5000, "fe74684d1211a92eaccba00957ca308b43bff286f230cd62db89412fb753b488"),
        "id_test": (10000, "4178c1a97951afab348d63734d426d7e3cb0d13ddf3c0aebae331467d723fd73"),
    }
    id_policy: dict[str, Any] = {}
    for key, (count, digest) in id_expected.items():
        item = datasets.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"dataset config is missing {key}")
        if int(item.get("expected_count", -1)) != count or item.get("expected_sha256") != digest:
            raise ValueError(f"dataset config {key} identity mismatch")
        id_policy[key] = {
            "config_key": key,
            "dataset_name": item["dataset_name"],
            "logical_role": {
                "id_train": "fit_and_primary_geometry",
                "id_validation": "temperature_and_heldout_control",
                "id_test": "id_test_primary",
            }[key],
            "count": count,
            "imglist_sha256": digest,
        }

    ood_policy: dict[str, Any] = {}
    for key, expected in OOD_SPLIT_POLICY.items():
        item = datasets.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"dataset config is missing OOD split {key}")
        if item.get("dataset_name") != expected["dataset_name"] or item.get("group") != expected["group"]:
            raise ValueError(f"dataset config OOD split {key} identity mismatch")
        ood_policy[key] = {"config_key": key, **expected}

    forbidden = {
        "id_test_openood": {
            "reason": "compatibility_only_9k_not_executed",
            "count": 9000,
            "imglist_sha256": "84559ec5225425e7f7363058d5f9442ed21a39c7c66b63c3f1bb8b1ddb935e13",
        },
        "ood_validation_tin": {
            "reason": "compatibility_only_not_used_for_selection_or_evaluation",
            "count": 1000,
            "imglist_sha256": "39e22c5c6c2d95f95ab5ba43106f43ad7aa254ff62316104f406f7f10811f7c2",
        },
    }
    for key in forbidden:
        if key not in datasets:
            raise ValueError(f"dataset config must retain excluded provenance {key}")

    policy = {
        "protocol_name": config["protocol_name"],
        "dataset_config_path": "configs/datasets/oge_cifar10_holdout_v1.yaml",
        "dataset_config_sha256": _sha256_file(path),
        "membership_manifest": {
            "row_count": int(membership["row_count"]),
            "sha256": membership["sha256"],
        },
        "id_splits": id_policy,
        "ood_splits": ood_policy,
        "near_order": ["cifar100", "tin"],
        "far_order": ["mnist", "svhn", "texture", "places365"],
        "excluded_splits": forbidden,
        "openood_source_commit": OPENOOD_SOURCE_COMMIT,
        "aggregation": "per_dataset_then_arithmetic_macro_mean",
    }
    return policy, canonical_sha256(policy)


def _source_locator(record: Mapping[str, Any], checkpoint_role: str) -> dict[str, Any]:
    source = str(record["source"])
    host_id = str(record["host_id"])
    trial_id = str(record["source_trial_id"])
    attempt_ids = [str(value) for value in record["attempt_ids"]]
    if not attempt_ids:
        raise ValueError(f"source record {trial_id} has no preserved attempt")
    terminal_attempt_id = attempt_ids[-1]
    if source == "seed0_grid_reuse":
        execution_id = "seed0_20260728"
        stage = "canary" if trial_id == "grid-sgd-06" else "remaining"
        prefix = f"hf://buckets/contra333/ICLR_RUN/servers/{host_id}/{execution_id}/{stage}"
    elif source == "followup_execution":
        execution_id = "role_pair_followup_20260730"
        stage = "complete"
        prefix = f"hf://buckets/contra333/ICLR_RUN/servers/{host_id}/{execution_id}"
    else:
        raise ValueError(f"unsupported aggregate source {source!r}")
    filename = {"last": "last.pt", "best_val": "best_val.pt"}[checkpoint_role]
    relative = (
        f"trials/{trial_id}/attempts/{terminal_attempt_id}/run/checkpoints/{filename}"
    )
    return {
        "source_kind": source,
        "source_host": host_id,
        "source_execution_id": execution_id,
        "source_stage": stage,
        "source_trial_id": trial_id,
        "preserved_attempt_ids": attempt_ids,
        "terminal_attempt_id": terminal_attempt_id,
        "artifact_relative_path": relative,
        "hf_checkpoint_uri": f"{prefix}/{relative}",
    }


def _is_role_record(record: Mapping[str, Any]) -> bool:
    return any(str(label).startswith("role:") for label in record["labels"])


def _training_identity(record: Mapping[str, Any], authorized: bool) -> dict[str, Any]:
    checkpoints = {
        role: {
            "role": role,
            "reporting_role": ROLE_REPORTING[role],
            "completed_epoch": int(record["checkpoints"][role]["completed_epoch"]),
            "sha256": record["checkpoints"][role]["sha256"],
            "source": _source_locator(record, role),
        }
        for role in CHECKPOINT_ROLES
    }
    return {
        "training_identity_id": f"{record['config_hash']}:{int(record['training_seed'])}",
        "aggregate_trial_id": record["aggregate_trial_id"],
        "config_hash": record["config_hash"],
        "training_seed": int(record["training_seed"]),
        "optimizer_family": record["optimizer_family"],
        "optimizer": dict(record["optimizer"]),
        "labels": sorted(str(label) for label in record["labels"]),
        "training_git_sha": record["git_sha"],
        "training_provenance_hash": record["provenance_hash"],
        "dataset_membership_hashes": dict(record["dataset_membership_hashes"]),
        "source_host": record["host_id"],
        "disposition": "authorized_role" if authorized else "excluded_pair_only",
        "exclusion_reason": None if authorized else "card07_role_config_hashes_only",
        "checkpoints": checkpoints,
    }


def build_checkpoint_inventory(
    *,
    aggregate_path: str | Path,
    aggregate_sha256_path: str | Path,
    dataset_config_path: str | Path,
) -> dict[str, Any]:
    """Build the frozen 42-source/30-authorized/60-job inventory."""
    aggregate, aggregate_file_sha = _load_verified_aggregate(
        aggregate_path, aggregate_sha256_path
    )
    records = aggregate.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SOURCE_IDENTITIES:
        raise ValueError("source aggregate must contain exactly 42 records")
    dataset_policy, dataset_policy_hash = _load_dataset_policy(dataset_config_path)
    identities = [_training_identity(record, _is_role_record(record)) for record in records]
    identities.sort(
        key=lambda item: (
            str(item["optimizer_family"]),
            str(item["config_hash"]),
            int(item["training_seed"]),
        )
    )

    jobs: list[dict[str, Any]] = []
    for identity in identities:
        if identity["disposition"] != "authorized_role":
            continue
        for role in CHECKPOINT_ROLES:
            checkpoint = identity["checkpoints"][role]
            checkpoint_sha = str(checkpoint["sha256"])
            jobs.append(
                {
                    "job_id": f"eval-{identity['aggregate_trial_id']}-{role}",
                    "training_identity_id": identity["training_identity_id"],
                    "aggregate_trial_id": identity["aggregate_trial_id"],
                    "config_hash": identity["config_hash"],
                    "training_seed": identity["training_seed"],
                    "optimizer_family": identity["optimizer_family"],
                    "labels": list(identity["labels"]),
                    "checkpoint_role": role,
                    "reporting_role": checkpoint["reporting_role"],
                    "checkpoint_sha256": checkpoint_sha,
                    "completed_epoch": checkpoint["completed_epoch"],
                    "assigned_host": identity["source_host"],
                    "source": dict(checkpoint["source"]),
                    "bindings": {
                        "definition_version": DEFINITION_VERSION,
                        "runtime_merge_git_sha": RUNTIME_MERGE_GIT_SHA,
                        "dataset_policy_hash": dataset_policy_hash,
                        "membership_manifest_sha256": dataset_policy[
                            "membership_manifest"
                        ]["sha256"],
                        "fit_dtype": "float64",
                        "protected_authorization_policy": (
                            "separate_active_execution_issue_required"
                        ),
                    },
                    "output_namespace": (
                        f"evaluations/{DEFINITION_VERSION}/{checkpoint_sha}/"
                    ),
                    "hf_output_uri": (
                        "hf://buckets/contra333/ICLR_RUN/"
                        f"evaluations/{DEFINITION_VERSION}/{checkpoint_sha}"
                    ),
                    "planned_status": "not_started",
                }
            )

    payload: dict[str, Any] = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "definition_version": DEFINITION_VERSION,
        "planning_issue": 55,
        "runtime_merge_git_sha": RUNTIME_MERGE_GIT_SHA,
        "source_aggregate": {
            "path": (
                "results/wrn28_10_optimizer_hpo_v1_2/"
                "followup_20260730/aggregate.json"
            ),
            "file_sha256": aggregate_file_sha,
            "aggregate_hash": aggregate["aggregate_hash"],
            "role_freeze_hash": aggregate["role_freeze_hash"],
            "followup_plan_hash": aggregate["followup_plan_hash"],
        },
        "dataset_policy": dataset_policy,
        "dataset_policy_hash": dataset_policy_hash,
        "authorization": {
            "planning_issue_authorizes_protected_access": False,
            "required_launch_authorization": "separate_active_execution_issue",
            "pair_only_protected_evaluation": "not_authorized",
        },
        "counts": {
            "source_training_identities": len(identities),
            "source_configurations": len({item["config_hash"] for item in identities}),
            "authorized_training_identities": sum(
                item["disposition"] == "authorized_role" for item in identities
            ),
            "authorized_configurations": len(
                {
                    item["config_hash"]
                    for item in identities
                    if item["disposition"] == "authorized_role"
                }
            ),
            "excluded_pair_only_training_identities": sum(
                item["disposition"] == "excluded_pair_only" for item in identities
            ),
            "excluded_pair_only_checkpoint_references": 2
            * sum(item["disposition"] == "excluded_pair_only" for item in identities),
            "checkpoint_jobs": len(jobs),
        },
        "training_identities": identities,
        "checkpoint_jobs": jobs,
    }
    validate_checkpoint_inventory(payload)
    payload["inventory_hash"] = canonical_sha256(payload)
    return payload


def validate_checkpoint_inventory(payload: Mapping[str, Any]) -> None:
    """Reject identity, role, assignment, split, and namespace drift."""
    if payload.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint-inventory schema version")
    if payload.get("protocol_version") != EVALUATION_PROTOCOL_VERSION:
        raise ValueError("checkpoint-inventory protocol version mismatch")
    if payload.get("definition_version") != DEFINITION_VERSION:
        raise ValueError("checkpoint-inventory definition version mismatch")
    if payload.get("runtime_merge_git_sha") != RUNTIME_MERGE_GIT_SHA:
        raise ValueError("checkpoint-inventory runtime merge SHA mismatch")
    if "inventory_hash" in payload:
        unhashed = dict(payload)
        observed = unhashed.pop("inventory_hash")
        if observed != canonical_sha256(unhashed):
            raise ValueError("checkpoint-inventory hash mismatch")

    source = payload.get("source_aggregate")
    if not isinstance(source, Mapping):
        raise ValueError("checkpoint inventory is missing source aggregate identity")
    if source.get("file_sha256") != SOURCE_AGGREGATE_FILE_SHA256:
        raise ValueError("checkpoint inventory aggregate file SHA mismatch")
    if source.get("aggregate_hash") != SOURCE_AGGREGATE_HASH:
        raise ValueError("checkpoint inventory aggregate hash mismatch")

    policy = payload.get("dataset_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("checkpoint inventory is missing dataset policy")
    if payload.get("dataset_policy_hash") != canonical_sha256(policy):
        raise ValueError("checkpoint inventory dataset policy hash mismatch")
    if list(policy.get("near_order", [])) != ["cifar100", "tin"]:
        raise ValueError("checkpoint inventory near-OOD policy mismatch")
    if list(policy.get("far_order", [])) != [
        "mnist",
        "svhn",
        "texture",
        "places365",
    ]:
        raise ValueError("checkpoint inventory far-OOD policy mismatch")
    id_splits = policy.get("id_splits")
    if not isinstance(id_splits, Mapping) or {
        key: int(value["count"]) for key, value in id_splits.items()
    } != {"id_train": 45000, "id_validation": 5000, "id_test": 10000}:
        raise ValueError("checkpoint inventory ID split policy mismatch")
    if set(policy.get("excluded_splits", {})) != {
        "id_test_openood",
        "ood_validation_tin",
    }:
        raise ValueError("checkpoint inventory excluded-split policy mismatch")

    identities = payload.get("training_identities")
    jobs = payload.get("checkpoint_jobs")
    if not isinstance(identities, list) or len(identities) != EXPECTED_SOURCE_IDENTITIES:
        raise ValueError("checkpoint inventory must contain 42 source identities")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_CHECKPOINT_JOBS:
        raise ValueError("checkpoint inventory must contain 60 checkpoint jobs")
    identity_keys = [
        (str(item["config_hash"]), int(item["training_seed"])) for item in identities
    ]
    if len(identity_keys) != len(set(identity_keys)):
        raise ValueError("checkpoint inventory repeats a config/seed identity")
    by_config: dict[str, set[int]] = defaultdict(set)
    for config_hash, seed in identity_keys:
        by_config[config_hash].add(seed)
    if len(by_config) != 14 or any(seeds != {0, 1, 2} for seeds in by_config.values()):
        raise ValueError("checkpoint inventory source configs must cover seeds 0,1,2")

    authorized = [
        item for item in identities if item.get("disposition") == "authorized_role"
    ]
    excluded = [
        item for item in identities if item.get("disposition") == "excluded_pair_only"
    ]
    if len(authorized) != EXPECTED_AUTHORIZED_IDENTITIES or len(excluded) != EXPECTED_EXCLUDED_IDENTITIES:
        raise ValueError("checkpoint inventory authorization counts mismatch")
    if any(not any(str(label).startswith("role:") for label in item["labels"]) for item in authorized):
        raise ValueError("authorized identity lacks a frozen role label")
    if any(any(str(label).startswith("role:") for label in item["labels"]) for item in excluded):
        raise ValueError("role identity was incorrectly excluded")
    if any(item.get("exclusion_reason") != "card07_role_config_hashes_only" for item in excluded):
        raise ValueError("pair-only exclusion reason mismatch")
    authorized_configs: dict[str, set[int]] = defaultdict(set)
    for item in authorized:
        authorized_configs[str(item["config_hash"])].add(int(item["training_seed"]))
    if len(authorized_configs) != 10 or any(
        seeds != {0, 1, 2} for seeds in authorized_configs.values()
    ):
        raise ValueError("authorized role configs must cover 10 configs x three seeds")

    expected_jobs: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in authorized:
        checkpoints = item.get("checkpoints")
        if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(CHECKPOINT_ROLES):
            raise ValueError("authorized identity must retain last and best_val")
        for role in CHECKPOINT_ROLES:
            expected_jobs[(str(item["training_identity_id"]), role)] = checkpoints[role]
    if any(job.get("checkpoint_role") not in CHECKPOINT_ROLES for job in jobs):
        raise ValueError("checkpoint inventory contains an invalid role")
    observed_job_keys = [
        (str(job["training_identity_id"]), str(job["checkpoint_role"])) for job in jobs
    ]
    if len(observed_job_keys) != len(set(observed_job_keys)) or set(observed_job_keys) != set(expected_jobs):
        raise ValueError("checkpoint jobs do not cover authorized role pairs exactly once")

    job_ids = [str(job["job_id"]) for job in jobs]
    checkpoint_shas = [str(job["checkpoint_sha256"]) for job in jobs]
    outputs = [str(job["output_namespace"]) for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("checkpoint inventory repeats a job ID")
    if len(checkpoint_shas) != len(set(checkpoint_shas)):
        raise ValueError("checkpoint inventory repeats a checkpoint SHA")
    if len(outputs) != len(set(outputs)):
        raise ValueError("checkpoint inventory output namespace collision")

    host_counts = Counter(str(job["assigned_host"]) for job in jobs)
    if host_counts != Counter({host: EXPECTED_HOST_JOB_COUNT for host in APPROVED_HOSTS}):
        raise ValueError("checkpoint inventory host assignment must be 20/20/20")
    dataset_hash = str(payload["dataset_policy_hash"])
    membership_hash = policy["membership_manifest"]["sha256"]
    for job in jobs:
        key = (str(job["training_identity_id"]), str(job["checkpoint_role"]))
        checkpoint = expected_jobs[key]
        if job["reporting_role"] != ROLE_REPORTING[job["checkpoint_role"]]:
            raise ValueError("checkpoint inventory checkpoint reporting role mismatch")
        if (
            job["checkpoint_sha256"] != checkpoint["sha256"]
            or int(job["completed_epoch"]) != int(checkpoint["completed_epoch"])
        ):
            raise ValueError("checkpoint job differs from its aggregate reference")
        if job["assigned_host"] != job["source"]["source_host"]:
            raise ValueError("checkpoint job is not source-local")
        bindings = job.get("bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError("checkpoint job is missing immutable bindings")
        expected_bindings = {
            "definition_version": DEFINITION_VERSION,
            "runtime_merge_git_sha": RUNTIME_MERGE_GIT_SHA,
            "dataset_policy_hash": dataset_hash,
            "membership_manifest_sha256": membership_hash,
            "fit_dtype": "float64",
            "protected_authorization_policy": "separate_active_execution_issue_required",
        }
        if dict(bindings) != expected_bindings:
            raise ValueError("checkpoint job immutable bindings mismatch")
        expected_output = f"evaluations/{DEFINITION_VERSION}/{job['checkpoint_sha256']}/"
        if job["output_namespace"] != expected_output:
            raise ValueError("checkpoint job output must be checkpoint-centric")
        if job.get("planned_status") != "not_started":
            raise ValueError("checkpoint inventory must remain not_started")

    counts = payload.get("counts")
    expected_counts = {
        "source_training_identities": 42,
        "source_configurations": 14,
        "authorized_training_identities": 30,
        "authorized_configurations": 10,
        "excluded_pair_only_training_identities": 12,
        "excluded_pair_only_checkpoint_references": 24,
        "checkpoint_jobs": 60,
    }
    if not isinstance(counts, Mapping) or dict(counts) != expected_counts:
        raise ValueError("checkpoint inventory summary counts mismatch")


def build_execution_manifest(
    inventory: Mapping[str, Any],
    *,
    inventory_file_sha256: str,
) -> dict[str, Any]:
    validate_checkpoint_inventory(inventory)
    hosts: dict[str, Any] = {}
    jobs = inventory["checkpoint_jobs"]
    for host_id in APPROVED_HOSTS:
        job_ids = [
            str(job["job_id"]) for job in jobs if job["assigned_host"] == host_id
        ]
        hosts[host_id] = {
            "coordinator": host_id == COORDINATOR_HOST,
            "concurrency": HOST_CONCURRENCY[host_id],
            "checkpoint_job_count": len(job_ids),
            "checkpoint_job_ids": job_ids,
        }
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "definition_version": DEFINITION_VERSION,
        "execution_id": "metric_contract_v1_2_protected_eval_planned_20260804",
        "planning_issue": 55,
        "runtime_merge_git_sha": RUNTIME_MERGE_GIT_SHA,
        "inventory": {
            "path": (
                "configs/evaluation/wrn28_10_cifar10_metric_v1_2/"
                "checkpoint_inventory.json"
            ),
            "file_sha256": inventory_file_sha256,
            "inventory_hash": inventory["inventory_hash"],
            "checkpoint_job_count": len(jobs),
        },
        "launch_authorization": {
            "status": "pending_separate_execution_issue",
            "planning_issue_authorizes_protected_access": False,
            "required_execution_git_sha": "merged_commit_containing_this_plan",
        },
        "state": {
            "execution_started": False,
            "protected_accessed": False,
            "server_sync": "not_started",
            "hf_upload": "not_started",
        },
        "coordinator_host": COORDINATOR_HOST,
        "assignment_policy": "source_local_checkpoint_no_transfer",
        "hosts": hosts,
        "dataset_policy_hash": inventory["dataset_policy_hash"],
        "output": {
            "layout": f"evaluations/{DEFINITION_VERSION}/{{checkpoint_sha256}}/",
            "hf_template": (
                "hf://buckets/contra333/ICLR_RUN/"
                f"evaluations/{DEFINITION_VERSION}/{{checkpoint_sha256}}"
            ),
            "identity": "checkpoint_sha256",
            "upload_policy": "dry_run_then_no_delete",
            "completion_marker": "REMOTE_COMPLETE.json",
            "coordinator_validation_required": True,
        },
        "counts": {
            "authorized_training_identities": 30,
            "checkpoint_jobs": 60,
            "jobs_per_host": 20,
            "excluded_pair_only_training_identities": 12,
        },
    }
    validate_execution_manifest(payload, inventory)
    payload["execution_hash"] = canonical_sha256(payload)
    return payload


def validate_execution_manifest(
    payload: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> None:
    validate_checkpoint_inventory(inventory)
    if payload.get("schema_version") != EXECUTION_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation execution schema version")
    if payload.get("protocol_version") != EVALUATION_PROTOCOL_VERSION:
        raise ValueError("evaluation execution protocol version mismatch")
    if payload.get("definition_version") != DEFINITION_VERSION:
        raise ValueError("evaluation execution definition version mismatch")
    if payload.get("runtime_merge_git_sha") != RUNTIME_MERGE_GIT_SHA:
        raise ValueError("evaluation execution runtime merge SHA mismatch")
    if "execution_hash" in payload:
        unhashed = dict(payload)
        observed = unhashed.pop("execution_hash")
        if observed != canonical_sha256(unhashed):
            raise ValueError("evaluation execution hash mismatch")
    state = payload.get("state")
    if not isinstance(state, Mapping) or dict(state) != {
        "execution_started": False,
        "protected_accessed": False,
        "server_sync": "not_started",
        "hf_upload": "not_started",
    }:
        raise ValueError("planning manifest must not record started execution")
    authorization = payload.get("launch_authorization")
    if not isinstance(authorization, Mapping) or authorization.get("status") != "pending_separate_execution_issue":
        raise ValueError("planning manifest requires a separate launch authorization")
    if authorization.get("planning_issue_authorizes_protected_access") is not False:
        raise ValueError("planning Issue must not authorize protected access")
    if payload.get("coordinator_host") != COORDINATOR_HOST:
        raise ValueError("evaluation execution coordinator must be curie")
    if payload.get("assignment_policy") != "source_local_checkpoint_no_transfer":
        raise ValueError("evaluation execution assignment policy mismatch")

    inventory_reference = payload.get("inventory")
    if not isinstance(inventory_reference, Mapping):
        raise ValueError("execution manifest is missing inventory identity")
    if (
        inventory_reference.get("inventory_hash") != inventory["inventory_hash"]
        or int(inventory_reference.get("checkpoint_job_count", -1)) != 60
    ):
        raise ValueError("execution manifest inventory identity mismatch")
    hosts = payload.get("hosts")
    if not isinstance(hosts, Mapping) or tuple(hosts) != APPROVED_HOSTS:
        raise ValueError("execution manifest approved hosts mismatch")
    inventory_jobs = {
        str(job["job_id"]): job for job in inventory["checkpoint_jobs"]
    }
    assigned: list[str] = []
    for host_id in APPROVED_HOSTS:
        host = hosts[host_id]
        if not isinstance(host, Mapping):
            raise ValueError(f"execution host {host_id} must be a mapping")
        job_ids = host.get("checkpoint_job_ids")
        if not isinstance(job_ids, list) or len(job_ids) != EXPECTED_HOST_JOB_COUNT:
            raise ValueError(f"execution host {host_id} must contain 20 jobs")
        if len(job_ids) != len(set(job_ids)):
            raise ValueError(f"execution host {host_id} repeats a job")
        if host.get("checkpoint_job_count") != EXPECTED_HOST_JOB_COUNT:
            raise ValueError(f"execution host {host_id} count mismatch")
        if host.get("concurrency") != HOST_CONCURRENCY[host_id]:
            raise ValueError(f"execution host {host_id} concurrency mismatch")
        if host.get("coordinator") is not (host_id == COORDINATOR_HOST):
            raise ValueError(f"execution host {host_id} coordinator flag mismatch")
        for job_id in job_ids:
            if job_id not in inventory_jobs:
                raise ValueError(f"execution host {host_id} contains an unknown job")
        assigned.extend(str(job_id) for job_id in job_ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(inventory_jobs):
        raise ValueError("execution manifest must assign all jobs exactly once")
    for host_id in APPROVED_HOSTS:
        for job_id in hosts[host_id]["checkpoint_job_ids"]:
            if inventory_jobs[job_id]["assigned_host"] != host_id:
                raise ValueError(f"execution job {job_id} is not source-local")

    role_hosts: dict[str, set[str]] = defaultdict(set)
    for job_id in assigned:
        job = inventory_jobs[job_id]
        role_hosts[str(job["training_identity_id"])].add(str(job["assigned_host"]))
    if any(len(values) != 1 for values in role_hosts.values()):
        raise ValueError("last and best_val roles must remain on the same host")
    output = payload.get("output")
    if not isinstance(output, Mapping) or output.get("layout") != (
        f"evaluations/{DEFINITION_VERSION}/{{checkpoint_sha256}}/"
    ):
        raise ValueError("execution output layout must be checkpoint-centric")
    if output.get("upload_policy") != "dry_run_then_no_delete":
        raise ValueError("execution upload policy must forbid deletes")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _yaml_bytes(value: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(value),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")


def render_evaluation_plan_files(
    inventory: Mapping[str, Any],
) -> dict[str, bytes]:
    validate_checkpoint_inventory(inventory)
    inventory_bytes = _json_bytes(inventory)
    execution = build_execution_manifest(
        inventory, inventory_file_sha256=_sha256_bytes(inventory_bytes)
    )
    execution_bytes = _yaml_bytes(execution)
    return {
        "checkpoint_inventory.json": inventory_bytes,
        "checkpoint_inventory.json.sha256": (
            f"{_sha256_bytes(inventory_bytes)}  checkpoint_inventory.json\n"
        ).encode("utf-8"),
        "execution.yaml": execution_bytes,
        "execution.yaml.sha256": (
            f"{_sha256_bytes(execution_bytes)}  execution.yaml\n"
        ).encode("utf-8"),
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_evaluation_plan(
    inventory: Mapping[str, Any],
    destination: str | Path,
) -> dict[str, str]:
    files = render_evaluation_plan_files(inventory)
    root = Path(destination)
    for name, content in files.items():
        _atomic_write(root / name, content)
    return {name: _sha256_bytes(content) for name, content in files.items()}


def load_materialized_evaluation_plan(
    source: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(source)
    inventory_path = root / "checkpoint_inventory.json"
    execution_path = root / "execution.yaml"
    for path in (inventory_path, execution_path):
        sidecar = path.with_name(f"{path.name}.sha256")
        expected = _read_sha256_sidecar(sidecar, path.name)
        if _sha256_file(path) != expected:
            raise ValueError(f"materialized plan checksum mismatch for {path.name}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    if not isinstance(inventory, dict) or not isinstance(execution, dict):
        raise ValueError("materialized evaluation plan files must contain mappings")
    validate_checkpoint_inventory(inventory)
    validate_execution_manifest(execution, inventory)
    if execution["inventory"]["file_sha256"] != _sha256_file(inventory_path):
        raise ValueError("execution manifest inventory file checksum mismatch")
    return inventory, execution


def verify_materialized_evaluation_plan(
    source: str | Path,
    expected_inventory: Mapping[str, Any],
) -> None:
    loaded_inventory, _ = load_materialized_evaluation_plan(source)
    if loaded_inventory != dict(expected_inventory):
        raise ValueError("materialized checkpoint inventory differs from generation")
    expected_files = render_evaluation_plan_files(expected_inventory)
    root = Path(source)
    for name, content in expected_files.items():
        if (root / name).read_bytes() != content:
            raise ValueError(f"materialized evaluation plan differs at {name}")
