"""Paired-control and checkpoint provenance for the ResNet-18 replication."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from oge.studies.hashing import canonical_sha256

from .provenance import (
    dataset_membership_sha256,
    initial_rng_state_hashes,
    model_initialization_sha256,
)
from .resnet18_replication_plan import (
    RESNET18_REPLICATION_STUDY_ID,
    validate_resnet18_replication_training_config,
)


RESNET18_REPLICATION_PAIRED_PROVENANCE_SCHEMA_VERSION = (
    "resnet18_cifar10_paired_control_provenance_v1"
)
RESNET18_REPLICATION_CHECKPOINT_PROVENANCE_SCHEMA_VERSION = (
    "resnet18_cifar10_checkpoint_provenance_v1"
)

_RESUME_FIELDS = (
    "sibling_group_id",
    "cross_lr_pairing_block_id",
    "training_seed",
    "data_stream_id",
    "initialization_sha256",
    "initial_python_rng_sha256",
    "initial_numpy_rng_sha256",
    "initial_torch_rng_sha256",
    "initial_dataloader_rng_sha256",
    "dataset_membership_sha256",
    "branch_neutral_config_sha256",
    "cross_lr_neutral_config_sha256",
    "execution_only",
)
_DATA_STREAM_FIELDS = (
    "data_stream_id",
    "initial_python_rng_sha256",
    "initial_numpy_rng_sha256",
    "initial_torch_rng_sha256",
    "initial_dataloader_rng_sha256",
    "dataset_membership_sha256",
    "cross_lr_neutral_config_sha256",
    "first_minibatch_ordered_sample_id_sha256",
    "first_minibatch_transformed_image_sha256",
    "first_minibatch_size",
)


def create_initial_resnet18_replication_provenance(
    *,
    resolved_config: Mapping[str, Any],
    model: Any,
    initial_rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    validate_resnet18_replication_training_config(resolved_config)
    replication = resolved_config["resnet18_replication"]
    return {
        "schema_version": RESNET18_REPLICATION_PAIRED_PROVENANCE_SCHEMA_VERSION,
        "run_plan_id": str(replication["run_id"]),
        "sibling_group_id": str(replication["sibling_group_id"]),
        "cross_lr_pairing_block_id": str(replication["cross_lr_pairing_block_id"]),
        "training_seed": int(resolved_config["training"]["seed"]),
        "data_stream_id": str(replication["data_stream_id"]),
        "branch_policy": str(replication["branch_policy"]),
        "initialization_sha256": model_initialization_sha256(model),
        **initial_rng_state_hashes(initial_rng_state),
        "dataset_membership_sha256": dataset_membership_sha256(resolved_config),
        "branch_neutral_config_sha256": str(
            replication["branch_neutral_config_sha256"]
        ),
        "cross_lr_neutral_config_sha256": str(
            replication["cross_lr_neutral_config_sha256"]
        ),
        "first_minibatch_ordered_sample_id_sha256": None,
        "first_minibatch_transformed_image_sha256": None,
        "first_minibatch_size": None,
        "first_minibatch_witness_status": "pending",
        "execution_only": bool(replication["execution_only"]),
    }


def validate_resume_resnet18_replication_identity(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    if saved.get("schema_version") != RESNET18_REPLICATION_PAIRED_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("resume checkpoint has unsupported replication provenance schema")
    if saved.get("first_minibatch_witness_status") != "observed":
        raise ValueError("resume checkpoint is missing the replication minibatch witness")
    for field in _RESUME_FIELDS:
        if saved.get(field) != current.get(field):
            raise ValueError(f"resume replication identity changed: {field}")
    if saved.get("branch_policy") != current.get("branch_policy"):
        raise ValueError("resume replication identity changed: branch_policy")


def resnet18_replication_data_stream_sha256(provenance: Mapping[str, Any]) -> str:
    if provenance.get("first_minibatch_witness_status") != "observed":
        raise ValueError("replication data-stream identity requires an observed first batch")
    missing = [field for field in _DATA_STREAM_FIELDS if provenance.get(field) is None]
    if missing:
        raise ValueError(f"replication data-stream identity is missing fields: {missing}")
    return canonical_sha256({field: provenance[field] for field in _DATA_STREAM_FIELDS})


def build_resnet18_replication_checkpoint_provenance(
    *,
    resolved_config: Mapping[str, Any],
    paired_control_provenance: Mapping[str, Any],
    checkpoint_epoch: int,
    checkpoint_role: str,
    oge_git_sha: str,
    run_id: str,
) -> dict[str, Any] | None:
    replication = resolved_config.get("resnet18_replication")
    if not isinstance(replication, Mapping):
        return None
    validate_resnet18_replication_training_config(resolved_config)
    if paired_control_provenance.get("first_minibatch_witness_status") != "observed":
        if checkpoint_role == "snapshot" and checkpoint_epoch == 0:
            return None
        raise ValueError("replication checkpoint provenance requires an observed first batch")
    if run_id != replication.get("run_id") or run_id != paired_control_provenance.get(
        "run_plan_id"
    ):
        raise ValueError("replication checkpoint run_id differs from the run plan")
    initialization_sha256 = str(paired_control_provenance["initialization_sha256"])
    data_stream_sha256 = resnet18_replication_data_stream_sha256(
        paired_control_provenance
    )
    members = {}
    for role, member in replication["sibling_members"].items():
        members[str(role)] = {
            **copy.deepcopy(dict(member)),
            "initialization_sha256": initialization_sha256,
            "data_stream_sha256": data_stream_sha256,
        }
    sibling_role = str(replication["sibling_role"])
    current = members[sibling_role]
    value = {
        "schema_version": RESNET18_REPLICATION_CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
        "study_id": RESNET18_REPLICATION_STUDY_ID,
        "run_id": run_id,
        "training_seed": int(resolved_config["training"]["seed"]),
        "branch_policy": str(current["branch_policy"]),
        "initial_lr": float(current["initial_lr"]),
        "weight_decay": float(current["weight_decay"]),
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_role": str(checkpoint_role),
        "oge_git_sha": str(oge_git_sha),
        "execution_only": bool(replication["execution_only"]),
        "initialization_sha256": initialization_sha256,
        "data_stream_sha256": data_stream_sha256,
        "sibling_group_id": str(replication["sibling_group_id"]),
        "cross_lr_pairing_block_id": str(replication["cross_lr_pairing_block_id"]),
        "branch_neutral_config_sha256": str(
            replication["branch_neutral_config_sha256"]
        ),
        "cross_lr_neutral_config_sha256": str(
            replication["cross_lr_neutral_config_sha256"]
        ),
        "sibling_role": sibling_role,
        "sibling_members": members,
        "model_config": copy.deepcopy(dict(resolved_config["model"])),
    }
    return validate_resnet18_replication_checkpoint_provenance(value)


def validate_resnet18_replication_checkpoint_provenance(
    value: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schema_version",
        "study_id",
        "run_id",
        "training_seed",
        "branch_policy",
        "initial_lr",
        "weight_decay",
        "checkpoint_epoch",
        "checkpoint_role",
        "oge_git_sha",
        "execution_only",
        "initialization_sha256",
        "data_stream_sha256",
        "sibling_group_id",
        "cross_lr_pairing_block_id",
        "branch_neutral_config_sha256",
        "cross_lr_neutral_config_sha256",
        "sibling_role",
        "sibling_members",
        "model_config",
    }
    if set(value) != required:
        raise ValueError("replication checkpoint provenance fields differ from v1")
    if value["schema_version"] != RESNET18_REPLICATION_CHECKPOINT_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported replication checkpoint provenance schema")
    if value["study_id"] != RESNET18_REPLICATION_STUDY_ID:
        raise ValueError("replication checkpoint study_id mismatch")
    if value["model_config"] != {
        "name": "resnet18",
        "variant": "cifar",
        "num_classes": 10,
    }:
        raise ValueError("replication checkpoint must identify native ResNet-18/CIFAR-10")
    if value["branch_policy"] not in {"adam_coupled", "adamw_decoupled"}:
        raise ValueError("replication checkpoint has an unsupported branch policy")
    if value["checkpoint_role"] not in {"last", "best_val", "snapshot"}:
        raise ValueError("replication checkpoint has an unsupported checkpoint role")
    if not isinstance(value["checkpoint_epoch"], int) or value["checkpoint_epoch"] < 0:
        raise ValueError("replication checkpoint epoch must be non-negative")
    if not re.fullmatch(r"[0-9a-f]{40}", str(value["oge_git_sha"])):
        raise ValueError("replication checkpoint oge_git_sha is invalid")
    for field in (
        "initialization_sha256",
        "data_stream_sha256",
        "branch_neutral_config_sha256",
        "cross_lr_neutral_config_sha256",
    ):
        digest = value[field]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"replication checkpoint {field} is invalid")
    role = value["sibling_role"]
    members = value["sibling_members"]
    if not isinstance(members, Mapping) or set(members) != {
        "adam_coupled",
        "adamw_decoupled",
    } or role not in members:
        raise ValueError("replication checkpoint sibling members are incomplete")
    current = members[role]
    for field in (
        "run_id",
        "training_seed",
        "branch_policy",
        "initial_lr",
        "weight_decay",
        "initialization_sha256",
        "data_stream_sha256",
    ):
        if value[field] != current[field]:
            raise ValueError(f"replication checkpoint {field} differs from sibling plan")
    return copy.deepcopy(dict(value))


def validate_resnet18_replication_checkpoint_payload(
    payload: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "checkpoint_type",
        "completed_epoch",
        "model_state",
        "oge_git_sha",
        "run_id",
        "resnet18_replication_provenance",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"replication checkpoint is missing fields: {sorted(missing)}")
    provenance = validate_resnet18_replication_checkpoint_provenance(
        payload["resnet18_replication_provenance"]
    )
    for checkpoint_key, provenance_key in {
        "checkpoint_type": "checkpoint_role",
        "completed_epoch": "checkpoint_epoch",
        "oge_git_sha": "oge_git_sha",
        "run_id": "run_id",
    }.items():
        if payload[checkpoint_key] != provenance[provenance_key]:
            raise ValueError(
                f"checkpoint {checkpoint_key} differs from replication provenance"
            )
    if not isinstance(payload["model_state"], Mapping):
        raise ValueError("replication checkpoint model_state must be a mapping")
    return provenance
