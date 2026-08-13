"""Passive same-initialization and data-stream witnesses for Task F training."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from oge.studies.hashing import canonical_sha256


PAIRED_PROVENANCE_SCHEMA_VERSION = "task_f_paired_control_provenance_v1"
_PRE_BATCH_IDENTITY_FIELDS = (
    "sibling_group_id",
    "training_seed",
    "data_stream_id",
    "initialization_sha256",
    "initial_python_rng_sha256",
    "initial_numpy_rng_sha256",
    "initial_torch_rng_sha256",
    "initial_dataloader_rng_sha256",
    "dataset_membership_sha256",
    "branch_neutral_config_sha256",
    "execution_only",
)
_FULL_SIBLING_IDENTITY_FIELDS = (
    *_PRE_BATCH_IDENTITY_FIELDS,
    "first_minibatch_ordered_sample_id_sha256",
    "first_minibatch_transformed_image_sha256",
)

_DATA_STREAM_DIGEST_FIELDS = (
    "data_stream_id",
    "initial_python_rng_sha256",
    "initial_numpy_rng_sha256",
    "initial_torch_rng_sha256",
    "initial_dataloader_rng_sha256",
    "dataset_membership_sha256",
    "branch_neutral_config_sha256",
    "first_minibatch_ordered_sample_id_sha256",
    "first_minibatch_transformed_image_sha256",
    "first_minibatch_size",
)


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, float):
        digest.update(b"float\0" + struct.pack("!d", value))
    elif isinstance(value, (str, int, bool)) or value is None:
        digest.update(type(value).__name__.encode("utf-8") + b"\0")
        digest.update(repr(value).encode("utf-8"))
    else:
        raise TypeError(f"unsupported provenance digest value: {type(value).__name__}")


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return state_sha256(value)


def model_initialization_sha256(model: nn.Module) -> str:
    return state_sha256(model.state_dict())


def initial_rng_state_hashes(state: Mapping[str, Any]) -> dict[str, str]:
    required = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "train_dataloader_generator",
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(f"initial RNG state is missing fields: {sorted(missing)}")
    return {
        "initial_python_rng_sha256": state_sha256(state["python"]),
        "initial_numpy_rng_sha256": state_sha256(state["numpy"]),
        "initial_torch_rng_sha256": state_sha256(
            {"cpu": state["torch_cpu"], "cuda": state["torch_cuda"]}
        ),
        "initial_dataloader_rng_sha256": state_sha256(
            state["train_dataloader_generator"]
        ),
    }


def dataset_membership_sha256(resolved_config: Mapping[str, Any]) -> str:
    dataset = resolved_config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("resolved config is missing dataset provenance")
    membership = dataset.get("membership")
    manifest = dataset.get("membership_manifest")
    if not isinstance(membership, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("Task F requires resolved dataset membership and manifest")
    return canonical_sha256({"membership": membership, "membership_manifest": manifest})


def create_initial_paired_provenance(
    *,
    resolved_config: Mapping[str, Any],
    model: nn.Module,
    initial_rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    task_f = resolved_config.get("task_f")
    if not isinstance(task_f, Mapping):
        raise ValueError("Task F paired provenance requires a task_f config")
    neutral = task_f.get("branch_neutral_config")
    neutral_hash = task_f.get("branch_neutral_config_sha256")
    if not isinstance(neutral, Mapping) or canonical_sha256(neutral) != neutral_hash:
        raise ValueError("Task F branch-neutral config/hash mismatch")
    return {
        "schema_version": PAIRED_PROVENANCE_SCHEMA_VERSION,
        "run_plan_id": str(task_f["run_id"]),
        "sibling_group_id": str(task_f["sibling_group_id"]),
        "training_seed": int(resolved_config["training"]["seed"]),
        "data_stream_id": str(task_f["data_stream_id"]),
        "branch_policy": str(task_f["branch_policy"]),
        "initialization_sha256": model_initialization_sha256(model),
        **initial_rng_state_hashes(initial_rng_state),
        "dataset_membership_sha256": dataset_membership_sha256(resolved_config),
        "branch_neutral_config_sha256": str(neutral_hash),
        "first_minibatch_ordered_sample_id_sha256": None,
        "first_minibatch_transformed_image_sha256": None,
        "first_minibatch_size": None,
        "first_minibatch_witness_status": "pending",
        "execution_only": bool(task_f["execution_only"]),
    }


def observe_first_minibatch(
    provenance: dict[str, Any], batch: Mapping[str, Any]
) -> None:
    """Record witnesses from the already-yielded first batch without re-iteration."""

    if provenance.get("first_minibatch_witness_status") != "pending":
        raise ValueError("first-minibatch provenance may be observed exactly once")
    sample_ids = batch.get("sample_id")
    images = batch.get("image")
    if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
        raise ValueError("Task F first batch must contain ordered sample_id values")
    if not all(isinstance(sample_id, str) and sample_id for sample_id in sample_ids):
        raise ValueError("Task F first-batch sample IDs must be non-empty strings")
    if not isinstance(images, torch.Tensor):
        raise ValueError("Task F first batch must contain a transformed image tensor")
    if int(images.shape[0]) != len(sample_ids):
        raise ValueError("Task F first-batch sample IDs and image tensor length differ")
    provenance["first_minibatch_ordered_sample_id_sha256"] = canonical_sha256(
        list(sample_ids)
    )
    provenance["first_minibatch_transformed_image_sha256"] = tensor_sha256(images)
    provenance["first_minibatch_size"] = len(sample_ids)
    provenance["first_minibatch_witness_status"] = "observed"


def validate_resume_paired_identity(
    saved: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    if saved.get("schema_version") != PAIRED_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("resume checkpoint has unsupported paired provenance schema")
    if saved.get("first_minibatch_witness_status") != "observed":
        raise ValueError("resume checkpoint is missing the first-minibatch witness")
    for field in _PRE_BATCH_IDENTITY_FIELDS:
        if saved.get(field) != current.get(field):
            raise ValueError(f"resume paired-control identity changed: {field}")
    if saved.get("branch_policy") != current.get("branch_policy"):
        raise ValueError("resume paired-control identity changed: branch_policy")


def validate_sibling_provenance(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) < 2:
        raise ValueError("sibling provenance validation requires at least two records")
    reference = records[0]
    if reference.get("first_minibatch_witness_status") != "observed":
        raise ValueError("sibling provenance requires observed first-minibatch witnesses")
    run_plan_ids = set()
    for record in records:
        if record.get("schema_version") != PAIRED_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("unsupported sibling provenance schema")
        if record.get("first_minibatch_witness_status") != "observed":
            raise ValueError("sibling provenance requires observed first-minibatch witnesses")
        for field in _FULL_SIBLING_IDENTITY_FIELDS:
            if record.get(field) != reference.get(field):
                raise ValueError(f"sibling paired-control identity mismatch: {field}")
        if not str(record.get("branch_policy", "")):
            raise ValueError("sibling branch policies must be non-empty")
        run_plan_id = str(record.get("run_plan_id", ""))
        if not run_plan_id or run_plan_id in run_plan_ids:
            raise ValueError("sibling run-plan identities must be non-empty and unique")
        run_plan_ids.add(run_plan_id)


def paired_data_stream_sha256(provenance: Mapping[str, Any]) -> str:
    """Hash the complete observed, branch-neutral Task F data-stream witness."""

    if provenance.get("first_minibatch_witness_status") != "observed":
        raise ValueError("Task F data-stream identity requires an observed first batch")
    missing = [field for field in _DATA_STREAM_DIGEST_FIELDS if provenance.get(field) is None]
    if missing:
        raise ValueError(f"Task F data-stream identity is missing fields: {missing}")
    return canonical_sha256(
        {field: provenance[field] for field in _DATA_STREAM_DIGEST_FIELDS}
    )


def build_task_f_checkpoint_provenance(
    *,
    resolved_config: Mapping[str, Any],
    paired_control_provenance: Mapping[str, Any],
    checkpoint_epoch: int,
    checkpoint_role: str,
    oge_git_sha: str,
    run_id: str,
) -> dict[str, Any] | None:
    """Build the exact Task F-B checkpoint interface for anchor/pilot exports."""

    task_f = resolved_config.get("task_f")
    if not isinstance(task_f, Mapping):
        return None
    artifact_export = task_f.get("artifact_export")
    if artifact_export is None:
        return None
    if not isinstance(artifact_export, Mapping):
        raise ValueError("Task F artifact_export must be a mapping")
    if paired_control_provenance.get("first_minibatch_witness_status") != "observed":
        if checkpoint_role == "snapshot" and checkpoint_epoch == 0:
            return None
        raise ValueError("Task F-B checkpoint provenance requires an observed first batch")
    if run_id != task_f.get("run_id") or run_id != paired_control_provenance.get(
        "run_plan_id"
    ):
        raise ValueError("Task F checkpoint run_id differs from the frozen run plan")
    initialization_sha256 = str(
        paired_control_provenance["initialization_sha256"]
    )
    data_stream_sha256 = paired_data_stream_sha256(paired_control_provenance)
    members = {}
    planned_members = artifact_export.get("sibling_members")
    if not isinstance(planned_members, Mapping):
        raise ValueError("Task F-B sibling plan must be a mapping")
    for role, member in planned_members.items():
        if not isinstance(member, Mapping):
            raise ValueError("Task F-B sibling member must be a mapping")
        members[str(role)] = {
            **dict(member),
            "initialization_sha256": initialization_sha256,
            "data_stream_sha256": data_stream_sha256,
        }
    sibling_role = str(artifact_export["sibling_role"])
    if sibling_role not in members:
        raise ValueError("Task F-B current sibling is absent from the sibling plan")
    current = members[sibling_role]
    model = resolved_config["model"]
    provenance = {
        "schema_version": "task_f_checkpoint_provenance_v1",
        "run_id": run_id,
        "training_seed": int(resolved_config["training"]["seed"]),
        "branch_policy": current["branch_policy"],
        "total_weight_decay": current["total_weight_decay"],
        "coupled_ratio": current["coupled_ratio"],
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_role": checkpoint_role,
        "oge_git_sha": oge_git_sha,
        "execution_only": bool(task_f["execution_only"]),
        "initialization_sha256": initialization_sha256,
        "data_stream_sha256": data_stream_sha256,
        "sibling_group_id": str(task_f["sibling_group_id"]),
        "sibling_role": sibling_role,
        "sibling_members": members,
        "model_config": {
            "name": model["name"],
            "num_classes": int(model.get("num_classes", 10)),
            "depth": int(model.get("depth", 28)),
            "widen_factor": int(model.get("widen_factor", 10)),
            "dropout_rate": float(model["dropout_rate"]),
            "init_policy": model.get("init_policy", "msr_fan_in"),
        },
    }
    # The B validator is the authority for this cross-workstream interface.
    from oge.feature_export import validate_task_f_checkpoint_provenance

    return validate_task_f_checkpoint_provenance(provenance)
