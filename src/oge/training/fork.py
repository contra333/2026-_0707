"""Shared-prefix optimizer-branch restoration without weakening resume rules."""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from oge.studies.hashing import canonical_sha256

from .checkpoint import CHECKPOINT_SCHEMA_VERSION, restore_rng_state


FORK_SCHEMA_VERSION = "shared_prefix_fork_v1"
_ADAM_FAMILY = {"adam", "adamw", "adam_coupled_decoupled"}
_SGD_FAMILY = {"sgd", "sgdw", "sgd_coupled_decoupled"}
_DECAY_FIELDS = {
    "name",
    "weight_decay",
    "total_weight_decay",
    "coupled_ratio",
}


def optimizer_family(name: Any) -> str:
    normalized = str(name).lower()
    if normalized in _ADAM_FAMILY:
        return "adam"
    if normalized in _SGD_FAMILY:
        return "sgd"
    raise ValueError(f"optimizer is outside the fork-compatible families: {name!r}")


def optimizer_parameter_names(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    output = []
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            if id(parameter) not in names:
                raise ValueError("optimizer contains a parameter outside the model")
            group_names.append(names[id(parameter)])
        output.append(group_names)
    return output


def _configured_decay(optimizer: Mapping[str, Any]) -> float:
    name = str(optimizer.get("name", "")).lower()
    key = (
        "total_weight_decay"
        if name in {"adam_coupled_decoupled", "sgd_coupled_decoupled"}
        else "weight_decay"
    )
    value = float(optimizer.get(key, 0.0))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("configured weight decay must be finite and non-negative")
    return value


def _first_difference(left: Any, right: Any, path: str = "") -> str:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left).union(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                return child
            difference = _first_difference(left[key], right[key], child)
            if difference:
                return difference
        return ""
    if left != right:
        return path or "<root>"
    return ""


def _fork_invariants(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.get("training", {}).pop("max_epochs", None)
    value.get("checkpoint", {}).pop("snapshot_epochs", None)
    value.get("runtime", {}).pop("device", None)
    optimizer = dict(value.get("optimizer", {}))
    family = optimizer_family(optimizer.get("name"))
    for field in _DECAY_FIELDS:
        optimizer.pop(field, None)
    optimizer["fork_family"] = family
    value["optimizer"] = optimizer
    return value


def validate_fork_configuration(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    completed_epoch: int,
) -> None:
    """Allow only decay-coupling changes within an optimizer family."""

    saved_optimizer = saved.get("optimizer")
    current_optimizer = current.get("optimizer")
    if not isinstance(saved_optimizer, Mapping) or not isinstance(
        current_optimizer, Mapping
    ):
        raise ValueError("fork configurations require optimizer mappings")
    if _configured_decay(saved_optimizer) != 0.0:
        raise ValueError("fork prefix must be an exact zero-decay checkpoint")
    saved_family = optimizer_family(saved_optimizer.get("name"))
    current_family = optimizer_family(current_optimizer.get("name"))
    if saved_family != current_family:
        raise ValueError("fork cannot cross optimizer families")
    difference = _first_difference(_fork_invariants(saved), _fork_invariants(current))
    if difference:
        raise ValueError(
            f"incompatible fork configuration: field {difference!r} changed; "
            "only decay endpoint/coupling and future run length may change"
        )
    current_max = int(current["training"]["max_epochs"])
    if current_max < completed_epoch:
        raise ValueError("fork max_epochs must not precede the prefix boundary")


def _copy_dynamic_optimizer_state(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    saved_names = payload["optimizer_parameter_names"]
    current_names = optimizer_parameter_names(model, optimizer)
    if saved_names != current_names:
        raise ValueError("fork optimizer parameter names/order differ from the prefix")
    saved_state = payload["optimizer_state"]
    if not isinstance(saved_state, Mapping):
        raise ValueError("prefix optimizer_state must be a mapping")
    saved_groups = saved_state.get("param_groups")
    saved_values = saved_state.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_values, Mapping):
        raise ValueError("prefix optimizer_state has an invalid structure")
    current_state = optimizer.state_dict()
    current_groups = current_state["param_groups"]
    if len(saved_groups) != len(current_groups) or len(saved_groups) != len(saved_names):
        raise ValueError("fork optimizer group count differs from the prefix")
    transferred: dict[Any, Any] = {}
    for names, saved_group, current_group in zip(
        saved_names, saved_groups, current_groups
    ):
        saved_ids = saved_group.get("params")
        current_ids = current_group.get("params")
        if not isinstance(saved_ids, list) or len(saved_ids) != len(names):
            raise ValueError("prefix optimizer parameter IDs do not match names")
        if not isinstance(current_ids, list) or len(current_ids) != len(names):
            raise ValueError("current optimizer parameter IDs do not match names")
        for saved_id, current_id in zip(saved_ids, current_ids):
            if saved_id in saved_values:
                transferred[current_id] = copy.deepcopy(saved_values[saved_id])
        # LR is schedule state at the fork boundary.  Decay/coupling and all
        # static branch hyperparameters remain those of the new optimizer.
        current_group["lr"] = float(saved_group["lr"])
        if "initial_lr" in saved_group:
            current_group["initial_lr"] = float(saved_group["initial_lr"])
    current_state["state"] = transferred
    optimizer.load_state_dict(current_state)


def _hash_value(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_value(digest, str(key))
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _hash_value(digest, item)
    elif isinstance(value, float):
        digest.update(b"float\0" + struct.pack("!d", value))
    elif isinstance(value, (str, int, bool)) or value is None:
        digest.update(type(value).__name__.encode() + b"\0" + repr(value).encode())
    else:
        raise TypeError(f"unsupported digest value: {type(value).__name__}")


def transferred_state_digest(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    rng_state: Mapping[str, Any],
) -> str:
    """Hash only state that must be identical across sibling branches."""

    state = optimizer.state_dict()
    names = optimizer_parameter_names(model, optimizer)
    by_name = {}
    for group_names, group in zip(names, state["param_groups"]):
        for name, parameter_id in zip(group_names, group["params"]):
            if parameter_id in state["state"]:
                by_name[name] = state["state"][parameter_id]
    payload = {
        "model_state": model.state_dict(),
        "optimizer_dynamic_state": by_name,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": rng_state,
    }
    digest = hashlib.sha256()
    _hash_value(digest, payload)
    return digest.hexdigest()


def restore_fork_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_generator: torch.Generator,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore a zero-decay prefix into a new decay-coupling branch."""

    required = {
        "schema_version",
        "checkpoint_type",
        "completed_epoch",
        "global_step",
        "model_state",
        "optimizer_state",
        "optimizer_parameter_names",
        "scheduler_state",
        "best_validation",
        "rng_state",
        "resolved_config",
        "run_id",
        "history",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"fork checkpoint is missing fields: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported fork checkpoint schema_version")
    if payload["checkpoint_type"] != "last":
        raise ValueError("fork requires an epoch-boundary last.pt checkpoint")
    completed_epoch = int(payload["completed_epoch"])
    validate_fork_configuration(
        payload["resolved_config"], resolved_config, completed_epoch=completed_epoch
    )
    model.load_state_dict(payload["model_state"], strict=True)
    saved_scheduler = payload["scheduler_state"]
    if scheduler is None and saved_scheduler is not None:
        raise ValueError("prefix has scheduler state but the fork branch does not")
    if scheduler is not None and saved_scheduler is None:
        raise ValueError("fork branch scheduler is missing prefix state")
    _copy_dynamic_optimizer_state(payload, model=model, optimizer=optimizer)
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)
    restore_rng_state(payload["rng_state"], train_generator)
    digest = transferred_state_digest(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rng_state=payload["rng_state"],
    )
    return {
        "schema_version": FORK_SCHEMA_VERSION,
        "source_run_id": str(payload["run_id"]),
        "source_config_sha256": canonical_sha256(payload["resolved_config"]),
        "source_training_seed": int(payload["resolved_config"]["training"]["seed"]),
        "completed_epoch": completed_epoch,
        "global_step": int(payload["global_step"]),
        "optimizer_family": optimizer_family(resolved_config["optimizer"]["name"]),
        "branch_optimizer": copy.deepcopy(resolved_config["optimizer"]),
        "branch_config_sha256": canonical_sha256(resolved_config),
        "branch_training_seed": int(resolved_config["training"]["seed"]),
        "transferred_state_digest": digest,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
