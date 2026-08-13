"""Card 13 v12 Task F ID-only multi-depth feature artifacts.

This module intentionally does not reuse or reinterpret the Metric Contract
v1.2 raw-feature artifact.  Task F has its own strict provenance, identity,
and serialization contract.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import platform
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np
import torch

from oge.models import WideResNet, make_model
from oge.models.wide_resnet import (
    WRN_FEATURE_TAP_CONTRACT_VERSION,
    WRN_FEATURE_TAP_NAMES,
)
from oge.studies.artifacts import sha256_file
from oge.studies.hashing import canonical_json_bytes, canonical_sha256
from oge.training import load_torch_artifact


TASK_F_ARTIFACT_SCHEMA_VERSION: Final = "task_f_id_feature_artifact_v3"
TASK_F_CHECKPOINT_PROVENANCE_SCHEMA_VERSION: Final = (
    "task_f_checkpoint_provenance_v2"
)
TASK_F_SIBLING_SCHEMA_VERSION: Final = "task_f_paired_sibling_identity_v2"
TASK_F_SPECIFICATION_VERSION: Final = "task_f_id_feature_export_specification_v3"
TASK_F_OUTPUT_IDENTITY_VERSION: Final = "task_f_feature_output_identity_v3"
TASK_F_ID_SPLITS: Final = (
    "id_train",
    "id_validation",
    "id_probe",
    "synthetic_id_fixture",
)
TASK_F_ALPHA_SIBLING_ROLES: Final = ("zero", "alpha_0", "alpha_0_5", "alpha_1")
TASK_F_GENERIC_BRANCH_POLICIES: Final = ("adam", "adamw", "sgd", "sgdw")
TASK_F_FEATURE_DTYPE: Final = "float32"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_PROTECTED_REFERENCE_PATTERN = re.compile(
    r"(?:^|[^a-z0-9])(?:ood|protected|id[_-]?test|near[_-]?ood|far[_-]?ood|"
    r"cifar[_-]?100|svhn|mnist|textures?|places[_-]?365|tiny[_-]?imagenet|"
    r"openood|tin)(?:$|[^a-z0-9])",
    flags=re.IGNORECASE,
)
_THREAD_ENVIRONMENT_KEYS: Final = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_RUNTIME_KEYS: Final = {
    "python_version",
    "python_implementation",
    "numpy_version",
    "pytorch_version",
    "device_type",
    "device",
    "platform_system",
    "platform_machine",
    "cpu_count",
    "torch_num_threads",
    "torch_num_interop_threads",
    "thread_environment",
    "numpy_blas",
    "accelerator",
}
_ACCELERATOR_KEYS: Final = {
    "backend",
    "local_device_index",
    "device_name",
    "device_uuid",
    "total_memory_bytes",
    "cuda_runtime_version",
    "cudnn_version",
    "cuda_visible_devices",
}

_CHECKPOINT_PROVENANCE_KEYS: Final = {
    "schema_version",
    "run_id",
    "training_seed",
    "branch_policy",
    "total_weight_decay",
    "coupled_ratio",
    "checkpoint_epoch",
    "checkpoint_role",
    "oge_git_sha",
    "execution_only",
    "initialization_sha256",
    "data_stream_sha256",
    "sibling_group_id",
    "sibling_role",
    "sibling_members",
    "model_config",
}
_SIBLING_MEMBER_KEYS: Final = {
    "run_id",
    "training_seed",
    "branch_policy",
    "total_weight_decay",
    "coupled_ratio",
    "initialization_sha256",
    "data_stream_sha256",
}
_MODEL_CONFIG_KEYS: Final = {
    "name",
    "num_classes",
    "depth",
    "widen_factor",
    "dropout_rate",
    "init_policy",
}
_MANIFEST_KEYS: Final = {
    "schema_version",
    "artifact_role",
    "tap_contract_version",
    "tap_names",
    "run_id",
    "training_seed",
    "branch_policy",
    "total_weight_decay",
    "coupled_ratio",
    "checkpoint_epoch",
    "checkpoint_role",
    "checkpoint_sha256",
    "depth_tap",
    "dataset_split",
    "ordered_sample_id_sha256",
    "feature_shape",
    "feature_dtype",
    "oge_git_sha",
    "specification_sha256",
    "execution_only",
    "initialization_sha256",
    "data_stream_sha256",
    "model_config",
    "sibling_identity",
    "serialization",
    "output_identity_sha256",
    "runtime",
}
_ARTIFACT_IDENTITY_KEYS: Final = (
    "run_id",
    "training_seed",
    "branch_policy",
    "total_weight_decay",
    "coupled_ratio",
    "checkpoint_epoch",
    "checkpoint_role",
    "checkpoint_sha256",
    "depth_tap",
    "dataset_split",
    "ordered_sample_id_sha256",
    "feature_shape",
    "feature_dtype",
    "oge_git_sha",
    "execution_only",
    "initialization_sha256",
    "data_stream_sha256",
    "model_config",
)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected.difference(value)
    unexpected = set(value).difference(expected)
    if missing or unexpected:
        pieces = []
        if missing:
            pieces.append(f"missing fields: {sorted(missing)}")
        if unexpected:
            pieces.append(f"unexpected fields: {sorted(unexpected)}")
        raise ValueError(f"{label} has " + "; ".join(pieces))


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_git_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40-character Git SHA")
    return value


def _reject_protected_reference(value: str | Path, label: str) -> None:
    text = str(value).replace("\\", "/")
    if _PROTECTED_REFERENCE_PATTERN.search(text):
        raise ValueError(f"{label} contains a protected-OOD or protected-ID reference")


def _validate_dataset_split(dataset_split: Any, *, execution_only: bool) -> str:
    if dataset_split not in TASK_F_ID_SPLITS:
        raise ValueError(
            f"dataset_split must be one of the Task F ID-only splits: {TASK_F_ID_SPLITS}"
        )
    _reject_protected_reference(str(dataset_split), "dataset_split")
    if dataset_split == "synthetic_id_fixture" and not execution_only:
        raise ValueError("synthetic_id_fixture requires execution_only=true")
    return str(dataset_split)


def _require_export_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type == "cpu":
        if target.index is not None:
            raise ValueError("Task F exporter CPU device must be exactly 'cpu'")
        return target
    if target.type != "cuda":
        raise ValueError("Task F exporter device must be 'cpu' or explicit 'cuda:<index>'")
    if target.index is None:
        raise ValueError("Task F exporter CUDA device must include an explicit local index")
    if not torch.cuda.is_available():
        raise RuntimeError("Task F exporter requested CUDA but torch.cuda.is_available() is false")
    device_count = torch.cuda.device_count()
    if target.index < 0 or target.index >= device_count:
        raise ValueError(
            f"Task F exporter CUDA index {target.index} is outside the visible device count "
            f"{device_count}"
        )
    return target


def _validate_role_semantics(role: str, member: Mapping[str, Any]) -> None:
    total = _require_nonnegative_number(
        member["total_weight_decay"], f"sibling_members.{role}.total_weight_decay"
    )
    ratio = member["coupled_ratio"]
    policy = member["branch_policy"]
    if role == "zero":
        if policy != "zero_decay" or total != 0.0 or ratio is not None:
            raise ValueError(
                "zero sibling requires branch_policy='zero_decay', "
                "total_weight_decay=0, and coupled_ratio=null"
            )
        return
    if role in {"alpha_0", "alpha_0_5", "alpha_1"}:
        expected_ratio = {"alpha_0": 0.0, "alpha_0_5": 0.5, "alpha_1": 1.0}[role]
        if policy != "adam_coupled_decoupled" or total <= 0.0:
            raise ValueError(
                f"{role} requires branch_policy='adam_coupled_decoupled' and positive "
                "total_weight_decay"
            )
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError(f"{role} coupled_ratio must be {expected_ratio}")
        if not math.isfinite(float(ratio)) or float(ratio) != expected_ratio:
            raise ValueError(f"{role} coupled_ratio must be {expected_ratio}")
        return
    if policy not in TASK_F_GENERIC_BRANCH_POLICIES or total <= 0.0 or ratio is not None:
        raise ValueError(
            f"generic sibling {role!r} requires branch_policy in "
            f"{TASK_F_GENERIC_BRANCH_POLICIES}, positive total_weight_decay, "
            "and coupled_ratio=null"
        )


def _validated_sibling_members(
    value: Any,
    *,
    training_seed: int,
    initialization_sha256: str,
    data_stream_sha256: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("sibling_members must be a mapping")
    if len(value) < 2 or "zero" not in value:
        raise ValueError("sibling_members must contain zero and at least one comparison")
    roles = tuple(sorted(value))
    if any(not isinstance(role, str) or not role for role in roles):
        raise ValueError("sibling role keys must be non-empty strings")
    for role in roles:
        _reject_protected_reference(role, f"sibling role {role!r}")
    alpha_present = set(roles).intersection(TASK_F_ALPHA_SIBLING_ROLES)
    if alpha_present.difference({"zero"}) and not set(TASK_F_ALPHA_SIBLING_ROLES).issubset(
        roles
    ):
        raise ValueError("any alpha sibling requires the complete zero/alpha quartet")
    normalized: dict[str, dict[str, Any]] = {}
    run_ids: set[str] = set()
    alpha_total: float | None = None
    for role in roles:
        member = value[role]
        if not isinstance(member, Mapping):
            raise ValueError(f"sibling_members.{role} must be a mapping")
        _require_exact_keys(member, _SIBLING_MEMBER_KEYS, f"sibling_members.{role}")
        run_id = _require_nonempty_string(
            member["run_id"], f"sibling_members.{role}.run_id"
        )
        _reject_protected_reference(run_id, f"sibling_members.{role}.run_id")
        if run_id in run_ids:
            raise ValueError("sibling run_id values must be unique")
        run_ids.add(run_id)
        if _require_integer(
            member["training_seed"], f"sibling_members.{role}.training_seed"
        ) != training_seed:
            raise ValueError("all siblings must share training_seed")
        if _require_sha256(
            member["initialization_sha256"],
            f"sibling_members.{role}.initialization_sha256",
        ) != initialization_sha256:
            raise ValueError("all siblings must share initialization_sha256")
        if _require_sha256(
            member["data_stream_sha256"],
            f"sibling_members.{role}.data_stream_sha256",
        ) != data_stream_sha256:
            raise ValueError("all siblings must share data_stream_sha256")
        _validate_role_semantics(role, member)
        if role in {"alpha_0", "alpha_0_5", "alpha_1"}:
            member_total = float(member["total_weight_decay"])
            if alpha_total is None:
                alpha_total = member_total
            elif member_total != alpha_total:
                raise ValueError("alpha siblings must share total_weight_decay")
        normalized[role] = json.loads(canonical_json_bytes(dict(member)))
    return normalized


def _sibling_identity_payload(
    group_id: str, members: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": TASK_F_SIBLING_SCHEMA_VERSION,
        "group_id": group_id,
        "members": dict(members),
    }


def _validated_model_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("task_f_provenance.model_config must be a mapping")
    _require_exact_keys(value, _MODEL_CONFIG_KEYS, "task_f_provenance.model_config")
    if (
        value["name"] != "wrn28_10"
        or value["depth"] != 28
        or value["widen_factor"] != 10
    ):
        raise ValueError("Task F model_config must explicitly identify WRN-28-10")
    _require_integer(value["num_classes"], "model_config.num_classes", minimum=1)
    if value["init_policy"] != "msr_fan_in":
        raise ValueError("Task F WRN-28-10 init_policy must be msr_fan_in")
    dropout_rate = value["dropout_rate"]
    if (
        isinstance(dropout_rate, bool)
        or not isinstance(dropout_rate, (int, float))
        or not math.isfinite(float(dropout_rate))
        or not 0.0 <= float(dropout_rate) < 1.0
    ):
        raise ValueError("Task F WRN-28-10 dropout_rate must be finite and in [0, 1)")
    return json.loads(canonical_json_bytes(dict(value)))


def validate_task_f_checkpoint_provenance(value: Any) -> dict[str, Any]:
    """Validate the exact provenance Task F-C must place in every checkpoint."""

    if not isinstance(value, Mapping):
        raise ValueError("checkpoint task_f_provenance must be a mapping")
    _require_exact_keys(value, _CHECKPOINT_PROVENANCE_KEYS, "task_f_provenance")
    if value["schema_version"] != TASK_F_CHECKPOINT_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported task_f_provenance schema_version")
    run_id = _require_nonempty_string(value["run_id"], "task_f_provenance.run_id")
    _reject_protected_reference(run_id, "task_f_provenance.run_id")
    training_seed = _require_integer(
        value["training_seed"], "task_f_provenance.training_seed"
    )
    checkpoint_epoch = _require_integer(
        value["checkpoint_epoch"], "task_f_provenance.checkpoint_epoch"
    )
    if value["checkpoint_role"] not in {"last", "best_val", "snapshot"}:
        raise ValueError("checkpoint_role must be last, best_val, or snapshot")
    _require_git_sha(value["oge_git_sha"], "task_f_provenance.oge_git_sha")
    if not isinstance(value["execution_only"], bool):
        raise ValueError("task_f_provenance.execution_only must be boolean")
    initialization_sha256 = _require_sha256(
        value["initialization_sha256"], "task_f_provenance.initialization_sha256"
    )
    data_stream_sha256 = _require_sha256(
        value["data_stream_sha256"], "task_f_provenance.data_stream_sha256"
    )
    sibling_group_id = _require_nonempty_string(
        value["sibling_group_id"], "task_f_provenance.sibling_group_id"
    )
    _reject_protected_reference(sibling_group_id, "task_f_provenance.sibling_group_id")
    sibling_role = value["sibling_role"]
    members = _validated_sibling_members(
        value["sibling_members"],
        training_seed=training_seed,
        initialization_sha256=initialization_sha256,
        data_stream_sha256=data_stream_sha256,
    )
    if sibling_role not in members:
        raise ValueError("sibling_role must identify one declared sibling member")
    current = members[str(sibling_role)]
    for key in (
        "run_id",
        "training_seed",
        "branch_policy",
        "total_weight_decay",
        "coupled_ratio",
        "initialization_sha256",
        "data_stream_sha256",
    ):
        if value[key] != current[key]:
            raise ValueError(f"task_f_provenance.{key} differs from current sibling")
    _validated_model_config(value["model_config"])
    normalized = json.loads(canonical_json_bytes(dict(value)))
    normalized["checkpoint_epoch"] = checkpoint_epoch
    return normalized


def validate_task_f_checkpoint_payload(payload: Any) -> dict[str, Any]:
    """Require Task F provenance without inferring it from legacy fields."""

    if not isinstance(payload, Mapping):
        raise ValueError("Task F checkpoint must contain a mapping")
    required = {
        "checkpoint_type",
        "completed_epoch",
        "model_state",
        "oge_git_sha",
        "run_id",
        "task_f_provenance",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Task F checkpoint is missing fields: {sorted(missing)}")
    provenance = validate_task_f_checkpoint_provenance(payload["task_f_provenance"])
    _require_integer(payload["completed_epoch"], "checkpoint completed_epoch")
    comparisons = {
        "checkpoint_type": "checkpoint_role",
        "completed_epoch": "checkpoint_epoch",
        "oge_git_sha": "oge_git_sha",
        "run_id": "run_id",
    }
    for checkpoint_key, provenance_key in comparisons.items():
        if payload[checkpoint_key] != provenance[provenance_key]:
            raise ValueError(
                f"checkpoint {checkpoint_key} differs from Task F provenance"
            )
    if not isinstance(payload["model_state"], Mapping):
        raise ValueError("Task F checkpoint model_state must be a mapping")
    return provenance


def ordered_sample_id_sha256(sample_ids: Sequence[str] | np.ndarray) -> str:
    digest = hashlib.sha256()
    observed: set[str] = set()
    for index, sample_id in enumerate(sample_ids):
        value = str(sample_id)
        if not value or "\0" in value:
            raise ValueError(f"sample_ids[{index}] must be non-empty and contain no NUL")
        _reject_protected_reference(value, f"sample_ids[{index}]")
        if value in observed:
            raise ValueError("sample_ids must be unique and ordered deterministically")
        observed.add(value)
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    if not observed:
        raise ValueError("sample_ids must not be empty")
    return digest.hexdigest()


def _numpy_blas_identity() -> dict[str, Any]:
    config = getattr(np.__config__, "CONFIG", {})
    if not isinstance(config, Mapping):
        return {"available": False}
    dependencies = config.get("Build Dependencies", {})
    if not isinstance(dependencies, Mapping):
        return {"available": False}
    result: dict[str, Any] = {"available": True}
    for library in ("blas", "lapack"):
        details = dependencies.get(library, {})
        if not isinstance(details, Mapping):
            result[library] = None
            continue
        result[library] = {
            key: details.get(key)
            for key in ("name", "found", "version", "openblas configuration")
        }
    return result


def collect_runtime_provenance(device: str | torch.device) -> dict[str, Any]:
    target = _require_export_device(device)
    try:
        interop_threads: int | None = torch.get_num_interop_threads()
    except RuntimeError:
        interop_threads = None
    accelerator: dict[str, Any] = {
        "backend": target.type,
        "local_device_index": None,
        "device_name": None,
        "device_uuid": None,
        "total_memory_bytes": None,
        "cuda_runtime_version": None,
        "cudnn_version": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if target.type == "cuda":
        properties = torch.cuda.get_device_properties(target.index)
        device_uuid = getattr(properties, "uuid", None)
        cudnn_version = torch.backends.cudnn.version()
        accelerator.update(
            {
                "local_device_index": target.index,
                "device_name": str(properties.name),
                "device_uuid": None if device_uuid is None else str(device_uuid),
                "total_memory_bytes": int(properties.total_memory),
                "cuda_runtime_version": (
                    None if torch.version.cuda is None else str(torch.version.cuda)
                ),
                "cudnn_version": None if cudnn_version is None else int(cudnn_version),
            }
        )
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": str(np.__version__),
        "pytorch_version": str(torch.__version__),
        "device_type": target.type,
        "device": str(target),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": interop_threads,
        "thread_environment": {
            key: os.environ.get(key) for key in _THREAD_ENVIRONMENT_KEYS
        },
        "numpy_blas": _numpy_blas_identity(),
        "accelerator": accelerator,
    }


def _validate_runtime(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("runtime must be a mapping")
    _require_exact_keys(value, _RUNTIME_KEYS, "runtime")
    for key in (
        "python_version",
        "python_implementation",
        "numpy_version",
        "pytorch_version",
        "device_type",
        "device",
        "platform_system",
        "platform_machine",
    ):
        _require_nonempty_string(value[key], f"runtime.{key}")
    for key in ("torch_num_threads",):
        _require_integer(value[key], f"runtime.{key}", minimum=1)
    if value["cpu_count"] is not None:
        _require_integer(value["cpu_count"], "runtime.cpu_count", minimum=1)
    if value["torch_num_interop_threads"] is not None:
        _require_integer(
            value["torch_num_interop_threads"],
            "runtime.torch_num_interop_threads",
            minimum=1,
        )
    thread_environment = value["thread_environment"]
    if not isinstance(thread_environment, Mapping):
        raise ValueError("runtime.thread_environment must be a mapping")
    _require_exact_keys(
        thread_environment,
        set(_THREAD_ENVIRONMENT_KEYS),
        "runtime.thread_environment",
    )
    if any(item is not None and not isinstance(item, str) for item in thread_environment.values()):
        raise ValueError("runtime thread environment values must be strings or null")
    if not isinstance(value["numpy_blas"], Mapping):
        raise ValueError("runtime.numpy_blas must be a mapping")
    accelerator = value["accelerator"]
    if not isinstance(accelerator, Mapping):
        raise ValueError("runtime.accelerator must be a mapping")
    _require_exact_keys(accelerator, _ACCELERATOR_KEYS, "runtime.accelerator")
    if accelerator["backend"] != value["device_type"]:
        raise ValueError("runtime accelerator backend must match device_type")
    visible_devices = accelerator["cuda_visible_devices"]
    if visible_devices is not None and not isinstance(visible_devices, str):
        raise ValueError("runtime.accelerator.cuda_visible_devices must be a string or null")
    if value["device_type"] == "cpu":
        if value["device"] != "cpu":
            raise ValueError("CPU runtime device must be exactly 'cpu'")
        cpu_null_fields = (
            "local_device_index",
            "device_name",
            "device_uuid",
            "total_memory_bytes",
            "cuda_runtime_version",
            "cudnn_version",
        )
        if any(accelerator[key] is not None for key in cpu_null_fields):
            raise ValueError("CPU runtime must not declare CUDA accelerator identity")
    elif value["device_type"] == "cuda":
        index = _require_integer(
            accelerator["local_device_index"],
            "runtime.accelerator.local_device_index",
        )
        if value["device"] != f"cuda:{index}":
            raise ValueError("CUDA runtime device must match its explicit local index")
        _require_nonempty_string(
            accelerator["device_name"], "runtime.accelerator.device_name"
        )
        _require_integer(
            accelerator["total_memory_bytes"],
            "runtime.accelerator.total_memory_bytes",
            minimum=1,
        )
        for key in ("device_uuid", "cuda_runtime_version"):
            item = accelerator[key]
            if item is not None:
                _require_nonempty_string(item, f"runtime.accelerator.{key}")
        if accelerator["cudnn_version"] is not None:
            _require_integer(
                accelerator["cudnn_version"],
                "runtime.accelerator.cudnn_version",
                minimum=1,
            )
    else:
        raise ValueError("runtime.device_type must be cpu or cuda")
    return json.loads(canonical_json_bytes(dict(value)))


def specification_payload() -> dict[str, Any]:
    """Return the static canonical contract payload frozen before Task F runs."""

    return {
        "specification_version": TASK_F_SPECIFICATION_VERSION,
        "artifact_schema_version": TASK_F_ARTIFACT_SCHEMA_VERSION,
        "checkpoint_provenance_schema_version": (
            TASK_F_CHECKPOINT_PROVENANCE_SCHEMA_VERSION
        ),
        "artifact_role": "task_f_id_feature_export",
        "tap_contract": {
            "version": WRN_FEATURE_TAP_CONTRACT_VERSION,
            "names": list(WRN_FEATURE_TAP_NAMES),
            "wrn28_10_widths": {
                "stage1": 160,
                "stage2": 320,
                "stage3": 640,
                "penultimate": 640,
            },
            "non_epoch_200_rule": "penultimate_only",
        },
        "manifest_fields": sorted(_MANIFEST_KEYS),
        "card13_minimum_fields": sorted(
            {
                "run_id",
                "training_seed",
                "branch_policy",
                "total_weight_decay",
                "coupled_ratio",
                "checkpoint_epoch",
                "checkpoint_sha256",
                "depth_tap",
                "dataset_split",
                "ordered_sample_id_sha256",
                "feature_shape",
                "feature_dtype",
                "oge_git_sha",
                "specification_sha256",
                "execution_only",
            }
        ),
        "checkpoint_provenance_fields": sorted(_CHECKPOINT_PROVENANCE_KEYS),
        "sibling_contract": {
            "schema_version": TASK_F_SIBLING_SCHEMA_VERSION,
            "required_zero_member": True,
            "minimum_member_count": 2,
            "member_fields": sorted(_SIBLING_MEMBER_KEYS),
            "zero": {
                "branch_policy": "zero_decay",
                "total_weight_decay": 0.0,
                "coupled_ratio": None,
            },
            "alpha": {
                "branch_policy": "adam_coupled_decoupled",
                "roles_to_coupled_ratio": {
                    "alpha_0": 0.0,
                    "alpha_0_5": 0.5,
                    "alpha_1": 1.0,
                },
                "requires_shared_positive_total_weight_decay": True,
                "roles": list(TASK_F_ALPHA_SIBLING_ROLES),
                "complete_quartet_required_if_any_alpha_role_is_present": True,
            },
            "generic_paired_roles": {
                "role_names_are_stable_manifest_keys": True,
                "allowed_branch_policies": list(TASK_F_GENERIC_BRANCH_POLICIES),
                "requires_positive_total_weight_decay": True,
                "coupled_ratio": None,
            },
            "requires_shared_training_seed": True,
            "requires_shared_initialization_sha256": True,
            "requires_shared_data_stream_sha256": True,
        },
        "id_only_contract": {
            "allowed_dataset_splits": list(TASK_F_ID_SPLITS),
            "synthetic_id_fixture_requires_execution_only": True,
            "input_npz_fields": ["images", "is_id", "sample_ids"],
            "all_is_id_values_must_be_true": True,
            "protected_reference_pattern": _PROTECTED_REFERENCE_PATTERN.pattern,
        },
        "serialization_contract": {
            "features_file": "features.npy",
            "sample_ids_file": "sample_ids.npy",
            "feature_dtype": TASK_F_FEATURE_DTYPE,
            "sample_id_encoding": "numpy-unicode-no-pickle",
            "ordered_sample_id_digest": "sha256-null-terminated-utf8-v1",
            "manifest_json": "canonical-json-sorted-utf8-no-nan-v1",
            "checksums_file": "checksums.sha256",
            "publication": "same-filesystem-temporary-directory-then-rename-no-overwrite",
        },
        "runtime_contract": {
            "fields": sorted(_RUNTIME_KEYS),
            "allowed_device_types": ["cpu", "cuda"],
            "cuda_requires_explicit_local_index": True,
            "accelerator_fields": sorted(_ACCELERATOR_KEYS),
        },
        "output_identity": {
            "schema_version": TASK_F_OUTPUT_IDENTITY_VERSION,
            "includes_artifact_identity_fields": list(_ARTIFACT_IDENTITY_KEYS),
            "includes_sibling_identity": True,
            "includes_runtime": True,
            "includes_feature_and_sample_id_file_sha256_and_bytes": True,
        },
    }


def _output_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    serialization = manifest["serialization"]
    return {
        "schema_version": TASK_F_OUTPUT_IDENTITY_VERSION,
        "specification_sha256": manifest["specification_sha256"],
        "artifact_identity": {
            key: manifest[key] for key in _ARTIFACT_IDENTITY_KEYS
        },
        "sibling_identity": manifest["sibling_identity"],
        "runtime": manifest["runtime"],
        "files": [
            {
                "path": "features.npy",
                "sha256": serialization["features_sha256"],
                "bytes": serialization["features_bytes"],
            },
            {
                "path": "sample_ids.npy",
                "sha256": serialization["sample_ids_sha256"],
                "bytes": serialization["sample_ids_bytes"],
            },
        ],
    }


def _validate_sibling_identity(
    value: Any,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("sibling_identity must be a mapping")
    _require_exact_keys(
        value,
        {"schema_version", "group_id", "role", "members", "identity_sha256"},
        "sibling_identity",
    )
    if value["schema_version"] != TASK_F_SIBLING_SCHEMA_VERSION:
        raise ValueError("unsupported sibling_identity schema_version")
    group_id = _require_nonempty_string(value["group_id"], "sibling_identity.group_id")
    role = value["role"]
    members = _validated_sibling_members(
        value["members"],
        training_seed=manifest["training_seed"],
        initialization_sha256=manifest["initialization_sha256"],
        data_stream_sha256=manifest["data_stream_sha256"],
    )
    if role not in members:
        raise ValueError("sibling_identity.role must identify one declared member")
    current = members[str(role)]
    for key in (
        "run_id",
        "training_seed",
        "branch_policy",
        "total_weight_decay",
        "coupled_ratio",
        "initialization_sha256",
        "data_stream_sha256",
    ):
        if manifest[key] != current[key]:
            raise ValueError(f"manifest {key} differs from current sibling")
    expected_identity = canonical_sha256(_sibling_identity_payload(group_id, members))
    if value["identity_sha256"] != expected_identity:
        raise ValueError("sibling_identity.identity_sha256 mismatch")
    return json.loads(canonical_json_bytes(dict(value)))


def validate_task_f_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Task F manifest must be a mapping")
    _require_exact_keys(value, _MANIFEST_KEYS, "Task F manifest")
    if value["schema_version"] != TASK_F_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported Task F artifact schema_version")
    if value["artifact_role"] != "task_f_id_feature_export":
        raise ValueError("invalid Task F artifact_role")
    if value["tap_contract_version"] != WRN_FEATURE_TAP_CONTRACT_VERSION:
        raise ValueError("Task F tap contract version mismatch")
    if value["tap_names"] != list(WRN_FEATURE_TAP_NAMES):
        raise ValueError("Task F tap names/order mismatch")
    _require_nonempty_string(value["run_id"], "run_id")
    _reject_protected_reference(value["run_id"], "run_id")
    _require_integer(value["training_seed"], "training_seed")
    _require_nonnegative_number(value["total_weight_decay"], "total_weight_decay")
    checkpoint_epoch = _require_integer(value["checkpoint_epoch"], "checkpoint_epoch")
    if value["checkpoint_role"] not in {"last", "best_val", "snapshot"}:
        raise ValueError("checkpoint_role must be last, best_val, or snapshot")
    _require_sha256(value["checkpoint_sha256"], "checkpoint_sha256")
    depth_tap = value["depth_tap"]
    if depth_tap not in WRN_FEATURE_TAP_NAMES:
        raise ValueError(f"depth_tap must be one of {WRN_FEATURE_TAP_NAMES}")
    if checkpoint_epoch != 200 and depth_tap != "penultimate":
        raise ValueError("non-epoch-200 Task F snapshots may export penultimate only")
    if not isinstance(value["execution_only"], bool):
        raise ValueError("execution_only must be boolean")
    _validate_dataset_split(value["dataset_split"], execution_only=value["execution_only"])
    _require_sha256(value["ordered_sample_id_sha256"], "ordered_sample_id_sha256")
    shape = value["feature_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise ValueError("feature_shape must be [positive sample count, positive width]")
    if value["feature_dtype"] != TASK_F_FEATURE_DTYPE:
        raise ValueError(f"feature_dtype must be {TASK_F_FEATURE_DTYPE}")
    _require_git_sha(value["oge_git_sha"], "oge_git_sha")
    _require_sha256(value["specification_sha256"], "specification_sha256")
    _require_sha256(value["initialization_sha256"], "initialization_sha256")
    _require_sha256(value["data_stream_sha256"], "data_stream_sha256")
    _validated_model_config(value["model_config"])
    _validate_sibling_identity(value["sibling_identity"], manifest=value)
    serialization = value["serialization"]
    if not isinstance(serialization, Mapping):
        raise ValueError("serialization must be a mapping")
    _require_exact_keys(
        serialization,
        {
            "features_file",
            "sample_ids_file",
            "features_sha256",
            "sample_ids_sha256",
            "features_bytes",
            "sample_ids_bytes",
        },
        "serialization",
    )
    if serialization["features_file"] != "features.npy":
        raise ValueError("features_file must be features.npy")
    if serialization["sample_ids_file"] != "sample_ids.npy":
        raise ValueError("sample_ids_file must be sample_ids.npy")
    _require_sha256(serialization["features_sha256"], "serialization.features_sha256")
    _require_sha256(serialization["sample_ids_sha256"], "serialization.sample_ids_sha256")
    _require_integer(
        serialization["features_bytes"],
        "serialization.features_bytes",
        minimum=1,
    )
    _require_integer(
        serialization["sample_ids_bytes"], "serialization.sample_ids_bytes", minimum=1
    )
    _require_sha256(value["output_identity_sha256"], "output_identity_sha256")
    _validate_runtime(value["runtime"])
    expected_specification = canonical_sha256(specification_payload())
    if value["specification_sha256"] != expected_specification:
        raise ValueError("specification_sha256 mismatch")
    expected_output = canonical_sha256(_output_identity_payload(value))
    if value["output_identity_sha256"] != expected_output:
        raise ValueError("output_identity_sha256 mismatch")
    canonical_json_bytes(value)
    return json.loads(canonical_json_bytes(dict(value)))


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _write_checksums(root: Path) -> None:
    names = ("features.npy", "manifest.json", "sample_ids.npy")
    content = "".join(f"{sha256_file(root / name)}  {name}\n" for name in names)
    _write_bytes(root / "checksums.sha256", content.encode("utf-8"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest_from_inputs(
    *,
    features: np.ndarray,
    sample_ids: np.ndarray,
    checkpoint_sha256: str,
    provenance: Mapping[str, Any],
    depth_tap: str,
    dataset_split: str,
    runtime: Mapping[str, Any],
    features_path: Path,
    sample_ids_path: Path,
) -> dict[str, Any]:
    members = provenance["sibling_members"]
    group_id = provenance["sibling_group_id"]
    sibling_identity = {
        "schema_version": TASK_F_SIBLING_SCHEMA_VERSION,
        "group_id": group_id,
        "role": provenance["sibling_role"],
        "members": members,
        "identity_sha256": canonical_sha256(_sibling_identity_payload(group_id, members)),
    }
    manifest: dict[str, Any] = {
        "schema_version": TASK_F_ARTIFACT_SCHEMA_VERSION,
        "artifact_role": "task_f_id_feature_export",
        "tap_contract_version": WRN_FEATURE_TAP_CONTRACT_VERSION,
        "tap_names": list(WRN_FEATURE_TAP_NAMES),
        "run_id": provenance["run_id"],
        "training_seed": provenance["training_seed"],
        "branch_policy": provenance["branch_policy"],
        "total_weight_decay": provenance["total_weight_decay"],
        "coupled_ratio": provenance["coupled_ratio"],
        "checkpoint_epoch": provenance["checkpoint_epoch"],
        "checkpoint_role": provenance["checkpoint_role"],
        "checkpoint_sha256": checkpoint_sha256,
        "depth_tap": depth_tap,
        "dataset_split": dataset_split,
        "ordered_sample_id_sha256": ordered_sample_id_sha256(sample_ids),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "oge_git_sha": provenance["oge_git_sha"],
        "specification_sha256": "0" * 64,
        "execution_only": provenance["execution_only"],
        "initialization_sha256": provenance["initialization_sha256"],
        "data_stream_sha256": provenance["data_stream_sha256"],
        "model_config": provenance["model_config"],
        "sibling_identity": sibling_identity,
        "serialization": {
            "features_file": "features.npy",
            "sample_ids_file": "sample_ids.npy",
            "features_sha256": sha256_file(features_path),
            "sample_ids_sha256": sha256_file(sample_ids_path),
            "features_bytes": features_path.stat().st_size,
            "sample_ids_bytes": sample_ids_path.stat().st_size,
        },
        "output_identity_sha256": "0" * 64,
        "runtime": dict(runtime),
    }
    manifest["specification_sha256"] = canonical_sha256(specification_payload())
    manifest["output_identity_sha256"] = canonical_sha256(
        _output_identity_payload(manifest)
    )
    return manifest


def write_task_f_artifact(
    *,
    artifact_root: str | Path,
    features: np.ndarray,
    sample_ids: Sequence[str] | np.ndarray,
    checkpoint_sha256: str,
    checkpoint_provenance: Mapping[str, Any],
    depth_tap: str,
    dataset_split: str,
    runtime: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically publish one immutable Task F feature artifact directory."""

    provenance = validate_task_f_checkpoint_provenance(checkpoint_provenance)
    _require_sha256(checkpoint_sha256, "checkpoint_sha256")
    if depth_tap not in WRN_FEATURE_TAP_NAMES:
        raise ValueError(f"depth_tap must be one of {WRN_FEATURE_TAP_NAMES}")
    if provenance["checkpoint_epoch"] != 200 and depth_tap != "penultimate":
        raise ValueError("non-epoch-200 Task F snapshots may export penultimate only")
    dataset_split = _validate_dataset_split(
        dataset_split, execution_only=provenance["execution_only"]
    )
    feature_array = np.asarray(features)
    if feature_array.ndim != 2 or feature_array.shape[0] == 0:
        raise ValueError("features must be a non-empty rank-2 array")
    expected_width = {"stage1": 160, "stage2": 320, "stage3": 640, "penultimate": 640}[
        depth_tap
    ]
    if feature_array.shape[1] != expected_width:
        raise ValueError(
            f"{depth_tap} feature width must be {expected_width}, got {feature_array.shape[1]}"
        )
    if feature_array.dtype != np.float32:
        raise ValueError("Task F features must be float32")
    if not np.isfinite(feature_array).all():
        raise ValueError("Task F features must be finite")
    sample_array = np.asarray(sample_ids)
    if sample_array.ndim != 1 or sample_array.shape[0] != feature_array.shape[0]:
        raise ValueError("sample_ids must be rank-1 and match the feature row count")
    if sample_array.dtype.kind not in {"U", "S"}:
        sample_array = sample_array.astype(str)
    else:
        sample_array = sample_array.astype(str, copy=False)
    ordered_sample_id_sha256(sample_array)
    runtime_record = _validate_runtime(
        collect_runtime_provenance("cpu") if runtime is None else runtime
    )

    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".task-f-{uuid.uuid4().hex}.tmp"
    temporary.mkdir(exist_ok=False)
    try:
        features_path = temporary / "features.npy"
        sample_ids_path = temporary / "sample_ids.npy"
        _write_npy(features_path, feature_array)
        _write_npy(sample_ids_path, sample_array)
        manifest = _manifest_from_inputs(
            features=feature_array,
            sample_ids=sample_array,
            checkpoint_sha256=checkpoint_sha256,
            provenance=provenance,
            depth_tap=depth_tap,
            dataset_split=dataset_split,
            runtime=runtime_record,
            features_path=features_path,
            sample_ids_path=sample_ids_path,
        )
        validate_task_f_manifest(manifest)
        _write_bytes(
            temporary / "manifest.json", canonical_json_bytes(manifest) + b"\n"
        )
        _write_checksums(temporary)
        verify_task_f_artifact(temporary, require_identity_directory=False)
        destination = root / manifest["output_identity_sha256"]
        if destination.exists():
            raise FileExistsError(f"completed Task F artifact already exists: {destination}")
        _fsync_directory(temporary)
        try:
            os.rename(temporary, destination)
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY} or destination.exists():
                raise FileExistsError(
                    f"completed Task F artifact already exists: {destination}"
                ) from exc
            raise
        _fsync_directory(root)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_task_f_artifact(
    path: str | Path, *, require_identity_directory: bool = True
) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_path.read_bytes() != canonical_json_bytes(manifest) + b"\n":
        raise ValueError("manifest.json is not canonical JSON")
    manifest = validate_task_f_manifest(manifest)
    expected_names = {"features.npy", "manifest.json", "sample_ids.npy"}
    observed: dict[str, str] = {}
    checksum_lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        if "  " not in line:
            raise ValueError("invalid checksums.sha256 row")
        digest, relative = line.split("  ", 1)
        if relative in observed or relative not in expected_names:
            raise ValueError("checksums.sha256 contains an unexpected or duplicate path")
        _require_sha256(digest, f"checksum for {relative}")
        actual = sha256_file(root / relative)
        if actual != digest:
            raise ValueError(f"checksum mismatch for {relative}")
        observed[relative] = actual
    if set(observed) != expected_names:
        raise ValueError("checksums.sha256 does not cover the complete artifact")
    features = np.load(root / "features.npy", allow_pickle=False)
    sample_ids = np.load(root / "sample_ids.npy", allow_pickle=False)
    if list(features.shape) != manifest["feature_shape"]:
        raise ValueError("features.npy shape differs from manifest")
    if str(features.dtype) != manifest["feature_dtype"] or not np.isfinite(features).all():
        raise ValueError("features.npy dtype/finiteness differs from manifest")
    if sample_ids.ndim != 1 or sample_ids.shape[0] != features.shape[0]:
        raise ValueError("sample_ids.npy shape differs from features.npy")
    if sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("sample_ids.npy must use a non-pickle string dtype")
    if ordered_sample_id_sha256(sample_ids) != manifest["ordered_sample_id_sha256"]:
        raise ValueError("sample ID order digest mismatch")
    serialization = manifest["serialization"]
    for name, prefix in (("features.npy", "features"), ("sample_ids.npy", "sample_ids")):
        target = root / name
        if sha256_file(target) != serialization[f"{prefix}_sha256"]:
            raise ValueError(f"{name} serialization checksum mismatch")
        if target.stat().st_size != serialization[f"{prefix}_bytes"]:
            raise ValueError(f"{name} serialization byte count mismatch")
    if require_identity_directory and root.name != manifest["output_identity_sha256"]:
        raise ValueError("artifact directory name differs from output identity")
    return {"manifest": manifest, "verified_files": observed}


def load_task_f_id_input(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an already evaluation-transformed, explicitly ID-only NPZ input."""

    source = Path(path)
    _reject_protected_reference(source, "input NPZ path")
    with np.load(source, allow_pickle=False) as payload:
        expected = {"images", "sample_ids", "is_id"}
        if set(payload.files) != expected:
            raise ValueError(f"Task F input NPZ must contain exactly {sorted(expected)}")
        images = np.asarray(payload["images"])
        sample_ids = np.asarray(payload["sample_ids"])
        is_id = np.asarray(payload["is_id"])
    if images.dtype != np.float32 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Task F images must be float32 [N,3,H,W]")
    if images.shape[0] == 0 or not np.isfinite(images).all():
        raise ValueError("Task F images must be non-empty and finite")
    if is_id.dtype != np.bool_ or is_id.shape != (images.shape[0],) or not is_id.all():
        raise ValueError("Task F input must explicitly mark every sample as ID")
    if sample_ids.shape != (images.shape[0],) or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("Task F sample_ids must be a rank-1 string array")
    sample_ids = sample_ids.astype(str, copy=False)
    ordered_sample_id_sha256(sample_ids)
    return images, sample_ids


def load_task_f_checkpoint(
    path: str | Path, *, device: str | torch.device
) -> tuple[WideResNet, dict[str, Any], str]:
    target = _require_export_device(device)
    source = Path(path)
    _reject_protected_reference(source, "checkpoint path")
    payload = load_torch_artifact(source, map_location="cpu")
    provenance = validate_task_f_checkpoint_payload(payload)
    model = make_model(provenance["model_config"])
    if not isinstance(model, WideResNet):
        raise ValueError("Task F checkpoint must construct WideResNet")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(target)
    model.eval()
    return model, provenance, sha256_file(source)


def extract_task_f_features(
    model: WideResNet,
    images: np.ndarray,
    *,
    depth_tap: str,
    device: str | torch.device,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    if depth_tap not in WRN_FEATURE_TAP_NAMES:
        raise ValueError(f"depth_tap must be one of {WRN_FEATURE_TAP_NAMES}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if model.feature_tap_contract_version != WRN_FEATURE_TAP_CONTRACT_VERSION:
        raise ValueError("model feature tap contract version mismatch")
    if tuple(model.feature_tap_names) != WRN_FEATURE_TAP_NAMES:
        raise ValueError("model feature tap names/order mismatch")
    target = _require_export_device(device)
    model.eval()
    chunks: list[np.ndarray] = []
    parity_checked = False
    with torch.inference_mode():
        for start in range(0, images.shape[0], batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).to(target)
            logits, taps = model(batch, return_feature_taps=True)
            if not parity_checked:
                ordinary_logits = model(batch)
                torch.testing.assert_close(logits, ordinary_logits, rtol=0.0, atol=0.0)
                parity_checked = True
            if tuple(taps) != WRN_FEATURE_TAP_NAMES:
                raise ValueError("model returned feature taps in an invalid order")
            selected = taps[depth_tap]
            expected_width = model.feature_tap_dims[depth_tap]
            if selected.ndim != 2 or selected.shape[1] != expected_width:
                raise ValueError("model returned an invalid Task F feature shape")
            if not torch.is_floating_point(selected) or not torch.isfinite(selected).all():
                raise ValueError("model returned non-floating or non-finite Task F features")
            chunks.append(selected.detach().cpu().to(torch.float32).numpy())
            if progress_callback is not None:
                progress_callback(min(start + batch_size, images.shape[0]), images.shape[0])
    if not parity_checked:
        raise ValueError("Task F input is empty")
    return np.concatenate(chunks).astype(np.float32, copy=False)


def export_task_f_from_files(
    *,
    checkpoint_path: str | Path,
    input_npz_path: str | Path,
    artifact_root: str | Path,
    dataset_split: str,
    depth_tap: str,
    device: str | torch.device = "cpu",
    batch_size: int = 128,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Extract one ID-only tap from an explicit Task F checkpoint and NPZ."""

    # Refuse protected inputs before opening either the data or checkpoint path.
    _validate_dataset_split(dataset_split, execution_only=True)
    _reject_protected_reference(input_npz_path, "input NPZ path")
    _reject_protected_reference(checkpoint_path, "checkpoint path")
    _require_export_device(device)
    images, sample_ids = load_task_f_id_input(input_npz_path)
    model, provenance, checkpoint_sha256 = load_task_f_checkpoint(
        checkpoint_path, device=device
    )
    _validate_dataset_split(dataset_split, execution_only=provenance["execution_only"])
    features = extract_task_f_features(
        model,
        images,
        depth_tap=depth_tap,
        device=device,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    return write_task_f_artifact(
        artifact_root=artifact_root,
        features=features,
        sample_ids=sample_ids,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_provenance=provenance,
        depth_tap=depth_tap,
        dataset_split=dataset_split,
        runtime=collect_runtime_provenance(device),
    )
