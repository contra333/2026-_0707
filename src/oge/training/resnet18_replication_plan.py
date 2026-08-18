"""Dedicated planning contract for the ResNet-18/CIFAR-10 replication.

This module deliberately does not reuse or extend the frozen Task F schemas.
It shares only the ordinary classifier runner and optimizer implementations.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from oge.studies.hashing import canonical_sha256


RESNET18_REPLICATION_PLAN_SCHEMA_VERSION = "resnet18_cifar10_replication_plan_v3"
RESNET18_REPLICATION_TRAINING_SCHEMA_VERSION = (
    "resnet18_cifar10_replication_training_v3"
)
RESNET18_REPLICATION_APPROVAL_SCHEMA_VERSION = (
    "resnet18_cifar10_replication_approval_packet_v3"
)
RESNET18_REPLICATION_PROTOCOL_ID = (
    "protocol_fixed_branch_refitted_md_architecture_replication_v3"
)
RESNET18_REPLICATION_STUDY_ID = "resnet18_cifar10_replication_v3"
RESNET18_REPLICATION_NUMERICAL_POLICY_ID = "strict_cuda_deterministic_cublas4096_v3"
RESNET18_REPLICATION_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
RESNET18_REPLICATION_SEEDS = (0, 1, 2, 3, 4)
RESNET18_REPLICATION_LRS = (1e-3, 3e-4)
RESNET18_REPLICATION_WEIGHT_DECAY = 1e-4
RESNET18_REPLICATION_ROLES = ("adam_coupled", "adamw_decoupled")
RESNET18_REPLICATION_SNAPSHOT_EPOCHS = (0, 200)
RESNET18_REPLICATION_ID_EQUIVALENCE_MARGINS = {
    "accuracy": 0.01,
    "nll": 0.08,
    "ece": 0.02,
}

_PROTECTED_PATTERNS = (
    re.compile(r"(^|[/_.-])ood($|[/_.-])", re.IGNORECASE),
    re.compile(r"openood", re.IGNORECASE),
    re.compile(r"(^|[/_.-])protected[_-]?ood($|[/_.-])", re.IGNORECASE),
    re.compile(r"(^|[/_.-])nearood($|[/_.-])", re.IGNORECASE),
    re.compile(r"(^|[/_.-])farood($|[/_.-])", re.IGNORECASE),
)


def _validate_seed_list(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if len(seeds) != 5:
        raise ValueError("research_seeds must contain exactly 5 seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("research_seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("research_seeds must not contain duplicates")
    return seeds


def _float_token(value: float) -> str:
    return format(value, ".0e").replace("+", "")


def _optimizer(role: str, *, lr: float) -> dict[str, Any]:
    common = {
        "lr": lr,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": RESNET18_REPLICATION_WEIGHT_DECAY,
        "weight_decay_policy": "weights_only_no_bias_norm",
    }
    if role == "adam_coupled":
        return {"name": "adam", **common}
    if role == "adamw_decoupled":
        return {"name": "adamw", **common}
    raise ValueError(f"unsupported replication role: {role!r}")


def _run_id(*, lr: float, seed: int, role: str) -> str:
    return (
        f"resnet18-c10-rep-v3-lr{_float_token(lr)}-"
        f"wd{_float_token(RESNET18_REPLICATION_WEIGHT_DECAY)}-seed{seed}-"
        f"{role.replace('_', '-')}"
    )


def _record(*, lr: float, seed: int, role: str) -> dict[str, Any]:
    lr_token = _float_token(lr)
    sibling_group_id = f"resnet18-c10-rep-v3-lr{lr_token}-seed{seed}"
    cross_lr_pairing_block_id = f"resnet18-c10-rep-v3-cross-lr-seed{seed}"
    return {
        "run_id": _run_id(lr=lr, seed=seed, role=role),
        "cell_id": f"resnet18_c10_lr{lr_token}_wd1e-4",
        "training_seed": seed,
        "sibling_group_id": sibling_group_id,
        "data_stream_id": cross_lr_pairing_block_id,
        "cross_lr_pairing_block_id": cross_lr_pairing_block_id,
        "branch_policy": role,
        "initial_lr": lr,
        "nominal_weight_decay": RESNET18_REPLICATION_WEIGHT_DECAY,
        "optimizer": _optimizer(role, lr=lr),
        "execution_only": False,
        "aggregate_eligible": True,
        "from_scratch": True,
        "fork_from_prefix": None,
    }


def _sibling_members(runs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    for run in sorted(runs, key=lambda item: str(item["branch_policy"])):
        role = str(run["branch_policy"])
        members[role] = {
            "run_id": str(run["run_id"]),
            "training_seed": int(run["training_seed"]),
            "branch_policy": role,
            "initial_lr": float(run["initial_lr"]),
            "weight_decay": float(run["nominal_weight_decay"]),
        }
    return members


def _attach_sibling_members(runs: Sequence[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_group[str(run["sibling_group_id"])].append(run)
    for group_runs in by_group.values():
        members = _sibling_members(group_runs)
        for run in group_runs:
            run["sibling_role"] = str(run["branch_policy"])
            run["sibling_members"] = copy.deepcopy(members)


def generate_resnet18_replication_matrix(
    *, research_seeds: Sequence[int] = RESNET18_REPLICATION_SEEDS
) -> dict[str, Any]:
    """Generate the exact 20-run, two-LR, C/D replication matrix."""

    seeds = _validate_seed_list(research_seeds)
    runs = [
        _record(lr=lr, seed=seed, role=role)
        for seed in seeds
        for lr in RESNET18_REPLICATION_LRS
        for role in RESNET18_REPLICATION_ROLES
    ]
    _attach_sibling_members(runs)
    plan = {
        "schema_version": RESNET18_REPLICATION_PLAN_SCHEMA_VERSION,
        "protocol_id": RESNET18_REPLICATION_PROTOCOL_ID,
        "study_id": RESNET18_REPLICATION_STUDY_ID,
        "numerical_policy_id": RESNET18_REPLICATION_NUMERICAL_POLICY_ID,
        "cublas_workspace_config": RESNET18_REPLICATION_CUBLAS_WORKSPACE_CONFIG,
        "research_seeds": list(seeds),
        "snapshot_epochs": list(RESNET18_REPLICATION_SNAPSHOT_EPOCHS),
        "runs": runs,
    }
    validate_resnet18_replication_matrix(plan)
    return plan


def resnet18_replication_count_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    runs = list(plan.get("runs", []))
    return {
        "total_research_runs": len(runs),
        "cell_counts": dict(sorted(Counter(run["cell_id"] for run in runs).items())),
        "role_counts": dict(
            sorted(Counter(run["branch_policy"] for run in runs).items())
        ),
    }


def validate_resnet18_replication_matrix(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != RESNET18_REPLICATION_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported ResNet-18 replication plan schema_version")
    if plan.get("protocol_id") != RESNET18_REPLICATION_PROTOCOL_ID:
        raise ValueError("ResNet-18 replication protocol identity mismatch")
    if plan.get("study_id") != RESNET18_REPLICATION_STUDY_ID:
        raise ValueError("ResNet-18 replication study identity mismatch")
    if plan.get("numerical_policy_id") != RESNET18_REPLICATION_NUMERICAL_POLICY_ID:
        raise ValueError("ResNet-18 replication numerical policy mismatch")
    if (
        plan.get("cublas_workspace_config")
        != RESNET18_REPLICATION_CUBLAS_WORKSPACE_CONFIG
    ):
        raise ValueError("ResNet-18 replication CuBLAS workspace policy mismatch")
    runs = plan.get("runs")
    if not isinstance(runs, list) or len(runs) != 20:
        raise ValueError("ResNet-18 replication matrix must contain exactly 20 runs")
    if len({run.get("run_id") for run in runs}) != 20:
        raise ValueError("ResNet-18 replication run IDs must be unique")
    if any(bool(run.get("execution_only")) or not bool(run.get("aggregate_eligible")) for run in runs):
        raise ValueError("research runs must be aggregate-eligible and not execution-only")
    if any(run.get("fork_from_prefix") is not None or not bool(run.get("from_scratch")) for run in runs):
        raise ValueError("ResNet-18 replication research runs must start from scratch")

    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        seed = int(run["training_seed"])
        by_seed[seed].append(run)
        by_group[str(run["sibling_group_id"])].append(run)
        if run.get("data_stream_id") != run.get("cross_lr_pairing_block_id"):
            raise ValueError("all four seed arms must declare one cross-LR data stream")
        if float(run["nominal_weight_decay"]) != RESNET18_REPLICATION_WEIGHT_DECAY:
            raise ValueError("replication weight decay must be exactly 1e-4")
        optimizer = run.get("optimizer")
        if not isinstance(optimizer, Mapping):
            raise ValueError("replication optimizer must be a mapping")
        expected_name = "adam" if run["branch_policy"] == "adam_coupled" else "adamw"
        if optimizer.get("name") != expected_name:
            raise ValueError("replication optimizer does not match branch policy")
        if float(optimizer.get("lr", math.nan)) != float(run["initial_lr"]):
            raise ValueError("replication optimizer LR differs from run record")

    if len(by_seed) != 5:
        raise ValueError("ResNet-18 replication must contain five paired seeds")
    for seed, seed_runs in by_seed.items():
        expected_block = f"resnet18-c10-rep-v3-cross-lr-seed{seed}"
        if {run["cross_lr_pairing_block_id"] for run in seed_runs} != {expected_block}:
            raise ValueError("cross-LR pairing block identity mismatch")
        combinations = {
            (float(run["initial_lr"]), str(run["branch_policy"])) for run in seed_runs
        }
        expected = set((lr, role) for lr in RESNET18_REPLICATION_LRS for role in RESNET18_REPLICATION_ROLES)
        if combinations != expected:
            raise ValueError("each seed must contain the exact four LR-by-coupling arms")
    for group_runs in by_group.values():
        if len(group_runs) != 2 or {run["branch_policy"] for run in group_runs} != set(
            RESNET18_REPLICATION_ROLES
        ):
            raise ValueError("each sibling group must contain coupled and decoupled arms")
        members = group_runs[0]["sibling_members"]
        if any(run["sibling_members"] != members for run in group_runs):
            raise ValueError("sibling member declarations differ within a group")


def _model_config() -> dict[str, Any]:
    return {"name": "resnet18", "variant": "cifar", "num_classes": 10}


def _branch_neutral_config(
    config: Mapping[str, Any], *, run: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "protocol_id": RESNET18_REPLICATION_PROTOCOL_ID,
        "study_id": RESNET18_REPLICATION_STUDY_ID,
        "sibling_group_id": str(run["sibling_group_id"]),
        "training_seed": int(run["training_seed"]),
        "dataset": copy.deepcopy(config["dataset"]),
        "model": copy.deepcopy(config["model"]),
        "loss": copy.deepcopy(config["loss"]),
        "optimizer_shared": {
            "family": "adam",
            "lr": float(run["initial_lr"]),
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": RESNET18_REPLICATION_WEIGHT_DECAY,
            "weight_decay_policy": "weights_only_no_bias_norm",
        },
        "scheduler": copy.deepcopy(config["scheduler"]),
        "training": copy.deepcopy(config["training"]),
        "checkpoint": copy.deepcopy(config["checkpoint"]),
    }


def _cross_lr_neutral_config(
    branch_neutral: Mapping[str, Any], *, run: Mapping[str, Any]
) -> dict[str, Any]:
    neutral = copy.deepcopy(dict(branch_neutral))
    neutral["cross_lr_pairing_block_id"] = str(run["cross_lr_pairing_block_id"])
    neutral.pop("sibling_group_id")
    neutral["optimizer_shared"].pop("lr")
    neutral["scheduler"] = {
        **neutral["scheduler"],
        "initial_lr_source": "optimizer_arm",
    }
    return neutral


def build_resnet18_replication_training_config(
    base_config: Mapping[str, Any], run: Mapping[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    config["model"] = _model_config()
    config["optimizer"] = copy.deepcopy(run["optimizer"])
    config["training"]["seed"] = int(run["training_seed"])
    config["training"]["max_epochs"] = 200
    config["training"]["deterministic"] = True
    config["checkpoint"]["snapshot_epochs"] = list(
        RESNET18_REPLICATION_SNAPSHOT_EPOCHS
    )
    branch_neutral = _branch_neutral_config(config, run=run)
    cross_lr_neutral = _cross_lr_neutral_config(branch_neutral, run=run)
    config["resnet18_replication"] = {
        "schema_version": RESNET18_REPLICATION_TRAINING_SCHEMA_VERSION,
        "protocol_id": RESNET18_REPLICATION_PROTOCOL_ID,
        "study_id": RESNET18_REPLICATION_STUDY_ID,
        "numerical_policy_id": RESNET18_REPLICATION_NUMERICAL_POLICY_ID,
        "cublas_workspace_config": RESNET18_REPLICATION_CUBLAS_WORKSPACE_CONFIG,
        "run_id": str(run["run_id"]),
        "sibling_group_id": str(run["sibling_group_id"]),
        "data_stream_id": str(run["data_stream_id"]),
        "cross_lr_pairing_block_id": str(run["cross_lr_pairing_block_id"]),
        "branch_policy": str(run["branch_policy"]),
        "branch_neutral_config": branch_neutral,
        "branch_neutral_config_sha256": canonical_sha256(branch_neutral),
        "cross_lr_neutral_config": cross_lr_neutral,
        "cross_lr_neutral_config_sha256": canonical_sha256(cross_lr_neutral),
        "execution_only": bool(run["execution_only"]),
        "aggregate_eligible": bool(run["aggregate_eligible"]),
        "from_scratch": bool(run["from_scratch"]),
        "defer_id_test": True,
        "artifact_namespace": RESNET18_REPLICATION_STUDY_ID,
        "sibling_role": str(run["sibling_role"]),
        "sibling_members": copy.deepcopy(run["sibling_members"]),
    }
    validate_resnet18_replication_training_config(config)
    return config


def validate_no_protected_references(value: object, path: str = "<root>") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            validate_no_protected_references(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            validate_no_protected_references(child, f"{path}[{index}]")
    elif isinstance(value, str) and any(pattern.search(value) for pattern in _PROTECTED_PATTERNS):
        raise ValueError(f"protected reference is forbidden at {path}")


def validate_resnet18_replication_training_config(config: Mapping[str, Any]) -> None:
    if "task_f" in config:
        raise ValueError("ResNet-18 replication must not use the frozen Task F schema")
    replication = config.get("resnet18_replication")
    if not isinstance(replication, Mapping):
        raise ValueError("ResNet-18 replication config requires its dedicated mapping")
    if replication.get("schema_version") != RESNET18_REPLICATION_TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported ResNet-18 replication training schema_version")
    if config.get("model") != _model_config():
        raise ValueError("replication model must be native ResNet-18 CIFAR with 10 classes")
    if config["dataset"].get("protocol") != "oge_cifar10_holdout_v1":
        raise ValueError("replication dataset must use oge_cifar10_holdout_v1")
    if config["training"].get("deterministic") is not True:
        raise ValueError("replication v3 requires training.deterministic=true")
    if (
        replication.get("numerical_policy_id")
        != RESNET18_REPLICATION_NUMERICAL_POLICY_ID
    ):
        raise ValueError("replication v3 numerical policy identity mismatch")
    if (
        replication.get("cublas_workspace_config")
        != RESNET18_REPLICATION_CUBLAS_WORKSPACE_CONFIG
    ):
        raise ValueError("replication v3 CuBLAS workspace policy mismatch")
    if not bool(replication.get("from_scratch")):
        raise ValueError("ResNet-18 replication must start from scratch")
    if not bool(replication.get("defer_id_test")):
        raise ValueError("ResNet-18 replication must defer ID-test evaluation")
    if bool(replication.get("execution_only")) == bool(replication.get("aggregate_eligible")):
        raise ValueError("execution_only and aggregate_eligible must be opposites")
    branch_neutral = replication.get("branch_neutral_config")
    cross_lr_neutral = replication.get("cross_lr_neutral_config")
    if not isinstance(branch_neutral, Mapping) or canonical_sha256(branch_neutral) != replication.get(
        "branch_neutral_config_sha256"
    ):
        raise ValueError("replication branch-neutral config/hash mismatch")
    if not isinstance(cross_lr_neutral, Mapping) or canonical_sha256(cross_lr_neutral) != replication.get(
        "cross_lr_neutral_config_sha256"
    ):
        raise ValueError("replication cross-LR neutral config/hash mismatch")
    if branch_neutral.get("training") != config["training"]:
        raise ValueError("replication branch-neutral training policy mismatch")
    if cross_lr_neutral.get("training") != config["training"]:
        raise ValueError("replication cross-LR training policy mismatch")
    if branch_neutral.get("sibling_group_id") != replication.get("sibling_group_id"):
        raise ValueError("replication sibling identity differs from neutral config")
    if cross_lr_neutral.get("cross_lr_pairing_block_id") != replication.get(
        "cross_lr_pairing_block_id"
    ):
        raise ValueError("replication cross-LR identity differs from neutral config")
    if replication.get("data_stream_id") != replication.get("cross_lr_pairing_block_id"):
        raise ValueError("replication data stream must be shared across both LR arms")
    if tuple(config["checkpoint"]["snapshot_epochs"]) != RESNET18_REPLICATION_SNAPSHOT_EPOCHS and not bool(
        replication["execution_only"]
    ):
        raise ValueError("research runs require endpoint-only snapshots [0, 200]")
    members = replication.get("sibling_members")
    role = replication.get("sibling_role")
    if not isinstance(members, Mapping) or role not in members:
        raise ValueError("replication sibling plan is incomplete")
    current = members[role]
    if current.get("run_id") != replication.get("run_id"):
        raise ValueError("replication current sibling differs from run config")
    validate_no_protected_references(
        {
            "run_id": replication["run_id"],
            "dataset": {
                "config_path": config["dataset"]["config_path"],
                "train_split": config["dataset"]["train_split"],
                "validation_split": config["dataset"]["validation_split"],
            },
            "artifact_namespace": replication["artifact_namespace"],
        }
    )


def generate_resnet18_execution_only_pilot_configs(
    *, base_config: Mapping[str, Any], seed: int = 9000, max_epochs: int = 2
) -> list[dict[str, Any]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("pilot seed must be a non-negative integer")
    if max_epochs != 2:
        raise ValueError("the frozen replication pilot must use exactly two epochs")
    pilot_runs = [
        _record(lr=lr, seed=seed, role=role)
        for lr in RESNET18_REPLICATION_LRS
        for role in RESNET18_REPLICATION_ROLES
    ]
    for run in pilot_runs:
        run.update({"execution_only": True, "aggregate_eligible": False})
        run["run_id"] = "pilot-" + str(run["run_id"])
    _attach_sibling_members(pilot_runs)
    configs = []
    for run in pilot_runs:
        config = build_resnet18_replication_training_config(base_config, run)
        config["training"]["max_epochs"] = max_epochs
        config["checkpoint"]["snapshot_epochs"] = [0, 1, 2]
        branch_neutral = _branch_neutral_config(config, run=run)
        cross_lr_neutral = _cross_lr_neutral_config(branch_neutral, run=run)
        replication = config["resnet18_replication"]
        replication["branch_neutral_config"] = branch_neutral
        replication["branch_neutral_config_sha256"] = canonical_sha256(branch_neutral)
        replication["cross_lr_neutral_config"] = cross_lr_neutral
        replication["cross_lr_neutral_config_sha256"] = canonical_sha256(cross_lr_neutral)
        validate_resnet18_replication_training_config(config)
        configs.append(config)
    return configs


def generate_resnet18_approval_packet(
    *,
    execution_sha: str,
    pilot_configs: Sequence[Mapping[str, Any]] | None = None,
    pilot_approval_reference: str | None = None,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", execution_sha) is None:
        raise ValueError("execution_sha must be a lowercase 40-character Git SHA")
    pilot = {
        "status": "UNMATERIALIZED",
        "seed": 9000,
        "arm_count": 4,
        "max_epochs": 2,
    }
    if pilot_configs is not None:
        if len(pilot_configs) != 4:
            raise ValueError("pilot must contain exactly four configs")
        for config in pilot_configs:
            validate_resnet18_replication_training_config(config)
            if not bool(config["resnet18_replication"]["execution_only"]):
                raise ValueError("approval packet pilot configs must be execution-only")
        pilot["status"] = "MATERIALIZED_NOT_RUN"
    if pilot_approval_reference is not None:
        if not isinstance(pilot_approval_reference, str) or not pilot_approval_reference:
            raise ValueError("pilot_approval_reference must be a non-empty string")
        if pilot_configs is None:
            raise ValueError("pilot approval requires materialized pilot configs")
        pilot["status"] = "EXPLICITLY_APPROVED_NOT_RUN"
        pilot["approval_reference"] = pilot_approval_reference
    return {
        "schema_version": RESNET18_REPLICATION_APPROVAL_SCHEMA_VERSION,
        "protocol_id": RESNET18_REPLICATION_PROTOCOL_ID,
        "study_id": RESNET18_REPLICATION_STUDY_ID,
        "numerical_policy_id": RESNET18_REPLICATION_NUMERICAL_POLICY_ID,
        "cublas_workspace_config": RESNET18_REPLICATION_CUBLAS_WORKSPACE_CONFIG,
        "execution_sha": execution_sha,
        "research_matrix": {
            "run_count": 20,
            "learning_rates": list(RESNET18_REPLICATION_LRS),
            "weight_decay": RESNET18_REPLICATION_WEIGHT_DECAY,
            "roles": list(RESNET18_REPLICATION_ROLES),
            "seeds": list(RESNET18_REPLICATION_SEEDS),
        },
        "pilot": pilot,
        "resource_estimates": {
            "seconds_per_epoch": "PENDING_PILOT",
            "peak_vram_bytes": "PENDING_PILOT",
            "expected_gpu_hours": "PENDING_PILOT",
            "expected_storage_bytes": "PENDING_PILOT",
        },
        "approval_boundaries": {
            "pilot_gpu_approval": (
                "EXPLICITLY_APPROVED"
                if pilot_approval_reference is not None
                else "REQUIRED_BEFORE_EXECUTION"
            ),
            "main_training_gpu_approval": "SEPARATE_REQUIRED_AFTER_PILOT",
            "protected_evaluation": "NOT_AUTHORIZED",
            "pilot_is_research_evidence": False,
        },
    }
