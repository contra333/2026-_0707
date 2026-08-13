"""Declarative Card 13 v12 run planning without launching training."""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from oge.studies.hashing import canonical_sha256


TASK_F_PLAN_SCHEMA_VERSION = "task_f_paired_run_plan_v1"
TASK_F_TRAINING_SCHEMA_VERSION = "task_f_paired_training_v1"
TASK_F_APPROVAL_PACKET_SCHEMA_VERSION = "task_f_approval_packet_v1"
TASK_F_PROTOCOL_ID = "fixed_readout_pair_ranking_multiplicity_paired_trajectory_v12"
TASK_F_STUDY_ID = "card13_v12_fresh_paired_mechanism"
TASK_F_SNAPSHOT_EPOCHS = (0, 1, 10, 30, 60, 61, 120, 121, 160, 161, 200)

_EXPECTED_CELL_COUNTS = {
    "adam_lr1e-3_wd1e-4_anchor": 20,
    "adam_lr1e-3_wd1e-3": 6,
    "adam_lr3e-4_wd1e-4": 9,
    "adam_lr3e-4_wd1e-3": 6,
    "sgdm_lr0.1_wd5e-4": 9,
}
_EXPECTED_CELL_POLICIES = {
    "adam_lr1e-3_wd1e-4_anchor": {
        "zero",
        "adamw_alpha_0",
        "adam_mixed_alpha_0_5",
        "adam_alpha_1",
    },
    "adam_lr1e-3_wd1e-3": {"adam_coupled", "adamw_decoupled"},
    "adam_lr3e-4_wd1e-4": {"zero", "adam_coupled", "adamw_decoupled"},
    "adam_lr3e-4_wd1e-3": {"adam_coupled", "adamw_decoupled"},
    "sgdm_lr0.1_wd5e-4": {"zero", "sgdm_coupled", "sgdw_decoupled"},
}
_PROTECTED_PATTERNS = (
    re.compile(r"(^|[/_.-])ood($|[/_.-])", re.IGNORECASE),
    re.compile(r"openood", re.IGNORECASE),
    re.compile(r"(^|[/_.-])protected[_-]?ood($|[/_.-])", re.IGNORECASE),
    re.compile(r"(^|[/_.-])nearood($|[/_.-])", re.IGNORECASE),
    re.compile(r"(^|[/_.-])farood($|[/_.-])", re.IGNORECASE),
)


def _validate_seed_list(name: str, values: Sequence[int], expected_count: int) -> tuple[int, ...]:
    seeds = tuple(values)
    if len(seeds) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError(f"{name} must contain non-negative integer seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must not contain duplicate seeds")
    return seeds


def _float_token(value: float) -> str:
    return format(value, ".0e").replace("+", "")


def _run_id(*, family: str, lr: float, seed: int, branch: str, nominal_wd: float) -> str:
    return (
        f"task-f-{family}-lr{_float_token(lr)}-wd{_float_token(nominal_wd)}-"
        f"seed{seed}-{branch.replace('_', '-')}"
    )


def _adam_optimizer(branch: str, *, lr: float, nominal_wd: float) -> dict[str, Any]:
    shared = {
        "lr": lr,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay_policy": "weights_only_no_bias_norm",
    }
    if branch == "zero":
        return {"name": "adam", **shared, "weight_decay": 0.0}
    if branch == "adam_coupled":
        return {"name": "adam", **shared, "weight_decay": nominal_wd}
    if branch == "adamw_decoupled":
        return {"name": "adamw", **shared, "weight_decay": nominal_wd}
    alpha = {
        "adamw_alpha_0": 0.0,
        "adam_mixed_alpha_0_5": 0.5,
        "adam_alpha_1": 1.0,
    }.get(branch)
    if alpha is None:
        raise ValueError(f"unsupported Adam branch: {branch!r}")
    return {
        "name": "adam_coupled_decoupled",
        **shared,
        "total_weight_decay": nominal_wd,
        "coupled_ratio": alpha,
    }


def _sgdm_optimizer(branch: str, *, lr: float, nominal_wd: float) -> dict[str, Any]:
    shared = {
        "lr": lr,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay_policy": "weights_only_no_bias_norm",
    }
    if branch == "zero":
        return {"name": "sgd", **shared, "weight_decay": 0.0}
    if branch == "sgdm_coupled":
        return {"name": "sgd", **shared, "weight_decay": nominal_wd}
    if branch == "sgdw_decoupled":
        return {"name": "sgdw", **shared, "weight_decay": nominal_wd}
    raise ValueError(f"unsupported SGDM branch: {branch!r}")


def _record(
    *,
    family: str,
    cell_id: str,
    lr: float,
    nominal_wd: float,
    seed: int,
    branch_policy: str,
    sibling_group_id: str,
    optimizer: Mapping[str, Any],
    zero_reference_run_id: str | None,
) -> dict[str, Any]:
    run_id = _run_id(
        family=family,
        lr=lr,
        seed=seed,
        branch=branch_policy,
        nominal_wd=nominal_wd,
    )
    return {
        "run_id": run_id,
        "family": family,
        "cell_id": cell_id,
        "training_seed": seed,
        "sibling_group_id": sibling_group_id,
        "data_stream_id": sibling_group_id,
        "branch_policy": branch_policy,
        "initial_lr": lr,
        "nominal_weight_decay": nominal_wd,
        "optimizer": copy.deepcopy(dict(optimizer)),
        "zero_reference_run_id": zero_reference_run_id,
        "execution_only": False,
        "aggregate_eligible": True,
        "from_scratch": True,
        "fork_from_prefix": None,
    }


def generate_research_run_matrix(
    *,
    anchor_seeds: Sequence[int],
    adam_factorial_seeds: Sequence[int],
    sgdm_seeds: Sequence[int],
) -> dict[str, Any]:
    """Generate the exact 41-Adam plus 9-SGDM Card 13 matrix.

    Seed identities are caller inputs.  The three Adam factorial seeds must be
    a subset of the five anchor seeds so the lr=1e-3 zero baseline is genuinely
    shared rather than duplicated.
    """

    anchor = _validate_seed_list("anchor_seeds", anchor_seeds, 5)
    factorial = _validate_seed_list("adam_factorial_seeds", adam_factorial_seeds, 3)
    sgdm = _validate_seed_list("sgdm_seeds", sgdm_seeds, 3)
    if not set(factorial).issubset(anchor):
        raise ValueError("adam_factorial_seeds must be a subset of anchor_seeds")

    runs: list[dict[str, Any]] = []
    anchor_cell = "adam_lr1e-3_wd1e-4_anchor"
    high_wd_cell = "adam_lr1e-3_wd1e-3"
    for seed in anchor:
        group = f"task-f-adam-lr1e-3-seed{seed}"
        zero_id = _run_id(
            family="adam", lr=1e-3, seed=seed, branch="zero", nominal_wd=0.0
        )
        runs.append(
            _record(
                family="adam",
                cell_id=anchor_cell,
                lr=1e-3,
                nominal_wd=0.0,
                seed=seed,
                branch_policy="zero",
                sibling_group_id=group,
                optimizer=_adam_optimizer("zero", lr=1e-3, nominal_wd=0.0),
                zero_reference_run_id=zero_id,
            )
        )
        for branch in ("adamw_alpha_0", "adam_mixed_alpha_0_5", "adam_alpha_1"):
            runs.append(
                _record(
                    family="adam",
                    cell_id=anchor_cell,
                    lr=1e-3,
                    nominal_wd=1e-4,
                    seed=seed,
                    branch_policy=branch,
                    sibling_group_id=group,
                    optimizer=_adam_optimizer(branch, lr=1e-3, nominal_wd=1e-4),
                    zero_reference_run_id=zero_id,
                )
            )
        if seed in factorial:
            for branch in ("adam_coupled", "adamw_decoupled"):
                runs.append(
                    _record(
                        family="adam",
                        cell_id=high_wd_cell,
                        lr=1e-3,
                        nominal_wd=1e-3,
                        seed=seed,
                        branch_policy=branch,
                        sibling_group_id=group,
                        optimizer=_adam_optimizer(branch, lr=1e-3, nominal_wd=1e-3),
                        zero_reference_run_id=zero_id,
                    )
                )

    low_wd_cell = "adam_lr3e-4_wd1e-4"
    low_lr_high_wd_cell = "adam_lr3e-4_wd1e-3"
    for seed in factorial:
        group = f"task-f-adam-lr3e-4-seed{seed}"
        zero_id = _run_id(
            family="adam", lr=3e-4, seed=seed, branch="zero", nominal_wd=0.0
        )
        runs.append(
            _record(
                family="adam",
                cell_id=low_wd_cell,
                lr=3e-4,
                nominal_wd=0.0,
                seed=seed,
                branch_policy="zero",
                sibling_group_id=group,
                optimizer=_adam_optimizer("zero", lr=3e-4, nominal_wd=0.0),
                zero_reference_run_id=zero_id,
            )
        )
        for branch in ("adam_coupled", "adamw_decoupled"):
            runs.append(
                _record(
                    family="adam",
                    cell_id=low_wd_cell,
                    lr=3e-4,
                    nominal_wd=1e-4,
                    seed=seed,
                    branch_policy=branch,
                    sibling_group_id=group,
                    optimizer=_adam_optimizer(branch, lr=3e-4, nominal_wd=1e-4),
                    zero_reference_run_id=zero_id,
                )
            )
        for branch in ("adam_coupled", "adamw_decoupled"):
            runs.append(
                _record(
                    family="adam",
                    cell_id=low_lr_high_wd_cell,
                    lr=3e-4,
                    nominal_wd=1e-3,
                    seed=seed,
                    branch_policy=branch,
                    sibling_group_id=group,
                    optimizer=_adam_optimizer(branch, lr=3e-4, nominal_wd=1e-3),
                    zero_reference_run_id=zero_id,
                )
            )

    sgdm_cell = "sgdm_lr0.1_wd5e-4"
    for seed in sgdm:
        group = f"task-f-sgdm-lr0.1-seed{seed}"
        zero_id = _run_id(
            family="sgdm", lr=0.1, seed=seed, branch="zero", nominal_wd=0.0
        )
        for branch, wd in (
            ("zero", 0.0),
            ("sgdm_coupled", 5e-4),
            ("sgdw_decoupled", 5e-4),
        ):
            runs.append(
                _record(
                    family="sgdm",
                    cell_id=sgdm_cell,
                    lr=0.1,
                    nominal_wd=wd,
                    seed=seed,
                    branch_policy=branch,
                    sibling_group_id=group,
                    optimizer=_sgdm_optimizer(branch, lr=0.1, nominal_wd=wd),
                    zero_reference_run_id=zero_id,
                )
            )

    plan = {
        "schema_version": TASK_F_PLAN_SCHEMA_VERSION,
        "protocol_id": TASK_F_PROTOCOL_ID,
        "study_id": TASK_F_STUDY_ID,
        "research_seed_inputs": {
            "anchor_seeds": list(anchor),
            "adam_factorial_seeds": list(factorial),
            "sgdm_seeds": list(sgdm),
        },
        "snapshot_epochs": list(TASK_F_SNAPSHOT_EPOCHS),
        "runs": runs,
    }
    validate_research_run_matrix(plan)
    return plan


def matrix_count_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    runs = list(plan.get("runs", []))
    return {
        "total_research_runs": len(runs),
        "family_counts": dict(sorted(Counter(run["family"] for run in runs).items())),
        "cell_counts": dict(sorted(Counter(run["cell_id"] for run in runs).items())),
    }


def validate_research_run_matrix(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != TASK_F_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported Task F run-plan schema_version")
    runs = plan.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Task F run plan must contain a runs list")
    if any(bool(run.get("execution_only")) for run in runs):
        raise ValueError("execution_only runs are forbidden in the research matrix")
    if any(not bool(run.get("aggregate_eligible")) for run in runs):
        raise ValueError("every research-matrix run must be aggregate eligible")
    if len({run.get("run_id") for run in runs}) != len(runs):
        raise ValueError("Task F research run IDs must be unique")
    summary = matrix_count_summary(plan)
    if summary["total_research_runs"] != 50:
        raise ValueError("Task F research matrix must contain exactly 50 runs")
    if summary["family_counts"] != {"adam": 41, "sgdm": 9}:
        raise ValueError("Task F research matrix must contain 41 Adam and 9 SGDM runs")
    if summary["cell_counts"] != dict(sorted(_EXPECTED_CELL_COUNTS.items())):
        raise ValueError("Task F research matrix cell counts do not match Card 13 v12")

    by_cell_seed: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for run in runs:
        by_cell_seed[str(run["cell_id"])][int(run["training_seed"])].add(
            str(run["branch_policy"])
        )
    expected_seed_counts = {
        "adam_lr1e-3_wd1e-4_anchor": 5,
        "adam_lr1e-3_wd1e-3": 3,
        "adam_lr3e-4_wd1e-4": 3,
        "adam_lr3e-4_wd1e-3": 3,
        "sgdm_lr0.1_wd5e-4": 3,
    }
    for cell_id, expected_policies in _EXPECTED_CELL_POLICIES.items():
        seed_groups = by_cell_seed[cell_id]
        if len(seed_groups) != expected_seed_counts[cell_id]:
            raise ValueError(f"Task F cell {cell_id} has an invalid paired-seed count")
        if any(policies != expected_policies for policies in seed_groups.values()):
            raise ValueError(f"Task F cell {cell_id} has an invalid sibling policy set")

    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_run_id = {str(run["run_id"]): run for run in runs}
    for run in runs:
        if run.get("fork_from_prefix") is not None or not bool(run.get("from_scratch")):
            raise ValueError("Task F research runs must start from scratch")
        group = str(run.get("sibling_group_id", ""))
        if not group or run.get("data_stream_id") != group:
            raise ValueError("Task F siblings require an explicit shared data stream")
        by_group[group].append(run)
        zero_id = str(run.get("zero_reference_run_id", ""))
        zero = by_run_id.get(zero_id)
        if zero is None or zero.get("branch_policy") != "zero":
            raise ValueError("every Task F run must reference its shared zero baseline")
        if zero.get("sibling_group_id") != group:
            raise ValueError("zero baseline must belong to the same sibling group")
        branch = str(run["branch_policy"])
        optimizer = run.get("optimizer")
        if not isinstance(optimizer, Mapping):
            raise ValueError("Task F run optimizer must be a mapping")
        if branch == "zero":
            if float(optimizer.get("weight_decay", -1.0)) != 0.0 or "coupled_ratio" in optimizer:
                raise ValueError("zero decay must be a separate sibling, not an alpha value")
        if branch in {"adamw_alpha_0", "adam_mixed_alpha_0_5", "adam_alpha_1"}:
            expected_ratio = {
                "adamw_alpha_0": 0.0,
                "adam_mixed_alpha_0_5": 0.5,
                "adam_alpha_1": 1.0,
            }[branch]
            if (
                optimizer.get("name") != "adam_coupled_decoupled"
                or float(optimizer.get("total_weight_decay", -1.0)) != 1e-4
                or float(optimizer.get("coupled_ratio", -1.0)) != expected_ratio
            ):
                raise ValueError("anchor alpha branch does not match exact optimizer semantics")

    for group_runs in by_group.values():
        seeds = {int(run["training_seed"]) for run in group_runs}
        streams = {str(run["data_stream_id"]) for run in group_runs}
        if len(seeds) != 1 or len(streams) != 1:
            raise ValueError("sibling pairing requires one shared seed and data stream")

    anchor_runs = [run for run in runs if run["cell_id"] == "adam_lr1e-3_wd1e-4_anchor"]
    anchor_by_seed: dict[int, set[str]] = defaultdict(set)
    for run in anchor_runs:
        anchor_by_seed[int(run["training_seed"])].add(str(run["branch_policy"]))
    expected_anchor = {"zero", "adamw_alpha_0", "adam_mixed_alpha_0_5", "adam_alpha_1"}
    if any(branches != expected_anchor for branches in anchor_by_seed.values()):
        raise ValueError("each anchor seed must contain zero and alpha 0/0.5/1 siblings")


def _optimizer_family_neutral(optimizer: Mapping[str, Any]) -> dict[str, Any]:
    name = str(optimizer.get("name", "")).lower()
    family = "adam" if name.startswith("adam") else "sgdm" if name.startswith("sgd") else name
    excluded = {"name", "weight_decay", "total_weight_decay", "coupled_ratio"}
    return {
        "family": family,
        **{key: copy.deepcopy(value) for key, value in optimizer.items() if key not in excluded},
    }


def branch_neutral_paired_control_config(
    config: Mapping[str, Any], *, sibling_group_id: str, data_stream_id: str
) -> dict[str, Any]:
    training = config["training"]
    return {
        "protocol_id": TASK_F_PROTOCOL_ID,
        "study_id": TASK_F_STUDY_ID,
        "sibling_group_id": sibling_group_id,
        "data_stream_id": data_stream_id,
        "training_seed": int(training["seed"]),
        "dataset": {
            key: copy.deepcopy(config["dataset"][key])
            for key in ("protocol", "config_path", "train_split", "validation_split", "test_split")
        },
        "model": copy.deepcopy(config["model"]),
        "loss": copy.deepcopy(config["loss"]),
        "optimizer_shared": _optimizer_family_neutral(config["optimizer"]),
        "scheduler": copy.deepcopy(config["scheduler"]),
        "training": copy.deepcopy(dict(training)),
        "checkpoint": {"snapshot_epochs": list(config["checkpoint"]["snapshot_epochs"])},
    }


def build_task_f_training_config(
    base_config: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    audit_steps: Sequence[int] = (),
) -> dict[str, Any]:
    """Attach a run-plan record to the existing training runner configuration."""

    config = copy.deepcopy(dict(base_config))
    config["optimizer"] = copy.deepcopy(run["optimizer"])
    config["training"]["seed"] = int(run["training_seed"])
    config["training"]["max_epochs"] = 200
    config["checkpoint"]["snapshot_epochs"] = list(TASK_F_SNAPSHOT_EPOCHS)
    sibling_group_id = str(run["sibling_group_id"])
    data_stream_id = str(run["data_stream_id"])
    neutral = branch_neutral_paired_control_config(
        config,
        sibling_group_id=sibling_group_id,
        data_stream_id=data_stream_id,
    )
    config["task_f"] = {
        "schema_version": TASK_F_TRAINING_SCHEMA_VERSION,
        "protocol_id": TASK_F_PROTOCOL_ID,
        "study_id": TASK_F_STUDY_ID,
        "run_id": str(run["run_id"]),
        "sibling_group_id": sibling_group_id,
        "data_stream_id": data_stream_id,
        "branch_policy": str(run["branch_policy"]),
        "branch_neutral_config": neutral,
        "branch_neutral_config_sha256": canonical_sha256(neutral),
        "execution_only": bool(run["execution_only"]),
        "aggregate_eligible": bool(run["aggregate_eligible"]),
        "from_scratch": bool(run["from_scratch"]),
        "defer_id_test": True,
        "update_telemetry": {
            "schema_version": "task_f_update_telemetry_v1",
            "audit_steps": list(audit_steps),
            "include_batch_norm": True,
            "activation_residual_diagnostics": [],
        },
    }
    validate_task_f_training_config(config)
    return config


def generate_execution_only_pilot(
    *,
    base_config: Mapping[str, Any],
    seed: int,
    max_epochs: int,
    audit_steps: Sequence[int] = (),
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("pilot seed must be a non-negative integer")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or not 1 <= max_epochs < 200:
        raise ValueError("pilot max_epochs must be a positive shortened budget below 200")
    run = _record(
        family="adam",
        cell_id="execution_only_alpha_0_5_pilot",
        lr=1e-3,
        nominal_wd=1e-4,
        seed=seed,
        branch_policy="adam_mixed_alpha_0_5",
        sibling_group_id=f"task-f-execution-only-pilot-seed{seed}",
        optimizer=_adam_optimizer("adam_mixed_alpha_0_5", lr=1e-3, nominal_wd=1e-4),
        zero_reference_run_id=None,
    )
    run.update(
        {
            "run_id": f"task-f-execution-only-alpha0.5-seed{seed}-epochs{max_epochs}",
            "execution_only": True,
            "aggregate_eligible": False,
        }
    )
    config = build_task_f_training_config(base_config, run, audit_steps=audit_steps)
    config["training"]["max_epochs"] = max_epochs
    config["checkpoint"]["snapshot_epochs"] = [
        epoch for epoch in TASK_F_SNAPSHOT_EPOCHS if epoch <= max_epochs
    ]
    neutral = branch_neutral_paired_control_config(
        config,
        sibling_group_id=run["sibling_group_id"],
        data_stream_id=run["data_stream_id"],
    )
    config["task_f"]["branch_neutral_config"] = neutral
    config["task_f"]["branch_neutral_config_sha256"] = canonical_sha256(neutral)
    validate_task_f_training_config(config)
    return config


def research_aggregate_runs(runs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        run
        for run in runs
        if not bool(run.get("execution_only")) and bool(run.get("aggregate_eligible"))
    ]


def validate_no_protected_ood_references(value: object, path: str = "<root>") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            validate_no_protected_ood_references(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            validate_no_protected_ood_references(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if any(pattern.search(value) for pattern in _PROTECTED_PATTERNS):
            raise ValueError(f"protected OOD reference is forbidden at {path}")


def validate_task_f_training_config(config: Mapping[str, Any]) -> None:
    task_f = config.get("task_f")
    if not isinstance(task_f, Mapping):
        raise ValueError("Task F training config requires a task_f mapping")
    if task_f.get("schema_version") != TASK_F_TRAINING_SCHEMA_VERSION:
        raise ValueError("unsupported Task F training schema_version")
    required = {
        "run_id",
        "sibling_group_id",
        "data_stream_id",
        "branch_policy",
        "branch_neutral_config",
        "branch_neutral_config_sha256",
        "execution_only",
        "aggregate_eligible",
        "from_scratch",
        "defer_id_test",
        "update_telemetry",
    }
    missing = required.difference(task_f)
    if missing:
        raise ValueError(f"Task F training config is missing fields: {sorted(missing)}")
    if not bool(task_f["from_scratch"]):
        raise ValueError("Task F main/pilot runs must start from scratch")
    if not bool(task_f["defer_id_test"]):
        raise ValueError("Task F training must defer ID-test evaluation")
    if bool(task_f["execution_only"]) == bool(task_f["aggregate_eligible"]):
        raise ValueError("execution_only and aggregate_eligible must be logical opposites")
    neutral = task_f["branch_neutral_config"]
    if canonical_sha256(neutral) != task_f["branch_neutral_config_sha256"]:
        raise ValueError("Task F branch-neutral config hash mismatch")
    if neutral.get("sibling_group_id") != task_f["sibling_group_id"]:
        raise ValueError("Task F sibling_group_id differs from branch-neutral config")
    if neutral.get("data_stream_id") != task_f["data_stream_id"]:
        raise ValueError("Task F data_stream_id differs from branch-neutral config")
    if int(neutral.get("training_seed", -1)) != int(config["training"]["seed"]):
        raise ValueError("Task F training seed differs from branch-neutral config")
    telemetry = task_f["update_telemetry"]
    if not isinstance(telemetry, Mapping):
        raise ValueError("Task F update_telemetry must be a mapping")
    if telemetry.get("schema_version") != "task_f_update_telemetry_v1":
        raise ValueError("unsupported Task F update telemetry schema_version")
    steps = telemetry.get("audit_steps")
    if (
        not isinstance(steps, list)
        or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
        or steps != sorted(set(steps))
    ):
        raise ValueError("Task F audit_steps must be unique increasing positive steps")
    if not bool(task_f["execution_only"]) and tuple(config["checkpoint"]["snapshot_epochs"]) != TASK_F_SNAPSHOT_EPOCHS:
        raise ValueError("Task F research runs require the exact Card 13 v12 snapshot list")
    validate_no_protected_ood_references(
        {
            "run_id": task_f["run_id"],
            "branch_policy": task_f["branch_policy"],
            "dataset": {
                "config_path": config["dataset"]["config_path"],
                "train_split": config["dataset"]["train_split"],
                "validation_split": config["dataset"]["validation_split"],
            },
            "branch_neutral_dataset": neutral.get("dataset"),
        }
    )


def generate_approval_packet(
    *,
    execution_sha: str,
    pilot_config: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    task_f_b_specification_sha256: str | None = None,
    audit_schedule_frozen: bool = False,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", execution_sha):
        raise ValueError("execution_sha must be a lowercase 40-character Git SHA")
    calibration = dict(calibration or {})
    resource_keys = ("expected_gpu_hours", "expected_wall_time", "expected_storage")
    resources = {}
    for key in resource_keys:
        value = calibration.get(key, "UNKNOWN")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} calibration must be finite")
        resources[key] = value

    unresolved = [
        "accuracy_nll_ece_equivalence_margins",
        "exact_research_seed_ids",
        "gpu_resource_approval",
        "rtmd_gate_3_rule",
    ]
    if pilot_config is None:
        pilot = {
            "config_path": "UNRESOLVED",
            "training_seed": "UNRESOLVED",
            "max_epochs": "UNRESOLVED",
            "execution_only": True,
        }
        unresolved.extend(["exact_pilot_length", "pilot_seed"])
    else:
        validate_task_f_training_config(pilot_config)
        if not bool(pilot_config["task_f"]["execution_only"]):
            raise ValueError("approval packet pilot_config must be execution_only")
        pilot = {
            "config_path": "pilot/execution_only.yaml",
            "training_seed": int(pilot_config["training"]["seed"]),
            "max_epochs": int(pilot_config["training"]["max_epochs"]),
            "execution_only": True,
        }
    specification = task_f_b_specification_sha256 or "UNKNOWN"
    if specification != "UNKNOWN" and not re.fullmatch(r"[0-9a-f]{64}", specification):
        raise ValueError("Task F-B specification hash must be a lowercase SHA-256")
    if specification == "UNKNOWN":
        unresolved.append("task_f_b_specification_sha256")
    if not audit_schedule_frozen:
        unresolved.append("update_telemetry_audit_schedule")
    return {
        "schema_version": TASK_F_APPROVAL_PACKET_SCHEMA_VERSION,
        "protocol_id": TASK_F_PROTOCOL_ID,
        "execution_sha": execution_sha,
        "pilot": pilot,
        "resource_estimates": resources,
        "task_f_b_specification_sha256": specification,
        "approval_boundaries": {
            "pilot_approval": "REQUIRED_BEFORE_EXECUTION_ONLY_PILOT",
            "full_training_approval": "SEPARATE_REQUIRED_BEFORE_RESEARCH_TRAINING",
            "pilot_approval_does_not_authorize_full_training": True,
        },
        "unresolved_inputs": sorted(unresolved),
    }
