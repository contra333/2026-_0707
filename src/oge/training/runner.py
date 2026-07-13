"""Resolved configuration, artifacts, checkpointing, and epoch-boundary resume."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from torch import nn
from torch.utils.data import DataLoader

from oge.data import build_openood_cifar10_loaders, load_dataset_config
from oge.models import make_model
from oge.optimizers import make_optimizer
from oge.train_utils.param_groups import DEFAULT_WEIGHT_DECAY_POLICY

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    atomic_torch_save,
    capture_rng_state,
    load_torch_artifact,
    restore_rng_state,
)
from .engine import (
    current_learning_rates,
    evaluate_classifier,
    is_better_validation,
    make_scheduler,
    train_one_epoch,
)

RUN_SCHEMA_VERSION = "1.0"
PROTOCOL_NAME = "openood_v1_5_aligned_cifar10"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: object) -> None:
    _atomic_text_write(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_yaml(path: Path, payload: object) -> None:
    _atomic_text_write(path, yaml.safe_dump(payload, sort_keys=False))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON mapping in {path}")
    return payload


def _write_history(path: Path, history: list[dict[str, object]]) -> None:
    lines = [json.dumps(row, sort_keys=True, allow_nan=False) for row in history]
    _atomic_text_write(path, "".join(f"{line}\n" for line in lines))


def load_training_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("training config must contain a mapping")
    return config


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"training config must contain a {key!r} mapping")
    return value


def _materialize_config_defaults(config: dict[str, Any]) -> None:
    model = _require_mapping(config, "model")
    model.setdefault("num_classes", 10)
    model.setdefault("depth", 28)
    model.setdefault("widen_factor", 10)

    loss = _require_mapping(config, "loss")
    loss.setdefault("label_smoothing", 0.0)

    optimizer = _require_mapping(config, "optimizer")
    optimizer.setdefault("weight_decay_policy", DEFAULT_WEIGHT_DECAY_POLICY)
    optimizer_name = str(optimizer.get("name", "")).lower()
    if optimizer_name in {"sgd", "sgdw"}:
        optimizer.setdefault("momentum", 0.0)
        optimizer.setdefault("nesterov", False)
        optimizer.setdefault("weight_decay", 0.0)
    elif optimizer_name in {"adam", "adamw"}:
        optimizer.setdefault("beta1", 0.9)
        optimizer.setdefault("beta2", 0.999)
        optimizer.setdefault("eps", 1e-8)
        optimizer.setdefault("weight_decay", 0.0)
    elif optimizer_name == "sgd_coupled_decoupled":
        optimizer.setdefault("momentum", 0.0)
        optimizer.setdefault("nesterov", False)
        optimizer.setdefault("total_weight_decay", 0.0)
        optimizer.setdefault("coupled_ratio", 0.5)
    elif optimizer_name == "adam_coupled_decoupled":
        optimizer.setdefault("beta1", 0.9)
        optimizer.setdefault("beta2", 0.999)
        optimizer.setdefault("eps", 1e-8)
        optimizer.setdefault("total_weight_decay", 0.0)
        optimizer.setdefault("coupled_ratio", 0.5)

    scheduler = _require_mapping(config, "scheduler")
    scheduler.setdefault("name", "none")
    if str(scheduler["name"]).lower() == "multistep":
        scheduler.setdefault("gamma", 0.1)
        scheduler.setdefault("step_timing", "end_of_epoch")

    training = _require_mapping(config, "training")
    training.setdefault("pin_memory", False)
    training.setdefault("drop_last", False)
    training.setdefault("persistent_workers", False)
    training.setdefault("deterministic", False)


def _validate_training_config(config: dict[str, Any]) -> None:
    if str(config.get("schema_version")) != RUN_SCHEMA_VERSION:
        raise ValueError(f"training config schema_version must be {RUN_SCHEMA_VERSION!r}")
    dataset = _require_mapping(config, "dataset")
    model = _require_mapping(config, "model")
    loss = _require_mapping(config, "loss")
    optimizer = _require_mapping(config, "optimizer")
    scheduler = _require_mapping(config, "scheduler")
    training = _require_mapping(config, "training")
    checkpoint = _require_mapping(config, "checkpoint")

    if dataset.get("protocol") != PROTOCOL_NAME:
        raise ValueError(f"dataset.protocol must be {PROTOCOL_NAME!r}")
    for key in ("config_path", "train_split", "validation_split", "test_split"):
        if not dataset.get(key):
            raise ValueError(f"dataset.{key} is required")
    expected_splits = {
        "train_split": "id_train",
        "validation_split": "id_validation",
        "test_split": "id_test",
    }
    for key, expected in expected_splits.items():
        if dataset[key] != expected:
            raise ValueError(f"dataset.{key} must be {expected!r}")
    if model.get("name") != "wrn28_10" or float(model.get("dropout_rate", -1.0)) != 0.0:
        raise ValueError("the first training protocol requires wrn28_10 dropout_rate=0.0")
    if loss.get("name") != "cross_entropy":
        raise ValueError("only loss.name='cross_entropy' is supported")
    label_smoothing = float(loss.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("loss.label_smoothing must be in [0, 1)")
    if not optimizer.get("name") or "lr" not in optimizer:
        raise ValueError("optimizer.name and optimizer.lr are required")

    scheduler_name = str(scheduler.get("name", "none")).lower()
    if scheduler_name not in {"none", "multistep"}:
        raise ValueError("scheduler.name must be 'none' or 'multistep'")
    if scheduler_name == "multistep":
        milestones = scheduler.get("milestones")
        if (
            not isinstance(milestones, list)
            or not milestones
            or any(type(value) is not int or value <= 0 for value in milestones)
            or milestones != sorted(set(milestones))
        ):
            raise ValueError("scheduler.milestones must be unique increasing positive epochs")
        if float(scheduler.get("gamma", 0.0)) <= 0.0:
            raise ValueError("scheduler.gamma must be positive")
        if scheduler.get("step_timing") != "end_of_epoch":
            raise ValueError("scheduler.step_timing must be 'end_of_epoch'")

    integer_fields = ("max_epochs", "batch_size", "seed", "num_workers")
    for key in integer_fields:
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"training.{key} must be an integer")
    if training["max_epochs"] <= 0 or training["batch_size"] <= 0:
        raise ValueError("training.max_epochs and training.batch_size must be positive")
    if training["num_workers"] != 0:
        raise ValueError("the first training protocol requires training.num_workers=0")
    if training.get("precision") != "fp32":
        raise ValueError("the first training protocol requires training.precision='fp32'")
    for key in ("pin_memory", "drop_last", "persistent_workers", "deterministic"):
        if not isinstance(training.get(key), bool):
            raise ValueError(f"training.{key} must be boolean")
    if training["persistent_workers"]:
        raise ValueError("the first training protocol requires persistent_workers=false")

    snapshot_epochs = checkpoint.get("snapshot_epochs")
    if not isinstance(snapshot_epochs, list):
        raise ValueError("checkpoint.snapshot_epochs must be a list")
    if (
        any(type(value) is not int or value < 0 for value in snapshot_epochs)
        or snapshot_epochs != sorted(set(snapshot_epochs))
    ):
        raise ValueError(
            "checkpoint.snapshot_epochs must be unique increasing non-negative epochs"
        )


def _resolve_path(path: str | Path, *, repository_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repository_root / candidate


def _target_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
    return target


def _selected_membership(
    dataset_config: dict[str, Any],
    dataset_selection: dict[str, Any],
    *,
    data_root: Path,
) -> dict[str, dict[str, object]]:
    membership: dict[str, dict[str, object]] = {}
    datasets = dataset_config["datasets"]
    for role, selection_key in (
        ("train", dataset_selection["train_split"]),
        ("validation", dataset_selection["validation_split"]),
        ("test", dataset_selection["test_split"]),
    ):
        if selection_key not in datasets:
            raise ValueError(
                f"dataset definition does not contain selected split {selection_key!r}"
            )
        item = datasets[selection_key]
        expected_split = {"train": "train", "validation": "validation", "test": "test"}[
            role
        ]
        if (
            not bool(item.get("is_id"))
            or item.get("group") != "id"
            or item.get("dataset_name") != "cifar10"
            or item.get("split") != expected_split
        ):
            raise ValueError(f"selected {role} split does not match the CIFAR-10 ID role")
        imglist = Path(item["imglist"])
        content = (data_root / imglist).read_bytes()
        membership[role] = {
            "selection_key": selection_key,
            "imglist": imglist.as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "line_count": len(content.decode("utf-8").splitlines()),
        }
    return membership


def resolve_training_config(
    config: dict[str, Any],
    *,
    data_root: str | Path,
    device: str,
    max_epochs: int | None = None,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Validate and resolve the protocol config and immutable ID memberships."""
    resolved = copy.deepcopy(config)
    _materialize_config_defaults(resolved)
    if max_epochs is not None:
        _require_mapping(resolved, "training")["max_epochs"] = max_epochs
    _validate_training_config(resolved)

    root = Path(repository_root).resolve()
    actual_data_root = Path(data_root).resolve()
    dataset = resolved["dataset"]
    dataset_config_path = _resolve_path(dataset["config_path"], repository_root=root)
    dataset_definition = load_dataset_config(dataset_config_path)
    if dataset_definition.get("protocol_name") != dataset["protocol"]:
        raise ValueError("training and dataset protocol names do not match")
    dataset["config_path"] = str(dataset_config_path.resolve())
    dataset["data_root"] = str(actual_data_root)
    dataset["definition"] = dataset_definition
    dataset["membership"] = _selected_membership(
        dataset_definition,
        dataset,
        data_root=actual_data_root,
    )
    resolved["runtime"] = {"device": str(_target_device(device))}
    return resolved


def _semantic_resume_config(config: dict[str, Any]) -> dict[str, Any]:
    semantic = copy.deepcopy(config)
    semantic["training"].pop("max_epochs", None)
    semantic["checkpoint"].pop("snapshot_epochs", None)
    return semantic


def _first_config_difference(left: object, right: object, path: str = "") -> str:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left).union(right)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                return child_path
            difference = _first_config_difference(left[key], right[key], child_path)
            if difference:
                return difference
        return ""
    if left != right:
        return path or "<root>"
    return ""


def validate_resume_configuration(
    saved: dict[str, Any],
    current: dict[str, Any],
    *,
    completed_epoch: int,
) -> None:
    """Reject every resume change except max-epoch extension and future snapshots."""
    saved_semantic = _semantic_resume_config(saved)
    current_semantic = _semantic_resume_config(current)
    difference = _first_config_difference(saved_semantic, current_semantic)
    if difference:
        raise ValueError(
            f"Incompatible resume configuration: field {difference!r} changed; "
            "only max_epochs and future snapshots may change"
        )
    saved_max = int(saved["training"]["max_epochs"])
    current_max = int(current["training"]["max_epochs"])
    if current_max < saved_max or current_max < completed_epoch:
        raise ValueError(
            "Incompatible resume configuration: max_epochs must be unchanged or increased "
            "and must not precede completed_epoch"
        )
    saved_snapshots = set(int(value) for value in saved["checkpoint"]["snapshot_epochs"])
    current_snapshots = set(int(value) for value in current["checkpoint"]["snapshot_epochs"])
    if not saved_snapshots.issubset(current_snapshots):
        raise ValueError("Incompatible resume configuration: existing snapshots were removed")
    added = current_snapshots.difference(saved_snapshots)
    if any(epoch <= completed_epoch for epoch in added):
        raise ValueError(
            "Incompatible resume configuration: added snapshots must be after completed_epoch"
        )


def seed_everything(seed: int, *, deterministic: bool) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    return torch.Generator().manual_seed(seed)


def _optimizer_parameter_names(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group["params"]:
            try:
                group_names.append(names[id(parameter)])
            except KeyError as exc:
                raise ValueError("optimizer contains a parameter outside the model") from exc
        groups.append(group_names)
    return groups


def _environment(device: str, *, deterministic: bool) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "recorded_at": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": device,
        "deterministic_algorithms": deterministic,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if cuda_available
            else []
        ),
    }


def repository_git_state(repository_root: str | Path = REPOSITORY_ROOT) -> tuple[str, bool]:
    root = Path(repository_root)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, dirty


def _checkpoint_payload(
    *,
    checkpoint_type: str,
    completed_epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    best_validation: dict[str, float | int],
    train_generator: torch.Generator,
    resolved_config: dict[str, Any],
    oge_git_sha: str,
    run_id: str,
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": checkpoint_type,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_parameter_names": _optimizer_parameter_names(model, optimizer),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_validation": copy.deepcopy(best_validation),
        "rng_state": capture_rng_state(train_generator),
        "resolved_config": copy.deepcopy(resolved_config),
        "oge_git_sha": oge_git_sha,
        "run_id": run_id,
        "history": copy.deepcopy(history),
    }


def _snapshot_payload(
    *,
    completed_epoch: int,
    model: nn.Module,
    resolved_config: dict[str, Any],
    oge_git_sha: str,
    run_id: str,
) -> dict[str, object]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "checkpoint_type": "snapshot",
        "completed_epoch": completed_epoch,
        "model_state": model.state_dict(),
        "protocol_name": resolved_config["dataset"]["protocol"],
        "model_name": resolved_config["model"]["name"],
        "oge_git_sha": oge_git_sha,
        "run_id": run_id,
    }


def _validate_checkpoint(payload: dict[str, object]) -> None:
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
        "oge_git_sha",
        "run_id",
        "history",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"resume checkpoint is missing fields: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema_version")
    if payload["checkpoint_type"] != "last":
        raise ValueError("resume requires an epoch-boundary last.pt checkpoint")


def _restore_checkpoint(
    payload: dict[str, object],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_generator: torch.Generator,
) -> None:
    current_names = _optimizer_parameter_names(model, optimizer)
    if payload["optimizer_parameter_names"] != current_names:
        raise ValueError("optimizer parameter ordering differs from the checkpoint")
    model.load_state_dict(payload["model_state"], strict=True)
    scheduler_state = payload["scheduler_state"]
    if scheduler is None and scheduler_state is not None:
        raise ValueError("checkpoint has scheduler state but current scheduler is none")
    if scheduler is not None and scheduler_state is None:
        raise ValueError("checkpoint is missing scheduler state")
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    optimizer.load_state_dict(payload["optimizer_state"])
    restore_rng_state(payload["rng_state"], train_generator)


def _artifact_paths(run_dir: Path) -> dict[str, Path]:
    checkpoint_dir = run_dir / "checkpoints"
    return {
        "resolved_config": run_dir / "resolved_config.yaml",
        "metadata": run_dir / "run_metadata.json",
        "environment": run_dir / "environment.json",
        "history": run_dir / "history.jsonl",
        "summary": run_dir / "summary.json",
        "last": checkpoint_dir / "last.pt",
        "best": checkpoint_dir / "best_val.pt",
        "snapshots": checkpoint_dir / "snapshots",
        "evaluation": run_dir / "evaluation",
    }


def _prepare_artifacts(
    *,
    run_dir: Path,
    resolved_config: dict[str, Any],
    oge_git_sha: str,
    git_dirty: bool,
    resume_payload: dict[str, object] | None,
    resume_from: Path | None,
) -> tuple[dict[str, Path], str]:
    paths = _artifact_paths(run_dir)
    if resume_payload is None:
        run_dir.mkdir(parents=True, exist_ok=True)
        if any(run_dir.iterdir()):
            raise ValueError("fresh run_dir must be empty")
        run_id = str(uuid.uuid4())
        metadata = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "protocol_name": resolved_config["dataset"]["protocol"],
            "model_name": resolved_config["model"]["name"],
            "artifact_role": "classifier_training_run",
            "oge_git_sha": oge_git_sha,
            "git_dirty": git_dirty,
            "created_at": _utc_now(),
            "resume_events": [],
        }
        environments = {"schema_version": RUN_SCHEMA_VERSION, "executions": []}
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(resume_payload["run_id"])
        metadata = (
            _read_json(paths["metadata"])
            if paths["metadata"].is_file()
            else {
                "schema_version": RUN_SCHEMA_VERSION,
                "run_id": run_id,
                "protocol_name": resolved_config["dataset"]["protocol"],
                "model_name": resolved_config["model"]["name"],
                "artifact_role": "classifier_training_run",
                "oge_git_sha": oge_git_sha,
                "git_dirty": git_dirty,
                "created_at": _utc_now(),
                "resume_events": [],
            }
        )
        if metadata.get("run_id") != run_id:
            raise ValueError("run metadata and checkpoint run_id do not match")
        environments = (
            _read_json(paths["environment"])
            if paths["environment"].is_file()
            else {"schema_version": RUN_SCHEMA_VERSION, "executions": []}
        )
        metadata.setdefault("resume_events", []).append(
            {
                "resumed_at": _utc_now(),
                "checkpoint": str(resume_from),
                "completed_epoch": int(resume_payload["completed_epoch"]),
            }
        )

    paths["snapshots"].mkdir(parents=True, exist_ok=True)
    paths["evaluation"].mkdir(parents=True, exist_ok=True)
    _write_yaml(paths["resolved_config"], resolved_config)
    _write_json(paths["metadata"], metadata)
    environments.setdefault("executions", []).append(
        _environment(
            resolved_config["runtime"]["device"],
            deterministic=resolved_config["training"]["deterministic"],
        )
    )
    _write_json(paths["environment"], environments)
    return paths, run_id


def _evaluation_payload(
    *,
    role: str,
    checkpoint_path: Path,
    checkpoint: dict[str, object],
    metrics: dict[str, float | int],
    split: str,
) -> dict[str, object]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "checkpoint_role": role,
        "checkpoint_path": str(checkpoint_path),
        "completed_epoch": int(checkpoint["completed_epoch"]),
        "split": split,
        "sample_count": int(metrics["sample_count"]),
        "nll": float(metrics["nll"]),
        "accuracy": float(metrics["accuracy"]),
    }


def _reconcile_epoch_artifacts(
    *,
    paths: dict[str, Path],
    checkpoint: dict[str, object],
    model: nn.Module,
    resolved_config: dict[str, Any],
    oge_git_sha: str,
    run_id: str,
) -> None:
    """Finish artifacts belonging to the epoch committed by atomic last.pt."""
    completed_epoch = int(checkpoint["completed_epoch"])
    best_epoch = int(checkpoint["best_validation"]["epoch"])
    best_valid = False
    if paths["best"].is_file():
        try:
            best_payload = load_torch_artifact(paths["best"], map_location="cpu")
            best_valid = (
                best_payload.get("checkpoint_type") == "best_val"
                and int(best_payload.get("completed_epoch", -1)) == best_epoch
                and best_payload.get("run_id") == run_id
            )
        except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError):
            best_valid = False
    if not best_valid:
        if best_epoch != completed_epoch:
            raise ValueError("best_val.pt is missing or inconsistent with last.pt")
        best_payload = dict(checkpoint)
        best_payload["checkpoint_type"] = "best_val"
        atomic_torch_save(best_payload, paths["best"])

    snapshot_epochs = set(
        int(value) for value in resolved_config["checkpoint"]["snapshot_epochs"]
    )
    if completed_epoch in snapshot_epochs:
        snapshot_path = paths["snapshots"] / f"epoch_{completed_epoch:04d}.pt"
        snapshot_valid = False
        if snapshot_path.is_file():
            try:
                snapshot = load_torch_artifact(snapshot_path, map_location="cpu")
                snapshot_valid = (
                    snapshot.get("checkpoint_type") == "snapshot"
                    and int(snapshot.get("completed_epoch", -1)) == completed_epoch
                    and snapshot.get("run_id") == run_id
                )
            except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError):
                snapshot_valid = False
        if not snapshot_valid:
            atomic_torch_save(
                _snapshot_payload(
                    completed_epoch=completed_epoch,
                    model=model,
                    resolved_config=resolved_config,
                    oge_git_sha=oge_git_sha,
                    run_id=run_id,
                ),
                snapshot_path,
            )


def fit_classifier(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    train_generator: torch.Generator,
    resolved_config: dict[str, Any],
    run_dir: str | Path,
    device: str | torch.device,
    oge_git_sha: str,
    git_dirty: bool = False,
    resume_from: str | Path | None = None,
) -> dict[str, object]:
    """Run or resume the configured classifier training job."""
    target_device = _target_device(device)
    model_devices = {parameter.device for parameter in model.parameters()}
    model_devices.update(buffer.device for buffer in model.buffers())
    if model_devices and model_devices != {target_device}:
        raise ValueError(
            "model must be moved to the target device before optimizer construction"
        )
    output_dir = Path(run_dir)
    resume_path = Path(resume_from) if resume_from is not None else None
    if resume_path is not None:
        expected_resume_path = output_dir / "checkpoints/last.pt"
        if resume_path.resolve() != expected_resume_path.resolve():
            raise ValueError("resume checkpoint must be this run_dir's checkpoints/last.pt")
    resume_payload = (
        load_torch_artifact(resume_path, map_location="cpu") if resume_path is not None else None
    )

    if resume_payload is not None:
        _validate_checkpoint(resume_payload)
        validate_resume_configuration(
            resume_payload["resolved_config"],
            resolved_config,
            completed_epoch=int(resume_payload["completed_epoch"]),
        )
        if resume_payload["oge_git_sha"] != oge_git_sha:
            raise ValueError("resume checkpoint repository Git SHA differs from the current run")

    paths, run_id = _prepare_artifacts(
        run_dir=output_dir,
        resolved_config=resolved_config,
        oge_git_sha=oge_git_sha,
        git_dirty=git_dirty,
        resume_payload=resume_payload,
        resume_from=resume_path,
    )
    snapshot_epochs = set(int(value) for value in resolved_config["checkpoint"]["snapshot_epochs"])

    if resume_payload is None:
        completed_epoch = 0
        global_step = 0
        best_validation: dict[str, float | int] | None = None
        history: list[dict[str, object]] = []
        _write_history(paths["history"], history)
        if 0 in snapshot_epochs:
            atomic_torch_save(
                _snapshot_payload(
                    completed_epoch=0,
                    model=model,
                    resolved_config=resolved_config,
                    oge_git_sha=oge_git_sha,
                    run_id=run_id,
                ),
                paths["snapshots"] / "epoch_0000.pt",
            )
    else:
        _restore_checkpoint(
            resume_payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_generator=train_generator,
        )
        completed_epoch = int(resume_payload["completed_epoch"])
        global_step = int(resume_payload["global_step"])
        best_validation = copy.deepcopy(resume_payload["best_validation"])
        history = copy.deepcopy(resume_payload["history"])
        if len(history) != completed_epoch or (
            history and int(history[-1]["epoch"]) != completed_epoch
        ):
            raise ValueError("checkpoint history does not match completed_epoch")
        _write_history(paths["history"], history)
        _reconcile_epoch_artifacts(
            paths=paths,
            checkpoint=resume_payload,
            model=model,
            resolved_config=resolved_config,
            oge_git_sha=oge_git_sha,
            run_id=run_id,
        )

    max_epochs = int(resolved_config["training"]["max_epochs"])
    for epoch in range(completed_epoch + 1, max_epochs + 1):
        epoch_started = time.perf_counter()
        learning_rates = current_learning_rates(optimizer)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=target_device,
        )
        global_step += int(train_metrics["step_count"])
        validation_metrics = evaluate_classifier(
            model,
            validation_loader,
            criterion,
            device=target_device,
        )
        candidate = {
            "epoch": epoch,
            "accuracy": float(validation_metrics["accuracy"]),
            "nll": float(validation_metrics["nll"]),
        }
        became_best = is_better_validation(candidate, best_validation)
        if became_best:
            best_validation = candidate
        if scheduler is not None:
            scheduler.step()

        row: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": learning_rates[0],
            "learning_rates": learning_rates,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "validation_nll": float(validation_metrics["nll"]),
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "is_best": became_best,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        _write_history(paths["history"], history)
        assert best_validation is not None
        last_payload = _checkpoint_payload(
            checkpoint_type="last",
            completed_epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_validation=best_validation,
            train_generator=train_generator,
            resolved_config=resolved_config,
            oge_git_sha=oge_git_sha,
            run_id=run_id,
            history=history,
        )
        atomic_torch_save(last_payload, paths["last"])
        if became_best:
            best_payload = dict(last_payload)
            best_payload["checkpoint_type"] = "best_val"
            atomic_torch_save(best_payload, paths["best"])
        if epoch in snapshot_epochs:
            atomic_torch_save(
                _snapshot_payload(
                    completed_epoch=epoch,
                    model=model,
                    resolved_config=resolved_config,
                    oge_git_sha=oge_git_sha,
                    run_id=run_id,
                ),
                paths["snapshots"] / f"epoch_{epoch:04d}.pt",
            )

    final_checkpoint = load_torch_artifact(paths["last"], map_location="cpu")
    best_checkpoint = load_torch_artifact(paths["best"], map_location="cpu")
    evaluation_results: dict[str, dict[str, object]] = {}
    for role, checkpoint_path, checkpoint in (
        ("final", paths["last"], final_checkpoint),
        ("best_val", paths["best"], best_checkpoint),
    ):
        model.load_state_dict(checkpoint["model_state"], strict=True)
        metrics = evaluate_classifier(model, test_loader, criterion, device=target_device)
        payload = _evaluation_payload(
            role=role,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            metrics=metrics,
            split=resolved_config["dataset"]["test_split"],
        )
        filename = "final_id_test.json" if role == "final" else "best_val_id_test.json"
        _write_json(paths["evaluation"] / filename, payload)
        evaluation_results[role] = payload

    assert best_validation is not None
    final_history = history[-1]
    summary: dict[str, object] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id,
        "completed_epoch": max_epochs,
        "global_step": global_step,
        "final_validation": {
            "epoch": int(final_history["epoch"]),
            "accuracy": float(final_history["validation_accuracy"]),
            "nll": float(final_history["validation_nll"]),
        },
        "best_validation": best_validation,
        "final_id_test": evaluation_results["final"],
        "best_val_id_test": evaluation_results["best_val"],
        "checkpoint_paths": {
            "last": str(paths["last"]),
            "best_val": str(paths["best"]),
            "snapshots": str(paths["snapshots"]),
        },
        "artifact_paths": {
            "resolved_config": str(paths["resolved_config"]),
            "run_metadata": str(paths["metadata"]),
            "environment": str(paths["environment"]),
            "history": str(paths["history"]),
            "summary": str(paths["summary"]),
            "final_id_test": str(paths["evaluation"] / "final_id_test.json"),
            "best_val_id_test": str(paths["evaluation"] / "best_val_id_test.json"),
        },
    }
    _write_json(paths["summary"], summary)
    return summary


def run_training_from_config(
    *,
    config_path: str | Path,
    data_root: str | Path,
    run_dir: str | Path,
    device: str,
    resume_from: str | Path | None = None,
    max_epochs: int | None = None,
) -> dict[str, object]:
    """Resolve real protocol inputs, construct shared factories, and run training."""
    raw_config = load_training_config(config_path)
    resolved_config = resolve_training_config(
        raw_config,
        data_root=data_root,
        device=device,
        max_epochs=max_epochs,
    )
    training = resolved_config["training"]
    train_generator = seed_everything(
        int(training["seed"]),
        deterministic=bool(training["deterministic"]),
    )
    dataset_config = resolved_config["dataset"]["definition"]
    loaders = build_openood_cifar10_loaders(
        dataset_config,
        data_root=resolved_config["dataset"]["data_root"],
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
        pin_memory=bool(training["pin_memory"]),
        drop_last=bool(training["drop_last"]),
        persistent_workers=bool(training["persistent_workers"]),
        train_generator=train_generator,
    )
    target_device = _target_device(device)
    model = make_model(resolved_config["model"]).to(target_device)
    optimizer = make_optimizer(model, resolved_config["optimizer"])
    scheduler = make_scheduler(optimizer, resolved_config["scheduler"])
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(resolved_config["loss"].get("label_smoothing", 0.0))
    )
    id_loaders = loaders["id"]
    dataset = resolved_config["dataset"]
    git_sha, git_dirty = repository_git_state()
    return fit_classifier(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=id_loaders[dataset["train_split"]],
        validation_loader=id_loaders[dataset["validation_split"]],
        test_loader=id_loaders[dataset["test_split"]],
        train_generator=train_generator,
        resolved_config=resolved_config,
        run_dir=run_dir,
        device=device,
        oge_git_sha=git_sha,
        git_dirty=git_dirty,
        resume_from=resume_from,
    )
