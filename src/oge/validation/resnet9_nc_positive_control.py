"""Lightweight ResNet9/MNIST Neural Collapse positive-control runner."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torchvision
import yaml
from torch import nn

from oge.data import build_mnist_positive_control_loaders
from oge.evaluation.geometry import (
    fit_geometry_statistics,
    neural_collapse_metrics,
    neural_collapse_structure_metrics,
)
from oge.models import make_model
from oge.optimizers import make_optimizer
from oge.training.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    load_torch_artifact,
    restore_rng_state,
)
from oge.training.engine import (
    current_learning_rates,
    evaluate_classifier,
    make_scheduler,
    train_one_epoch,
)

RESNET9_NC_POSITIVE_CONTROL_PROTOCOL = "resnet9_mnist_nc_positive_control_v1"
UPSTREAM_COMMIT = "7cab4a59bc28da6e356cee1e793ec67a694933b9"
PAPER_VERSION = "arXiv:2602.16642v3; ICLR 2026 published paper"
SCHEMA_VERSION = "1.0"
EXPECTED_RATIOS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
STRUCTURE_KEYS = (
    "nc0_row_sum_raw",
    "nc0_eq12_per_dim",
    "nc0_theory_squared",
    "nc2_equinorm",
    "nc2_equiangular",
    "nc2_etf_raw",
    "nc2_etf_eq5_scaled",
    "nc2w_etf_raw",
    "nc2w_equinorm",
    "nc2w_equiangular",
    "nc3_self_duality_raw",
    "nc3_eq10_scaled",
)
ENDPOINT_KEYS = STRUCTURE_KEYS + (
    "nc1_pinv",
    "nc1_svd_diagnostic",
    "nc1_trace_quotient_diagnostic",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    content = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    _atomic_write_text(path, content)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(config: Mapping[str, object], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def load_positive_control_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("positive-control config must contain a mapping")
    validate_positive_control_config(config)
    return config


def validate_positive_control_config(config: Mapping[str, object]) -> None:
    if str(config.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if config.get("protocol") != RESNET9_NC_POSITIVE_CONTROL_PROTOCOL:
        raise ValueError("unexpected positive-control protocol")

    source = _require_mapping(config, "source")
    if source.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("source.upstream_commit differs from the pinned audit commit")
    if source.get("paper_version") != PAPER_VERSION:
        raise ValueError("source.paper_version differs from the audited paper version")

    model = _require_mapping(config, "model")
    if model != {"name": "resnet9", "num_classes": 10, "in_channels": 1}:
        raise ValueError("model must be the pinned one-channel 10-class ResNet9")

    dataset = _require_mapping(config, "dataset")
    expected_dataset = {
        "name": "mnist",
        "train_count": 60000,
        "test_count": 10000,
        "resize": 32,
        "mean": [0.1307],
        "std": [0.3081],
        "augmentation": "none",
    }
    if dataset != expected_dataset:
        raise ValueError("dataset mapping differs from the pinned MNIST recipe")

    loss = _require_mapping(config, "loss")
    if loss != {"name": "cross_entropy", "label_smoothing": 0.0}:
        raise ValueError("loss must be unsmoothed cross entropy")

    optimizer = _require_mapping(config, "optimizer")
    expected_optimizer = {
        "name": "adam_coupled_decoupled",
        "lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "total_weight_decay": 0.0005,
        "coupled_ratios": list(EXPECTED_RATIOS),
        "weight_decay_policy": "all_parameters",
        "decoupled_update_order": "pre_adaptive_step_project_standard",
    }
    if optimizer != expected_optimizer:
        raise ValueError("optimizer mapping differs from the frozen interpolation")

    scheduler = _require_mapping(config, "scheduler")
    if scheduler != {
        "name": "multistep",
        "milestones": [66, 133],
        "gamma": 0.1,
        "step_timing": "end_of_epoch",
    }:
        raise ValueError("scheduler differs from int(E/3), int(2E/3) step decay")

    training = _require_mapping(config, "training")
    if training != {
        "max_epochs": 200,
        "batch_size": 128,
        "seeds": [3141],
        "num_workers": 0,
        "pin_memory": True,
        "precision": "fp32",
        "log_every": 13,
    }:
        raise ValueError("training mapping differs from the bounded one-seed protocol")


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    return torch.Generator().manual_seed(seed)


def _model_state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _repository_state(repository_root: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, dirty


def _runtime_environment(device: torch.device) -> dict[str, object]:
    cuda = torch.cuda.is_available()
    return {
        "recorded_at": _utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def _scalar_metrics(payload: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in keys:
        metric = payload[key]
        if not isinstance(metric, dict):
            raise ValueError(f"geometry metric {key!r} is not a metric record")
        result[key] = {
            "value": metric.get("value"),
            "status": metric.get("status"),
            "reason_codes": metric.get("reason_codes"),
        }
    return result


def _collect_train_geometry(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    collect_features: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    model.eval()
    sums = np.zeros((10, model.feature_dim), dtype=np.float64)
    counts = np.zeros(10, dtype=np.int64)
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["class_label"].numpy().astype(np.int64, copy=False)
            _, features = model(images, return_features=True)
            values = features.detach().cpu().numpy()
            for class_index in range(10):
                mask = labels == class_index
                if np.any(mask):
                    sums[class_index] += np.sum(values[mask], axis=0, dtype=np.float64)
                    counts[class_index] += int(np.count_nonzero(mask))
            if collect_features:
                feature_batches.append(values)
                label_batches.append(labels.copy())
    if np.any(counts == 0):
        raise ValueError("MNIST geometry collection found an empty class")
    means = sums / counts[:, None]
    if not collect_features:
        return means, None, None
    return means, np.concatenate(feature_batches), np.concatenate(label_batches)


def _collect_test_outputs(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            batch_logits, batch_features = model(images, return_features=True)
            features.append(batch_features.detach().cpu().numpy())
            logits.append(batch_logits.detach().cpu().numpy())
            labels.append(batch["class_label"].numpy().astype(np.int64, copy=False))
    return np.concatenate(features), np.concatenate(logits), np.concatenate(labels)


def _geometry_record(
    model: nn.Module,
    train_eval_loader,
    test_loader,
    criterion: nn.Module,
    *,
    device: torch.device,
    terminal: bool,
) -> dict[str, object]:
    means, train_features, train_labels = _collect_train_geometry(
        model,
        train_eval_loader,
        device=device,
        collect_features=terminal,
    )
    weight = model.classifier.weight.detach().cpu().numpy()
    structure = neural_collapse_structure_metrics(means, weight)
    if not terminal:
        test = evaluate_classifier(model, test_loader, criterion, device=device)
        return {
            "test_accuracy": float(test["accuracy"]),
            "test_nll": float(test["nll"]),
            "metrics": _scalar_metrics(structure, STRUCTURE_KEYS),
            "full_endpoint": False,
        }

    if train_features is None or train_labels is None:
        raise RuntimeError("terminal geometry did not retain training features")
    query_features, query_logits, query_labels = _collect_test_outputs(
        model,
        test_loader,
        device=device,
    )
    statistics = fit_geometry_statistics(train_features, train_labels, num_classes=10)
    full = neural_collapse_metrics(
        statistics,
        weight,
        model.classifier.bias.detach().cpu().numpy(),
        query_features=query_features,
        query_logits=query_logits,
        query_labels=query_labels,
    )
    test_accuracy = float(np.mean(np.argmax(query_logits, axis=1) == query_labels))
    endpoint = _scalar_metrics(full, ENDPOINT_KEYS)
    nc4 = full["nc4"]
    return {
        "test_accuracy": test_accuracy,
        "metrics": endpoint,
        "nc4": {
            "agreement_with_bias": nc4["agreement_with_bias"],
            "agreement_without_bias": nc4["agreement_without_bias"],
            "classifier_accuracy": nc4["classifier_accuracy"],
            "ncc_accuracy": nc4["ncc_accuracy"],
        },
        "class_counts": statistics.class_counts.tolist(),
        "nc1_retained_rank": full["nc1_diagnostics"]["retained_rank"],
        "full_endpoint": True,
    }


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_generator: torch.Generator,
    completed_epoch: int,
    global_step: int,
    resolved_config: dict[str, Any],
    history: list[dict[str, object]],
    trajectory: list[dict[str, object]],
    repository_sha: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": RESNET9_NC_POSITIVE_CONTROL_PROTOCOL,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "rng_state": capture_rng_state(train_generator),
        "resolved_config": resolved_config,
        "history": history,
        "trajectory": trajectory,
        "repository_sha": repository_sha,
    }


def run_resnet9_nc_positive_control(
    *,
    config_path: str | Path,
    data_root: str | Path,
    run_dir: str | Path,
    device: str,
    coupled_ratio: float,
    seed: int,
    max_epochs: int | None = None,
    download: bool = False,
    resume_from: str | Path | None = None,
) -> dict[str, object]:
    """Run one arm; multi-GPU orchestration intentionally stays outside this process."""
    started = time.monotonic()
    config = load_positive_control_config(config_path)
    allowed_ratios = tuple(float(value) for value in config["optimizer"]["coupled_ratios"])
    if not any(abs(coupled_ratio - value) < 1.0e-12 for value in allowed_ratios):
        raise ValueError(f"coupled_ratio must be one of {allowed_ratios}")
    if seed not in [int(value) for value in config["training"]["seeds"]]:
        raise ValueError("seed is not frozen in the positive-control config")
    scientific_epochs = int(config["training"]["max_epochs"])
    effective_epochs = scientific_epochs if max_epochs is None else int(max_epochs)
    if effective_epochs <= 0 or effective_epochs > scientific_epochs:
        raise ValueError("max_epochs must be in [1, training.max_epochs]")

    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() and resume_from is None:
        raise FileExistsError("run_dir already contains a completed summary")

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    repository_root = Path(__file__).resolve().parents[3]
    repository_sha, repository_dirty = _repository_state(repository_root)

    resolved = copy.deepcopy(config)
    resolved["arm"] = {
        "seed": seed,
        "coupled_ratio": float(coupled_ratio),
        "weight_decay_coupled": float(
            coupled_ratio * config["optimizer"]["total_weight_decay"]
        ),
        "weight_decay_decoupled": float(
            (1.0 - coupled_ratio) * config["optimizer"]["total_weight_decay"]
        ),
    }
    resolved["runtime"] = {
        "data_root": str(Path(data_root).expanduser().resolve()),
        "run_dir": str(output),
        "device": str(target),
        "effective_max_epochs": effective_epochs,
        "smoke_only": effective_epochs < scientific_epochs,
        "download": bool(download),
    }
    resolved_sha = _canonical_sha256(resolved)

    train_generator = _seed_everything(seed)
    train_loader, train_eval_loader, test_loader, sampler = (
        build_mnist_positive_control_loaders(
            data_root=data_root,
            batch_size=int(config["training"]["batch_size"]),
            train_generator=train_generator,
            num_workers=int(config["training"]["num_workers"]),
            pin_memory=bool(config["training"]["pin_memory"]),
            download=download,
        )
    )
    if len(train_loader.dataset) != 60000 or len(test_loader.dataset) != 10000:
        raise ValueError("MNIST membership counts differ from the frozen recipe")

    model = make_model(config["model"]).to(target)
    initial_model_digest = _model_state_digest(model)
    optimizer_config = copy.deepcopy(config["optimizer"])
    optimizer_config.pop("coupled_ratios")
    optimizer_config.pop("decoupled_update_order")
    optimizer_config["coupled_ratio"] = float(coupled_ratio)
    optimizer = make_optimizer(model, optimizer_config)
    scheduler = make_scheduler(optimizer, config["scheduler"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    history: list[dict[str, object]] = []
    trajectory: list[dict[str, object]] = []
    completed_epoch = 0
    global_step = 0
    if resume_from is not None:
        checkpoint = load_torch_artifact(resume_from, map_location=target)
        if checkpoint.get("protocol") != RESNET9_NC_POSITIVE_CONTROL_PROTOCOL:
            raise ValueError("resume checkpoint protocol mismatch")
        if checkpoint.get("resolved_config") != resolved:
            raise ValueError("resume resolved config mismatch")
        if checkpoint.get("repository_sha") != repository_sha:
            raise ValueError("resume repository SHA mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng_state(checkpoint["rng_state"], train_generator)
        completed_epoch = int(checkpoint["completed_epoch"])
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint["history"])
        trajectory = list(checkpoint["trajectory"])

    _atomic_write_text(output / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=False))
    _write_json(
        output / "run_metadata.json",
        {
            "protocol": RESNET9_NC_POSITIVE_CONTROL_PROTOCOL,
            "started_at": _utc_now(),
            "repository_sha": repository_sha,
            "repository_dirty": repository_dirty,
            "resolved_config_sha256": resolved_sha,
            "initial_model_state_sha256": initial_model_digest,
            "optimizer_parameter_group_count": len(optimizer.param_groups),
            "optimizer_parameter_counts": [
                int(sum(parameter.numel() for parameter in group["params"]))
                for group in optimizer.param_groups
            ],
            "environment": _runtime_environment(target),
        },
    )

    log_every = int(config["training"]["log_every"])
    first_epoch_order_digest: str | None = None
    for epoch in range(completed_epoch + 1, effective_epochs + 1):
        lr_used = current_learning_rates(optimizer)
        train = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=target,
            start_global_step=global_step,
        )
        global_step += int(train["step_count"])
        if scheduler is not None:
            scheduler.step()
        if sampler.last_order_digest is None:
            raise RuntimeError("training sampler did not record an epoch order")
        if epoch == 1:
            first_epoch_order_digest = sampler.last_order_digest

        row: dict[str, object] = {
            "completed_epoch": epoch,
            "global_step": global_step,
            "lr_used": lr_used,
            "train_loss": float(train["loss"]),
            "train_accuracy": float(train["accuracy"]),
            "train_sample_count": int(train["sample_count"]),
            "train_order_sha256": sampler.last_order_digest,
        }
        should_measure = (epoch - 1) % log_every == 0 or epoch == effective_epochs
        if should_measure:
            terminal = epoch == effective_epochs
            geometry = _geometry_record(
                model,
                train_eval_loader,
                test_loader,
                criterion,
                device=target,
                terminal=terminal,
            )
            point = {"completed_epoch": epoch, **geometry}
            trajectory.append(point)
            row["geometry"] = geometry
            atomic_torch_save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_generator=train_generator,
                    completed_epoch=epoch,
                    global_step=global_step,
                    resolved_config=resolved,
                    history=history + [row],
                    trajectory=trajectory,
                    repository_sha=repository_sha,
                ),
                output / "last.pt",
            )
        history.append(row)
        _write_jsonl(output / "history.jsonl", history)
        _write_json(output / "trajectory.json", trajectory)
        print(
            f"epoch={epoch}/{effective_epochs} loss={train['loss']:.6f} "
            f"acc={train['accuracy']:.6f} lr={lr_used[0]:.8g}",
            flush=True,
        )

    terminal = trajectory[-1]
    summary = {
        "status": "completed",
        "protocol": RESNET9_NC_POSITIVE_CONTROL_PROTOCOL,
        "completed_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "completed_epoch": effective_epochs,
        "global_step": global_step,
        "smoke_only": effective_epochs < scientific_epochs,
        "seed": seed,
        "coupled_ratio": float(coupled_ratio),
        "repository_sha": repository_sha,
        "repository_dirty": repository_dirty,
        "resolved_config_sha256": resolved_sha,
        "initial_model_state_sha256": initial_model_digest,
        "first_epoch_train_order_sha256": first_epoch_order_digest,
        "terminal": terminal,
    }
    _write_json(output / "summary.json", summary)
    return summary
