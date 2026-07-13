"""MSP inference and bounded OOD artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .metrics import METRIC_DEFINITIONS, compute_ood_metrics

OPENOOD_SOURCE_COMMIT = "3c35632ee91b54b09d1f085d04f94744cece7d0b"
SCHEMA_VERSION = "1.0"
PROTOCOL_NAME = "openood_v1_5_aligned_cifar10"


def infer_msp(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    target_device = torch.device(device)
    output: dict[str, list] = {
        "sample_id": [],
        "prediction": [],
        "class_label": [],
        "is_id": [],
        "id_like_score": [],
    }
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(target_device))
            probabilities = torch.softmax(logits, dim=1)
            scores, predictions = probabilities.max(dim=1)
            output["sample_id"].extend(batch["sample_id"])
            output["prediction"].extend(predictions.cpu().tolist())
            output["class_label"].extend(batch["class_label"].cpu().tolist())
            output["is_id"].extend(batch["is_id"].cpu().tolist())
            output["id_like_score"].extend(scores.cpu().tolist())
    return {
        "sample_id": np.asarray(output["sample_id"], dtype=str),
        "prediction": np.asarray(output["prediction"], dtype=np.int64),
        "class_label": np.asarray(output["class_label"], dtype=np.int64),
        "is_id": np.asarray(output["is_id"], dtype=np.bool_),
        "id_like_score": np.asarray(output["id_like_score"], dtype=np.float64),
    }


def _save_scores(path: Path, scores: Mapping[str, np.ndarray]) -> None:
    np.savez(path, **scores)


def _mean_metrics(per_dataset: dict[str, dict[str, float]], names: list[str]) -> dict[str, float]:
    if not names:
        raise ValueError("metric group must contain at least one dataset")
    metric_names = per_dataset[names[0]].keys()
    return {
        metric_name: float(np.mean([per_dataset[name][metric_name] for name in names]))
        for metric_name in metric_names
    }


def evaluate_msp_protocol(
    model: nn.Module,
    loaders: dict[str, object],
    *,
    resolved_config: dict,
    output_dir: str | Path,
    model_name: str,
    model_is_random_or_untrained: bool,
    device: str,
    oge_git_sha: str,
) -> dict[str, object]:
    output_path = Path(output_dir)
    score_dir = output_path / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    model.to(torch.device(device))

    id_loaders = loaders["id"]
    if "id_test" not in id_loaders:
        raise ValueError("dataset config must define an 'id_test' loader")
    id_scores = infer_msp(model, id_loaders["id_test"], device=device)
    _save_scores(score_dir / "cifar10.npz", id_scores)

    per_dataset: dict[str, dict[str, float]] = {}
    group_names: dict[str, list[str]] = {"near": [], "far": []}
    for group in ("near", "far"):
        for dataset_key, loader in loaders[group].items():
            scores = infer_msp(model, loader, device=device)
            _save_scores(score_dir / f"{dataset_key}.npz", scores)
            per_dataset[dataset_key] = compute_ood_metrics(
                id_scores["id_like_score"], scores["id_like_score"]
            )
            group_names[group].append(dataset_key)

    metrics_payload: dict[str, object] = {
        "per_dataset": per_dataset,
        "near_mean": _mean_metrics(per_dataset, group_names["near"]),
        "far_mean": _mean_metrics(per_dataset, group_names["far"]),
        "metric_definitions": METRIC_DEFINITIONS,
        "smoke_only": bool(model_is_random_or_untrained),
    }
    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "openood_source_commit": OPENOOD_SOURCE_COMMIT,
        "oge_git_sha": oge_git_sha,
        "model_name": model_name,
        "model_is_random_or_untrained": bool(model_is_random_or_untrained),
        "score_name": "msp",
        "device": device,
    }
    with (output_path / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)
    with (output_path / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True)
    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, sort_keys=True)
    return metrics_payload
