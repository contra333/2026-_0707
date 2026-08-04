"""Checkpoint-centric metric result records and atomic bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import DEFINITION_VERSION, REFERENCE_ATOL, REFERENCE_RTOL
from .extraction import sha256_file
from .registry import MetricDefinition


@dataclass(frozen=True)
class ResultIdentity:
    checkpoint_role: str
    checkpoint_sha256: str
    completed_epoch: int
    training_seed: int
    config_hash: str
    dataset_protocol: str
    membership_manifest_sha256: str
    evaluation_git_sha: str
    smoke_only: bool


def metric_result_record(
    *,
    definition: MetricDefinition,
    identity: ResultIdentity,
    value: float | int | None,
    status: str,
    reason_codes: Sequence[str],
    fit_split: str | None,
    query_split: str,
    transform: str,
    sample_count: int,
    class_counts: Sequence[int] | None,
    source_commit: str,
    runtime_diagnostics: Mapping[str, Any] | None = None,
    fixed_hyperparameters: Mapping[str, Any] | None = None,
    fit_dtype: str = "float64",
    evaluation_dtype: str = "float64",
) -> dict[str, Any]:
    if status not in {"success", "degenerate", "failed"}:
        raise ValueError(f"invalid metric status: {status}")
    if value is not None and not np.isfinite(value):
        raise ValueError("metric value must be finite or null")
    return {
        "metric_name": definition.artifact_key,
        "formula_id": definition.formula_id,
        "definition_version": DEFINITION_VERSION,
        "reporting_tier": definition.reporting_tier,
        **asdict(identity),
        "fit_split": fit_split,
        "query_split": query_split,
        "feature_endpoint": "model(x, return_features=True)",
        "transform": transform,
        "score_direction": definition.direction,
        "fit_dtype": fit_dtype,
        "evaluation_dtype": evaluation_dtype,
        "sample_count": int(sample_count),
        "class_counts": None if class_counts is None else [int(value) for value in class_counts],
        "fixed_hyperparameters": dict(fixed_hyperparameters or {}),
        "runtime_diagnostics": dict(runtime_diagnostics or {}),
        "source_commit": source_commit,
        "artifact_key": definition.artifact_key,
        "status": status,
        "reason_codes": list(reason_codes),
        "value": None if value is None else float(value),
        "tolerance": {"atol": REFERENCE_ATOL, "rtol": REFERENCE_RTOL},
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_metric_bundle(
    *,
    output_root: str | Path,
    identity: ResultIdentity,
    records: Sequence[Mapping[str, Any]],
    intermediates: Mapping[str, Mapping[str, np.ndarray]],
) -> Path:
    destination = Path(output_root) / identity.checkpoint_sha256
    if destination.exists():
        raise FileExistsError(f"metric bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{identity.checkpoint_sha256}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        payload = {
            "schema_version": "1.0",
            "definition_version": DEFINITION_VERSION,
            "identity": asdict(identity),
            "records": [_json_safe(dict(record)) for record in records],
        }
        (temporary / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        intermediate_root = temporary / "intermediates"
        intermediate_root.mkdir()
        index: dict[str, Any] = {}
        for artifact_key, arrays in sorted(intermediates.items()):
            safe_name = artifact_key.replace("/", "__").replace("{", "").replace("}", "")
            path = intermediate_root / f"{safe_name}.npz"
            np.savez(path, **arrays)
            index[artifact_key] = {
                "path": path.relative_to(temporary).as_posix(),
                "sha256": sha256_file(path),
                "arrays": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in arrays.items()
                },
            }
        (temporary / "intermediates.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_checksums(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
