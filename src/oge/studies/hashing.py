"""Canonical JSON and scientific-configuration hashing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def _reject_non_finite(value: object, path: str = "<root>") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_non_finite(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON with recursively sorted keys and no non-finite values."""
    _reject_non_finite(value)
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical-JSON serializable") from exc
    return text.encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that define the scientific training configuration."""
    required = {"optimizer", "model", "loss", "dataset", "scheduler", "training"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"scientific config is missing sections: {sorted(missing)}")

    dataset = config["dataset"]
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset must be a mapping")
    logical_dataset_keys = (
        "protocol",
        "train_split",
        "validation_split",
        "test_split",
    )
    logical_dataset = {key: copy.deepcopy(dataset[key]) for key in logical_dataset_keys}

    training = config["training"]
    if not isinstance(training, Mapping):
        raise ValueError("training must be a mapping")
    scientific_training = {
        key: copy.deepcopy(value)
        for key, value in training.items()
        if key != "seed"
    }

    return {
        "optimizer": copy.deepcopy(config["optimizer"]),
        "model": copy.deepcopy(config["model"]),
        "loss": copy.deepcopy(config["loss"]),
        "dataset": logical_dataset,
        "scheduler": copy.deepcopy(config["scheduler"]),
        "training": scientific_training,
    }


def scientific_config_hash(config: Mapping[str, Any]) -> str:
    return canonical_sha256(scientific_config(config))


def provenance_identity_hash(
    *,
    git_sha: str,
    dataset_membership_hashes: Mapping[str, str],
) -> str:
    """Hash execution provenance separately from scientific config identity."""
    if not git_sha or not dataset_membership_hashes:
        raise ValueError("Git SHA and dataset membership hashes are required provenance")
    return canonical_sha256(
        {
            "git_sha": git_sha,
            "dataset_membership_hashes": dict(dataset_membership_hashes),
        }
    )
