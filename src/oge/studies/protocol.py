"""Frozen protocol-v1.1 constants and discovery-table generation."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .hashing import canonical_json_bytes, canonical_sha256, scientific_config_hash

PROTOCOL_VERSION = "wrn28_10_optimizer_hpo_v1.1"
STUDY_CONFIG_SCHEMA_VERSION = "1.0"
DISCOVERY_TABLE_SCHEMA_VERSION = "1.0"
DISCOVERY_MANIFEST_SCHEMA_VERSION = "1.0"
SEARCH_SPACE_VERSION = "wrn28_10_optimizer_search_v1.1"
SAMPLER_NAME = "numpy_generator_pcg64"
SAMPLER_SEED = 0
DISCOVERY_TRAINING_SEED = 0
CONFIRMATION_TRAINING_SEEDS = [1, 2, 3]
FINAL_TRAINING_SEEDS = [4, 5, 6, 7, 8]
OPTIMIZER_ORDER = ["sgd", "sgdw", "adam", "adamw"]
ROWS_PER_OPTIMIZER = 16
RANDOM_ROWS_PER_OPTIMIZER = 14

SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "sgd": {
        "lr": [1e-2, 3e-1],
        "weight_decay": [1e-5, 1e-3],
        "positive_anchor": {"lr": 1e-1, "weight_decay": 5e-4},
        "no_decay_anchor": {"lr": 1e-1, "weight_decay": 0.0},
        "fixed": {"momentum": 0.9, "nesterov": True},
    },
    "sgdw": {
        "lr": [1e-2, 3e-1],
        "weight_decay": [1e-5, 2e-3],
        "positive_anchor": {"lr": 1e-1, "weight_decay": 5e-4},
        "no_decay_anchor": {"lr": 1e-1, "weight_decay": 0.0},
        "fixed": {"momentum": 0.9, "nesterov": True},
    },
    "adam": {
        "lr": [1e-4, 3e-3],
        "weight_decay": [1e-6, 1e-3],
        "positive_anchor": {"lr": 1e-3, "weight_decay": 1e-4},
        "no_decay_anchor": {"lr": 1e-3, "weight_decay": 0.0},
        "fixed": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    },
    "adamw": {
        "lr": [1e-4, 3e-3],
        "weight_decay": [1e-4, 1e-1],
        "positive_anchor": {"lr": 1e-3, "weight_decay": 1e-2},
        "no_decay_anchor": {"lr": 1e-3, "weight_decay": 0.0},
        "fixed": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    },
}


def load_study_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study config must contain a mapping")
    validate_study_config(payload)
    return payload


def validate_study_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != STUDY_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported study config schema_version")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("study config protocol_version does not match active protocol")
    sampler = config.get("sampler")
    if sampler != {"name": SAMPLER_NAME, "seed": SAMPLER_SEED}:
        raise ValueError("study sampler must be NumPy Generator(PCG64(0))")
    seeds = config.get("seeds")
    expected_seeds = {
        "discovery_training": DISCOVERY_TRAINING_SEED,
        "confirmation_training": CONFIRMATION_TRAINING_SEEDS,
        "final_training": FINAL_TRAINING_SEEDS,
    }
    if seeds != expected_seeds:
        raise ValueError("study training seed roles do not match protocol v1.1")
    seed_sets = [
        {DISCOVERY_TRAINING_SEED},
        set(CONFIRMATION_TRAINING_SEEDS),
        set(FINAL_TRAINING_SEEDS),
    ]
    if any(left.intersection(right) for index, left in enumerate(seed_sets) for right in seed_sets[index + 1 :]):
        raise ValueError("discovery, confirmation, and final training seeds must be disjoint")
    if config.get("optimizer_order") != OPTIMIZER_ORDER:
        raise ValueError("optimizer_order does not match the frozen protocol")
    if config.get("assigned_discovery_rows_per_optimizer") != ROWS_PER_OPTIMIZER:
        raise ValueError("assigned discovery budget must be 16 rows per optimizer")
    if config.get("search_spaces") != SEARCH_SPACES:
        raise ValueError("search spaces do not match protocol v1.1")
    if config.get("search_space_version") != SEARCH_SPACE_VERSION:
        raise ValueError("search_space_version does not match protocol v1.1")
    objective = config.get("objective")
    if objective != {
        "primary": "highest_best_id_validation_accuracy",
        "tie_breaks": [
            "lowest_corresponding_id_validation_nll",
            "earliest_best_validation_epoch",
            "ascending_trial_id",
            "ascending_canonical_config_hash",
        ],
    }:
        raise ValueError("discovery objective and tie-breaks do not match protocol v1.1")
    checkpoint_policy = config.get("checkpoint_policy")
    if checkpoint_policy != {
        "hpo_selection": "best_val.pt",
        "scientific_endpoint": "last.pt",
        "selection_snapshot_epochs": [],
        "final_snapshot_epochs": [0, 60, 61, 120, 121, 160, 161, 200],
    }:
        raise ValueError("checkpoint policy does not match protocol v1.1")
    declarations = config.get("selection_declarations")
    if declarations != {
        "id_test": "forbidden",
        "ood": "forbidden",
        "geometry_nc": "forbidden",
        "detector_metrics": "forbidden",
    }:
        raise ValueError("selection declarations must forbid non-validation evidence")


def _log_uniform(bounds: list[float], quantile: float) -> float:
    low, high = bounds
    return float(math.exp(math.log(low) + quantile * (math.log(high) - math.log(low))))


def _resolved_scientific_config(
    base_training_config: dict[str, Any],
    optimizer_name: str,
    sampled: dict[str, float],
) -> dict[str, Any]:
    config = copy.deepcopy(base_training_config)
    optimizer = {
        "name": optimizer_name,
        "lr": sampled["lr"],
        "weight_decay": sampled["weight_decay"],
        "weight_decay_policy": "weights_only_no_bias_norm",
        **SEARCH_SPACES[optimizer_name]["fixed"],
    }
    config["optimizer"] = optimizer
    config["training"]["max_epochs"] = 200
    config["training"]["batch_size"] = 128
    config["training"]["seed"] = DISCOVERY_TRAINING_SEED
    config["training"]["precision"] = "fp32"
    config["checkpoint"]["snapshot_epochs"] = []
    return config


def _table(
    optimizer_name: str,
    quantiles: np.ndarray,
    base_training_config: dict[str, Any],
) -> dict[str, Any]:
    space = SEARCH_SPACES[optimizer_name]
    sampled_rows = [space["positive_anchor"], space["no_decay_anchor"]]
    sampled_rows.extend(
        {
            "lr": _log_uniform(space["lr"], float(lr_q)),
            "weight_decay": _log_uniform(space["weight_decay"], float(wd_q)),
        }
        for lr_q, wd_q in quantiles
    )
    rows = []
    for slot, sampled in enumerate(sampled_rows):
        config = _resolved_scientific_config(base_training_config, optimizer_name, sampled)
        row = {
            "trial_id": f"discovery-{optimizer_name}-{slot:02d}",
            "assigned_slot": slot,
            "row_kind": "positive_anchor" if slot == 0 else "no_decay_anchor" if slot == 1 else "random_draw",
            "optimizer_family": optimizer_name,
            "training_seed": DISCOVERY_TRAINING_SEED,
            "scientific_config": config,
            "config_hash": scientific_config_hash(config),
        }
        row["row_hash"] = canonical_sha256(row)
        rows.append(row)
    table = {
        "schema_version": DISCOVERY_TABLE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "search_space_version": SEARCH_SPACE_VERSION,
        "sampler": {"name": SAMPLER_NAME, "seed": SAMPLER_SEED},
        "optimizer_family": optimizer_name,
        "assigned_rows": ROWS_PER_OPTIMIZER,
        "rows": rows,
    }
    table["table_hash"] = canonical_sha256(table)
    return table


def generate_discovery_bundle(
    study_config: dict[str, Any],
    base_training_config: dict[str, Any],
) -> dict[str, Any]:
    """Generate the four tables from one PCG64 stream and fixed pair quantiles."""
    validate_study_config(study_config)
    rng = np.random.Generator(np.random.PCG64(SAMPLER_SEED))
    sgd_pair_quantiles = rng.random((RANDOM_ROWS_PER_OPTIMIZER, 2))
    adam_pair_quantiles = rng.random((RANDOM_ROWS_PER_OPTIMIZER, 2))
    tables = {
        "sgd": _table("sgd", sgd_pair_quantiles, base_training_config),
        "sgdw": _table("sgdw", sgd_pair_quantiles, base_training_config),
        "adam": _table("adam", adam_pair_quantiles, base_training_config),
        "adamw": _table("adamw", adam_pair_quantiles, base_training_config),
    }
    manifest = {
        "schema_version": DISCOVERY_MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "search_space_version": SEARCH_SPACE_VERSION,
        "search_space_hash": canonical_sha256(SEARCH_SPACES),
        "sampler": {"name": SAMPLER_NAME, "seed": SAMPLER_SEED},
        "optimizer_order": OPTIMIZER_ORDER,
        "table_hashes": {name: tables[name]["table_hash"] for name in OPTIMIZER_ORDER},
        "assigned_rows_per_optimizer": ROWS_PER_OPTIMIZER,
        "total_assigned_rows": ROWS_PER_OPTIMIZER * len(OPTIMIZER_ORDER),
        "seed_roles": {
            "discovery_table_sampler": SAMPLER_SEED,
            "discovery_training": DISCOVERY_TRAINING_SEED,
            "confirmation_training": CONFIRMATION_TRAINING_SEEDS,
            "final_training": FINAL_TRAINING_SEEDS,
        },
        "selection_declarations": copy.deepcopy(study_config["selection_declarations"]),
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    bundle = {"tables": tables, "manifest": manifest}
    validate_discovery_bundle(bundle)
    return bundle


def _hash_without(payload: dict[str, Any], field: str) -> str:
    value = copy.deepcopy(payload)
    value.pop(field, None)
    return canonical_sha256(value)


def validate_discovery_bundle(bundle: dict[str, Any]) -> None:
    tables = bundle.get("tables")
    manifest = bundle.get("manifest")
    if not isinstance(tables, dict) or not isinstance(manifest, dict):
        raise ValueError("discovery bundle must contain tables and manifest mappings")
    if list(tables) != OPTIMIZER_ORDER:
        raise ValueError("discovery tables must use the frozen optimizer order")
    trial_ids: set[str] = set()
    for optimizer_name in OPTIMIZER_ORDER:
        table = tables[optimizer_name]
        if table.get("optimizer_family") != optimizer_name:
            raise ValueError("table optimizer_family mismatch")
        rows = table.get("rows")
        if not isinstance(rows, list) or len(rows) != ROWS_PER_OPTIMIZER:
            raise ValueError("each optimizer table must contain exactly 16 rows")
        if [row.get("row_kind") for row in rows[:2]] != ["positive_anchor", "no_decay_anchor"]:
            raise ValueError("anchors must be the first two ordered rows")
        for slot, row in enumerate(rows):
            if row.get("assigned_slot") != slot:
                raise ValueError("assigned slots must be ordered and contiguous")
            trial_id = row.get("trial_id")
            if not isinstance(trial_id, str) or trial_id in trial_ids:
                raise ValueError("trial IDs must be stable and globally unique")
            trial_ids.add(trial_id)
            if row.get("training_seed") != DISCOVERY_TRAINING_SEED:
                raise ValueError("discovery rows must use training seed 0")
            if row.get("config_hash") != scientific_config_hash(row["scientific_config"]):
                raise ValueError("discovery config hash mismatch")
            if row.get("row_hash") != _hash_without(row, "row_hash"):
                raise ValueError("discovery row hash mismatch")
            forbidden = {"id_test_metrics", "ood", "geometry", "neural_collapse", "detector_metrics"}
            if forbidden.intersection(row):
                raise ValueError("discovery rows contain forbidden evaluation fields")
        if table.get("table_hash") != _hash_without(table, "table_hash"):
            raise ValueError("ordered discovery table hash mismatch")
    if manifest.get("total_assigned_rows") != 64:
        raise ValueError("complete assigned discovery budget must be 64")
    expected_table_hashes = {name: tables[name]["table_hash"] for name in OPTIMIZER_ORDER}
    if manifest.get("table_hashes") != expected_table_hashes:
        raise ValueError("manifest table hashes do not match tables")
    if manifest.get("manifest_hash") != _hash_without(manifest, "manifest_hash"):
        raise ValueError("discovery manifest hash mismatch")


def materialize_discovery_bundle(bundle: dict[str, Any], destination: str | Path) -> None:
    validate_discovery_bundle(bundle)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for optimizer_name in OPTIMIZER_ORDER:
        path = root / f"{optimizer_name}.json"
        path.write_bytes(canonical_json_bytes(bundle["tables"][optimizer_name]) + b"\n")
    (root / "manifest.json").write_bytes(canonical_json_bytes(bundle["manifest"]) + b"\n")


def load_materialized_discovery_bundle(source: str | Path) -> dict[str, Any]:
    root = Path(source)
    tables = {
        name: json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        for name in OPTIMIZER_ORDER
    }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bundle = {"tables": tables, "manifest": manifest}
    validate_discovery_bundle(bundle)
    return bundle


def verify_materialized_discovery_bundle(
    source: str | Path,
    generated: dict[str, Any],
) -> None:
    """Require both semantic/hash parity and exact canonical bytes."""
    validate_discovery_bundle(generated)
    loaded = load_materialized_discovery_bundle(source)
    if loaded != generated:
        raise ValueError("materialized discovery bundle differs from deterministic generation")
    root = Path(source)
    for optimizer_name in OPTIMIZER_ORDER:
        expected = canonical_json_bytes(generated["tables"][optimizer_name]) + b"\n"
        if (root / f"{optimizer_name}.json").read_bytes() != expected:
            raise ValueError(f"{optimizer_name} frozen table is not byte-canonical")
    expected_manifest = canonical_json_bytes(generated["manifest"]) + b"\n"
    if (root / "manifest.json").read_bytes() != expected_manifest:
        raise ValueError("frozen discovery manifest is not byte-canonical")
