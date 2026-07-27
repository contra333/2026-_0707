"""Frozen WRN-28-10 optimizer-grid protocol v1.2."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .hashing import canonical_json_bytes, canonical_sha256, scientific_config_hash

PROTOCOL_VERSION = "wrn28_10_optimizer_hpo_v1.2"
STUDY_CONFIG_SCHEMA_VERSION = "1.0"
GRID_TABLE_SCHEMA_VERSION = "1.0"
GRID_MANIFEST_SCHEMA_VERSION = "1.0"
GRID_VERSION = "wrn28_10_optimizer_grid_v1.2"
GRID_TRAINING_SEED = 0
ROLE_REPLICATION_SEEDS = [1, 2]
OPTIMIZER_ORDER = ["sgd", "adam", "adamw"]
ROWS_PER_OPTIMIZER = 12

GRIDS: dict[str, dict[str, Any]] = {
    "sgd": {
        "lr": [0.03, 0.1, 0.3],
        "weight_decay": [0.0, 1e-4, 5e-4, 1e-3],
        "fixed": {"momentum": 0.9, "nesterov": True},
    },
    "adam": {
        "lr": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-4, 5e-4, 1e-3],
        "fixed": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    },
    "adamw": {
        "lr": [3e-4, 1e-3, 3e-3],
        "weight_decay": [1e-4, 1e-3, 1e-2, 1e-1],
        "fixed": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    },
}

SELECTION_POLICY = {
    "valid_accuracy_floor": 0.90,
    "primary_band": 0.002,
    "widened_band": 0.005,
    "c4_mean_gap": 0.005,
    "checkpoint": "last.pt",
    "checkpoint_epoch": 200,
    "bands_are": "predeclared_operational_not_statistical_significance",
}

PAIR_CONTROLS = {
    "adam_adamw": [
        {"lr": 1e-3, "weight_decay": 1e-4},
        {"lr": 3e-3, "weight_decay": 1e-3},
    ],
    "sgd_sgdw": {
        "anchor_a": {"lr": 0.1, "weight_decay": 5e-4},
        "anchor_b": "highest_ranked_sgd_cell_other_than_anchor_a",
    },
}

OBJECTIVE = {
    "primary": "highest_last_epoch_200_id_validation_accuracy",
    "tie_breaks": [
        "lowest_same_checkpoint_id_validation_nll",
        "earliest_best_validation_epoch",
        "ascending_lr_rank",
        "ascending_weight_decay_rank",
        "ascending_canonical_config_hash",
    ],
}

CHECKPOINT_POLICY = {
    "selection": "last.pt_epoch_200",
    "best_validation_epoch": "best_val.pt_metadata_only",
    "snapshot_epochs": [0, 60, 61, 120, 121, 160, 161, 200],
}

SELECTION_DECLARATIONS = {
    "id_test": "forbidden_until_role_freeze",
    "ood": "forbidden_until_role_freeze",
    "geometry_nc": "forbidden_until_role_freeze",
    "detector_metrics": "forbidden_until_role_freeze",
}


def load_study_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study config must contain a mapping")
    validate_study_config(payload)
    return payload


def validate_study_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != STUDY_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported study config schema_version")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("study config protocol_version does not match active protocol")
    if config.get("grid_version") != GRID_VERSION:
        raise ValueError("grid_version does not match protocol v1.2")
    if config.get("optimizer_order") != OPTIMIZER_ORDER:
        raise ValueError("optimizer_order does not match protocol v1.2")
    if config.get("assigned_grid_rows_per_optimizer") != ROWS_PER_OPTIMIZER:
        raise ValueError("grid budget must be exactly 12 rows per optimizer")
    if config.get("grids") != GRIDS:
        raise ValueError("optimizer grids do not match protocol v1.2")
    if config.get("seeds") != {
        "grid_training": GRID_TRAINING_SEED,
        "role_replication": ROLE_REPLICATION_SEEDS,
    }:
        raise ValueError("study seed roles do not match protocol v1.2")
    if config.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("selection policy does not match protocol v1.2")
    if config.get("pair_controls") != PAIR_CONTROLS:
        raise ValueError("pair controls do not match protocol v1.2")
    if config.get("objective") != OBJECTIVE:
        raise ValueError("objective and tie-breaks do not match protocol v1.2")
    if config.get("checkpoint_policy") != CHECKPOINT_POLICY:
        raise ValueError("checkpoint policy does not match protocol v1.2")
    if config.get("selection_declarations") != SELECTION_DECLARATIONS:
        raise ValueError("protected evidence declarations do not match protocol v1.2")
    if not isinstance(config.get("base_training_config"), str):
        raise ValueError("base_training_config must be a repository-relative path")
    if not isinstance(config.get("frozen_grid_dir"), str):
        raise ValueError("frozen_grid_dir must be a repository-relative path")


def _portable_resolved_config(resolved: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(dict(resolved))
    if config.get("dataset", {}).get("protocol") != "oge_cifar10_holdout_v1":
        raise ValueError("v1.2 grid requires oge_cifar10_holdout_v1")
    dataset = config["dataset"]
    if not isinstance(dataset.get("membership"), Mapping):
        raise ValueError("base config must contain resolved dataset membership hashes")
    if not isinstance(dataset.get("membership_manifest"), Mapping):
        raise ValueError("base config must contain the resolved membership manifest")
    dataset["config_path"] = "configs/datasets/oge_cifar10_holdout_v1.yaml"
    for key in ("data_root", "imglist_config_root", "definition"):
        dataset.pop(key, None)
    config.pop("runtime", None)
    return config


def _resolved_grid_config(
    base_resolved_config: Mapping[str, Any],
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    config = _portable_resolved_config(base_resolved_config)
    config["optimizer"] = {
        "name": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "weight_decay_policy": "weights_only_no_bias_norm",
        **GRIDS[optimizer_name]["fixed"],
    }
    config["model"]["init_policy"] = "msr_fan_in"
    config["training"]["max_epochs"] = 200
    config["training"]["batch_size"] = 128
    config["training"]["seed"] = GRID_TRAINING_SEED
    config["training"]["precision"] = "fp32"
    config["checkpoint"]["snapshot_epochs"] = CHECKPOINT_POLICY["snapshot_epochs"]
    return config


def _hash_without(payload: Mapping[str, Any], field: str) -> str:
    value = copy.deepcopy(dict(payload))
    value.pop(field, None)
    return canonical_sha256(value)


def _table(
    optimizer_name: str,
    base_resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    slot = 0
    for lr_rank, lr in enumerate(GRIDS[optimizer_name]["lr"]):
        for weight_decay_rank, weight_decay in enumerate(
            GRIDS[optimizer_name]["weight_decay"]
        ):
            config = _resolved_grid_config(
                base_resolved_config,
                optimizer_name,
                lr,
                weight_decay,
            )
            row = {
                "trial_id": f"grid-{optimizer_name}-{slot:02d}",
                "assigned_slot": slot,
                "optimizer_family": optimizer_name,
                "lr_rank": lr_rank,
                "weight_decay_rank": weight_decay_rank,
                "training_seed": GRID_TRAINING_SEED,
                "scientific_config": config,
                "config_hash": scientific_config_hash(config),
            }
            row["row_hash"] = canonical_sha256(row)
            rows.append(row)
            slot += 1
    table = {
        "schema_version": GRID_TABLE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "grid_version": GRID_VERSION,
        "optimizer_family": optimizer_name,
        "assigned_rows": ROWS_PER_OPTIMIZER,
        "rows": rows,
    }
    table["table_hash"] = canonical_sha256(table)
    return table


def _dataset_identity(base_resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    portable = _portable_resolved_config(base_resolved_config)
    dataset = portable["dataset"]
    identity = {
        "protocol": dataset["protocol"],
        "membership": copy.deepcopy(dataset["membership"]),
        "membership_manifest": copy.deepcopy(dataset["membership_manifest"]),
    }
    identity["dataset_identity_hash"] = canonical_sha256(identity)
    return identity


def generate_grid_bundle(
    study_config: Mapping[str, Any],
    base_resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate the complete ordered 36-cell grid without random sampling."""
    validate_study_config(study_config)
    tables = {
        optimizer: _table(optimizer, base_resolved_config)
        for optimizer in OPTIMIZER_ORDER
    }
    manifest = {
        "schema_version": GRID_MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "grid_version": GRID_VERSION,
        "grid_hash": canonical_sha256(GRIDS),
        "optimizer_order": OPTIMIZER_ORDER,
        "table_hashes": {
            optimizer: tables[optimizer]["table_hash"]
            for optimizer in OPTIMIZER_ORDER
        },
        "assigned_rows_per_optimizer": ROWS_PER_OPTIMIZER,
        "total_assigned_rows": ROWS_PER_OPTIMIZER * len(OPTIMIZER_ORDER),
        "seed_roles": {
            "grid_training": GRID_TRAINING_SEED,
            "role_replication": ROLE_REPLICATION_SEEDS,
        },
        "dataset_identity": _dataset_identity(base_resolved_config),
        "selection_policy": copy.deepcopy(SELECTION_POLICY),
        "pair_controls": copy.deepcopy(PAIR_CONTROLS),
        "selection_declarations": copy.deepcopy(SELECTION_DECLARATIONS),
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    bundle = {"tables": tables, "manifest": manifest}
    validate_grid_bundle(bundle)
    return bundle


def validate_grid_bundle(bundle: Mapping[str, Any]) -> None:
    tables = bundle.get("tables")
    manifest = bundle.get("manifest")
    if not isinstance(tables, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("grid bundle must contain tables and manifest mappings")
    if list(tables) != OPTIMIZER_ORDER:
        raise ValueError("grid tables must use the frozen optimizer order")
    trial_ids: set[str] = set()
    config_hashes: set[str] = set()
    for optimizer_name in OPTIMIZER_ORDER:
        table = tables[optimizer_name]
        if table.get("optimizer_family") != optimizer_name:
            raise ValueError("table optimizer_family mismatch")
        rows = table.get("rows")
        if not isinstance(rows, list) or len(rows) != ROWS_PER_OPTIMIZER:
            raise ValueError("each optimizer table must contain exactly 12 rows")
        expected_pairs = [
            (lr, weight_decay)
            for lr in GRIDS[optimizer_name]["lr"]
            for weight_decay in GRIDS[optimizer_name]["weight_decay"]
        ]
        actual_pairs = []
        for slot, row in enumerate(rows):
            if row.get("assigned_slot") != slot:
                raise ValueError("assigned slots must be ordered and contiguous")
            trial_id = row.get("trial_id")
            if not isinstance(trial_id, str) or trial_id in trial_ids:
                raise ValueError("trial IDs must be stable and globally unique")
            trial_ids.add(trial_id)
            if row.get("training_seed") != GRID_TRAINING_SEED:
                raise ValueError("grid rows must use training seed 0")
            config = row.get("scientific_config")
            if row.get("config_hash") != scientific_config_hash(config):
                raise ValueError("grid config hash mismatch")
            if row["config_hash"] in config_hashes:
                raise ValueError("grid scientific config hashes must be globally unique")
            config_hashes.add(row["config_hash"])
            if row.get("row_hash") != _hash_without(row, "row_hash"):
                raise ValueError("grid row hash mismatch")
            if config["model"].get("init_policy") != "msr_fan_in":
                raise ValueError("grid row does not materialize msr_fan_in")
            if config["dataset"].get("protocol") != "oge_cifar10_holdout_v1":
                raise ValueError("grid row does not use the holdout protocol")
            if "runtime" in config or "data_root" in config["dataset"]:
                raise ValueError("grid rows must not freeze host-specific runtime paths")
            optimizer = config["optimizer"]
            actual_pairs.append((optimizer["lr"], optimizer["weight_decay"]))
            if row.get("lr_rank") != slot // 4 or row.get("weight_decay_rank") != slot % 4:
                raise ValueError("grid rank metadata does not match ordered cells")
            forbidden = {
                "id_test_metrics",
                "ood_metrics",
                "geometry",
                "neural_collapse",
                "detector_metrics",
            }
            if forbidden.intersection(row):
                raise ValueError("grid rows contain protected evaluation fields")
        if actual_pairs != expected_pairs:
            raise ValueError("grid rows do not match the exact declared Cartesian product")
        if table.get("table_hash") != _hash_without(table, "table_hash"):
            raise ValueError("ordered grid table hash mismatch")
    if manifest.get("total_assigned_rows") != 36:
        raise ValueError("complete assigned grid budget must be 36")
    expected_table_hashes = {
        optimizer: tables[optimizer]["table_hash"] for optimizer in OPTIMIZER_ORDER
    }
    if manifest.get("table_hashes") != expected_table_hashes:
        raise ValueError("manifest table hashes do not match tables")
    dataset_identity = manifest.get("dataset_identity")
    if not isinstance(dataset_identity, Mapping):
        raise ValueError("grid manifest is missing dataset identity")
    if dataset_identity.get("dataset_identity_hash") != _hash_without(
        dataset_identity, "dataset_identity_hash"
    ):
        raise ValueError("dataset identity hash mismatch")
    if manifest.get("manifest_hash") != _hash_without(manifest, "manifest_hash"):
        raise ValueError("grid manifest hash mismatch")


def materialize_grid_bundle(bundle: Mapping[str, Any], destination: str | Path) -> None:
    validate_grid_bundle(bundle)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for optimizer_name in OPTIMIZER_ORDER:
        (root / f"{optimizer_name}.json").write_bytes(
            canonical_json_bytes(bundle["tables"][optimizer_name]) + b"\n"
        )
    (root / "manifest.json").write_bytes(
        canonical_json_bytes(bundle["manifest"]) + b"\n"
    )


def load_materialized_grid_bundle(source: str | Path) -> dict[str, Any]:
    root = Path(source)
    tables = {
        name: json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        for name in OPTIMIZER_ORDER
    }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bundle = {"tables": tables, "manifest": manifest}
    validate_grid_bundle(bundle)
    return bundle


def verify_materialized_grid_bundle(
    source: str | Path,
    generated: Mapping[str, Any],
) -> None:
    """Require semantic, hash, and exact canonical-byte parity."""
    validate_grid_bundle(generated)
    loaded = load_materialized_grid_bundle(source)
    if loaded != generated:
        raise ValueError("materialized grid bundle differs from deterministic generation")
    root = Path(source)
    for optimizer_name in OPTIMIZER_ORDER:
        expected = canonical_json_bytes(generated["tables"][optimizer_name]) + b"\n"
        if (root / f"{optimizer_name}.json").read_bytes() != expected:
            raise ValueError(f"{optimizer_name} frozen table is not byte-canonical")
    expected_manifest = canonical_json_bytes(generated["manifest"]) + b"\n"
    if (root / "manifest.json").read_bytes() != expected_manifest:
        raise ValueError("frozen grid manifest is not byte-canonical")
