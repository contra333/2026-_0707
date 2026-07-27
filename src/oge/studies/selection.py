"""Deterministic v1.2 role selection, reuse planning, and evidence gates."""

from __future__ import annotations

import copy
import itertools
import math
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from .hashing import canonical_sha256, scientific_config_hash
from .protocol import (
    GRID_TRAINING_SEED,
    OPTIMIZER_ORDER,
    PAIR_CONTROLS,
    ROLE_REPLICATION_SEEDS,
    SELECTION_POLICY,
    validate_grid_bundle,
)
from .schemas import seal_freeze, validate_freeze

TERMINAL_GRID_STATUSES = {
    "completed",
    "scientific_failed",
    "infrastructure_exhausted",
}
PROTECTED_EVIDENCE_FIELDS = {
    "id_test",
    "id_test_metrics",
    "ood",
    "ood_metrics",
    "geometry",
    "geometry_nc",
    "neural_collapse",
    "detector_metrics",
}


def _protected_fields(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in PROTECTED_EVIDENCE_FIELDS:
                found.add(str(key))
            found.update(_protected_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_protected_fields(child))
    return found


def grid_rank_key(record: Mapping[str, Any]) -> tuple[object, ...]:
    final = record["final_validation"]
    best = record["best_validation"]
    return (
        -float(final["accuracy"]),
        float(final["nll"]),
        int(best["epoch"]),
        int(record["lr_rank"]),
        int(record["weight_decay_rank"]),
        str(record["config_hash"]),
    )


def _validate_grid_results(
    records: Iterable[Mapping[str, Any]],
    grid_bundle: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    validate_grid_bundle(grid_bundle)
    rows = [
        row
        for optimizer in OPTIMIZER_ORDER
        for row in grid_bundle["tables"][optimizer]["rows"]
    ]
    expected = {str(row["trial_id"]): row for row in rows}
    values = [copy.deepcopy(dict(record)) for record in records]
    actual = {str(record.get("trial_id")): record for record in values}
    if len(actual) != len(values):
        raise ValueError("grid result trial IDs must be unique")
    if set(actual) != set(expected):
        raise ValueError("role freeze requires all and only the frozen 36 grid cells")
    grouped: dict[str, list[dict[str, Any]]] = {
        optimizer: [] for optimizer in OPTIMIZER_ORDER
    }
    for trial_id, row in expected.items():
        record = actual[trial_id]
        if record.get("status") not in TERMINAL_GRID_STATUSES:
            raise ValueError(f"cannot freeze roles before terminal accounting: {trial_id}")
        for key in (
            "optimizer_family",
            "assigned_slot",
            "config_hash",
            "lr_rank",
            "weight_decay_rank",
        ):
            if record.get(key) != row.get(key):
                raise ValueError(f"grid result identity differs at {trial_id}:{key}")
        if int(record.get("training_seed", -1)) != GRID_TRAINING_SEED:
            raise ValueError("role selection accepts only frozen seed-0 grid results")
        protected = _protected_fields(record)
        if protected:
            raise ValueError(
                "grid selection record contains protected downstream evidence: "
                f"{sorted(protected)}"
            )
        if record["status"] == "completed":
            final = record.get("final_validation")
            best = record.get("best_validation")
            if not isinstance(final, Mapping) or not isinstance(best, Mapping):
                raise ValueError("completed grid result is missing validation metrics")
            if int(final.get("epoch", -1)) != SELECTION_POLICY["checkpoint_epoch"]:
                raise ValueError("role selection requires last.pt epoch-200 validation")
            for label, metric in (("accuracy", final.get("accuracy")), ("nll", final.get("nll"))):
                if not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
                    raise ValueError(f"completed grid result has invalid final {label}")
            if not 0.0 <= float(final["accuracy"]) <= 1.0:
                raise ValueError("completed grid result accuracy must be in [0, 1]")
            grouped[row["optimizer_family"]].append(record)
    return grouped


def _valid_ranked(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    valid = [
        dict(record)
        for record in records
        if record.get("status") == "completed"
        and float(record["final_validation"]["accuracy"])
        >= SELECTION_POLICY["valid_accuracy_floor"]
    ]
    return sorted(valid, key=grid_rank_key)


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_optimizer = left["scientific_config"]["optimizer"]
    right_optimizer = right["scientific_config"]["optimizer"]
    lr_distance = abs(
        math.log10(float(left_optimizer["lr"]) / float(right_optimizer["lr"]))
    )
    left_wd = float(left_optimizer["weight_decay"])
    right_wd = float(right_optimizer["weight_decay"])
    if left_wd > 0.0 and right_wd > 0.0:
        wd_distance = abs(math.log10(left_wd / right_wd))
    elif (left_wd == 0.0) != (right_wd == 0.0):
        wd_distance = 2.0
    else:
        wd_distance = 0.0
    return lr_distance + wd_distance


def _select_c2(
    ranked: list[dict[str, Any]],
    c1: Mapping[str, Any],
) -> dict[str, Any]:
    for band in (
        SELECTION_POLICY["primary_band"],
        SELECTION_POLICY["widened_band"],
    ):
        candidates = [
            record
            for record in ranked
            if record["config_hash"] != c1["config_hash"]
            and abs(
                float(record["final_validation"]["accuracy"])
                - float(c1["final_validation"]["accuracy"])
            )
            <= band
        ]
        if candidates:
            candidates.sort(
                key=lambda record: (
                    -_distance(record, c1),
                    *grid_rank_key(record),
                )
            )
            return {
                "status": "selected",
                "band_used": band,
                "distance": _distance(candidates[0], c1),
                "cell": _cell_summary(candidates[0]),
            }
    return {
        "status": "absent",
        "band_used": SELECTION_POLICY["widened_band"],
        "distance": None,
        "cell": None,
    }


def _cell_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = record["scientific_config"]["optimizer"]
    return {
        "trial_id": record["trial_id"],
        "optimizer_family": record["optimizer_family"],
        "config_hash": record["config_hash"],
        "assigned_slot": record["assigned_slot"],
        "lr_rank": record["lr_rank"],
        "weight_decay_rank": record["weight_decay_rank"],
        "lr": optimizer["lr"],
        "weight_decay": optimizer["weight_decay"],
        "final_validation": copy.deepcopy(record["final_validation"]),
        "best_validation_epoch": int(record["best_validation"]["epoch"]),
    }


def _triple_summary(
    triple: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
) -> dict[str, Any]:
    accuracies = [float(record["final_validation"]["accuracy"]) for record in triple]
    nlls = [float(record["final_validation"]["nll"]) for record in triple]
    cells = {
        optimizer: _cell_summary(record)
        for optimizer, record in zip(OPTIMIZER_ORDER, triple)
    }
    return {
        "cells": cells,
        "accuracy_spread": max(accuracies) - min(accuracies),
        "mean_accuracy": statistics.fmean(accuracies),
        "mean_nll": statistics.fmean(nlls),
        "concatenated_config_hashes": "".join(
            cells[optimizer]["config_hash"] for optimizer in OPTIMIZER_ORDER
        ),
    }


def _qualifying_triples(
    ranked: Mapping[str, list[dict[str, Any]]],
    band: float,
) -> list[dict[str, Any]]:
    triples = [
        _triple_summary(triple)
        for triple in itertools.product(*(ranked[name] for name in OPTIMIZER_ORDER))
    ]
    eligible = [triple for triple in triples if triple["accuracy_spread"] <= band]
    eligible.sort(
        key=lambda triple: (
            -triple["mean_accuracy"],
            triple["mean_nll"],
            triple["concatenated_config_hashes"],
        )
    )
    return eligible


def freeze_grid_roles(
    records: Iterable[Mapping[str, Any]],
    grid_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze C1-C4 after every seed-0 grid cell is terminal."""
    grouped = _validate_grid_results(records, grid_bundle)
    ranked = {optimizer: _valid_ranked(grouped[optimizer]) for optimizer in OPTIMIZER_ORDER}
    per_optimizer: dict[str, Any] = {}
    for optimizer in OPTIMIZER_ORDER:
        c1 = ranked[optimizer][0] if ranked[optimizer] else None
        per_optimizer[optimizer] = {
            "valid_cell_count": len(ranked[optimizer]),
            "ranked_valid_cells": [
                _cell_summary(record) for record in ranked[optimizer]
            ],
            "c1": None if c1 is None else _cell_summary(c1),
            "c2": (
                {
                    "status": "absent",
                    "band_used": SELECTION_POLICY["widened_band"],
                    "distance": None,
                    "cell": None,
                }
                if c1 is None
                else _select_c2(ranked[optimizer], c1)
            ),
        }

    c3_candidates: list[dict[str, Any]] = []
    c3_band = None
    if all(ranked[optimizer] for optimizer in OPTIMIZER_ORDER):
        for band in (
            SELECTION_POLICY["primary_band"],
            SELECTION_POLICY["widened_band"],
        ):
            c3_candidates = _qualifying_triples(ranked, band)
            if c3_candidates:
                c3_band = band
                break
    c3 = c3_candidates[0] if c3_candidates else None
    c4 = None
    if c3 is not None:
        c4_candidates = [
            triple
            for triple in c3_candidates
            if triple["mean_accuracy"]
            <= c3["mean_accuracy"] - SELECTION_POLICY["c4_mean_gap"]
        ]
        c4 = c4_candidates[0] if c4_candidates else None

    role_labels: dict[str, list[str]] = {}
    for optimizer, roles in per_optimizer.items():
        if roles["c1"] is not None:
            role_labels.setdefault(roles["c1"]["config_hash"], []).append(
                f"{optimizer}:C1"
            )
        if roles["c2"]["cell"] is not None:
            role_labels.setdefault(roles["c2"]["cell"]["config_hash"], []).append(
                f"{optimizer}:C2"
            )
    for role_name, triple in (("C3", c3), ("C4", c4)):
        if triple is not None:
            for optimizer, cell in triple["cells"].items():
                role_labels.setdefault(cell["config_hash"], []).append(
                    f"{optimizer}:{role_name}"
                )
    role_labels = {
        config_hash: sorted(labels)
        for config_hash, labels in sorted(role_labels.items())
    }
    result_identity = [
        {
            "trial_id": record["trial_id"],
            "status": record["status"],
            "config_hash": record["config_hash"],
            "final_validation": copy.deepcopy(record.get("final_validation")),
            "best_validation": copy.deepcopy(record.get("best_validation")),
        }
        for optimizer in OPTIMIZER_ORDER
        for record in sorted(grouped[optimizer], key=lambda value: value["assigned_slot"])
    ]
    return seal_freeze(
        "role_selection",
        {
            "grid_manifest_hash": grid_bundle["manifest"]["manifest_hash"],
            "grid_result_identity_hash": canonical_sha256(result_identity),
            "selection_policy": copy.deepcopy(SELECTION_POLICY),
            "per_optimizer": per_optimizer,
            "c3": {
                "status": "selected" if c3 is not None else "unresolved",
                "band_used": c3_band,
                "triple": c3,
            },
            "c4": {
                "status": "selected" if c4 is not None else "omitted",
                "band_used": c3_band,
                "triple": c4,
            },
            "role_labels_by_config_hash": role_labels,
            "protected_evidence_released": True,
            "released_training_seeds": [
                GRID_TRAINING_SEED,
                *ROLE_REPLICATION_SEEDS,
            ],
        },
    )


def _grid_rows_by_hash(grid_bundle: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["config_hash"]: copy.deepcopy(row)
        for optimizer in OPTIMIZER_ORDER
        for row in grid_bundle["tables"][optimizer]["rows"]
    }


def _find_grid_row(
    grid_bundle: Mapping[str, Any],
    optimizer_family: str,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    matches = [
        row
        for row in grid_bundle["tables"][optimizer_family]["rows"]
        if float(row["scientific_config"]["optimizer"]["lr"]) == float(lr)
        and float(row["scientific_config"]["optimizer"]["weight_decay"])
        == float(weight_decay)
    ]
    if len(matches) != 1:
        raise ValueError("pair-control cell does not resolve uniquely in the grid")
    return copy.deepcopy(matches[0])


def _followup_row(
    grid_row: Mapping[str, Any],
    *,
    optimizer_family: str,
    training_seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(grid_row["scientific_config"])
    config["training"]["seed"] = training_seed
    if optimizer_family == "sgdw":
        config["optimizer"]["name"] = "sgdw"
    config_hash = scientific_config_hash(config)
    return {
        "trial_id": f"followup-{optimizer_family}-{config_hash[:12]}-seed-{training_seed}",
        "assigned_slot": None,
        "optimizer_family": optimizer_family,
        "lr_rank": grid_row["lr_rank"],
        "weight_decay_rank": grid_row["weight_decay_rank"],
        "training_seed": training_seed,
        "scientific_config": config,
        "config_hash": config_hash,
        "source_grid_config_hash": grid_row["config_hash"],
    }


def build_followup_plan(
    role_freeze: Mapping[str, Any],
    grid_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deduplicated role-replication and pair-control runs with explicit reuse."""
    validate_freeze(role_freeze, expected_kind="role_selection")
    validate_grid_bundle(grid_bundle)
    payload = role_freeze["payload"]
    if payload["grid_manifest_hash"] != grid_bundle["manifest"]["manifest_hash"]:
        raise ValueError("role freeze and grid manifest identities differ")
    rows_by_hash = _grid_rows_by_hash(grid_bundle)
    requested: dict[tuple[str, int], dict[str, Any]] = {}
    reuse: dict[tuple[str, int], dict[str, Any]] = {}

    def require(
        grid_row: Mapping[str, Any],
        optimizer_family: str,
        seed: int,
        label: str,
    ) -> None:
        candidate = _followup_row(
            grid_row,
            optimizer_family=optimizer_family,
            training_seed=seed,
        )
        key = (candidate["config_hash"], seed)
        target = reuse if optimizer_family != "sgdw" and seed == 0 else requested
        record = target.setdefault(
            key,
            {
                **candidate,
                "labels": [],
                "reuse_source": (
                    "seed0_grid" if optimizer_family != "sgdw" and seed == 0 else None
                ),
            },
        )
        record["labels"].append(label)

    for config_hash, labels in payload["role_labels_by_config_hash"].items():
        row = rows_by_hash[config_hash]
        for seed in (GRID_TRAINING_SEED, *ROLE_REPLICATION_SEEDS):
            for label in labels:
                require(row, row["optimizer_family"], seed, f"role:{label}")

    for cell in PAIR_CONTROLS["adam_adamw"]:
        for optimizer in ("adam", "adamw"):
            row = _find_grid_row(
                grid_bundle,
                optimizer,
                cell["lr"],
                cell["weight_decay"],
            )
            for seed in (GRID_TRAINING_SEED, *ROLE_REPLICATION_SEEDS):
                require(row, optimizer, seed, "pair:adam_adamw")

    anchor_a = PAIR_CONTROLS["sgd_sgdw"]["anchor_a"]
    sgd_a = _find_grid_row(
        grid_bundle,
        "sgd",
        anchor_a["lr"],
        anchor_a["weight_decay"],
    )
    ranked_sgd = [
        cell
        for cell in payload["per_optimizer"]["sgd"]["ranked_valid_cells"]
        if cell["config_hash"] != sgd_a["config_hash"]
    ]
    if not ranked_sgd:
        raise ValueError(
            "SGD pair-control cell B is unresolved because no ranked valid non-anchor cell exists"
        )
    sgd_b = rows_by_hash[ranked_sgd[0]["config_hash"]]
    for row, label in ((sgd_a, "pair:sgd_sgdw:A"), (sgd_b, "pair:sgd_sgdw:B")):
        for seed in (GRID_TRAINING_SEED, *ROLE_REPLICATION_SEEDS):
            require(row, "sgd", seed, label)
            require(row, "sgdw", seed, label)

    for collection in (requested, reuse):
        for record in collection.values():
            record["labels"] = sorted(set(record["labels"]))
    scheduled = sorted(
        requested.values(),
        key=lambda record: (
            record["optimizer_family"],
            record["source_grid_config_hash"],
            record["training_seed"],
        ),
    )
    reused = sorted(
        reuse.values(),
        key=lambda record: (
            record["optimizer_family"],
            record["source_grid_config_hash"],
            record["training_seed"],
        ),
    )
    for slot, record in enumerate(scheduled):
        record["assigned_slot"] = slot
    plan = {
        "role_freeze_hash": role_freeze["freeze_hash"],
        "grid_manifest_hash": grid_bundle["manifest"]["manifest_hash"],
        "scheduled": scheduled,
        "reused": reused,
    }
    plan["plan_hash"] = canonical_sha256(plan)
    return plan


def validate_protected_evidence_access(
    role_freeze: Mapping[str, Any],
    config_hash: str,
    training_seed: int,
) -> None:
    """Allow ID-test/OOD/geometry/detectors only for frozen role configurations."""
    validate_freeze(role_freeze, expected_kind="role_selection")
    payload = role_freeze["payload"]
    if not payload.get("protected_evidence_released"):
        raise ValueError("role freeze does not release protected evidence")
    if config_hash not in payload["role_labels_by_config_hash"]:
        raise ValueError("config is not part of the immutable role freeze")
    if training_seed not in payload["released_training_seeds"]:
        raise ValueError("seed is not part of the immutable role freeze")
