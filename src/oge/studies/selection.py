"""Deterministic ranking, matching, pair-control, and freeze rules."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from typing import Any

from .protocol import CONFIRMATION_TRAINING_SEEDS, FINAL_TRAINING_SEEDS
from .schemas import seal_freeze, validate_freeze

TERMINAL_DISCOVERY_STATUSES = {"completed", "scientific_failed", "infrastructure_exhausted"}
PAIR_CONTROL_ANCHORS = {
    "sgd_sgdw": [
        {"lr": lr, "weight_decay": weight_decay}
        for lr in (0.03, 0.1)
        for weight_decay in (1e-4, 5e-4)
    ],
    "adam_adamw": [
        {"lr": lr, "weight_decay": weight_decay}
        for lr in (3e-4, 1e-3)
        for weight_decay in (1e-5, 1e-4)
    ],
}


def discovery_rank_key(record: Mapping[str, Any]) -> tuple[object, ...]:
    best = record["best_validation"]
    return (
        -float(best["accuracy"]),
        float(best["nll"]),
        int(best["epoch"]),
        str(record["trial_id"]),
        str(record["config_hash"]),
    )


def rank_discovery(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    completed = [dict(record) for record in records if record.get("status") == "completed"]
    return sorted(completed, key=discovery_rank_key)


def freeze_discovery_top_k(
    optimizer_family: str,
    records: Iterable[Mapping[str, Any]],
    assigned_trial_ids: Iterable[str],
    *,
    k: int = 3,
) -> dict[str, Any]:
    records_by_id = {str(record["trial_id"]): dict(record) for record in records}
    assigned = list(assigned_trial_ids)
    if len(assigned) != 16 or len(set(assigned)) != 16:
        raise ValueError("protocol v1.1 requires exactly 16 unique assigned discovery slots")
    if set(records_by_id) != set(assigned):
        raise ValueError("all and only assigned discovery slots must be accounted for")
    nonterminal = [
        trial_id
        for trial_id in assigned
        if records_by_id[trial_id].get("status") not in TERMINAL_DISCOVERY_STATUSES
    ]
    if nonterminal:
        raise ValueError(f"cannot freeze top-k before terminal accounting: {nonterminal}")
    ranked = rank_discovery(records_by_id.values())
    selected = ranked[:k]
    return seal_freeze(
        "discovery_top_k",
        {
            "optimizer_family": optimizer_family,
            "assigned_trial_ids": assigned,
            "completed_count": len(ranked),
            "requested_k": k,
            "selected": [
                {"trial_id": item["trial_id"], "config_hash": item["config_hash"]}
                for item in selected
            ],
            "replacement_trials": [],
        },
    )


def freeze_discovery_tables(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("total_assigned_rows") != 64:
        raise ValueError("discovery table freeze requires exactly 64 assigned rows")
    table_hashes = manifest.get("table_hashes")
    if not isinstance(table_hashes, Mapping) or set(table_hashes) != {"sgd", "sgdw", "adam", "adamw"}:
        raise ValueError("discovery table freeze requires all four optimizer table hashes")
    return seal_freeze(
        "discovery_tables",
        {
            "manifest_hash": manifest["manifest_hash"],
            "table_hashes": dict(table_hashes),
            "total_assigned_rows": 64,
        },
    )


def _confirmation_aggregate(record: Mapping[str, Any]) -> dict[str, Any] | None:
    observations = record.get("observations")
    if not isinstance(observations, list):
        return None
    by_seed = {int(item["training_seed"]): item for item in observations}
    if set(by_seed) != set(CONFIRMATION_TRAINING_SEEDS):
        return None
    if any(item.get("status") != "completed" for item in by_seed.values()):
        return None
    ordered = [by_seed[seed] for seed in CONFIRMATION_TRAINING_SEEDS]
    return {
        "trial_id": record["trial_id"],
        "config_hash": record["config_hash"],
        "mean_best_accuracy": statistics.fmean(float(item["best_validation"]["accuracy"]) for item in ordered),
        "mean_best_nll": statistics.fmean(float(item["best_validation"]["nll"]) for item in ordered),
        "mean_best_epoch": statistics.fmean(float(item["best_validation"]["epoch"]) for item in ordered),
        "mean_terminal_accuracy": statistics.fmean(float(item["terminal_validation"]["accuracy"]) for item in ordered),
        "mean_terminal_nll": statistics.fmean(float(item["terminal_validation"]["nll"]) for item in ordered),
        "terminal_accuracy_std": statistics.pstdev(float(item["terminal_validation"]["accuracy"]) for item in ordered),
    }


def confirmation_rank_key(aggregate: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        -float(aggregate["mean_best_accuracy"]),
        float(aggregate["mean_best_nll"]),
        float(aggregate["mean_best_epoch"]),
        str(aggregate["trial_id"]),
        str(aggregate["config_hash"]),
    )


def rank_confirmation(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aggregates = [_confirmation_aggregate(record) for record in records]
    return sorted(
        [aggregate for aggregate in aggregates if aggregate is not None],
        key=confirmation_rank_key,
    )


def freeze_tuned_winner(
    optimizer_family: str,
    confirmation_records: Iterable[Mapping[str, Any]],
    top_k_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    validate_freeze(top_k_freeze, expected_kind="discovery_top_k")
    allowed = {item["config_hash"] for item in top_k_freeze["payload"]["selected"]}
    records = list(confirmation_records)
    if any(record["config_hash"] not in allowed for record in records):
        raise ValueError("confirmation record is outside the frozen top-k")
    ranked = rank_confirmation(records)
    winner = ranked[0] if ranked else None
    return seal_freeze(
        "tuned_winner",
        {
            "optimizer_family": optimizer_family,
            "top_k_freeze_hash": top_k_freeze["freeze_hash"],
            "eligible_ranked": ranked,
            "winner": winner,
        },
    )


def freeze_accuracy_matching(
    tuned_winner_freezes: Iterable[Mapping[str, Any]],
    confirmed_candidates: Mapping[str, Iterable[Mapping[str, Any]]],
    candidate_top_k_freezes: Mapping[str, Mapping[str, Any]],
    *,
    tolerance: float = 0.002,
) -> dict[str, Any]:
    if tolerance != 0.002:
        raise ValueError("protocol v1.1 accuracy tolerance is fixed at 0.002")
    winners: dict[str, Mapping[str, Any]] = {}
    winner_freeze_hashes: dict[str, str] = {}
    for freeze in tuned_winner_freezes:
        validate_freeze(freeze, expected_kind="tuned_winner")
        family = str(freeze["payload"]["optimizer_family"])
        winner = freeze["payload"]["winner"]
        if winner is not None:
            winners[family] = winner
        winner_freeze_hashes[family] = str(freeze["freeze_hash"])
    target = min((float(item["mean_terminal_accuracy"]) for item in winners.values()), default=None)
    selections: dict[str, Any] = {}
    frozen_candidates: dict[str, Any] = {}
    for family, records in confirmed_candidates.items():
        if family not in candidate_top_k_freezes:
            raise ValueError("accuracy matching requires the optimizer's top-k freeze")
        top_k_freeze = candidate_top_k_freezes[family]
        validate_freeze(top_k_freeze, expected_kind="discovery_top_k")
        records = list(records)
        expected_hashes = {item["config_hash"] for item in top_k_freeze["payload"]["selected"]}
        actual_hashes = {str(record["config_hash"]) for record in records}
        if actual_hashes != expected_hashes:
            raise ValueError("accuracy-matching candidates must be exactly the frozen top-k pool")
        aggregates = rank_confirmation(records)
        frozen_candidates[family] = aggregates
        if target is None:
            selections[family] = {"status": "unmatched", "selected": None}
            continue
        eligible = [
            item
            for item in aggregates
            if abs(float(item["mean_terminal_accuracy"]) - target) <= tolerance
        ]
        eligible.sort(
            key=lambda item: (
                abs(float(item["mean_terminal_accuracy"]) - target),
                float(item["mean_terminal_nll"]),
                float(item["terminal_accuracy_std"]),
                str(item["config_hash"]),
            )
        )
        selections[family] = {
            "status": "matched" if eligible else "unmatched",
            "selected": eligible[0] if eligible else None,
        }
    return seal_freeze(
        "accuracy_matching",
        {
            "tuned_winner_freeze_hashes": winner_freeze_hashes,
            "target": target,
            "absolute_tolerance": tolerance,
            "checkpoint_role": "last.pt_epoch_200",
            "candidates": frozen_candidates,
            "selections": selections,
            "replacement_trials": [],
        },
    )


def _complete_endpoint(outcomes: Iterable[Mapping[str, Any]]) -> dict[str, float] | None:
    values = list(outcomes)
    by_seed = {int(item["training_seed"]): item for item in values}
    if set(by_seed) != set(CONFIRMATION_TRAINING_SEEDS):
        return None
    if any(item.get("status") != "completed" for item in by_seed.values()):
        return None
    ordered = [by_seed[seed] for seed in CONFIRMATION_TRAINING_SEEDS]
    return {
        "mean_best_accuracy": statistics.fmean(float(item["best_validation"]["accuracy"]) for item in ordered),
        "mean_best_nll": statistics.fmean(float(item["best_validation"]["nll"]) for item in ordered),
        "mean_best_epoch": statistics.fmean(float(item["best_validation"]["epoch"]) for item in ordered),
    }


def freeze_pair_control(pair_name: str, anchors: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    if pair_name not in PAIR_CONTROL_ANCHORS:
        raise ValueError("unknown protocol v1.1 pair-control name")
    anchors = list(anchors)
    actual_numeric = [dict(anchor["numeric_config"]) for anchor in anchors]
    if actual_numeric != PAIR_CONTROL_ANCHORS[pair_name]:
        raise ValueError("pair-control anchors must exactly match the predeclared ordered table")
    candidates = []
    all_anchors = []
    for anchor in anchors:
        endpoint_items = list(anchor["endpoints"].items())
        if len(endpoint_items) != 2:
            raise ValueError("pair-control anchor must have exactly two endpoints")
        endpoint_aggregates = {
            name: _complete_endpoint(outcomes) for name, outcomes in endpoint_items
        }
        frozen = {
            "shared_config_hash": anchor["shared_config_hash"],
            "numeric_config": dict(anchor["numeric_config"]),
            "endpoints": endpoint_aggregates,
        }
        all_anchors.append(frozen)
        if all(value is not None for value in endpoint_aggregates.values()):
            values = list(endpoint_aggregates.values())
            candidate = {
                **frozen,
                "minimum_mean_best_accuracy": min(value["mean_best_accuracy"] for value in values),
                "mean_endpoint_accuracy": statistics.fmean(value["mean_best_accuracy"] for value in values),
                "mean_endpoint_nll": statistics.fmean(value["mean_best_nll"] for value in values),
                "mean_endpoint_best_epoch": statistics.fmean(value["mean_best_epoch"] for value in values),
            }
            candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -item["minimum_mean_best_accuracy"],
            -item["mean_endpoint_accuracy"],
            item["mean_endpoint_nll"],
            item["mean_endpoint_best_epoch"],
            item["shared_config_hash"],
        )
    )
    return seal_freeze(
        "pair_control",
        {
            "pair_name": pair_name,
            "anchors": all_anchors,
            "eligible_ranked": candidates,
            "selected": candidates[0] if candidates else None,
            "replacement_anchors": [],
        },
    )


def freeze_final_authorization(
    *,
    selected_configs: Mapping[str, str],
    tuned_freezes: Iterable[Mapping[str, Any]],
    accuracy_matching_freeze: Mapping[str, Any],
    pair_control_freezes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    tuned_hashes = []
    for freeze in tuned_freezes:
        validate_freeze(freeze, expected_kind="tuned_winner")
        tuned_hashes.append(freeze["freeze_hash"])
    validate_freeze(accuracy_matching_freeze, expected_kind="accuracy_matching")
    pair_hashes = []
    for freeze in pair_control_freezes:
        validate_freeze(freeze, expected_kind="pair_control")
        pair_hashes.append(freeze["freeze_hash"])
    if not selected_configs:
        raise ValueError("final authorization requires at least one selected config")
    return seal_freeze(
        "final_authorization",
        {
            "selected_config_hashes": dict(sorted(selected_configs.items())),
            "final_training_seeds": FINAL_TRAINING_SEEDS,
            "checkpoint_roles": {
                "primary_scientific": "last.pt",
                "performance_control": "best_val.pt",
                "trajectory": "fixed_snapshots",
            },
            "tuned_freeze_hashes": tuned_hashes,
            "accuracy_matching_freeze_hash": accuracy_matching_freeze["freeze_hash"],
            "pair_control_freeze_hashes": pair_hashes,
            "id_test_release_authorized": True,
        },
    )


def validate_id_test_release(final_authorization: Mapping[str, Any], config_hash: str, seed: int) -> None:
    validate_freeze(final_authorization, expected_kind="final_authorization")
    payload = final_authorization["payload"]
    if not payload.get("id_test_release_authorized"):
        raise ValueError("final authorization does not release ID test")
    if config_hash not in set(payload["selected_config_hashes"].values()):
        raise ValueError("config is not part of the immutable final authorization")
    if seed not in payload["final_training_seeds"]:
        raise ValueError("seed is not part of the immutable final authorization")
