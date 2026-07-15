import copy

import pytest

from oge.studies.schemas import validate_freeze
from oge.studies.selection import (
    confirmation_rank_key,
    discovery_rank_key,
    freeze_accuracy_matching,
    freeze_discovery_tables,
    freeze_discovery_top_k,
    freeze_final_authorization,
    freeze_pair_control,
    freeze_tuned_winner,
    validate_id_test_release,
)


def _discovery(trial_id, config_hash, accuracy, nll, epoch, status="completed"):
    return {
        "trial_id": trial_id,
        "config_hash": config_hash,
        "status": status,
        "best_validation": None if status != "completed" else {"accuracy": accuracy, "nll": nll, "epoch": epoch},
    }


def _confirmation(trial_id, config_hash, best=(0.9, 0.5, 10), terminal=(0.88, 0.6), missing=()):
    observations = []
    for seed in (1, 2, 3):
        if seed in missing:
            continue
        observations.append(
            {
                "training_seed": seed,
                "status": "completed",
                "best_validation": {"accuracy": best[0], "nll": best[1], "epoch": best[2]},
                "terminal_validation": {"accuracy": terminal[0], "nll": terminal[1]},
            }
        )
    return {"trial_id": trial_id, "config_hash": config_hash, "observations": observations}


def _top_k(family="sgd", selected=None):
    selected = selected or [("d-0", "a"), ("d-1", "b"), ("d-2", "c")]
    records = [_discovery(trial, config_hash, 0.9 - index * 0.01, 0.4, 5) for index, (trial, config_hash) in enumerate(selected)]
    index = 0
    while len(records) < 16:
        trial_id = f"{family}-failed-{index:02d}"
        index += 1
        records.append(
            _discovery(
                trial_id,
                f"failed-{index}",
                0.0,
                0.0,
                0,
                status="scientific_failed",
            )
        )
    return freeze_discovery_top_k(
        family,
        records,
        [item["trial_id"] for item in records],
    )


def test_discovery_ranking_implements_every_tie_break_in_order():
    records = [
        _discovery("z", "z", 0.90, 0.4, 3),
        _discovery("z", "y", 0.90, 0.4, 3),
        _discovery("a", "z", 0.90, 0.4, 3),
        _discovery("nll", "x", 0.90, 0.3, 8),
        _discovery("epoch", "x", 0.90, 0.4, 2),
        _discovery("accuracy", "x", 0.91, 0.9, 20),
    ]
    ordered = sorted(records, key=discovery_rank_key)
    assert [item["trial_id"] for item in ordered] == ["accuracy", "nll", "epoch", "a", "z", "z"]
    assert [item["config_hash"] for item in ordered[-2:]] == ["y", "z"]


def test_top_k_requires_all_assigned_slots_terminal_and_keeps_fewer_than_three():
    records = [_discovery("a", "a", 0.9, 0.4, 2)]
    records.extend(
        _discovery(
            f"slot-{index}",
            f"hash-{index}",
            0.0,
            0.0,
            0,
            status="running" if index == 0 else "scientific_failed",
        )
        for index in range(15)
    )
    assigned = [record["trial_id"] for record in records]
    with pytest.raises(ValueError, match="terminal"):
        freeze_discovery_top_k("sgd", records, assigned)

    records[1] = _discovery("slot-0", "hash-0", 0.0, 0.0, 0, status="scientific_failed")
    freeze = freeze_discovery_top_k("sgd", records, assigned)
    validate_freeze(freeze, expected_kind="discovery_top_k")
    assert freeze["payload"]["completed_count"] == 1
    assert freeze["payload"]["selected"] == [{"trial_id": "a", "config_hash": "a"}]
    assert freeze["payload"]["replacement_trials"] == []


def test_confirmation_ranking_tie_breaks_and_requires_all_three_seeds():
    records = [
        _confirmation("z", "z", best=(0.90, 0.4, 3)),
        _confirmation("z", "y", best=(0.90, 0.4, 3)),
        _confirmation("a", "z", best=(0.90, 0.4, 3)),
        _confirmation("nll", "x", best=(0.90, 0.3, 8)),
        _confirmation("epoch", "x", best=(0.90, 0.4, 2)),
        _confirmation("accuracy", "x", best=(0.91, 0.9, 20)),
        _confirmation("missing", "m", missing=(3,)),
    ]
    from oge.studies.selection import rank_confirmation

    ordered = rank_confirmation(records)
    assert [item["trial_id"] for item in ordered] == ["accuracy", "nll", "epoch", "a", "z", "z"]
    assert [item["config_hash"] for item in ordered[-2:]] == ["y", "z"]
    assert "missing" not in [item["trial_id"] for item in ordered]
    assert confirmation_rank_key(ordered[0]) < confirmation_rank_key(ordered[1])


def test_tuned_winner_is_limited_to_frozen_top_k_and_final_results_cannot_reselect():
    top_k = _top_k(selected=[("d-0", "a"), ("d-1", "b")])
    freeze = freeze_tuned_winner(
        "sgd",
        [_confirmation("d-0", "a", best=(0.9, 0.4, 4)), _confirmation("d-1", "b", best=(0.91, 0.5, 5))],
        top_k,
    )
    assert freeze["payload"]["winner"]["config_hash"] == "b"
    with pytest.raises(ValueError, match="outside"):
        freeze_tuned_winner("sgd", [_confirmation("other", "x")], top_k)


def test_accuracy_matching_freezes_minimum_target_tolerance_tie_break_and_unmatched():
    sgd_top = _top_k("sgd", [("s0", "sa"), ("s1", "sb")])
    adam_top = _top_k("adam", [("a0", "aa")])
    sgd_tuned = freeze_tuned_winner(
        "sgd", [_confirmation("s0", "sa", terminal=(0.900, 0.3)), _confirmation("s1", "sb", terminal=(0.899, 0.2))], sgd_top
    )
    adam_tuned = freeze_tuned_winner("adam", [_confirmation("a0", "aa", terminal=(0.890, 0.4))], adam_top)
    candidates = {
        "sgd": [
            _confirmation("s0", "sa", terminal=(0.891, 0.4)),
            _confirmation("s1", "sb", terminal=(0.891, 0.2)),
        ],
        "adam": [_confirmation("a0", "aa", terminal=(0.900, 0.1))],
    }
    freeze = freeze_accuracy_matching(
        [sgd_tuned, adam_tuned],
        candidates,
        {"sgd": sgd_top, "adam": adam_top},
    )
    assert freeze["payload"]["target"] == pytest.approx(0.890)
    assert freeze["payload"]["absolute_tolerance"] == 0.002
    assert freeze["payload"]["selections"]["sgd"]["selected"]["config_hash"] == "sb"
    assert freeze["payload"]["selections"]["adam"]["status"] == "unmatched"
    assert freeze["payload"]["replacement_trials"] == []
    with pytest.raises(ValueError, match="fixed"):
        freeze_accuracy_matching(
            [sgd_tuned],
            candidates,
            {"sgd": sgd_top, "adam": adam_top},
            tolerance=0.01,
        )


def _endpoint(accuracy, nll=0.4, epoch=5, missing=()):
    return [
        {
            "training_seed": seed,
            "status": "completed",
            "best_validation": {"accuracy": accuracy, "nll": nll, "epoch": epoch},
        }
        for seed in (1, 2, 3)
        if seed not in missing
    ]


def test_pair_control_requires_complete_both_endpoints_and_uses_symmetric_ranking():
    anchors = [
        {
            "shared_config_hash": "incomplete",
            "numeric_config": {"lr": 0.03, "weight_decay": 0.0001},
            "endpoints": {"sgd": _endpoint(0.95), "sgdw": _endpoint(0.95, missing=(3,))},
        },
        {
            "shared_config_hash": "balanced",
            "numeric_config": {"lr": 0.03, "weight_decay": 0.0005},
            "endpoints": {"sgd": _endpoint(0.91), "sgdw": _endpoint(0.91)},
        },
        {
            "shared_config_hash": "lopsided",
            "numeric_config": {"lr": 0.1, "weight_decay": 0.0001},
            "endpoints": {"sgd": _endpoint(0.99), "sgdw": _endpoint(0.90)},
        },
        {
            "shared_config_hash": "fourth",
            "numeric_config": {"lr": 0.1, "weight_decay": 0.0005},
            "endpoints": {"sgd": _endpoint(0.89), "sgdw": _endpoint(0.89)},
        },
    ]
    freeze = freeze_pair_control("sgd_sgdw", anchors)
    assert freeze["payload"]["selected"]["shared_config_hash"] == "balanced"
    assert len(freeze["payload"]["anchors"]) == 4
    assert freeze["payload"]["replacement_anchors"] == []


def test_freeze_hash_detects_mutation_and_final_authorization_gates_id_test():
    top = _top_k(selected=[("d-0", "a")])
    tuned = freeze_tuned_winner("sgd", [_confirmation("d-0", "a")], top)
    matching = freeze_accuracy_matching(
        [tuned],
        {"sgd": [_confirmation("d-0", "a")]},
        {"sgd": top},
    )
    pair = freeze_pair_control(
        "sgd_sgdw",
        [
            {
                "shared_config_hash": f"p-{index}",
                "numeric_config": numeric,
                "endpoints": {"sgd": _endpoint(0.9), "sgdw": _endpoint(0.9)},
            }
            for index, numeric in enumerate(
                [
                    {"lr": 0.03, "weight_decay": 0.0001},
                    {"lr": 0.03, "weight_decay": 0.0005},
                    {"lr": 0.1, "weight_decay": 0.0001},
                    {"lr": 0.1, "weight_decay": 0.0005},
                ]
            )
        ],
    )
    final = freeze_final_authorization(
        selected_configs={"sgd_tuned": "a"},
        tuned_freezes=[tuned],
        accuracy_matching_freeze=matching,
        pair_control_freezes=[pair],
    )
    validate_id_test_release(final, "a", 4)
    with pytest.raises(ValueError, match="not part"):
        validate_id_test_release(final, "other", 4)
    with pytest.raises(ValueError, match="seed"):
        validate_id_test_release(final, "a", 1)
    mutated = copy.deepcopy(final)
    mutated["payload"]["final_training_seeds"] = [4]
    with pytest.raises(ValueError, match="hash"):
        validate_freeze(mutated)


def test_discovery_table_freeze_requires_complete_manifest_hashes():
    freeze = freeze_discovery_tables(
        {
            "total_assigned_rows": 64,
            "manifest_hash": "manifest",
            "table_hashes": {
                "sgd": "a",
                "sgdw": "b",
                "adam": "c",
                "adamw": "d",
            },
        }
    )
    validate_freeze(freeze, expected_kind="discovery_tables")
