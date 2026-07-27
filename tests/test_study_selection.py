import copy
from pathlib import Path

import pytest

from oge.studies.protocol import (
    generate_grid_bundle,
    load_study_config,
)
from oge.studies.schemas import validate_freeze
from oge.studies.selection import (
    build_followup_plan,
    freeze_grid_roles,
    grid_rank_key,
    validate_protected_evidence_access,
)
from oge.training import load_training_config, resolve_training_config

ROOT = Path(__file__).parents[1]
STUDY_PATH = ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml"


def _bundle():
    study = load_study_config(STUDY_PATH)
    base = load_training_config(ROOT / study["base_training_config"])
    resolved = resolve_training_config(base, data_root=ROOT, device="cpu")
    return generate_grid_bundle(study, resolved)


def _result(row, accuracy=0.94, nll=0.3, best_epoch=180, status="completed"):
    return {
        "trial_id": row["trial_id"],
        "optimizer_family": row["optimizer_family"],
        "assigned_slot": row["assigned_slot"],
        "config_hash": row["config_hash"],
        "lr_rank": row["lr_rank"],
        "weight_decay_rank": row["weight_decay_rank"],
        "training_seed": 0,
        "scientific_config": copy.deepcopy(row["scientific_config"]),
        "status": status,
        "best_validation": (
            None
            if status != "completed"
            else {"epoch": best_epoch, "accuracy": accuracy, "nll": nll}
        ),
        "final_validation": (
            None
            if status != "completed"
            else {"epoch": 200, "accuracy": accuracy, "nll": nll}
        ),
    }


def _complete_results(bundle):
    accuracies = {
        "sgd": {0: 0.9500, 1: 0.9485, 2: 0.9400},
        "adam": {0: 0.9495, 1: 0.9450, 2: 0.9400},
        "adamw": {0: 0.9490, 1: 0.9400, 2: 0.9400},
    }
    records = []
    for optimizer in ("sgd", "adam", "adamw"):
        for row in bundle["tables"][optimizer]["rows"]:
            accuracy = accuracies[optimizer].get(row["assigned_slot"], 0.91)
            records.append(_result(row, accuracy=accuracy, nll=0.4 + row["assigned_slot"] / 100))
    return records


def test_grid_ranking_uses_last_epoch_then_nll_best_epoch_cell_order_and_hash():
    bundle = _bundle()
    row = bundle["tables"]["sgd"]["rows"][0]
    records = [
        _result(row, accuracy=0.91, nll=0.4, best_epoch=4),
        {
            **_result(row, accuracy=0.91, nll=0.3, best_epoch=9),
            "trial_id": "nll",
        },
        {
            **_result(row, accuracy=0.92, nll=0.9, best_epoch=200),
            "trial_id": "accuracy",
        },
    ]
    assert [record["trial_id"] for record in sorted(records, key=grid_rank_key)] == [
        "accuracy",
        "nll",
        row["trial_id"],
    ]


def test_role_freeze_requires_all_36_terminal_and_rejects_protected_signals():
    bundle = _bundle()
    records = _complete_results(bundle)
    records[0]["status"] = "running"
    with pytest.raises(ValueError, match="terminal"):
        freeze_grid_roles(records, bundle)
    records = _complete_results(bundle)
    records[0]["ood_metrics"] = {"auroc": 1.0}
    with pytest.raises(ValueError, match="protected"):
        freeze_grid_roles(records, bundle)


def test_c1_c2_widening_absence_and_c3_c4_are_frozen_deterministically():
    bundle = _bundle()
    freeze = freeze_grid_roles(_complete_results(bundle), bundle)
    validate_freeze(freeze, expected_kind="role_selection")
    payload = freeze["payload"]
    assert payload["per_optimizer"]["sgd"]["c1"]["assigned_slot"] == 0
    assert payload["per_optimizer"]["sgd"]["c2"]["status"] == "selected"
    assert payload["per_optimizer"]["sgd"]["c2"]["band_used"] == 0.002
    assert payload["per_optimizer"]["adam"]["c2"]["status"] == "selected"
    assert payload["per_optimizer"]["adam"]["c2"]["band_used"] == 0.005
    assert payload["per_optimizer"]["adamw"]["c2"]["status"] == "absent"
    assert payload["c3"]["status"] == "selected"
    assert payload["c3"]["band_used"] == 0.002
    assert payload["c4"]["status"] == "selected"
    assert payload["c4"]["triple"]["mean_accuracy"] <= (
        payload["c3"]["triple"]["mean_accuracy"] - 0.005
    )
    repeated = freeze_grid_roles(_complete_results(bundle), bundle)
    assert repeated == freeze


def test_c3_widens_once_when_primary_band_has_no_triple():
    bundle = _bundle()
    records = _complete_results(bundle)
    offsets = {"sgd": 0.0, "adam": -0.002, "adamw": -0.004}
    for record in records:
        accuracy = 0.94 + offsets[record["optimizer_family"]]
        record["best_validation"]["accuracy"] = accuracy
        record["final_validation"]["accuracy"] = accuracy
    freeze = freeze_grid_roles(records, bundle)
    assert freeze["payload"]["c3"]["status"] == "selected"
    assert freeze["payload"]["c3"]["band_used"] == 0.005


def test_c3_can_be_unresolved_after_widening_and_c4_is_omitted():
    bundle = _bundle()
    records = _complete_results(bundle)
    offsets = {"sgd": 0.0, "adam": -0.003, "adamw": -0.006}
    for record in records:
        accuracy = 0.94 + offsets[record["optimizer_family"]]
        record["best_validation"]["accuracy"] = accuracy
        record["final_validation"]["accuracy"] = accuracy
    freeze = freeze_grid_roles(records, bundle)
    assert freeze["payload"]["c3"]["status"] == "unresolved"
    assert freeze["payload"]["c4"]["status"] == "omitted"


def test_followup_plan_deduplicates_roles_and_pair_controls_and_reuses_seed_zero():
    bundle = _bundle()
    role_freeze = freeze_grid_roles(_complete_results(bundle), bundle)
    plan = build_followup_plan(role_freeze, bundle)
    scheduled_keys = {
        (row["config_hash"], row["training_seed"]) for row in plan["scheduled"]
    }
    reused_keys = {
        (row["config_hash"], row["training_seed"]) for row in plan["reused"]
    }
    assert len(scheduled_keys) == len(plan["scheduled"])
    assert len(reused_keys) == len(plan["reused"])
    assert not scheduled_keys.intersection(reused_keys)
    assert all(row["training_seed"] in (1, 2) for row in plan["scheduled"] if row["optimizer_family"] != "sgdw")
    assert {
        row["training_seed"]
        for row in plan["scheduled"]
        if row["optimizer_family"] == "sgdw"
    } == {0, 1, 2}
    assert all(row["reuse_source"] == "seed0_grid" for row in plan["reused"])
    assert len(plan["plan_hash"]) == 64


def test_protected_evidence_gate_allows_only_frozen_role_configs_and_seeds_0_to_2():
    bundle = _bundle()
    role_freeze = freeze_grid_roles(_complete_results(bundle), bundle)
    selected_hash = next(iter(role_freeze["payload"]["role_labels_by_config_hash"]))
    validate_protected_evidence_access(role_freeze, selected_hash, 0)
    validate_protected_evidence_access(role_freeze, selected_hash, 2)
    with pytest.raises(ValueError, match="not part"):
        validate_protected_evidence_access(role_freeze, "other", 0)
    with pytest.raises(ValueError, match="seed"):
        validate_protected_evidence_access(role_freeze, selected_hash, 3)


def test_role_freeze_hash_detects_mutation():
    bundle = _bundle()
    freeze = freeze_grid_roles(_complete_results(bundle), bundle)
    mutated = copy.deepcopy(freeze)
    mutated["payload"]["released_training_seeds"] = [0, 1]
    with pytest.raises(ValueError, match="hash"):
        validate_freeze(mutated)
