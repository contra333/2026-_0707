from pathlib import Path

import pytest
import yaml

from oge.studies.execution import (
    host_trial_ids,
    load_followup_execution_plan,
    load_seed0_execution_plan,
    staged_host_trial_ids,
)
from oge.studies.protocol import load_materialized_grid_bundle, load_study_config

ROOT = Path(__file__).parents[1]
STUDY_PATH = ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml"
EXECUTION_PATH = (
    ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/seed0_execution.yaml"
)
FOLLOWUP_PLAN_PATH = (
    ROOT
    / "results/wrn28_10_optimizer_hpo_v1_2/seed0_20260728/followup_plan.json"
)
FOLLOWUP_EXECUTION_PATH = (
    ROOT
    / "configs/studies/wrn28_10_optimizer_hpo_v1_2/followup_execution.yaml"
)


def _bundle():
    study = load_study_config(STUDY_PATH)
    return load_materialized_grid_bundle(ROOT / study["frozen_grid_dir"])


def test_seed0_execution_plan_assigns_every_grid_cell_exactly_once():
    plan = load_seed0_execution_plan(EXECUTION_PATH, _bundle())
    assert {host: len(value["trial_ids"]) for host, value in plan["hosts"].items()} == {
        "curie": 14,
        "lise": 8,
        "precision_medicine": 14,
    }
    assert [plan["hosts"][host]["concurrency"] for host in plan["hosts"]] == [4, 2, 4]
    assert host_trial_ids(plan, "lise") == plan["hosts"]["lise"]["trial_ids"]
    assert staged_host_trial_ids(plan, "precision_medicine", "canary") == [
        "grid-sgd-06"
    ]
    assert "grid-sgd-06" not in staged_host_trial_ids(
        plan, "precision_medicine", "remaining"
    )
    with pytest.raises(ValueError, match="not assigned"):
        staged_host_trial_ids(plan, "lise", "canary")


def test_seed0_execution_plan_rejects_duplicate_or_missing_trials(tmp_path):
    plan = yaml.safe_load(EXECUTION_PATH.read_text(encoding="utf-8"))
    plan["hosts"]["lise"]["trial_ids"][0] = plan["hosts"]["curie"]["trial_ids"][0]
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="more than once"):
        load_seed0_execution_plan(path, _bundle())


def test_seed0_execution_plan_rejects_optimizer_absent_from_a_host(tmp_path):
    plan = yaml.safe_load(EXECUTION_PATH.read_text(encoding="utf-8"))
    plan["hosts"]["lise"]["trial_ids"] = [
        trial_id
        for trial_id in plan["hosts"]["lise"]["trial_ids"]
        if not trial_id.startswith("grid-adamw-")
    ]
    path = tmp_path / "missing-family.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="every optimizer family"):
        load_seed0_execution_plan(path, _bundle())


def _followup_plan():
    import json

    return json.loads(FOLLOWUP_PLAN_PATH.read_text(encoding="utf-8"))


def test_followup_execution_plan_has_exact_counts_rotation_and_pairs():
    plan = load_followup_execution_plan(FOLLOWUP_EXECUTION_PATH, _followup_plan())
    assert {host: len(value["trial_ids"]) for host, value in plan["hosts"].items()} == {
        "curie": 13,
        "lise": 10,
        "precision_medicine": 7,
    }
    assert [plan["hosts"][host]["concurrency"] for host in plan["hosts"]] == [4, 2, 4]
    assert host_trial_ids(plan, "curie") == plan["hosts"]["curie"]["trial_ids"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hash", "followup_plan_hash mismatch"),
        ("duplicate", "more than once"),
        ("rotation", "does not rotate"),
        ("pair", "not on the same host"),
    ],
)
def test_followup_execution_plan_rejects_invalid_contract(
    tmp_path,
    mutation,
    message,
):
    plan = yaml.safe_load(FOLLOWUP_EXECUTION_PATH.read_text(encoding="utf-8"))
    if mutation == "hash":
        plan["followup_plan_hash"] = "wrong"
    elif mutation == "duplicate":
        plan["hosts"]["lise"]["trial_ids"][0] = plan["hosts"]["curie"]["trial_ids"][0]
    elif mutation == "rotation":
        curie_id = "followup-adam-b701f780ad99-seed-2"
        lise_id = "followup-adam-db137e7d3c71-seed-1"
        curie_index = plan["hosts"]["curie"]["trial_ids"].index(curie_id)
        lise_index = plan["hosts"]["lise"]["trial_ids"].index(lise_id)
        plan["hosts"]["curie"]["trial_ids"][curie_index] = lise_id
        plan["hosts"]["lise"]["trial_ids"][lise_index] = curie_id
    else:
        curie_id = "followup-adam-0ac334244797-seed-1"
        precision_id = "followup-adam-0ac334244797-seed-2"
        curie_index = plan["hosts"]["curie"]["trial_ids"].index(curie_id)
        precision_index = plan["hosts"]["precision_medicine"]["trial_ids"].index(
            precision_id
        )
        plan["hosts"]["curie"]["trial_ids"][curie_index] = precision_id
        plan["hosts"]["precision_medicine"]["trial_ids"][
            precision_index
        ] = curie_id
    path = tmp_path / "invalid-followup.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_followup_execution_plan(path, _followup_plan())
