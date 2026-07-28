from pathlib import Path

import pytest
import yaml

from oge.studies.execution import (
    host_trial_ids,
    load_seed0_execution_plan,
    staged_host_trial_ids,
)
from oge.studies.protocol import load_materialized_grid_bundle, load_study_config

ROOT = Path(__file__).parents[1]
STUDY_PATH = ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml"
EXECUTION_PATH = (
    ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/seed0_execution.yaml"
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
