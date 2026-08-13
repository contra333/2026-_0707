import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch
import yaml

from oge.optimizers import AdamCoupledDecoupled, make_optimizer
from oge.training import (
    TASK_F_SNAPSHOT_EPOCHS,
    build_task_f_training_config,
    generate_approval_packet,
    generate_execution_only_pilot,
    generate_research_run_matrix,
    load_training_config,
    matrix_count_summary,
    research_aggregate_runs,
    validate_research_run_matrix,
    validate_task_f_training_config,
)


# Test-only identities. They are not proposed research seeds.
FIXTURE_ANCHOR_SEEDS = (101, 103, 107, 109, 113)
FIXTURE_FACTORIAL_SEEDS = (101, 107, 113)
FIXTURE_SGDM_SEEDS = (211, 223, 227)


def _plan():
    return generate_research_run_matrix(
        anchor_seeds=FIXTURE_ANCHOR_SEEDS,
        adam_factorial_seeds=FIXTURE_FACTORIAL_SEEDS,
        sgdm_seeds=FIXTURE_SGDM_SEEDS,
    )


def _base_config():
    return load_training_config(
        Path(__file__).parents[1]
        / "configs/training/cifar10_wrn28_10_holdout_v1.yaml"
    )


def _parameter():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    parameter.grad = torch.tensor([0.2, -0.4], dtype=torch.float64)
    return parameter


def test_committed_task_f_templates_are_parseable_and_leave_owner_inputs_unfrozen():
    root = Path(__file__).parents[1] / "configs/studies/task_f_paired_training_v1"
    matrix_template = yaml.safe_load((root / "run_matrix.template.yaml").read_text())
    pilot_template = yaml.safe_load(
        (root / "pilot_execution_only.template.yaml").read_text()
    )
    summary = json.loads((root / "matrix_summary.json").read_text())
    assert matrix_template["research_seed_inputs"]["anchor_seeds"]["values"] is None
    assert matrix_template["update_telemetry"]["audit_steps"]["values"] is None
    assert pilot_template["training_seed"]["value"] is None
    assert pilot_template["max_epochs"]["value"] is None
    assert pilot_template["execution_only"] is True
    assert summary["total_research_runs"] == 50


def test_task_f_matrix_conserves_41_adam_plus_9_sgdm_and_exact_cells():
    summary = matrix_count_summary(_plan())
    assert summary == {
        "total_research_runs": 50,
        "family_counts": {"adam": 41, "sgdm": 9},
        "cell_counts": {
            "adam_lr1e-3_wd1e-4_anchor": 20,
            "adam_lr1e-3_wd1e-3": 6,
            "adam_lr3e-4_wd1e-4": 9,
            "adam_lr3e-4_wd1e-3": 6,
            "sgdm_lr0.1_wd5e-4": 9,
        },
    }


def test_zero_baselines_are_shared_without_duplicate_research_runs():
    runs = _plan()["runs"]
    zeros = {run["run_id"]: run for run in runs if run["branch_policy"] == "zero"}
    assert Counter(run["family"] for run in zeros.values()) == {"adam": 8, "sgdm": 3}
    for run in runs:
        zero = zeros[run["zero_reference_run_id"]]
        assert zero["sibling_group_id"] == run["sibling_group_id"]
        assert zero["training_seed"] == run["training_seed"]
        assert zero["initial_lr"] == run["initial_lr"]
    high_wd = [run for run in runs if run["cell_id"] == "adam_lr1e-3_wd1e-3"]
    assert {run["zero_reference_run_id"] for run in high_wd}.issubset(zeros)
    assert not any(run["branch_policy"] == "zero" for run in high_wd)


def test_anchor_alpha_endpoints_use_exact_factory_semantics_and_midpoint():
    anchor = [
        run
        for run in _plan()["runs"]
        if run["cell_id"] == "adam_lr1e-3_wd1e-4_anchor"
        and run["training_seed"] == FIXTURE_ANCHOR_SEEDS[0]
    ]
    by_branch = {run["branch_policy"]: run for run in anchor}
    assert by_branch["zero"]["optimizer"]["weight_decay"] == 0.0
    for branch, ratio, expected_class in (
        ("adamw_alpha_0", 0.0, torch.optim.AdamW),
        ("adam_mixed_alpha_0_5", 0.5, AdamCoupledDecoupled),
        ("adam_alpha_1", 1.0, torch.optim.Adam),
    ):
        config = by_branch[branch]["optimizer"]
        assert config["name"] == "adam_coupled_decoupled"
        assert config["total_weight_decay"] == 1e-4
        assert config["coupled_ratio"] == ratio
        assert isinstance(make_optimizer([_parameter()], config), expected_class)

    for branch, reference_name in (
        ("adamw_alpha_0", "adamw"),
        ("adam_alpha_1", "adam"),
    ):
        left = _parameter()
        right = _parameter()
        endpoint = make_optimizer([left], by_branch[branch]["optimizer"])
        reference_config = copy.deepcopy(by_branch[branch]["optimizer"])
        reference_config["name"] = reference_name
        reference_config["weight_decay"] = reference_config.pop("total_weight_decay")
        reference_config.pop("coupled_ratio")
        reference = make_optimizer([right], reference_config)
        endpoint.step()
        reference.step()
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_sibling_branch_neutral_hash_is_identical_within_group():
    hashes = defaultdict(set)
    for run in _plan()["runs"]:
        config = build_task_f_training_config(_base_config(), run)
        hashes[run["sibling_group_id"]].add(
            config["task_f"]["branch_neutral_config_sha256"]
        )
    assert hashes
    assert all(len(group_hashes) == 1 for group_hashes in hashes.values())


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "anchor_seeds": (1, 2, 3, 4),
                "adam_factorial_seeds": (1, 2, 3),
                "sgdm_seeds": (4, 5, 6),
            },
            "exactly 5",
        ),
        (
            {
                "anchor_seeds": (1, 2, 3, 4, 5),
                "adam_factorial_seeds": (1, 1, 2),
                "sgdm_seeds": (4, 5, 6),
            },
            "duplicate",
        ),
        (
            {
                "anchor_seeds": (1, 2, 3, 4, 5),
                "adam_factorial_seeds": (6, 7, 8),
                "sgdm_seeds": (4, 5, 6),
            },
            "subset",
        ),
    ],
)
def test_research_seed_inputs_validate_counts_uniqueness_and_pairing(kwargs, match):
    with pytest.raises(ValueError, match=match):
        generate_research_run_matrix(**kwargs)


def test_snapshot_contract_and_strict_from_scratch_research_validation():
    plan = _plan()
    assert tuple(plan["snapshot_epochs"]) == TASK_F_SNAPSHOT_EPOCHS
    run = plan["runs"][0]
    config = build_task_f_training_config(_base_config(), run)
    assert tuple(config["checkpoint"]["snapshot_epochs"]) == TASK_F_SNAPSHOT_EPOCHS

    invalid = copy.deepcopy(plan)
    invalid["runs"][0]["fork_from_prefix"] = "prefix.pt"
    with pytest.raises(ValueError, match="from scratch"):
        validate_research_run_matrix(invalid)


def test_execution_only_pilot_is_excluded_and_approval_unknowns_are_explicit():
    plan = _plan()
    pilot = generate_execution_only_pilot(
        base_config=_base_config(),
        seed=997,  # fixture-only, not a proposed pilot or research seed
        max_epochs=2,
    )
    assert pilot["task_f"]["execution_only"] is True
    assert pilot["task_f"]["aggregate_eligible"] is False
    assert pilot["optimizer"]["coupled_ratio"] == 0.5
    assert research_aggregate_runs([*plan["runs"], pilot["task_f"]]) == plan["runs"]

    packet = generate_approval_packet(execution_sha="a" * 40, pilot_config=pilot)
    assert packet["resource_estimates"] == {
        "expected_gpu_hours": "UNKNOWN",
        "expected_wall_time": "UNKNOWN",
        "expected_storage": "UNKNOWN",
    }
    assert packet["approval_boundaries"][
        "pilot_approval_does_not_authorize_full_training"
    ] is True
    assert "task_f_b_specification_sha256" in packet["unresolved_inputs"]
    assert "exact_pilot_length" not in packet["unresolved_inputs"]


def test_unresolved_pilot_packet_is_deterministic_and_does_not_invent_resources():
    first = generate_approval_packet(execution_sha="b" * 40)
    second = generate_approval_packet(execution_sha="b" * 40)
    assert first == second
    assert first["pilot"] == {
        "config_path": "UNRESOLVED",
        "training_seed": "UNRESOLVED",
        "max_epochs": "UNRESOLVED",
        "execution_only": True,
    }
    assert {
        "exact_pilot_length",
        "pilot_seed",
        "task_f_b_specification_sha256",
        "update_telemetry_audit_schedule",
    }.issubset(first["unresolved_inputs"])


def test_execution_only_cannot_be_smuggled_into_research_matrix():
    plan = _plan()
    plan["runs"][0]["execution_only"] = True
    with pytest.raises(ValueError, match="execution_only"):
        validate_research_run_matrix(plan)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "task-f-protected_ood-run"),
        ("branch_policy", "nearood_probe"),
    ],
)
def test_task_f_training_config_rejects_protected_ood_names(field, value):
    run = _plan()["runs"][0]
    config = build_task_f_training_config(_base_config(), run)
    config["task_f"][field] = value
    with pytest.raises(ValueError, match="protected OOD"):
        validate_task_f_training_config(config)
