import copy
import json
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import yaml

from oge.models import make_model
from oge.training import capture_rng_state, observe_first_minibatch, seed_everything
from oge.training import runner as training_runner
from oge.training.resnet18_replication_provenance import (
    build_resnet18_replication_checkpoint_provenance,
    create_initial_resnet18_replication_provenance,
)
from oge.training.resnet18_replication_plan import (
    RESNET18_REPLICATION_LRS,
    RESNET18_REPLICATION_ROLES,
    build_resnet18_replication_training_config,
    generate_resnet18_approval_packet,
    generate_resnet18_execution_only_pilot_configs,
    generate_resnet18_replication_matrix,
    resnet18_replication_count_summary,
    validate_resnet18_replication_matrix,
    validate_resnet18_replication_training_config,
)
from oge.training.runner import load_training_config


ROOT = Path(__file__).parents[1]


def _base_config():
    return load_training_config(
        ROOT / "configs/training/cifar10_resnet18_replication_v1.yaml"
    )


def test_committed_replication_templates_are_parseable_and_gpu_safe():
    root = ROOT / "configs/studies/resnet18_cifar10_replication_v1"
    matrix = yaml.safe_load((root / "run_matrix.template.yaml").read_text())
    pilot = yaml.safe_load((root / "pilot_execution_only.template.yaml").read_text())
    summary = json.loads((root / "matrix_summary.json").read_text())
    approval = json.loads((root / "approval_packet.template.json").read_text())
    assert matrix["matrix"]["total_runs"] == summary["total_research_runs"] == 20
    assert pilot["seed"] == 9000
    assert pilot["arms"]["count"] == 4
    assert matrix["output_policy"]["gpu_launch_allowed"] is False
    assert approval["approval_boundaries"]["protected_evaluation"] == "NOT_AUTHORIZED"


def test_exact_20_run_matrix_and_four_arms_per_seed():
    plan = generate_resnet18_replication_matrix()
    assert resnet18_replication_count_summary(plan) == {
        "total_research_runs": 20,
        "cell_counts": {
            "resnet18_c10_lr1e-03_wd1e-4": 10,
            "resnet18_c10_lr3e-04_wd1e-4": 10,
        },
        "role_counts": {"adam_coupled": 10, "adamw_decoupled": 10},
    }
    by_seed = defaultdict(list)
    for run in plan["runs"]:
        by_seed[run["training_seed"]].append(run)
    assert set(by_seed) == {0, 1, 2, 3, 4}
    expected = set((lr, role) for lr in RESNET18_REPLICATION_LRS for role in RESNET18_REPLICATION_ROLES)
    for seed, runs in by_seed.items():
        assert {(run["initial_lr"], run["branch_policy"]) for run in runs} == expected
        assert {run["cross_lr_pairing_block_id"] for run in runs} == {
            f"resnet18-c10-rep-cross-lr-seed{seed}"
        }
        assert {run["data_stream_id"] for run in runs} == {
            f"resnet18-c10-rep-cross-lr-seed{seed}"
        }


def test_config_keeps_task_f_frozen_and_hashes_within_lr_and_cross_lr_controls():
    plan = generate_resnet18_replication_matrix()
    by_seed = defaultdict(list)
    for run in plan["runs"]:
        config = build_resnet18_replication_training_config(_base_config(), run)
        validated = copy.deepcopy(config)
        training_runner._materialize_config_defaults(validated)
        training_runner._validate_training_config(validated)
        assert "task_f" not in config
        assert config["model"] == {
            "name": "resnet18",
            "variant": "cifar",
            "num_classes": 10,
        }
        by_seed[run["training_seed"]].append(config)
    for configs in by_seed.values():
        by_lr = defaultdict(list)
        for config in configs:
            by_lr[config["optimizer"]["lr"]].append(config)
        assert all(
            len(
                {
                    config["resnet18_replication"]["branch_neutral_config_sha256"]
                    for config in lr_configs
                }
            )
            == 1
            for lr_configs in by_lr.values()
        )
        assert len(
            {
                config["resnet18_replication"]["cross_lr_neutral_config_sha256"]
                for config in configs
            }
        ) == 1


def test_matrix_and_training_validation_reject_identity_and_shape_drift():
    plan = generate_resnet18_replication_matrix()
    broken = copy.deepcopy(plan)
    broken["runs"][0]["cross_lr_pairing_block_id"] = "drift"
    with pytest.raises(ValueError, match="cross-LR"):
        validate_resnet18_replication_matrix(broken)

    config = build_resnet18_replication_training_config(_base_config(), plan["runs"][0])
    config["task_f"] = {}
    with pytest.raises(ValueError, match="frozen Task F"):
        validate_resnet18_replication_training_config(config)


def test_seed9000_four_arm_two_epoch_pilot_and_approval_packet():
    configs = generate_resnet18_execution_only_pilot_configs(base_config=_base_config())
    assert len(configs) == 4
    assert {
        (config["optimizer"]["lr"], config["resnet18_replication"]["branch_policy"])
        for config in configs
    } == set((lr, role) for lr in RESNET18_REPLICATION_LRS for role in RESNET18_REPLICATION_ROLES)
    assert {config["training"]["seed"] for config in configs} == {9000}
    assert {config["training"]["max_epochs"] for config in configs} == {2}
    assert {tuple(config["checkpoint"]["snapshot_epochs"]) for config in configs} == {
        (0, 1, 2)
    }
    assert len(
        {
            config["resnet18_replication"]["cross_lr_neutral_config_sha256"]
            for config in configs
        }
    ) == 1
    for lr in RESNET18_REPLICATION_LRS:
        same_lr = [config for config in configs if config["optimizer"]["lr"] == lr]
        assert len(
            {
                config["resnet18_replication"]["branch_neutral_config_sha256"]
                for config in same_lr
            }
        ) == 1
        assert same_lr[0]["resnet18_replication"]["sibling_members"] == same_lr[1][
            "resnet18_replication"
        ]["sibling_members"]
    assert all(config["resnet18_replication"]["execution_only"] for config in configs)
    packet = generate_resnet18_approval_packet(
        execution_sha="a" * 40, pilot_configs=configs
    )
    assert packet["pilot"]["status"] == "MATERIALIZED_NOT_RUN"
    assert packet["approval_boundaries"]["protected_evaluation"] == "NOT_AUTHORIZED"
    assert set(packet["resource_estimates"].values()) == {"PENDING_PILOT"}


def test_four_seed_arms_realize_identical_initialization_and_data_stream_identity():
    plan = generate_resnet18_replication_matrix()
    runs = [run for run in plan["runs"] if run["training_seed"] == 0]
    identities = []
    for run in runs:
        config = build_resnet18_replication_training_config(_base_config(), run)
        config["dataset"].update(
            {
                "membership": {
                    "train": {"sha256": "train", "line_count": 45000},
                    "validation": {"sha256": "validation", "line_count": 5000},
                    "test": {"sha256": "test", "line_count": 10000},
                },
                "membership_manifest": {"sha256": "manifest", "row_count": 1},
            }
        )
        generator = seed_everything(0, deterministic=True)
        initial_rng = capture_rng_state(generator)
        model = make_model(config["model"])
        provenance = create_initial_resnet18_replication_provenance(
            resolved_config=config,
            model=model,
            initial_rng_state=initial_rng,
        )
        observe_first_minibatch(
            provenance,
            {
                "sample_id": ["fixture:a", "fixture:b"],
                "image": torch.zeros(2, 3, 32, 32),
            },
        )
        checkpoint = build_resnet18_replication_checkpoint_provenance(
            resolved_config=config,
            paired_control_provenance=provenance,
            checkpoint_epoch=200,
            checkpoint_role="last",
            oge_git_sha="f" * 40,
            run_id=run["run_id"],
        )
        identities.append(checkpoint)
        snapshot = training_runner._snapshot_payload(
            completed_epoch=200,
            model=model,
            resolved_config=config,
            oge_git_sha="f" * 40,
            run_id=run["run_id"],
            paired_control_provenance=provenance,
        )
        assert "resnet18_replication_provenance" in snapshot
        assert "task_f_provenance" not in snapshot
    assert len({row["initialization_sha256"] for row in identities}) == 1
    assert len({row["data_stream_sha256"] for row in identities}) == 1
    assert len({row["cross_lr_neutral_config_sha256"] for row in identities}) == 1
    assert len({row["branch_neutral_config_sha256"] for row in identities}) == 2
