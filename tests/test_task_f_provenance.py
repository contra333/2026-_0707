import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from oge.training import (
    build_task_f_training_config,
    capture_rng_state,
    create_initial_paired_provenance,
    generate_research_run_matrix,
    load_training_config,
    observe_first_minibatch,
    seed_everything,
    validate_resume_paired_identity,
    validate_sibling_provenance,
)


FIXTURE_ANCHOR_SEEDS = (101, 103, 107, 109, 113)
FIXTURE_FACTORIAL_SEEDS = (101, 107, 113)
FIXTURE_SGDM_SEEDS = (211, 223, 227)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 2)


def _base_config():
    return load_training_config(
        Path(__file__).parents[1]
        / "configs/training/cifar10_wrn28_10_holdout_v1.yaml"
    )


def _sibling_runs():
    plan = generate_research_run_matrix(
        anchor_seeds=FIXTURE_ANCHOR_SEEDS,
        adam_factorial_seeds=FIXTURE_FACTORIAL_SEEDS,
        sgdm_seeds=FIXTURE_SGDM_SEEDS,
    )
    return [
        run
        for run in plan["runs"]
        if run["sibling_group_id"] == "task-f-adam-lr1e-3-seed101"
        and run["cell_id"] == "adam_lr1e-3_wd1e-4_anchor"
    ]


def _resolved_config(run):
    config = build_task_f_training_config(_base_config(), run)
    config["dataset"]["membership"] = {
        "train": {"sha256": "train", "line_count": 6},
        "validation": {"sha256": "validation", "line_count": 4},
        "test": {"sha256": "test", "line_count": 4},
    }
    config["dataset"]["membership_manifest"] = {
        "sha256": "manifest",
        "row_count": 1,
    }
    config["runtime"] = {"device": "cpu"}
    return config


def _provenance(run, batch):
    config = _resolved_config(run)
    generator = seed_everything(config["training"]["seed"], deterministic=True)
    initial_rng = capture_rng_state(generator)
    model = TinyModel()
    provenance = create_initial_paired_provenance(
        resolved_config=config,
        model=model,
        initial_rng_state=initial_rng,
    )
    observe_first_minibatch(provenance, batch)
    return provenance


def test_same_initialization_rng_first_batch_and_augmentation_witnesses_match():
    batch = {
        "sample_id": ["fixture:a", "fixture:b"],
        "image": torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2),
        "class_label": torch.tensor([0, 1]),
    }
    records = [_provenance(run, batch) for run in _sibling_runs()]
    validate_sibling_provenance(records)

    fields = (
        "initialization_sha256",
        "initial_python_rng_sha256",
        "initial_numpy_rng_sha256",
        "initial_torch_rng_sha256",
        "initial_dataloader_rng_sha256",
        "first_minibatch_ordered_sample_id_sha256",
        "first_minibatch_transformed_image_sha256",
        "branch_neutral_config_sha256",
    )
    for field in fields:
        assert len({record[field] for record in records}) == 1


def test_first_batch_observation_itself_does_not_advance_rng_or_consume_again():
    run = _sibling_runs()[0]
    config = _resolved_config(run)
    generator = seed_everything(config["training"]["seed"], deterministic=True)
    initial_rng = capture_rng_state(generator)
    provenance = create_initial_paired_provenance(
        resolved_config=config,
        model=TinyModel(),
        initial_rng_state=initial_rng,
    )
    batch = {
        "sample_id": ["fixture:a"],
        "image": torch.ones(1, 1, 2, 2),
    }
    before = torch.get_rng_state().clone()
    observe_first_minibatch(provenance, batch)
    assert torch.equal(torch.get_rng_state(), before)
    with pytest.raises(ValueError, match="exactly once"):
        observe_first_minibatch(provenance, batch)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sibling_group_id", "different-group"),
        ("training_seed", 999),
        ("data_stream_id", "different-stream"),
        ("first_minibatch_ordered_sample_id_sha256", "0" * 64),
        ("first_minibatch_transformed_image_sha256", "1" * 64),
    ],
)
def test_sibling_group_seed_and_stream_mismatches_are_rejected(field, value):
    batch = {
        "sample_id": ["fixture:a", "fixture:b"],
        "image": torch.ones(2, 1, 2, 2),
    }
    records = [_provenance(run, batch) for run in _sibling_runs()[:2]]
    records[1][field] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_sibling_provenance(records)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sibling_group_id", "changed"),
        ("data_stream_id", "changed"),
        ("branch_neutral_config_sha256", "f" * 64),
        ("execution_only", True),
        ("initialization_sha256", "e" * 64),
    ],
)
def test_resume_rejects_paired_provenance_identity_changes(field, value):
    batch = {
        "sample_id": ["fixture:a"],
        "image": torch.ones(1, 1, 2, 2),
    }
    saved = _provenance(_sibling_runs()[0], batch)
    current = copy.deepcopy(saved)
    current["first_minibatch_ordered_sample_id_sha256"] = None
    current["first_minibatch_transformed_image_sha256"] = None
    current["first_minibatch_witness_status"] = "pending"
    current[field] = value
    with pytest.raises(ValueError, match=field):
        validate_resume_paired_identity(saved, current)
