import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from oge.feature_export import (
    export_task_f_from_files,
    validate_task_f_checkpoint_payload,
    verify_task_f_artifact,
)
from oge.models import make_model
from oge.optimizers import make_optimizer
from oge.studies.hashing import canonical_sha256
from oge.training import runner as training_runner
from oge.training import (
    atomic_torch_save,
    capture_rng_state,
    fit_classifier,
    generate_execution_only_pilot,
    load_torch_artifact,
    load_training_config,
    make_scheduler,
    resolve_training_config,
    seed_everything,
    UpdateTelemetryRecorder,
    validate_resume_configuration,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3 * 4 * 4, 3)

    def forward(self, inputs):
        return self.classifier(inputs.flatten(1))


class RandomizedTrainDataset(Dataset):
    def __init__(self):
        self.images = torch.arange(6 * 3 * 4 * 4, dtype=torch.float32).reshape(6, 3, 4, 4)
        self.images = self.images / self.images.max()
        self.labels = torch.tensor([0, 1, 2, 0, 1, 2])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        jitter = random.random() + float(np.random.random()) + float(torch.rand(()))
        return {
            "image": self.images[index] + jitter * 1e-3,
            "class_label": self.labels[index],
            "sample_id": f"fixture:train-{index}",
        }


class FixedEvaluationDataset(Dataset):
    def __init__(self):
        self.images = torch.linspace(-1.0, 1.0, 4 * 3 * 4 * 4).reshape(4, 3, 4, 4)
        self.labels = torch.tensor([0, 1, 2, 0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"image": self.images[index], "class_label": self.labels[index]}


def _resolved_fixture_config(max_epochs, *, snapshots=None):
    return {
        "schema_version": "1.0",
        "dataset": {
            "protocol": "oge_cifar10_holdout_v1",
            "config_path": "/fixture/dataset.yaml",
            "data_root": "/fixture/data",
            "train_split": "id_train",
            "validation_split": "id_validation",
            "test_split": "id_test",
            "definition": {"fixture": True},
            "membership": {
                "train": {"sha256": "train", "line_count": 6},
                "validation": {"sha256": "validation", "line_count": 4},
                "test": {"sha256": "test", "line_count": 4},
            },
            "membership_manifest": {"sha256": "manifest", "row_count": 1},
        },
        "model": {"name": "wrn28_10", "num_classes": 10, "dropout_rate": 0.0},
        "loss": {"name": "cross_entropy", "label_smoothing": 0.0},
        "optimizer": {
            "name": "sgd",
            "lr": 0.05,
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 0.001,
            "weight_decay_policy": "weights_only_no_bias_norm",
        },
        "scheduler": {
            "name": "multistep",
            "milestones": [2],
            "gamma": 0.5,
            "step_timing": "end_of_epoch",
        },
        "training": {
            "max_epochs": max_epochs,
            "batch_size": 2,
            "seed": 17,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
            "persistent_workers": False,
            "precision": "fp32",
            "deterministic": True,
        },
        "checkpoint": {
            "snapshot_epochs": [0, 1, 2, 3] if snapshots is None else snapshots
        },
        "runtime": {"device": "cpu"},
    }


def _components(config):
    training = config["training"]
    generator = seed_everything(training["seed"], deterministic=True)
    initial_rng_state = capture_rng_state(generator)
    model = TinyClassifier()
    optimizer = make_optimizer(model, config["optimizer"])
    scheduler = make_scheduler(optimizer, config["scheduler"])
    train_loader = DataLoader(
        RandomizedTrainDataset(),
        batch_size=training["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(FixedEvaluationDataset(), batch_size=2)
    test_loader = DataLoader(FixedEvaluationDataset(), batch_size=2)
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": nn.CrossEntropyLoss(),
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
        "train_generator": generator,
        "initial_rng_state": initial_rng_state,
    }


def _run(
    config,
    run_dir,
    *,
    resume_from=None,
    fork_from_prefix=None,
    defer_id_test=False,
    test_loader=None,
):
    components = _components(config)
    if test_loader is not None:
        components["test_loader"] = test_loader
    summary = fit_classifier(
        **components,
        resolved_config=config,
        run_dir=run_dir,
        device="cpu",
        oge_git_sha="f" * 40,
        resume_from=resume_from,
        fork_from_prefix=fork_from_prefix,
        defer_id_test=defer_id_test,
    )
    return components["model"], summary


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _history_without_timing(history):
    return [
        {key: value for key, value in row.items() if key != "elapsed_seconds"}
        for row in history
    ]


def test_continuous_three_epochs_match_two_plus_one_resumed_epoch(tmp_path):
    continuous_dir = tmp_path / "continuous"
    split_dir = tmp_path / "split"
    _run(_resolved_fixture_config(3), continuous_dir)
    _run(_resolved_fixture_config(2), split_dir)
    _run(
        _resolved_fixture_config(3),
        split_dir,
        resume_from=split_dir / "checkpoints/last.pt",
    )

    continuous = load_torch_artifact(continuous_dir / "checkpoints/last.pt")
    resumed = load_torch_artifact(split_dir / "checkpoints/last.pt")

    assert continuous["completed_epoch"] == resumed["completed_epoch"] == 3
    assert continuous["global_step"] == resumed["global_step"]
    assert continuous["best_validation"] == resumed["best_validation"]
    assert _history_without_timing(continuous["history"]) == _history_without_timing(
        resumed["history"]
    )
    _assert_nested_equal(continuous["model_state"], resumed["model_state"])
    _assert_nested_equal(continuous["optimizer_state"], resumed["optimizer_state"])
    _assert_nested_equal(continuous["scheduler_state"], resumed["scheduler_state"])
    _assert_nested_equal(continuous["rng_state"], resumed["rng_state"])


def _zero_decay_config(max_epochs, *, optimizer_name="sgd", weight_decay=0.0):
    config = _resolved_fixture_config(max_epochs)
    optimizer = {
        "name": optimizer_name,
        "lr": 0.05,
        "weight_decay": weight_decay,
        "weight_decay_policy": "weights_only_no_bias_norm",
    }
    if optimizer_name in {"adam", "adamw"}:
        optimizer.update({"beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8})
    else:
        optimizer.update({"momentum": 0.9, "nesterov": True})
    config["optimizer"] = optimizer
    return config


@pytest.mark.parametrize("optimizer_name", ["sgd", "adam"])
def test_zero_decay_fork_matches_uninterrupted_zero_decay_continuation(
    tmp_path, optimizer_name
):
    import shutil

    prefix_dir = tmp_path / "prefix"
    resumed_dir = tmp_path / "ordinary_resume"
    fork_dir = tmp_path / "zero_fork"
    _run(_zero_decay_config(1, optimizer_name=optimizer_name), prefix_dir)
    shutil.copytree(prefix_dir, resumed_dir)

    _run(
        _zero_decay_config(2, optimizer_name=optimizer_name),
        resumed_dir,
        resume_from=resumed_dir / "checkpoints/last.pt",
    )
    _, fork_summary = _run(
        _zero_decay_config(2, optimizer_name=optimizer_name),
        fork_dir,
        fork_from_prefix=prefix_dir / "checkpoints/last.pt",
    )

    resumed = load_torch_artifact(resumed_dir / "checkpoints/last.pt")
    forked = load_torch_artifact(fork_dir / "checkpoints/last.pt")
    _assert_nested_equal(resumed["model_state"], forked["model_state"])
    _assert_nested_equal(resumed["optimizer_state"], forked["optimizer_state"])
    _assert_nested_equal(resumed["scheduler_state"], forked["scheduler_state"])
    _assert_nested_equal(resumed["rng_state"], forked["rng_state"])
    assert _history_without_timing(resumed["history"]) == _history_without_timing(
        forked["history"]
    )
    assert fork_summary["fork"]["status"] == "completed"
    manifest = json.loads((fork_dir / "fork_manifest.json").read_text())
    assert manifest["ordinary_resume_rules_changed"] is False
    assert manifest["source_run_id"] == load_torch_artifact(
        prefix_dir / "checkpoints/last.pt"
    )["run_id"]
    assert manifest["branch_run_id"] == forked["run_id"]
    assert manifest["source_training_seed"] == manifest["branch_training_seed"] == 17
    assert len(manifest["source_config_sha256"]) == 64
    assert len(manifest["branch_config_sha256"]) == 64
    assert len(manifest["transferred_state_digest"]) == 64


def test_decay_fork_preserves_prefix_state_but_keeps_new_branch_decay(tmp_path):
    prefix_dir = tmp_path / "prefix"
    branch_dir = tmp_path / "branch"
    _run(_zero_decay_config(1), prefix_dir)

    _, summary = _run(
        _zero_decay_config(2, optimizer_name="sgdw", weight_decay=0.01),
        branch_dir,
        fork_from_prefix=prefix_dir / "checkpoints/last.pt",
    )

    manifest = json.loads((branch_dir / "fork_manifest.json").read_text())
    assert manifest["optimizer_family"] == "sgd"
    assert manifest["branch_optimizer"]["name"] == "sgdw"
    assert manifest["branch_optimizer"]["weight_decay"] == 0.01
    assert summary["artifact_paths"]["fork_manifest"] == str(
        branch_dir / "fork_manifest.json"
    )


@pytest.mark.parametrize(
    ("prefix_optimizer", "sibling_optimizers"),
    [
        ("sgd", ("sgd", "sgdw")),
        ("adam", ("adam", "adamw")),
    ],
)
def test_sibling_forks_share_state_digest_rng_and_first_minibatch(
    tmp_path, prefix_optimizer, sibling_optimizers
):
    prefix_dir = tmp_path / "prefix"
    prefix_config = _zero_decay_config(1, optimizer_name=prefix_optimizer)
    _run(prefix_config, prefix_dir)
    payload = load_torch_artifact(prefix_dir / "checkpoints/last.pt")
    if prefix_optimizer == "adam":
        populated_states = list(payload["optimizer_state"]["state"].values())
        assert populated_states
        assert all(
            {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
            for state in populated_states
        )

    observations = []
    for optimizer_name in sibling_optimizers:
        config = _zero_decay_config(
            2, optimizer_name=optimizer_name, weight_decay=0.01
        )
        components = _components(config)
        record = training_runner.restore_fork_checkpoint(
            payload,
            model=components["model"],
            optimizer=components["optimizer"],
            scheduler=components["scheduler"],
            train_generator=components["train_generator"],
            resolved_config=config,
        )
        first_batch = next(iter(components["train_loader"]))
        observations.append((record, first_batch, components["optimizer"]))

        assert record["source_config_sha256"] == canonical_sha256(
            payload["resolved_config"]
        )
        assert record["branch_config_sha256"] == canonical_sha256(config)
        assert record["source_training_seed"] == record["branch_training_seed"] == 17
        if prefix_optimizer == "adam":
            transferred_states = list(
                components["optimizer"].state_dict()["state"].values()
            )
            assert len(transferred_states) == len(populated_states)
            for source_state, transferred_state in zip(
                populated_states, transferred_states
            ):
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    _assert_nested_equal(source_state[key], transferred_state[key])

    assert (
        observations[0][0]["transferred_state_digest"]
        == observations[1][0]["transferred_state_digest"]
    )
    torch.testing.assert_close(
        observations[0][1]["image"], observations[1][1]["image"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        observations[0][1]["class_label"],
        observations[1][1]["class_label"],
        rtol=0,
        atol=0,
    )
    assert any(group["weight_decay"] == 0.01 for group in observations[0][2].param_groups)
    assert any(group["weight_decay"] == 0.01 for group in observations[1][2].param_groups)


def test_fork_rejects_nonzero_prefix_and_cross_family():
    saved = _zero_decay_config(1)
    saved["optimizer"]["weight_decay"] = 0.001
    current = _zero_decay_config(2)
    with pytest.raises(ValueError, match="zero-decay"):
        training_runner.validate_fork_configuration(saved, current, completed_epoch=1)

    saved["optimizer"]["weight_decay"] = 0.0
    current["optimizer"] = {
        "name": "adamw",
        "lr": 0.05,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.001,
        "weight_decay_policy": "weights_only_no_bias_norm",
    }
    with pytest.raises(ValueError, match="cross optimizer families"):
        training_runner.validate_fork_configuration(saved, current, completed_epoch=1)


def test_run_artifacts_checkpoint_schema_and_reload_logits(tmp_path):
    run_dir = tmp_path / "run"
    trained_model, summary = _run(_resolved_fixture_config(1, snapshots=[0, 1]), run_dir)

    expected_files = {
        "resolved_config.yaml",
        "run_metadata.json",
        "environment.json",
        "history.jsonl",
        "summary.json",
    }
    assert expected_files.issubset(path.name for path in run_dir.iterdir())
    assert (run_dir / "evaluation/final_id_test.json").is_file()
    assert (run_dir / "evaluation/best_val_id_test.json").is_file()
    assert (run_dir / "checkpoints/snapshots/epoch_0000.pt").is_file()
    assert (run_dir / "checkpoints/snapshots/epoch_0001.pt").is_file()

    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert len(history) == 1
    assert {
        "epoch",
        "global_step",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "validation_nll",
        "validation_accuracy",
        "is_best",
        "elapsed_seconds",
    }.issubset(history[0])

    last = load_torch_artifact(run_dir / "checkpoints/last.pt")
    best = load_torch_artifact(run_dir / "checkpoints/best_val.pt")
    initial_snapshot = load_torch_artifact(
        run_dir / "checkpoints/snapshots/epoch_0000.pt"
    )
    epoch_snapshot = load_torch_artifact(
        run_dir / "checkpoints/snapshots/epoch_0001.pt"
    )
    assert last["checkpoint_type"] == "last"
    assert best["checkpoint_type"] == "best_val"
    assert {
        "schema_version",
        "completed_epoch",
        "global_step",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "best_validation",
        "rng_state",
        "resolved_config",
        "oge_git_sha",
    }.issubset(last)
    assert set(last["rng_state"]) == {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "train_dataloader_generator",
    }
    for snapshot, epoch in ((initial_snapshot, 0), (epoch_snapshot, 1)):
        assert snapshot["checkpoint_type"] == "snapshot"
        assert snapshot["completed_epoch"] == epoch
        assert {
            "schema_version",
            "model_state",
            "protocol_name",
            "model_name",
            "oge_git_sha",
            "run_id",
        }.issubset(snapshot)
    assert not torch.equal(
        initial_snapshot["model_state"]["classifier.weight"],
        epoch_snapshot["model_state"]["classifier.weight"],
    )

    restored_model = TinyClassifier().eval()
    restored_model.load_state_dict(last["model_state"])
    trained_model.eval()
    inputs = torch.randn(2, 3, 4, 4)
    with torch.no_grad():
        expected_logits = trained_model(inputs)
        reloaded_logits = restored_model(inputs)
    torch.testing.assert_close(reloaded_logits, expected_logits, rtol=0, atol=0)

    assert summary["status"] == "completed"
    assert summary["completed_epoch"] == 1
    assert summary["final_validation"]["epoch"] == 1
    assert summary["best_validation"]["epoch"] == 1
    assert summary["artifact_paths"]["final_id_test"] == str(
        run_dir / "evaluation/final_id_test.json"
    )
    assert json.loads((run_dir / "summary.json").read_text()) == summary
    assert yaml.safe_load((run_dir / "resolved_config.yaml").read_text())[
        "training"
    ]["max_epochs"] == 1
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["artifact_role"] == "classifier_training_run"
    environment = json.loads((run_dir / "environment.json").read_text())
    assert environment["schema_version"] == "1.0"
    assert environment["executions"][0]["device"] == "cpu"
    execution_environment = environment["executions"][0]
    assert execution_environment["python_executable"]
    assert execution_environment["hostname"]
    if torch.cuda.is_available():
        assert execution_environment["nvidia_driver"]
    else:
        assert execution_environment["nvidia_driver"] is None
    assert execution_environment["numerical_policy"]["precision_contract"] == {
        "parameters": "fp32",
        "activations": "fp32",
        "storage": "fp32",
        "amp": False,
        "bf16": False,
    }
    assert set(execution_environment["numerical_policy"]["tf32"]) == {
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    }


def test_task_f_synthetic_checkpoint_carries_exact_b_input_provenance(
    tmp_path,
):
    config = generate_execution_only_pilot(
        base_config=_resolved_fixture_config(1, snapshots=[0, 1]),
        seed=997,  # fixture-only; not a proposed pilot or research seed
        max_epochs=1,
        audit_steps=[1],
    )
    run_dir = tmp_path / "task_f_synthetic"
    _, summary = _run(config, run_dir, defer_id_test=True)

    last = load_torch_artifact(run_dir / "checkpoints/last.pt")
    best = load_torch_artifact(run_dir / "checkpoints/best_val.pt")
    initial = load_torch_artifact(run_dir / "checkpoints/snapshots/epoch_0000.pt")
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    provenance = last["paired_control_provenance"]
    required = {
        "sibling_group_id",
        "initialization_sha256",
        "initial_python_rng_sha256",
        "initial_numpy_rng_sha256",
        "initial_torch_rng_sha256",
        "initial_dataloader_rng_sha256",
        "first_minibatch_ordered_sample_id_sha256",
        "first_minibatch_transformed_image_sha256",
        "branch_neutral_config_sha256",
        "execution_only",
    }
    assert required.issubset(provenance)
    assert provenance["first_minibatch_witness_status"] == "observed"
    assert provenance["execution_only"] is True
    assert initial["paired_control_provenance"] == provenance
    assert metadata["paired_control_provenance"] == provenance
    assert summary["paired_control_provenance"] == provenance
    assert summary["id_test"]["status"] == "deferred"
    assert summary["update_telemetry"]["record_count"] > 0
    assert validate_task_f_checkpoint_payload(last)["checkpoint_role"] == "last"
    assert validate_task_f_checkpoint_payload(best)["checkpoint_role"] == "best_val"
    assert validate_task_f_checkpoint_payload(initial)["checkpoint_role"] == "snapshot"
    assert last["run_id"] == config["task_f"]["run_id"]


def test_task_f_synthetic_wrn_checkpoint_is_consumed_by_b_exporter(tmp_path):
    base = _resolved_fixture_config(1, snapshots=[0, 1])
    base["training"]["batch_size"] = 6
    base["model"].update(
        {"depth": 28, "widen_factor": 10, "init_policy": "msr_fan_in"}
    )
    config = generate_execution_only_pilot(
        base_config=base,
        seed=991,  # fixture-only; not a proposed pilot or research seed
        max_epochs=1,
        audit_steps=[1],
    )
    generator = seed_everything(config["training"]["seed"], deterministic=True)
    initial_rng_state = capture_rng_state(generator)
    model = make_model(config["model"])
    optimizer = make_optimizer(model, config["optimizer"])
    scheduler = make_scheduler(optimizer, config["scheduler"])
    train_loader = DataLoader(
        RandomizedTrainDataset(),
        batch_size=6,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    evaluation_loader = DataLoader(FixedEvaluationDataset(), batch_size=2)
    run_dir = tmp_path / "task_f_wrn"
    fit_classifier(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        validation_loader=evaluation_loader,
        test_loader=evaluation_loader,
        train_generator=generator,
        initial_rng_state=initial_rng_state,
        resolved_config=config,
        run_dir=run_dir,
        device="cpu",
        oge_git_sha="e" * 40,
        defer_id_test=True,
    )
    checkpoint_path = run_dir / "checkpoints/last.pt"
    checkpoint = load_torch_artifact(checkpoint_path)
    validate_task_f_checkpoint_payload(checkpoint)

    dataset = FixedEvaluationDataset()
    input_path = tmp_path / "synthetic_id_input.npz"
    np.savez(
        input_path,
        images=dataset.images.numpy().astype(np.float32),
        sample_ids=np.asarray([f"fixture:{index}" for index in range(len(dataset))]),
        is_id=np.ones(len(dataset), dtype=np.bool_),
    )
    artifact = export_task_f_from_files(
        checkpoint_path=checkpoint_path,
        input_npz_path=input_path,
        artifact_root=tmp_path / "artifacts",
        dataset_split="synthetic_id_fixture",
        depth_tap="penultimate",
        batch_size=2,
    )
    verified = verify_task_f_artifact(artifact)
    assert verified["manifest"]["run_id"] == config["task_f"]["run_id"]
    assert verified["manifest"]["execution_only"] is True


def test_instrumentation_on_off_preserves_trajectory_and_checkpoint_state(tmp_path):
    config = _resolved_fixture_config(1, snapshots=[])
    without_dir = tmp_path / "without_telemetry"
    with_dir = tmp_path / "with_telemetry"

    without = _components(config)
    fit_classifier(
        **without,
        resolved_config=config,
        run_dir=without_dir,
        device="cpu",
        oge_git_sha="f" * 40,
    )
    with_telemetry = _components(config)
    recorder = UpdateTelemetryRecorder(
        model=with_telemetry["model"],
        optimizer=with_telemetry["optimizer"],
        audit_steps=[1, 2, 3],
        include_batch_norm=False,
    )
    fit_classifier(
        **with_telemetry,
        resolved_config=config,
        run_dir=with_dir,
        device="cpu",
        oge_git_sha="f" * 40,
        update_telemetry_recorder=recorder,
    )

    checkpoint_without = load_torch_artifact(without_dir / "checkpoints/last.pt")
    checkpoint_with = load_torch_artifact(with_dir / "checkpoints/last.pt")
    for field in ("model_state", "optimizer_state", "scheduler_state", "rng_state"):
        _assert_nested_equal(checkpoint_without[field], checkpoint_with[field])
    assert _history_without_timing(checkpoint_without["history"]) == _history_without_timing(
        checkpoint_with["history"]
    )
    assert (with_dir / "update_telemetry.jsonl").is_file()
    assert not (without_dir / "update_telemetry.jsonl").exists()


class ExplodingIDTestLoader:
    def __iter__(self):
        raise AssertionError("deferred ID-test loader was iterated")


def test_study_deferred_id_test_skips_evaluator_loader_and_artifacts(tmp_path, monkeypatch):
    run_dir = tmp_path / "deferred"
    sentinel = ExplodingIDTestLoader()
    original_evaluator = training_runner.evaluate_classifier

    def guarded_evaluator(model, loader, criterion, *, device):
        if loader is sentinel:
            raise AssertionError("deferred ID-test evaluator was called")
        return original_evaluator(model, loader, criterion, device=device)

    monkeypatch.setattr(training_runner, "evaluate_classifier", guarded_evaluator)
    _, summary = _run(
        _resolved_fixture_config(1, snapshots=[]),
        run_dir,
        defer_id_test=True,
        test_loader=sentinel,
    )

    assert summary["id_test"] == {
        "status": "deferred",
        "metrics_available": False,
        "artifacts_created": False,
    }
    assert "final_id_test" not in summary
    assert "best_val_id_test" not in summary
    assert "final_id_test" not in summary["artifact_paths"]
    assert "best_val_id_test" not in summary["artifact_paths"]
    assert not (run_dir / "evaluation/final_id_test.json").exists()
    assert not (run_dir / "evaluation/best_val_id_test.json").exists()
    persisted = json.loads((run_dir / "summary.json").read_text())
    assert persisted == summary
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["id_test_evaluation"] == "deferred"


@pytest.mark.parametrize(
    "path,value",
    [
        (("optimizer", "name"), "adam"),
        (("optimizer", "lr"), 0.01),
        (("optimizer", "weight_decay"), 0.0),
        (("optimizer", "momentum"), 0.0),
        (("optimizer", "nesterov"), False),
        (("optimizer", "beta1"), 0.8),
        (("optimizer", "eps"), 1e-7),
        (("scheduler", "name"), "none"),
        (("scheduler", "milestones"), [1, 2]),
        (("scheduler", "gamma"), 0.1),
        (("model", "dropout_rate"), 0.3),
        (("dataset", "protocol"), "other"),
        (("dataset", "config_path"), "/other/dataset.yaml"),
        (("dataset", "data_root"), "/other/data"),
        (("dataset", "membership"), {"changed": True}),
        (("training", "seed"), 99),
    ],
)
def test_incompatible_resume_configuration_is_rejected(path, value):
    saved = _resolved_fixture_config(2)
    current = copy.deepcopy(saved)
    current["training"]["max_epochs"] = 3
    current[path[0]][path[1]] = value

    with pytest.raises(ValueError, match="Incompatible resume configuration"):
        validate_resume_configuration(saved, current, completed_epoch=2)


def test_incompatible_resume_error_identifies_changed_field():
    saved = _resolved_fixture_config(2)
    current = copy.deepcopy(saved)
    current["training"]["max_epochs"] = 3
    current["optimizer"]["lr"] = 0.01

    with pytest.raises(ValueError, match="optimizer.lr"):
        validate_resume_configuration(saved, current, completed_epoch=2)


@pytest.mark.parametrize("field,value", [("beta1", 0.8), ("beta2", 0.99), ("eps", 1e-7)])
def test_adam_resume_rejects_effective_hyperparameter_change(field, value):
    saved = _resolved_fixture_config(2)
    saved["optimizer"] = {
        "name": "adam",
        "lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0001,
        "weight_decay_policy": "weights_only_no_bias_norm",
    }
    current = copy.deepcopy(saved)
    current["training"]["max_epochs"] = 3
    current["optimizer"][field] = value

    with pytest.raises(ValueError, match=f"optimizer.{field}"):
        validate_resume_configuration(saved, current, completed_epoch=2)


def test_resume_allows_only_max_epoch_extension_and_future_snapshot_addition():
    saved = _resolved_fixture_config(3, snapshots=[0, 1])
    current = copy.deepcopy(saved)
    current["training"]["max_epochs"] = 4
    current["checkpoint"]["snapshot_epochs"] = [0, 1, 3, 4]

    validate_resume_configuration(saved, current, completed_epoch=2)

    removed = copy.deepcopy(current)
    removed["checkpoint"]["snapshot_epochs"] = [0, 3, 4]
    with pytest.raises(ValueError, match="existing snapshots were removed"):
        validate_resume_configuration(saved, removed, completed_epoch=2)

    past = copy.deepcopy(saved)
    past["training"]["max_epochs"] = 4
    past["checkpoint"]["snapshot_epochs"] = [0, 1, 2]
    with pytest.raises(ValueError, match="after completed_epoch"):
        validate_resume_configuration(saved, past, completed_epoch=2)


def test_same_boundary_resume_reconciles_best_snapshot_and_summary(tmp_path):
    run_dir = tmp_path / "run"
    config = _resolved_fixture_config(1, snapshots=[0, 1])
    _run(config, run_dir)
    (run_dir / "checkpoints/best_val.pt").unlink()
    (run_dir / "checkpoints/snapshots/epoch_0001.pt").unlink()
    (run_dir / "summary.json").unlink()

    _, summary = _run(
        config,
        run_dir,
        resume_from=run_dir / "checkpoints/last.pt",
    )

    assert summary["completed_epoch"] == 1
    assert (run_dir / "checkpoints/best_val.pt").is_file()
    assert (run_dir / "checkpoints/snapshots/epoch_0001.pt").is_file()
    assert (run_dir / "summary.json").is_file()


def test_atomic_checkpoint_failure_preserves_previous_complete_file(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_torch_save({"value": 1}, path)

    def fail_save(*args, **kwargs):
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="injected"):
        atomic_torch_save({"value": 2}, path)

    assert load_torch_artifact(path)["value"] == 1
    assert not list(tmp_path.glob(".last.pt.*.tmp"))


def test_none_scheduler_checkpoint_and_resume(tmp_path):
    run_dir = tmp_path / "none_scheduler"
    first = _resolved_fixture_config(1)
    first["scheduler"] = {"name": "none"}
    _run(first, run_dir)
    assert load_torch_artifact(run_dir / "checkpoints/last.pt")[
        "scheduler_state"
    ] is None

    resumed = copy.deepcopy(first)
    resumed["training"]["max_epochs"] = 2
    _run(
        resumed,
        run_dir,
        resume_from=run_dir / "checkpoints/last.pt",
    )
    checkpoint = load_torch_artifact(run_dir / "checkpoints/last.pt")
    assert checkpoint["completed_epoch"] == 2
    assert checkpoint["scheduler_state"] is None


def test_exact_validation_tie_does_not_replace_earliest_best_checkpoint(tmp_path):
    run_dir = tmp_path / "tie"
    config = _resolved_fixture_config(2)
    config["optimizer"].update(
        {"lr": 0.0, "momentum": 0.0, "nesterov": False, "weight_decay": 0.0}
    )
    config["scheduler"] = {"name": "none"}

    _run(config, run_dir)

    last = load_torch_artifact(run_dir / "checkpoints/last.pt")
    best = load_torch_artifact(run_dir / "checkpoints/best_val.pt")
    assert last["best_validation"]["epoch"] == 1
    assert best["completed_epoch"] == 1
    assert [row["is_best"] for row in last["history"]] == [True, False]


def _protocol_config(dataset_config_path):
    config_path = (
        Path(__file__).parents[1]
        / "configs/training/cifar10_wrn28_10_holdout_v1.yaml"
    )
    config = load_training_config(config_path)
    config["dataset"]["config_path"] = str(dataset_config_path)
    config["training"]["max_epochs"] = 1
    return config


def test_resolved_config_fingerprints_selected_imglist_membership(tmp_path):
    list_contents = {
        "train.txt": "a.png 0\nb.png 1\n",
        "validation.txt": "c.png 2\n",
        "test.txt": "d.png 3\n",
    }
    for name, content in list_contents.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    dataset_config = {
        "protocol_name": "oge_cifar10_holdout_v1",
        "datasets": {
            "id_train": {
                "dataset_name": "cifar10",
                "split": "train",
                "is_id": True,
                "group": "id",
                "imglist": "train.txt",
            },
            "id_validation": {
                "dataset_name": "cifar10",
                "split": "validation",
                "is_id": True,
                "group": "id",
                "imglist": "validation.txt",
            },
            "id_test": {
                "dataset_name": "cifar10",
                "split": "test",
                "is_id": True,
                "group": "id",
                "imglist": "test.txt",
            },
        },
    }
    manifest_header = {
        "record_type": "header",
        "protocol_name": "oge_cifar10_holdout_v1",
        "outputs": {
            "id_train": {
                "sha256": hashlib.sha256(list_contents["train.txt"].encode()).hexdigest(),
                "row_count": 2,
            },
            "id_validation": {
                "sha256": hashlib.sha256(
                    list_contents["validation.txt"].encode()
                ).hexdigest(),
                "row_count": 1,
            },
            "id_test": {
                "sha256": hashlib.sha256(list_contents["test.txt"].encode()).hexdigest(),
                "row_count": 1,
            },
        },
    }
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(manifest_header, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset_config["membership_manifest"] = {"path": "manifest.jsonl"}
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset_config), encoding="utf-8")

    resolved = resolve_training_config(
        _protocol_config(dataset_path),
        data_root=tmp_path,
        device="cpu",
    )

    assert resolved["dataset"]["data_root"] == str(tmp_path.resolve())
    assert resolved["dataset"]["definition"] == dataset_config
    assert resolved["dataset"]["membership"]["train"] == {
        "selection_key": "id_train",
        "imglist": "train.txt",
        "sha256": hashlib.sha256(list_contents["train.txt"].encode()).hexdigest(),
        "line_count": 2,
    }
    assert resolved["training"]["num_workers"] == 0
    assert resolved["training"]["precision"] == "fp32"

    defaults_config = _protocol_config(dataset_path)
    defaults_config["optimizer"] = {"name": "adam", "lr": 0.001}
    for key in ("num_classes", "depth", "widen_factor"):
        defaults_config["model"].pop(key)
    defaults_config["loss"].pop("label_smoothing")
    for key in ("pin_memory", "drop_last", "persistent_workers", "deterministic"):
        defaults_config["training"].pop(key)
    resolved_defaults = resolve_training_config(
        defaults_config,
        data_root=tmp_path,
        device="cpu",
    )
    assert resolved_defaults["model"]["num_classes"] == 10
    assert resolved_defaults["model"]["depth"] == 28
    assert resolved_defaults["model"]["widen_factor"] == 10
    assert resolved_defaults["model"]["init_policy"] == "msr_fan_in"
    assert resolved_defaults["loss"]["label_smoothing"] == 0.0
    assert resolved_defaults["optimizer"] == {
        "name": "adam",
        "lr": 0.001,
        "weight_decay_policy": "weights_only_no_bias_norm",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    assert resolved_defaults["training"]["deterministic"] is False


def test_first_protocol_rejects_workers_and_non_fp32(tmp_path):
    config = load_training_config(
        Path(__file__).parents[1]
        / "configs/training/cifar10_wrn28_10_holdout_v1.yaml"
    )
    config["training"]["num_workers"] = 1
    with pytest.raises(ValueError, match="num_workers=0"):
        resolve_training_config(config, data_root=tmp_path, device="cpu")

    config["training"]["num_workers"] = 0
    config["training"]["precision"] = "amp"
    with pytest.raises(ValueError, match="fp32"):
        resolve_training_config(config, data_root=tmp_path, device="cpu")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("train_split", "id_test", "train_split"),
        ("validation_split", "id_train", "validation_split"),
        ("test_split", "id_validation", "test_split"),
    ],
)
def test_first_protocol_rejects_noncanonical_split_roles(tmp_path, field, value, match):
    config = load_training_config(
        Path(__file__).parents[1]
        / "configs/training/cifar10_wrn28_10_holdout_v1.yaml"
    )
    config["dataset"][field] = value
    with pytest.raises(ValueError, match=match):
        resolve_training_config(config, data_root=tmp_path, device="cpu")


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("scheduler", "milestones", [60.9], "milestones"),
        ("checkpoint", "snapshot_epochs", [1.9], "snapshot_epochs"),
    ],
)
def test_epoch_configuration_rejects_non_integer_values(
    tmp_path, section, key, value, match
):
    config = load_training_config(
        Path(__file__).parents[1]
        / "configs/training/cifar10_wrn28_10_holdout_v1.yaml"
    )
    config[section][key] = value
    with pytest.raises(ValueError, match=match):
        resolve_training_config(config, data_root=tmp_path, device="cpu")


def test_production_runner_moves_model_before_optimizer_and_wires_id_loaders(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "training.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    resolved = _resolved_fixture_config(1)
    generator = torch.Generator().manual_seed(1)
    loader = DataLoader(FixedEvaluationDataset(), batch_size=2)
    loaders = {
        "id": {
            "id_train": loader,
            "id_validation": loader,
            "id_test": loader,
        }
    }

    class TrackedModel(TinyClassifier):
        moved = False

        def to(self, *args, **kwargs):
            result = super().to(*args, **kwargs)
            self.moved = True
            return result

    model = TrackedModel()
    optimizer_holder = {}

    def tracked_optimizer(actual_model, optimizer_config):
        assert actual_model is model
        assert model.moved is True
        optimizer = torch.optim.SGD(actual_model.parameters(), lr=optimizer_config["lr"])
        optimizer_holder["optimizer"] = optimizer
        return optimizer

    def tracked_fit(**kwargs):
        assert kwargs["model"] is model
        assert kwargs["optimizer"] is optimizer_holder["optimizer"]
        assert kwargs["train_loader"] is loader
        assert kwargs["validation_loader"] is loader
        assert kwargs["test_loader"] is loader
        return {"completed_epoch": 1}

    monkeypatch.setattr(training_runner, "resolve_training_config", lambda *a, **k: resolved)
    monkeypatch.setattr(training_runner, "seed_everything", lambda *a, **k: generator)
    monkeypatch.setattr(
        training_runner, "build_openood_cifar10_loaders", lambda *a, **k: loaders
    )
    monkeypatch.setattr(training_runner, "make_model", lambda config: model)
    monkeypatch.setattr(training_runner, "make_optimizer", tracked_optimizer)
    monkeypatch.setattr(training_runner, "make_scheduler", lambda optimizer, config: None)
    monkeypatch.setattr(training_runner, "repository_git_state", lambda: ("sha", False))
    monkeypatch.setattr(training_runner, "fit_classifier", tracked_fit)

    summary = training_runner.run_training_from_config(
        config_path=config_path,
        data_root=tmp_path,
        run_dir=tmp_path / "run",
        device="cpu",
    )

    assert summary == {"completed_epoch": 1}
