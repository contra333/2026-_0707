import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import oge.evaluation.resnet18_replication_production as production
from oge.evaluation.resnet18_replication import LARGE_CELL, SMALL_CELL
from oge.studies.hashing import canonical_json_bytes, canonical_sha256
from oge.training.resnet18_replication_plan import (
    generate_resnet18_replication_matrix,
)


def _sha(character: str) -> str:
    return character * 64


def _production_plan() -> dict:
    matrix = generate_resnet18_replication_matrix()
    records = []
    for index, row in enumerate(matrix["runs"]):
        records.append(
            {
                "run_id": row["run_id"],
                "cell_id": row["cell_id"],
                "training_seed": row["training_seed"],
                "branch_policy": row["branch_policy"],
                "sibling_group_id": row["sibling_group_id"],
                "cross_lr_pairing_block_id": row["cross_lr_pairing_block_id"],
                "host_id": production._HOST_BY_SEED[row["training_seed"]],
                "checkpoint_role": "last",
                "checkpoint_epoch": 200,
                "checkpoint_path": f"/fixture/{row['run_id']}/last.pt",
                "checkpoint_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "id_splits": list(production.ID_SPLITS),
                "protected_splits": list(production.PROTECTED_SPLITS),
            }
        )
    payload = {
        "schema_version": production.PLAN_SCHEMA_VERSION,
        "study_id": production.RESNET18_REPLICATION_STUDY_ID,
        "evaluation_git_sha": "e" * 40,
        "source_training_sha": production.SOURCE_TRAINING_SHA,
        "source_training_terminal_sha256": production.SOURCE_TRAINING_TERMINAL_SHA256,
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "depth_tap": "penultimate",
        "feature_dim": 512,
        "id_fit_split": "id_train",
        "id_validation_split": "id_validation",
        "protected_splits": list(production.PROTECTED_SPLITS),
        "score_targets": ["raw_md", "raw_rmd", "raw_marginal", "l2_md"],
        "pair_direction": "coupled_minus_decoupled",
        "records": records,
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return production.validate_production_plan(payload)


def _authorization(plan: dict) -> dict:
    return production.build_protected_authorization(
        plan=plan, evaluation_git_sha="e" * 40, approved_at="2026-08-19"
    )


def test_production_plan_binds_exact_training_terminal(monkeypatch, tmp_path):
    matrix = generate_resnet18_replication_matrix()
    runs = []
    for index, row in enumerate(matrix["runs"]):
        runs.append(
            {
                "run_id": row["run_id"],
                "seed": row["training_seed"],
                "branch_policy": row["branch_policy"],
                "sibling_group_id": row["sibling_group_id"],
                "cross_lr_pairing_block_id": row["cross_lr_pairing_block_id"],
                "initialization_sha256": _sha("a"),
                "data_stream_id": row["cross_lr_pairing_block_id"],
                "initial_dataloader_rng_sha256": _sha("b"),
                "first_minibatch_ordered_sample_id_sha256": _sha("c"),
                "first_minibatch_transformed_image_sha256": _sha("d"),
                "checkpoints": [
                    {
                        "checkpoint_role": "last",
                        "checkpoint_epoch": 200,
                        "path": f"/fixture/{row['run_id']}/last.pt",
                        "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                    }
                ],
            }
        )
    terminal = {
        "schema_version": "fixture",
        "status": "PASS",
        "execution_sha": production.SOURCE_TRAINING_SHA,
        "run_count": 20,
        "seed_count": 5,
        "runs": runs,
    }
    terminal_path = tmp_path / "terminal.json"
    terminal_path.write_bytes(canonical_json_bytes(terminal) + b"\n")
    monkeypatch.setattr(
        production,
        "SOURCE_TRAINING_TERMINAL_SHA256",
        production.sha256_file(terminal_path),
    )
    plan = production.build_production_plan(
        run_plan=matrix,
        training_terminal=terminal,
        training_terminal_path=terminal_path,
        evaluation_git_sha="e" * 40,
    )
    assert len(plan["records"]) == 20
    assert {row["host_id"] for row in plan["records"]} == {
        "curie",
        "lise",
        "precision_medicine",
    }
    assert {row["checkpoint_epoch"] for row in plan["records"]} == {200}


def test_authorization_rejects_scope_or_plan_drift():
    plan = _production_plan()
    authorization = _authorization(plan)
    assert (
        production.validate_protected_authorization(authorization, plan=plan)[
            "authorization_sha256"
        ]
        == authorization["authorization_sha256"]
    )
    broken = copy.deepcopy(authorization)
    broken["selection_or_tuning"] = True
    broken_without_hash = dict(broken)
    broken_without_hash.pop("authorization_sha256")
    broken["authorization_sha256"] = canonical_sha256(broken_without_hash)
    with pytest.raises(ValueError, match="no_tuning"):
        production.validate_protected_authorization(broken, plan=plan)


def test_id_fit_then_protected_pair_is_reconstructible(monkeypatch, tmp_path):
    plan = _production_plan()
    authorization = _authorization(plan)
    rng = np.random.default_rng(20260819)
    labels = np.tile(np.arange(10), 20)
    centers = rng.normal(size=(10, 12))
    arrays = {}
    manifests = {}
    role_paths = {}
    for role in ("adam_coupled", "adamw_decoupled"):
        record = next(
            row
            for row in plan["records"]
            if row["cell_id"] == LARGE_CELL
            and row["training_seed"] == 0
            and row["branch_policy"] == role
        )
        id_root = tmp_path / f"{role}-id"
        protected_root = tmp_path / f"{role}-protected"
        role_paths[role] = (id_root, protected_root)
        role_shift = 0.12 if role == "adam_coupled" else -0.05
        train = centers[labels] + rng.normal(scale=0.5, size=(len(labels), 12))
        train[:, 0] += role_shift
        validation_labels = np.tile(np.arange(10), 3)
        validation = centers[validation_labels] + rng.normal(
            scale=0.55, size=(len(validation_labels), 12)
        )
        arrays[(id_root, "id_train")] = {
            "features": train.astype(np.float32),
            "logits": rng.normal(size=(len(labels), 10)).astype(np.float32),
            "class_labels": labels.astype(np.int64),
            "predictions": np.zeros(len(labels), dtype=np.int64),
            "is_id": np.ones(len(labels), dtype=np.bool_),
            "sample_ids": np.asarray([f"train:{i}" for i in range(len(labels))]),
        }
        arrays[(id_root, "id_validation")] = {
            "features": validation.astype(np.float32),
            "logits": rng.normal(size=(len(validation), 10)).astype(np.float32),
            "class_labels": validation_labels.astype(np.int64),
            "predictions": np.zeros(len(validation), dtype=np.int64),
            "is_id": np.ones(len(validation), dtype=np.bool_),
            "sample_ids": np.asarray([f"val:{i}" for i in range(len(validation))]),
        }
        id_splits = {
            split: {
                "relative_directory": split,
                "ordered_sample_id_sha256": f"order-{split}",
                "sample_count": len(arrays[(id_root, split)]["features"]),
            }
            for split in production.ID_SPLITS
        }
        manifests[id_root] = {
            "checkpoint": {
                "training_run_id": record["run_id"],
                "sha256": record["checkpoint_sha256"],
                "role": "last",
                "completed_epoch": 200,
            },
            "dataset": {"splits": id_splits},
            "model": {"feature_dim": 512, "class_count": 10},
        }
        protected_splits = {}
        for index, split in enumerate(production.PROTECTED_SPLITS):
            count = 24 if split == "id_test" else 17
            query_labels = np.arange(count) % 10
            query = centers[query_labels] + rng.normal(
                scale=0.7 + 0.15 * index, size=(count, 12)
            )
            if split != "id_test":
                query[:, (index + 1) % 12] += 1.5 + index * 0.2
            query[:, 0] += role_shift
            logits = np.full((count, 10), -3.0, dtype=np.float32)
            logits[np.arange(count), query_labels] = 3.0
            arrays[(protected_root, split)] = {
                "features": query.astype(np.float32),
                "logits": logits,
                "class_labels": query_labels.astype(np.int64),
                "predictions": query_labels.astype(np.int64),
                "is_id": np.full(count, split == "id_test", dtype=np.bool_),
                "sample_ids": np.asarray([f"{split}:{i}" for i in range(count)]),
            }
            protected_splits[split] = {
                "relative_directory": split,
                "ordered_sample_id_sha256": f"order-{split}",
                "sample_count": count,
            }
        manifests[protected_root] = {
            "checkpoint": {
                "training_run_id": record["run_id"],
                "sha256": record["checkpoint_sha256"],
                "role": "last",
                "completed_epoch": 200,
            },
            "dataset": {"splits": protected_splits},
            "model": {"feature_dim": 512, "class_count": 10},
        }

    monkeypatch.setattr(
        production,
        "verify_raw_feature_artifact",
        lambda path: {"manifest": manifests[Path(path)]},
    )
    monkeypatch.setattr(
        production,
        "_load_split_arrays",
        lambda root, manifest, split: arrays[(Path(root), split)],
    )
    fits = {}
    for role, (id_root, _) in role_paths.items():
        run_id = manifests[id_root]["checkpoint"]["training_run_id"]
        fits[role] = production.fit_id_artifact(
            plan=plan,
            run_id=run_id,
            id_artifact=id_root,
            output_root=tmp_path / "fits",
        )
        assert production.verify_id_fit_artifact(fits[role])["manifest"][
            "protected_data_access"
        ] is False
    pair = production.evaluate_paired_endpoint(
        plan=plan,
        authorization=authorization,
        cell_id=LARGE_CELL,
        training_seed=0,
        coupled_fit=fits["adam_coupled"],
        decoupled_fit=fits["adamw_decoupled"],
        coupled_protected=role_paths["adam_coupled"][1],
        decoupled_protected=role_paths["adamw_decoupled"][1],
        output_root=tmp_path / "pairs",
        chunk_size=9,
    )
    record = production.verify_pair_artifact(pair)["record"]
    assert record["status"] == "PASS"
    assert set(record["datasets"]) == set(production.OOD_SPLITS)
    for split in production.OOD_SPLITS:
        row = record["datasets"][split]["md"]
        assert row["delta_auroc"] == pytest.approx(row["gain"] - row["loss"])
        assert row["pair_order_churn"] == pytest.approx(row["gain"] + row["loss"])

