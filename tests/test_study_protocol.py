import copy
from pathlib import Path

import pytest

from oge.studies.hashing import canonical_json_bytes
from oge.studies.protocol import (
    GRIDS,
    OPTIMIZER_ORDER,
    PROTOCOL_VERSION,
    generate_grid_bundle,
    load_study_config,
    materialize_grid_bundle,
    validate_grid_bundle,
    validate_study_config,
    verify_materialized_grid_bundle,
)
from oge.training import load_training_config, resolve_training_config

ROOT = Path(__file__).parents[1]
STUDY_PATH = ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml"
GOLDEN_TABLE_HASHES = {
    "sgd": "01d7a11eca9642502938060bcb1bcd823db6380251e865f81c1a2565d3f8e80f",
    "adam": "97bc7b301f9e8fc35fb795344343aa20befb210e38147d4d6cf9afbb0fcc41f1",
    "adamw": "7fcb493ad1b0996f231516abed385ecf05a5ada2766ec402011738183a887bbd",
}
GOLDEN_MANIFEST_HASH = "9f93f24e8b46c6d94b576d801f72913f3348a6fd8e67a018d4bd1e6f0f66fbd8"
GOLDEN_DATASET_IDENTITY_HASH = (
    "b094acb0ed0182188b8da634d18ebd73006da069d1429a76200b193b989e1215"
)


def _bundle():
    study = load_study_config(STUDY_PATH)
    base = load_training_config(ROOT / study["base_training_config"])
    resolved = resolve_training_config(base, data_root=ROOT, device="cpu")
    return study, generate_grid_bundle(study, resolved)


def test_grid_has_exact_cartesian_products_budget_order_and_golden_hashes():
    study, bundle = _bundle()
    validate_grid_bundle(bundle)
    manifest = bundle["manifest"]
    assert manifest["protocol_version"] == PROTOCOL_VERSION
    assert manifest["total_assigned_rows"] == 36
    assert manifest["table_hashes"] == GOLDEN_TABLE_HASHES
    assert manifest["manifest_hash"] == GOLDEN_MANIFEST_HASH
    assert (
        manifest["dataset_identity"]["dataset_identity_hash"]
        == GOLDEN_DATASET_IDENTITY_HASH
    )
    trial_ids = []
    for optimizer in OPTIMIZER_ORDER:
        rows = bundle["tables"][optimizer]["rows"]
        assert len(rows) == 12
        expected = [
            (lr, weight_decay)
            for lr in GRIDS[optimizer]["lr"]
            for weight_decay in GRIDS[optimizer]["weight_decay"]
        ]
        actual = [
            (
                row["scientific_config"]["optimizer"]["lr"],
                row["scientific_config"]["optimizer"]["weight_decay"],
            )
            for row in rows
        ]
        assert actual == expected
        assert [(row["lr_rank"], row["weight_decay_rank"]) for row in rows] == [
            (lr_rank, weight_decay_rank)
            for lr_rank in range(3)
            for weight_decay_rank in range(4)
        ]
        trial_ids.extend(row["trial_id"] for row in rows)
    assert len(trial_ids) == len(set(trial_ids)) == 36
    assert "sampler" not in study
    assert "sampler" not in manifest


def test_every_row_materializes_holdout_membership_fan_in_and_portable_paths():
    _, bundle = _bundle()
    for table in bundle["tables"].values():
        for row in table["rows"]:
            config = row["scientific_config"]
            assert config["model"]["init_policy"] == "msr_fan_in"
            assert config["dataset"]["protocol"] == "oge_cifar10_holdout_v1"
            assert set(config["dataset"]["membership"]) == {
                "train",
                "validation",
                "test",
            }
            assert config["dataset"]["membership_manifest"]["sha256"]
            assert config["dataset"]["config_path"] == (
                "configs/datasets/oge_cifar10_holdout_v1.yaml"
            )
            assert "data_root" not in config["dataset"]
            assert "runtime" not in config


def test_study_config_version_and_grid_mutation_are_rejected():
    study, bundle = _bundle()
    wrong = copy.deepcopy(study)
    wrong["protocol_version"] = "wrn28_10_optimizer_hpo_v1.1"
    with pytest.raises(ValueError, match="protocol_version"):
        validate_study_config(wrong)
    mutated = copy.deepcopy(bundle)
    mutated["tables"]["sgd"]["rows"][0]["training_seed"] = 99
    with pytest.raises(ValueError, match="training seed"):
        validate_grid_bundle(mutated)


def test_materialization_is_byte_identical_and_matches_checked_in_files(tmp_path):
    study, first = _bundle()
    _, second = _bundle()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    materialize_grid_bundle(first, first_dir)
    materialize_grid_bundle(second, second_dir)
    for filename in ["sgd.json", "adam.json", "adamw.json", "manifest.json"]:
        assert (first_dir / filename).read_bytes() == (
            second_dir / filename
        ).read_bytes()
    verify_materialized_grid_bundle(ROOT / study["frozen_grid_dir"], first)


def test_frozen_rows_have_no_selection_forbidden_result_fields():
    _, bundle = _bundle()
    forbidden = (
        b"id_test_metrics",
        b"ood_metrics",
        b"geometry",
        b"neural_collapse",
        b"detector_metrics",
    )
    for table in bundle["tables"].values():
        for row in table["rows"]:
            materialized = canonical_json_bytes(row)
            assert all(marker not in materialized for marker in forbidden)


def test_superseded_v1_1_execution_assets_are_absent_from_current_tree():
    old_study = ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_1"
    assert not old_study.exists() or not any(
        path.is_file() for path in old_study.rglob("*")
    )
    assert not (ROOT / "configs/training/cifar10_wrn28_10.yaml").exists()
    protocol_source = (ROOT / "src/oge/studies/protocol.py").read_text(
        encoding="utf-8"
    )
    runner_source = (ROOT / "scripts/run_optimizer_study.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "numpy_generator_pcg64",
        "PCG64",
        "_log_uniform",
        "generate_discovery_bundle",
        "frozen_discovery_dir",
    ):
        assert marker not in protocol_source
        assert marker not in runner_source
