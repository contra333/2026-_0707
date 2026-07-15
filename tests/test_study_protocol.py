import math
import copy
from pathlib import Path

import pytest

from oge.studies.hashing import canonical_json_bytes
from oge.studies.protocol import (
    CONFIRMATION_TRAINING_SEEDS,
    FINAL_TRAINING_SEEDS,
    OPTIMIZER_ORDER,
    PROTOCOL_VERSION,
    SEARCH_SPACES,
    generate_discovery_bundle,
    load_study_config,
    materialize_discovery_bundle,
    validate_study_config,
    validate_discovery_bundle,
    verify_materialized_discovery_bundle,
)
from oge.training import load_training_config

ROOT = Path(__file__).parents[1]
STUDY_PATH = ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_1/study.yaml"
GOLDEN_TABLE_HASHES = {
    "sgd": "a461338d71c1bee9861ae7497e09c53f736ecbb55f464aff3534cb2030c5baac",
    "sgdw": "be02253c3e1b6f2b110d754a21a6de5c19d319824ec12ea601002d0ee4004672",
    "adam": "14f05433c772be2ce543ce1e1b651d36c48c7dda900c5c47f6c5b1e04c7040a3",
    "adamw": "ab256621973438c0e4ed4d1ab906594fdc3b2925d139c6850a7fe1009ff99183",
}
GOLDEN_MANIFEST_HASH = "5b2f915d2337924cbb67077216bbcc4a49835f7927cea96c45004f0b76576b54"


def _bundle():
    study = load_study_config(STUDY_PATH)
    base = load_training_config(ROOT / study["base_training_config"])
    return study, generate_discovery_bundle(study, base)


def _log_quantile(value, bounds):
    return (math.log(value) - math.log(bounds[0])) / (math.log(bounds[1]) - math.log(bounds[0]))


def test_pcg64_tables_have_exact_budget_anchor_order_ranges_and_golden_hashes():
    study, bundle = _bundle()
    validate_discovery_bundle(bundle)
    assert bundle["manifest"]["protocol_version"] == PROTOCOL_VERSION
    assert bundle["manifest"]["total_assigned_rows"] == 64
    assert bundle["manifest"]["table_hashes"] == GOLDEN_TABLE_HASHES
    assert bundle["manifest"]["manifest_hash"] == GOLDEN_MANIFEST_HASH
    trial_ids = []
    for name in OPTIMIZER_ORDER:
        rows = bundle["tables"][name]["rows"]
        assert len(rows) == 16
        assert [row["row_kind"] for row in rows[:2]] == ["positive_anchor", "no_decay_anchor"]
        optimizer_configs = [row["scientific_config"]["optimizer"] for row in rows]
        assert optimizer_configs[0]["lr"] == SEARCH_SPACES[name]["positive_anchor"]["lr"]
        assert optimizer_configs[0]["weight_decay"] == SEARCH_SPACES[name]["positive_anchor"]["weight_decay"]
        assert optimizer_configs[1]["weight_decay"] == 0.0
        for config in optimizer_configs[2:]:
            assert SEARCH_SPACES[name]["lr"][0] <= config["lr"] <= SEARCH_SPACES[name]["lr"][1]
            assert SEARCH_SPACES[name]["weight_decay"][0] <= config["weight_decay"] <= SEARCH_SPACES[name]["weight_decay"][1]
        trial_ids.extend(row["trial_id"] for row in rows)
    assert len(trial_ids) == len(set(trial_ids)) == 64
    assert study["sampler"] == {"name": "numpy_generator_pcg64", "seed": 0}


def test_study_config_version_rejection_and_frozen_hash_mutation_detection():
    study, bundle = _bundle()
    wrong = copy.deepcopy(study)
    wrong["protocol_version"] = "wrn28_10_optimizer_hpo_v1"
    with pytest.raises(ValueError, match="protocol_version"):
        validate_study_config(wrong)
    mutated = copy.deepcopy(bundle)
    mutated["tables"]["sgd"]["rows"][0]["training_seed"] = 99
    with pytest.raises(ValueError, match="training seed"):
        validate_discovery_bundle(mutated)


def test_pair_members_share_common_uniform_quantiles():
    _, bundle = _bundle()
    for left, right in (("sgd", "sgdw"), ("adam", "adamw")):
        for left_row, right_row in zip(
            bundle["tables"][left]["rows"][2:],
            bundle["tables"][right]["rows"][2:],
        ):
            left_config = left_row["scientific_config"]["optimizer"]
            right_config = right_row["scientific_config"]["optimizer"]
            assert _log_quantile(left_config["lr"], SEARCH_SPACES[left]["lr"]) == pytest.approx(_log_quantile(
                right_config["lr"], SEARCH_SPACES[right]["lr"]
            ), abs=1e-15)
            assert _log_quantile(
                left_config["weight_decay"], SEARCH_SPACES[left]["weight_decay"]
            ) == pytest.approx(_log_quantile(
                right_config["weight_decay"], SEARCH_SPACES[right]["weight_decay"]
            ), abs=1e-15)


def test_seed_roles_are_distinct_and_training_phase_sets_are_disjoint():
    _, bundle = _bundle()
    roles = bundle["manifest"]["seed_roles"]
    assert roles["discovery_table_sampler"] == 0
    assert roles["discovery_training"] == 0
    phase_sets = [{roles["discovery_training"]}, set(CONFIRMATION_TRAINING_SEEDS), set(FINAL_TRAINING_SEEDS)]
    assert all(not left.intersection(right) for index, left in enumerate(phase_sets) for right in phase_sets[index + 1 :])


def test_materialization_is_byte_identical_and_matches_checked_in_frozen_files(tmp_path):
    study, first = _bundle()
    _, second = _bundle()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    materialize_discovery_bundle(first, first_dir)
    materialize_discovery_bundle(second, second_dir)
    for filename in ["sgd.json", "sgdw.json", "adam.json", "adamw.json", "manifest.json"]:
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()
    verify_materialized_discovery_bundle(ROOT / study["frozen_discovery_dir"], first)


def test_frozen_rows_have_no_selection_forbidden_result_fields():
    _, bundle = _bundle()
    forbidden = (b"id_test_metrics", b"ood_metrics", b"geometry", b"neural_collapse", b"detector_metrics")
    for table in bundle["tables"].values():
        for row in table["rows"]:
            materialized = canonical_json_bytes(row)
            assert all(marker not in materialized for marker in forbidden)
