import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from oge.analysis.metric_contract_v1_2 import (
    _bh_adjust,
    _bootstrap_correlations,
    _corr_columns,
    _paired_delta,
    _role_configs,
    load_authorized_configs,
    run_analysis,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
INVENTORY_PATH = (
    REPOSITORY_ROOT
    / "configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json"
)
ANALYSIS_DIR = REPOSITORY_ROOT / "docs/analysis/metric_contract_v1_2_c1_c4"


def test_frozen_role_matrix_and_hyperparameters_match_inventory():
    configs = load_authorized_configs(INVENTORY_PATH)
    role_configs = _role_configs(configs)

    assert len(configs) == 10
    assert len(role_configs) == 11
    assert ("C2", "adamw") not in role_configs
    assert role_configs[("C1", "adam")].config_hash == role_configs[
        ("C3", "adam")
    ].config_hash

    expected = {
        ("C1", "sgd"): (0.1, 5e-4),
        ("C1", "adam"): (3e-4, 0.0),
        ("C1", "adamw"): (3e-3, 0.1),
        ("C2", "sgd"): (0.3, 1e-4),
        ("C2", "adam"): (3e-4, 1e-3),
        ("C3", "sgd"): (0.1, 0.0),
        ("C3", "adam"): (3e-4, 0.0),
        ("C3", "adamw"): (1e-3, 1e-4),
        ("C4", "sgd"): (0.3, 0.0),
        ("C4", "adam"): (3e-4, 1e-4),
        ("C4", "adamw"): (3e-3, 1e-3),
    }
    assert {
        key: (config.lr, config.weight_decay)
        for key, config in role_configs.items()
    } == expected


def test_column_correlation_preserves_perfect_and_inverse_order():
    left = np.asarray([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    right = np.asarray([[5.0, 1.0], [6.0, 2.0], [7.0, 3.0]])

    observed = _corr_columns(left, right)

    np.testing.assert_allclose(observed, [[1.0, 1.0], [-1.0, -1.0]])


def test_benjamini_hochberg_adjustment_matches_hand_fixture():
    observed = _bh_adjust(np.asarray([[0.01, 0.04], [0.03, 0.002]]))

    np.testing.assert_allclose(observed, [[0.02, 0.04], [0.04, 0.008]])


def test_cluster_bootstrap_reranks_each_resampled_block_fixture():
    x = np.arange(30, dtype=np.float64)[:, None]
    y = x.copy()
    bootstrap_configs = np.asarray(
        [
            np.arange(10),
            [9, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=np.int64,
    )

    observed, low, high = _bootstrap_correlations(
        x,
        y,
        bootstrap_configs,
        batch_size=1,
    )

    np.testing.assert_allclose(observed, [[1.0]])
    np.testing.assert_allclose(low, [[1.0]])
    np.testing.assert_allclose(high, [[1.0]])


def test_seed_matched_delta_uses_sample_sd():
    left = {
        "mean": 4.0,
        "seed_values": {str(seed): {"value": value} for seed, value in enumerate([2.0, 4.0, 6.0])},
    }
    right = {
        "mean": 2.0,
        "seed_values": {str(seed): {"value": value} for seed, value in enumerate([1.0, 2.0, 3.0])},
    }

    observed = _paired_delta(left, right)

    assert observed["mean_delta"] == pytest.approx(2.0)
    assert observed["sample_sd_delta_ddof1"] == pytest.approx(1.0)
    assert [observed[f"seed{seed}_delta"] for seed in range(3)] == [1.0, 2.0, 3.0]


def test_analysis_refuses_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    (output_dir / "preexisting.txt").write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="output directory must be absent or empty"):
        run_analysis(
            aggregate_dir=tmp_path / "not-read",
            inventory_path=INVENTORY_PATH,
            output_dir=output_dir,
            resamples=100,
            make_figures=False,
        )

    assert (output_dir / "preexisting.txt").read_text(encoding="utf-8") == "preserve me"


def test_committed_analysis_manifest_and_payload_checksums_match():
    manifest_path = ANALYSIS_DIR / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["resamples"] == 10_000
    assert manifest["random_seed"] == 20_260_805
    assert manifest["analysis_dtype"] == "float64"
    assert manifest["comparison_tolerance"] == {"atol": 1e-12, "rtol": 1e-12}
    assert manifest["analysis_summary"]["seed_aggregate_count"] == 31_720
    assert manifest["analysis_summary"]["seed0_record_count"] == 31_720
    assert manifest["analysis_summary"]["association_record_count"] == 8_448
    assert manifest["analysis_summary"]["paired_delta_record_count"] == 5_700
    assert manifest["file_count"] == 203
    assert manifest["file_count"] == len(manifest["files"])
    for entry in manifest["files"]:
        path = ANALYSIS_DIR / entry["path"]
        assert path.stat().st_size == entry["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]

    expected_sidecar = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert (ANALYSIS_DIR / "output_manifest.json.sha256").read_text(
        encoding="utf-8"
    ) == f"{expected_sidecar}  output_manifest.json\n"


def test_committed_ood_tables_never_mix_datasets_or_checkpoints():
    datasets = ("cifar100", "tin", "mnist", "svhn", "texture", "places365")
    for dataset in datasets:
        for checkpoint in ("last", "best_val"):
            path = ANALYSIS_DIR / f"tables/ood/{dataset}/all_three_seed_{checkpoint}.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert rows
            assert {row["ood_dataset"] for row in rows} == {dataset}
            assert {row["checkpoint_role"] for row in rows} == {checkpoint}
            for endpoint in (
                "auroc",
                "fpr95_performance",
                "aupr_in_appendix",
                "aupr_out_appendix",
            ):
                association_path = (
                    ANALYSIS_DIR
                    / f"tables/associations/{checkpoint}/{dataset}_{endpoint}.csv"
                )
                with association_path.open(encoding="utf-8", newline="") as handle:
                    association_rows = list(csv.DictReader(handle))
                assert len(association_rows) == 16 * 11
                assert {row["ood_dataset"] for row in association_rows} == {dataset}
                assert {row["checkpoint_role"] for row in association_rows} == {
                    checkpoint
                }
                assert {row["endpoint"] for row in association_rows} == {endpoint}
                numeric_fields = (
                    "config_mean_spearman_rho",
                    "seed_level_clustered_rho",
                    "cluster_bootstrap_ci_low",
                    "cluster_bootstrap_ci_high",
                    "permutation_pvalue",
                    "bh_fdr_qvalue",
                )
                assert all(
                    math.isfinite(float(row[field]))
                    for row in association_rows
                    for field in numeric_fields
                )

    macro_path = ANALYSIS_DIR / "tables/cross_dataset_macro_audit/all_three_seed.csv"
    with macro_path.open(encoding="utf-8", newline="") as handle:
        macro_rows = list(csv.DictReader(handle))
    assert macro_rows
    assert {row["ood_dataset"] for row in macro_rows} == {""}


def test_committed_detector_panel_and_methods_crosswalk_are_complete():
    with (ANALYSIS_DIR / "tables/detector_panel.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        detector_rows = list(csv.DictReader(handle))
    assert len(detector_rows) == 19
    assert sum(row["paper_panel"] == "canonical_main" for row in detector_rows) == 11
    assert sum(row["paper_panel"] == "exhaustive_appendix" for row in detector_rows) == 8

    card = (
        REPOSITORY_ROOT / "docs/reference_cards/11_metric_contract_v1_2.md"
    ).read_text(encoding="utf-8")
    with (ANALYSIS_DIR / "tables/methods_crosswalk.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        crosswalk_rows = list(csv.DictReader(handle))
    assert crosswalk_rows
    for row in crosswalk_rows:
        assert f"[{row['formula_id']}]" in card
        assert not row["methods_sentence"].startswith("See Reference Card")
        assert row["authoritative_source"] == (
            "docs/reference_cards/11_metric_contract_v1_2.md"
        )
