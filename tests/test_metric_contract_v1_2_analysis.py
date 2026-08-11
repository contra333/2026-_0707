import hashlib
import json
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
    write_output_manifest,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
INVENTORY_PATH = (
    REPOSITORY_ROOT
    / "configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json"
)


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

    np.testing.assert_allclose(
        _corr_columns(left, right), [[1.0, 1.0], [-1.0, -1.0]]
    )


def test_benjamini_hochberg_adjustment_matches_hand_fixture():
    observed = _bh_adjust(np.asarray([[0.01, 0.04], [0.03, 0.002]]))
    np.testing.assert_allclose(observed, [[0.02, 0.04], [0.04, 0.008]])


def test_cluster_bootstrap_reranks_each_resampled_block_fixture():
    x = np.arange(30, dtype=np.float64)[:, None]
    bootstrap_configs = np.asarray(
        [np.arange(10), [9, 9, 8, 7, 6, 5, 4, 3, 2, 1]], dtype=np.int64
    )
    observed, low, high = _bootstrap_correlations(
        x, x.copy(), bootstrap_configs, batch_size=1
    )
    np.testing.assert_allclose(observed, [[1.0]])
    np.testing.assert_allclose(low, [[1.0]])
    np.testing.assert_allclose(high, [[1.0]])


def test_seed_matched_delta_uses_sample_sd():
    left = {
        "mean": 4.0,
        "seed_values": {
            str(seed): {"value": value}
            for seed, value in enumerate([2.0, 4.0, 6.0])
        },
    }
    right = {
        "mean": 2.0,
        "seed_values": {
            str(seed): {"value": value}
            for seed, value in enumerate([1.0, 2.0, 3.0])
        },
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


def test_output_manifest_uses_small_synthetic_payload(tmp_path):
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    payload = output_dir / "table.csv"
    payload.write_text("metric,value\nauroc,0.5\n", encoding="utf-8")

    manifest = write_output_manifest(
        output_dir,
        {"files": [{"path": "source.json", "sha256": "a" * 64, "size": 1}]},
        resamples=7,
        random_seed=11,
        figures_generated=False,
        analysis_summary={"seed_aggregate_count": 2},
    )

    assert manifest["file_count"] == 1
    assert manifest["files"][0]["path"] == "table.csv"
    assert manifest["files"][0]["sha256"] == hashlib.sha256(
        payload.read_bytes()
    ).hexdigest()
    manifest_path = output_dir / "output_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert (output_dir / "output_manifest.json.sha256").read_text(
        encoding="utf-8"
    ) == f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}  output_manifest.json\n"
