from __future__ import annotations

import math

import pytest

from oge.analysis.task_f_figure_inputs import _top_trace_share, normalized_role
from oge.analysis.task_f_frozen_paper_pack import (
    ALPHA_MARKERS,
    MEAN_MARKER_SIZE,
    RAW_SEED_MARKER_SIZE,
    _paired_effect,
    alpha_region_macro_seed_rows,
    context_absolute_raw_rows,
    context_heatmap_matrices,
    recovery_gain_seed_rows,
    sgdm_paired_raw_effect_seed_rows,
    sgdm_recovery_seed_rows,
    sgdm_region_recovery_seed_rows,
    summarize_rows,
)


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ({"sibling_role": "alpha_1", "branch_policy": "adam"}, "C"),
        ({"sibling_role": "alpha_0", "branch_policy": "adamw"}, "D"),
        ({"sibling_role": "alpha_0_5", "branch_policy": "mixed"}, "M"),
        ({"sibling_role": "zero", "branch_policy": "zero_decay"}, "Z"),
    ],
)
def test_normalized_role(manifest, expected):
    assert normalized_role(manifest) == expected


def test_top_trace_share_uses_nonnegative_trace():
    spectrum = {"raw_eigenvalues": [5.0, 3.0, 2.0, -1e-12], "trace": 10.0}
    assert _top_trace_share(spectrum, 2) == pytest.approx(0.8)


def test_geometry_effect_scales_are_frozen():
    assert _paired_effect("feature_norm", 4.0, 2.0) == pytest.approx(math.log(2.0))
    assert _paired_effect("within_trace", 9.0, 3.0) == pytest.approx(math.log(3.0))
    assert _paired_effect("effective_rank", 7.0, 9.0) == pytest.approx(-2.0)
    assert _paired_effect("cdnv", 0.1, 0.2) == pytest.approx(-0.1)


def test_recovery_gain_keeps_absolute_values_and_deltas(monkeypatch):
    monkeypatch.setattr("oge.analysis.task_f_frozen_paper_pack.CELLS", ("cell",))
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.EXPECTED_SEEDS", {"cell": (0,)}
    )
    monkeypatch.setattr("oge.analysis.task_f_frozen_paper_pack.DATASETS", ("cifar100",))
    rows = []
    for role, raw, rmd, l2 in (("C", 0.4, 0.9, 0.8), ("D", 0.6, 0.88, 0.82)):
        for readout, auroc, fpr95 in (
            ("Raw MD", raw, 0.9),
            ("RMD", rmd, 0.5),
            ("L2-MD", l2, 0.4),
        ):
            rows.append(
                {
                    "cell": "cell",
                    "dataset": "cifar100",
                    "region": "Near",
                    "seed": 0,
                    "role": role,
                    "readout": readout,
                    "auroc": auroc,
                    "fpr95": fpr95,
                }
            )
    output = recovery_gain_seed_rows(rows)
    coupled_rmd = next(row for row in output if row["role"] == "C" and row["readout"] == "RMD")
    assert coupled_rmd["raw_auroc"] == pytest.approx(0.4)
    assert coupled_rmd["readout_auroc"] == pytest.approx(0.9)
    assert coupled_rmd["delta_auroc_from_raw"] == pytest.approx(0.5)
    assert coupled_rmd["delta_fpr95_from_raw"] == pytest.approx(-0.4)


def test_alpha_macro_is_seed_first(monkeypatch):
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.EXPECTED_SEEDS",
        {"adam_lr1e-3_wd1e-4_anchor": (0,)},
    )
    rows = []
    for role in ("D", "M", "C"):
        for dataset, auroc in (
            ("cifar100", 0.6),
            ("tin", 0.4),
            ("mnist", 0.9),
            ("svhn", 0.7),
            ("texture", 0.5),
            ("places365", 0.3),
        ):
            rows.append(
                {"role": role, "seed": 0, "dataset": dataset, "auroc": auroc, "fpr95": 1-auroc}
            )
    output = alpha_region_macro_seed_rows(rows)
    near_d = next(row for row in output if row["role"] == "D" and row["region"] == "Near")
    far_d = next(row for row in output if row["role"] == "D" and row["region"] == "Far")
    assert near_d["auroc"] == pytest.approx(0.5)
    assert far_d["auroc"] == pytest.approx(0.6)


def test_alpha_marker_contract_is_redundant_and_balanced():
    assert set(ALPHA_MARKERS) == {"D", "M", "C"}
    assert len(set(ALPHA_MARKERS.values())) == 3
    assert RAW_SEED_MARKER_SIZE < MEAN_MARKER_SIZE
    assert MEAN_MARKER_SIZE / RAW_SEED_MARKER_SIZE < 2.0


def test_context_absolute_raw_rows_selects_all_cells_and_only_raw(monkeypatch):
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.CONTEXT_FIGURE_CELLS",
        ("cell_a", "cell_b"),
    )
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.EXPECTED_SEEDS",
        {"cell_a": (0,), "cell_b": (0,)},
    )
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.DATASETS", ("cifar100",)
    )
    rows = []
    for cell in ("cell_a", "cell_b"):
        for role, value in (("D", 0.7), ("C", 0.4)):
            rows.append(
                {
                    "cell": cell,
                    "dataset": "cifar100",
                    "region": "Near",
                    "seed": 0,
                    "role": role,
                    "readout": "Raw MD",
                    "auroc": value,
                    "fpr95": 1.0 - value,
                }
            )
            rows.append(
                {
                    "cell": cell,
                    "dataset": "cifar100",
                    "region": "Near",
                    "seed": 0,
                    "role": role,
                    "readout": "RMD",
                    "auroc": 0.9,
                    "fpr95": 0.2,
                }
            )
    output = context_absolute_raw_rows(rows)
    assert len(output) == 4
    assert {row["cell"] for row in output} == {"cell_a", "cell_b"}
    assert {row["role"] for row in output} == {"C", "D"}
    assert {row["readout"] for row in output} == {"Raw MD"}


def test_context_absolute_raw_rows_rejects_missing_role(monkeypatch):
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.CONTEXT_FIGURE_CELLS", ("cell",)
    )
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.EXPECTED_SEEDS", {"cell": (0,)}
    )
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.DATASETS", ("cifar100",)
    )
    rows = [
        {
            "cell": "cell",
            "dataset": "cifar100",
            "region": "Near",
            "seed": 0,
            "role": "D",
            "readout": "Raw MD",
            "auroc": 0.7,
            "fpr95": 0.3,
        }
    ]
    with pytest.raises(ValueError, match="coverage mismatch"):
        context_absolute_raw_rows(rows)


def test_context_heatmap_matrices_keep_absolute_levels_and_pair_delta(monkeypatch):
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.CONTEXT_FIGURE_CELLS",
        ("low", "primary"),
    )
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.EXPECTED_SEEDS",
        {"low": (0, 1), "primary": (0, 1)},
    )
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.DATASETS",
        ("cifar100", "svhn"),
    )
    rows = []
    for cell_index, cell in enumerate(("low", "primary")):
        for dataset_index, dataset in enumerate(("cifar100", "svhn")):
            for seed in (0, 1):
                adamw = 0.60 + 0.10 * cell_index + 0.02 * dataset_index + 0.01 * seed
                adam = adamw - 0.20 + 0.05 * seed
                for role, value in (("D", adamw), ("C", adam)):
                    rows.append(
                        {
                            "cell": cell,
                            "dataset": dataset,
                            "region": "Near" if dataset == "cifar100" else "Far",
                            "seed": seed,
                            "role": role,
                            "readout": "Raw MD",
                            "auroc": value,
                            "fpr95": 1.0 - value,
                        }
                    )

    matrices = context_heatmap_matrices(rows)

    assert matrices["adamw"][0][0] == pytest.approx(0.605)
    assert matrices["adam"][0][0] == pytest.approx(0.43)
    assert matrices["paired_delta"][0][0] == pytest.approx(-0.175)
    assert matrices["adam"][1][1] == pytest.approx(0.55)
    assert matrices["adamw"][1][1] == pytest.approx(0.725)
    assert matrices["paired_delta"][1][1] == pytest.approx(-0.175)


def _synthetic_sgdm_rows():
    rows = []
    readouts = (
        ("raw", "md", 0.0),
        ("raw", "rmd", 0.2),
        ("l2", "md", 0.1),
    )
    for dataset_index, dataset in enumerate(
        ("cifar100", "tin", "mnist", "svhn")
    ):
        for seed in (0, 1):
            for role, role_offset in (("Z", 0.0), ("D", 0.1), ("C", 0.2)):
                for transform, detector, readout_offset in readouts:
                    auroc = (
                        0.3
                        + 0.05 * dataset_index
                        + 0.01 * seed
                        + role_offset
                        + readout_offset
                    )
                    rows.append(
                        {
                            "cell": "sgdm_lr0.1_wd5e-4",
                            "dataset": dataset,
                            "seed": seed,
                            "role": role,
                            "checkpoint_epoch": 200,
                            "checkpoint_role": "last",
                            "depth_tap": "penultimate",
                            "transform": transform,
                            "detector": detector,
                            "auroc": auroc,
                            "fpr95": 1.0 - auroc,
                            "initialization_sha256": f"init-{dataset}-{seed}",
                            "data_stream_sha256": f"stream-{dataset}-{seed}",
                            "source_manifest_sha256": f"manifest-{seed}",
                        }
                    )
    return rows


def test_sgdm_recovery_and_region_macros_are_seed_first(monkeypatch):
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.DATASETS",
        ("cifar100", "tin", "mnist", "svhn"),
    )
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.SGDM_EXPECTED_SEEDS", (0, 1)
    )

    readout_rows = sgdm_recovery_seed_rows(_synthetic_sgdm_rows())
    paired_rows = sgdm_paired_raw_effect_seed_rows(readout_rows)
    region_rows = sgdm_region_recovery_seed_rows(readout_rows)

    assert len(readout_rows) == 4 * 2 * 3 * 3
    assert len(paired_rows) == 4 * 2
    assert len(region_rows) == 2 * 3 * 2
    assert all(
        row["delta_auroc_sgdm_minus_sgdw"] == pytest.approx(0.1)
        for row in paired_rows
    )
    near_decoupled_seed0 = next(
        row
        for row in region_rows
        if row["region"] == "Near" and row["role"] == "D" and row["seed"] == 0
    )
    assert near_decoupled_seed0["dataset_count"] == 2
    assert near_decoupled_seed0["raw_md_auroc"] == pytest.approx(0.425)
    assert near_decoupled_seed0["rmd_auroc"] == pytest.approx(0.625)
    assert near_decoupled_seed0["rmd_minus_raw"] == pytest.approx(0.2)
    assert near_decoupled_seed0["l2_md_minus_raw"] == pytest.approx(0.1)


def test_sgdm_recovery_rejects_missing_training_rule(monkeypatch):
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.DATASETS",
        ("cifar100", "tin", "mnist", "svhn"),
    )
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.SGDM_EXPECTED_SEEDS", (0, 1)
    )
    rows = [row for row in _synthetic_sgdm_rows() if row["role"] != "C"]
    with pytest.raises(ValueError, match="SGDM recovery coverage mismatch"):
        sgdm_recovery_seed_rows(rows)


def test_sgdm_paired_effect_rejects_identity_mismatch(monkeypatch):
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.DATASETS",
        ("cifar100", "tin", "mnist", "svhn"),
    )
    monkeypatch.setattr(
        "oge.analysis.task_f_frozen_paper_pack.SGDM_EXPECTED_SEEDS", (0, 1)
    )
    readout_rows = sgdm_recovery_seed_rows(_synthetic_sgdm_rows())
    mismatched = [dict(row) for row in readout_rows]
    target = next(
        row
        for row in mismatched
        if row["dataset"] == "cifar100"
        and row["seed"] == 0
        and row["role"] == "C"
        and row["readout"] == "Raw MD"
    )
    target["data_stream_sha256"] = "different-stream"
    with pytest.raises(ValueError, match="SGDM paired identity mismatch"):
        sgdm_paired_raw_effect_seed_rows(mismatched)


def test_summary_labels_only_within_seed_contrasts_as_paired():
    rows = [{"role": "C", "value": 0.4}, {"role": "C", "value": 0.6}]
    absolute = summarize_rows(rows, ("role",), ("value",))
    contrast = summarize_rows(rows, ("role",), ("value",), paired=True)
    assert absolute[0]["interval_definition"] == (
        "two-sided 90% t interval across training seeds"
    )
    assert contrast[0]["interval_definition"] == (
        "two-sided paired 90% t interval across training seeds"
    )
