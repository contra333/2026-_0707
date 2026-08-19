from __future__ import annotations

import math

import pytest

from oge.analysis.task_f_figure_inputs import _top_trace_share, normalized_role
from oge.analysis.task_f_frozen_paper_pack import (
    _paired_effect,
    alpha_region_macro_seed_rows,
    recovery_gain_seed_rows,
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
