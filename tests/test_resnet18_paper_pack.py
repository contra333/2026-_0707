from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as student_t

from oge.analysis import resnet18_paper_pack as paper_pack
from oge.analysis.fixed_readout_component_attribution import pair_transition_summary


def test_t90_interval_matches_student_t_oracle() -> None:
    values = [0.1, 0.2, 0.4, 0.5, 0.8]
    result = paper_pack._ordinary_summary(values, paired=True)
    expected_mean = statistics.fmean(values)
    expected_sd = statistics.stdev(values)
    half_width = student_t.ppf(0.95, 4) * expected_sd / math.sqrt(5)
    assert result["n"] == 5
    assert result["mean"] == pytest.approx(expected_mean)
    assert result["sd"] == pytest.approx(expected_sd)
    assert result["t90_low"] == pytest.approx(expected_mean - half_width)
    assert result["t90_high"] == pytest.approx(expected_mean + half_width)
    assert "paired" in result["interval_definition"]


def test_seed_first_macro_averages_datasets_within_seed() -> None:
    rows = []
    for seed, offset in ((0, 0.0), (1, 0.2)):
        for dataset, value in (("cifar100", 0.2), ("tin", 0.6)):
            rows.append(
                {
                    "context": "large",
                    "seed": seed,
                    "region": "Near",
                    "dataset": dataset,
                    "delta_auroc": value + offset,
                }
            )
    macro = paper_pack._macro_rows(
        rows,
        value_fields=("delta_auroc",),
        group_fields=("context", "seed", "region"),
    )
    assert [row["delta_auroc"] for row in macro] == pytest.approx([0.4, 0.6])
    assert [row["dataset_count"] for row in macro] == [2, 2]


def _absolute_and_paired_fixture() -> tuple[list[dict], list[dict]]:
    absolute: list[dict] = []
    paired: list[dict] = []
    for cell in paper_pack.CELLS:
        for dataset in paper_pack.DATASETS:
            for seed in range(5):
                role_values = {
                    "D": {"Raw MD": (0.5, 0.6), "RMD": (0.8, 0.3), "L2-MD": (0.7, 0.4)},
                    "C": {"Raw MD": (0.4, 0.7), "RMD": (0.75, 0.35), "L2-MD": (0.65, 0.45)},
                }
                for role, values in role_values.items():
                    for readout, (auroc, fpr95) in values.items():
                        absolute.append(
                            {
                                "cell": cell,
                                "dataset": dataset,
                                "seed": seed,
                                "role": role,
                                "readout": readout,
                                "auroc": auroc,
                                "fpr95": fpr95,
                            }
                        )
                for readout, delta in (("Raw MD", -0.1), ("RMD", -0.05), ("L2-MD", -0.05)):
                    paired.append(
                        {
                            "cell": cell,
                            "dataset": dataset,
                            "seed": seed,
                            "readout": readout,
                            "delta_auroc": delta,
                            "delta_fpr95": 0.1 if readout == "Raw MD" else 0.05,
                        }
                    )
    return absolute, paired


def test_recovery_and_contrast_attenuation_signs() -> None:
    recovery, attenuation = paper_pack._build_recovery_rows(
        *_absolute_and_paired_fixture()
    )
    rmd_d = next(
        row
        for row in recovery
        if row["context"] == "large"
        and row["dataset"] == "cifar100"
        and row["seed"] == 0
        and row["role"] == "D"
        and row["readout"] == "RMD"
    )
    assert rmd_d["auroc_recovery"] == pytest.approx(0.3)
    assert rmd_d["fpr95_recovery"] == pytest.approx(0.3)
    rmd_effect = next(
        row
        for row in attenuation
        if row["context"] == "large"
        and row["dataset"] == "cifar100"
        and row["seed"] == 0
        and row["readout"] == "RMD"
    )
    assert rmd_effect["contrast_attenuation"] == pytest.approx(0.05)


def test_gain_loss_churn_identities_with_ties() -> None:
    summary = pair_transition_summary(
        np.array([0.5, 0.8]),
        np.array([0.5, 0.7]),
        np.array([0.4, 0.9]),
        np.array([0.6, 0.7]),
    )
    gain, loss, churn = paper_pack._transition_rates(summary)
    utility = {"incorrect": 0.0, "tie": 0.5, "correct": 1.0}
    delta = sum(
        (utility[target] - utility[source]) * count
        for source, targets in summary["transitions"].items()
        for target, count in targets.items()
    ) / summary["pair_count"]
    assert gain - loss == pytest.approx(delta)
    assert gain + loss == pytest.approx(churn)


def _geometry_payload(host: str, seeds: list[int]) -> dict:
    rows = []
    for cell in paper_pack.CELLS:
        context = paper_pack.CELL_CONTEXT[cell]
        for seed in seeds:
            for role in ("D", "C"):
                factor = 2.0 if role == "C" else 1.0
                rows.append(
                    {
                        "run_id": f"{host}-{context}-{seed}-{role}",
                        "cell": cell,
                        "context": context,
                        "seed": seed,
                        "role": role,
                        "geometry_fit_split": "id_train",
                        "geometry_fit_count": 45000,
                        "held_out_geometry_split": "id_validation",
                        "held_out_geometry_count": 5000,
                        "feature_norm": factor,
                        "effective_rank": 20.0 + factor,
                        "within_trace": 4.0 * factor,
                        "cdnv": 2.0 + factor,
                        "nc0": 3.0 * factor,
                        "nc1": 1.0 + factor,
                        "nc2": 5.0 + factor,
                        "nc3": 7.0 + factor,
                        "top10_trace_share": 0.1 * factor,
                    }
                )
    payload = {
        "schema_version": paper_pack.GEOMETRY_SCHEMA_VERSION,
        "status": "PASS",
        "protected_data_access": False,
        "host": host,
        "rows": rows,
    }
    payload["output_identity_sha256"] = paper_pack.canonical_sha256(payload)
    return payload


def test_geometry_effect_scales_are_frozen(tmp_path: Path) -> None:
    host_seeds = {
        "curie": [0, 1],
        "lise": [2],
        "precision_medicine": [3, 4],
    }
    paths = []
    run_ids = set()
    for host, seeds in host_seeds.items():
        payload = _geometry_payload(host, seeds)
        run_ids.update(row["run_id"] for row in payload["rows"])
        path = tmp_path / f"{host}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    _, effects, audit = paper_pack._combine_geometry(paths, run_ids)
    by_metric = {row["metric"]: row for row in effects}
    assert audit["run_count"] == 20
    assert by_metric["feature_norm"]["effect_scale"] == "log(C/D)"
    assert by_metric["feature_norm"]["effect"] == pytest.approx(math.log(2.0))
    assert by_metric["within_trace"]["effect_scale"] == "log(C/D)"
    assert by_metric["nc0"]["effect_scale"] == "log(C/D)"
    assert by_metric["effective_rank"]["effect_scale"] == "C-D"
    assert by_metric["effective_rank"]["effect"] == pytest.approx(1.0)
    assert by_metric["top10_trace_share"]["effect_scale"] == "C-D"


def test_host_geometry_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = _geometry_payload("curie", [0, 1])
    payload["rows"][0]["feature_norm"] = 99.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        paper_pack.verify_host_geometry(path)


def test_build_failure_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**_: object) -> dict:
        raise ValueError("audit fixture failed")

    monkeypatch.setattr(paper_pack, "_build_pack_to", fail)
    output = tmp_path / "pack"
    with pytest.raises(ValueError, match="audit fixture failed"):
        paper_pack.build_paper_pack(
            evaluation_root=tmp_path,
            training_root=tmp_path,
            geometry_paths=[],
            wrn_root=tmp_path,
            output_root=output,
            analysis_git_sha="a" * 40,
        )
    assert sorted(path.name for path in output.iterdir()) == ["BLOCKED.json"]
    blocked = json.loads((output / "BLOCKED.json").read_text())
    assert blocked["status"] == "BLOCKED"
    assert blocked["partial_figures_written"] is False
