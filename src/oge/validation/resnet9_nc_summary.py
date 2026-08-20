"""Aggregate the frozen six-arm ResNet9/MNIST positive control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .resnet9_nc_positive_control import EXPECTED_RATIOS

PRIMARY_METRICS = (
    "nc0_row_sum_raw",
    "nc2_etf_raw",
    "nc3_self_duality_raw",
)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    ranked_x = _rank(x)
    ranked_y = _rank(y)
    if np.std(ranked_x) == 0.0 or np.std(ranked_y) == 0.0:
        return None
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def summarize_resnet9_nc_positive_control(run_root: str | Path) -> dict[str, Any]:
    paths = sorted(Path(run_root).glob("*/summary.json"))
    if len(paths) != len(EXPECTED_RATIOS):
        raise ValueError(f"expected six summary files, found {len(paths)}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_ratio = {float(item["coupled_ratio"]): item for item in summaries}
    if tuple(sorted(by_ratio)) != EXPECTED_RATIOS:
        raise ValueError("summary ratios do not match the frozen six-arm matrix")

    blocker_reasons: list[str] = []
    identity_fields = (
        "seed",
        "repository_sha",
        "initial_model_state_sha256",
        "first_epoch_train_order_sha256",
    )
    for field in identity_fields:
        if len({item[field] for item in summaries}) != 1:
            blocker_reasons.append(f"sibling_{field}_mismatch")
    if any(item.get("repository_dirty") for item in summaries):
        blocker_reasons.append("dirty_execution_repository")
    if any(item.get("smoke_only") for item in summaries):
        blocker_reasons.append("smoke_only_summary")
    if any(int(item.get("completed_epoch", -1)) != 200 for item in summaries):
        blocker_reasons.append("incomplete_epoch_coverage")

    ratios = np.asarray(EXPECTED_RATIOS, dtype=np.float64)
    arms: list[dict[str, object]] = []
    values_by_metric: dict[str, list[float]] = {key: [] for key in PRIMARY_METRICS}
    for ratio in EXPECTED_RATIOS:
        summary = by_ratio[ratio]
        terminal = summary["terminal"]
        metrics = terminal["metrics"]
        values = {key: float(metrics[key]["value"]) for key in PRIMARY_METRICS}
        if not all(np.isfinite(value) for value in values.values()):
            blocker_reasons.append(f"nonfinite_primary_metric_ratio_{ratio}")
        for key, value in values.items():
            values_by_metric[key].append(value)
        arms.append(
            {
                "coupled_ratio": ratio,
                "weight_decay_coupled": ratio * 5.0e-4,
                "weight_decay_decoupled": (1.0 - ratio) * 5.0e-4,
                "test_accuracy": float(terminal["test_accuracy"]),
                **values,
            }
        )

    endpoint_direction = {
        key: values_by_metric[key][-1] < values_by_metric[key][0]
        for key in PRIMARY_METRICS
    }
    direction_count = sum(endpoint_direction.values())
    accuracy_gap = abs(arms[-1]["test_accuracy"] - arms[0]["test_accuracy"])
    if blocker_reasons:
        verdict = "BLOCKED"
    elif direction_count == 3 and accuracy_gap <= 0.01:
        verdict = "PASS"
    elif direction_count == 2:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "protocol": "resnet9_mnist_nc_positive_control_v1",
        "verdict": verdict,
        "blocker_reasons": sorted(set(blocker_reasons)),
        "paper_target": (
            "Figure 8 directional comparison only: increasing coupled decay at fixed "
            "total decay decreases raw NC0, raw NC2 ETF, and raw NC3 while accuracy is stable"
        ),
        "comparison_level": "directional",
        "absolute_numeric_comparison": "not_comparable",
        "seed": summaries[0]["seed"],
        "repository_sha": summaries[0]["repository_sha"],
        "initial_model_state_sha256": summaries[0]["initial_model_state_sha256"],
        "first_epoch_train_order_sha256": summaries[0][
            "first_epoch_train_order_sha256"
        ],
        "endpoint_direction_coupled_smaller": endpoint_direction,
        "endpoint_direction_count": direction_count,
        "endpoint_accuracy_absolute_gap": accuracy_gap,
        "spearman_ratio_vs_metric": {
            key: _spearman(ratios, np.asarray(values, dtype=np.float64))
            for key, values in values_by_metric.items()
        },
        "arms": arms,
    }


def render_resnet9_nc_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ResNet9/MNIST NC positive control",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Comparison level: `{summary['comparison_level']}`",
        f"- Execution SHA: `{summary['repository_sha']}`",
        f"- Seed: `{summary['seed']}`",
        "- Smaller NC values indicate stronger collapse.",
        "",
        "| coupled ratio | coupled WD | decoupled WD | test acc | NC0 raw | NC2 ETF raw | NC3 raw |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in summary["arms"]:
        lines.append(
            "| {coupled_ratio:.1f} | {weight_decay_coupled:.1e} | "
            "{weight_decay_decoupled:.1e} | {test_accuracy:.6f} | "
            "{nc0_row_sum_raw:.6g} | {nc2_etf_raw:.6g} | "
            "{nc3_self_duality_raw:.6g} |".format(**arm)
        )
    lines.extend(
        [
            "",
            "## Frozen endpoint decision",
            "",
            f"- Matching directions: {summary['endpoint_direction_count']}/3",
            f"- Endpoint accuracy gap: {summary['endpoint_accuracy_absolute_gap']:.6f}",
            f"- Blockers: {summary['blocker_reasons'] or 'none'}",
            "",
            "Absolute NC levels are not compared to the paper because its printed scaling "
            "conflicts with the raw code/table convention. This control evaluates the "
            "within-architecture direction only.",
            "",
        ]
    )
    return "\n".join(lines)
