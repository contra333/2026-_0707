"""Build a seed-first, paper-facing Task F result pack.

The module consumes only the completed merged Task F analysis JSON.  It never
loads checkpoints, fits a detector, or accesses ID/OOD examples.  Every
Near/Far estimate is formed within a training seed before uncertainty is
computed across seeds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t


SCHEMA_VERSION = "task_f_paper_pack_v1"
PRIMARY_CELL = "adam_lr1e-3_wd1e-4_anchor"
CELLS = (
    PRIMARY_CELL,
    "adam_lr1e-3_wd1e-3",
    "adam_lr3e-4_wd1e-4",
    "adam_lr3e-4_wd1e-3",
)
CELL_LABELS = {
    PRIMARY_CELL: "LR 1e-3 / WD 1e-4",
    "adam_lr1e-3_wd1e-3": "LR 1e-3 / WD 1e-3",
    "adam_lr3e-4_wd1e-4": "LR 3e-4 / WD 1e-4",
    "adam_lr3e-4_wd1e-3": "LR 3e-4 / WD 1e-3",
}
EXPECTED_SEEDS = {
    PRIMARY_CELL: (0, 1, 2, 3, 4),
    "adam_lr1e-3_wd1e-3": (0, 1, 2),
    "adam_lr3e-4_wd1e-4": (0, 1, 2),
    "adam_lr3e-4_wd1e-3": (0, 1, 2),
}
DATASET_REGIONS = {
    "cifar100": "Near",
    "tin": "Near",
    "mnist": "Far",
    "svhn": "Far",
    "texture": "Far",
    "places365": "Far",
}
DATASETS = tuple(DATASET_REGIONS)
REGION_ORDER = ("Near", "Far")
CELL_GUARDRAILS = {
    PRIMARY_CELL: "Accuracy PASS; NLL FAIL; ECE PASS",
    "adam_lr1e-3_wd1e-3": "Accuracy FAIL; NLL PASS; ECE PASS",
    "adam_lr3e-4_wd1e-4": "Accuracy PASS; NLL PASS; ECE PASS",
    "adam_lr3e-4_wd1e-3": "Accuracy PASS; NLL FAIL; ECE PASS",
}
ENDPOINT_METRICS = (
    "left_auroc",
    "right_auroc",
    "delta_auroc",
    "delta_fpr95",
    "gain",
    "loss",
    "pair_order_churn",
)
LOCALIZATION_METRICS = (
    "delta_auroc",
    "phi_rmd",
    "phi_marginal",
    "phi_id",
    "phi_ood",
)
FORMATION_METRICS = (
    "left_auroc",
    "right_auroc",
    "delta_auroc",
    "gain",
    "loss",
    "pair_order_churn",
)
SUMMARY_KEYS = (
    "scope",
    "cell",
    "contrast",
    "transform",
    "detector",
    "axis",
    "axis_value",
    "region",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def mean_sd_t90(values: Iterable[float]) -> dict[str, Any]:
    """Return a two-sided 90% Student-t interval across independent seeds."""

    array = [float(value) for value in values]
    if not array:
        raise ValueError("cannot summarize an empty seed array")
    center = statistics.fmean(array)
    if len(array) == 1:
        return {
            "n": 1,
            "mean": center,
            "sd": None,
            "t90_low": None,
            "t90_high": None,
            "interval_definition": "not available for n=1",
        }
    sd = statistics.stdev(array)
    critical = float(student_t.ppf(0.95, len(array) - 1))
    half_width = critical * sd / math.sqrt(len(array))
    return {
        "n": len(array),
        "mean": center,
        "sd": sd,
        "t90_low": center - half_width,
        "t90_high": center + half_width,
        "interval_definition": "two-sided paired 90% t interval across training seeds",
    }


def _region(dataset: str) -> str:
    try:
        return DATASET_REGIONS[dataset]
    except KeyError as error:
        raise ValueError(f"unexpected OOD dataset: {dataset}") from error


def _macro_groups(rows: Sequence[Mapping[str, Any]]) -> Iterable[tuple[str, list[Mapping[str, Any]]]]:
    for region in REGION_ORDER:
        selected = [row for row in rows if _region(str(row["dataset"])) == region]
        if selected:
            yield region, selected


def _assert_unique(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, name: str) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate {name} keys")


def validate_source(data: Mapping[str, Any]) -> dict[str, Any]:
    """Reject incomplete, non-sibling, or numerically inconsistent source data."""

    validation = data.get("validation", {})
    if validation.get("status") != "PASS" or not validation.get(
        "all_sibling_identity_checks_pass"
    ):
        raise ValueError("source analysis validation or sibling identity is not PASS")
    expected_counts = {
        "endpoint_score_rows": 1512,
        "score_localization_rows": 504,
        "formation_score_rows": 960,
    }
    for name, expected in expected_counts.items():
        observed = len(data.get(name, []))
        if observed != expected:
            raise ValueError(f"{name} coverage mismatch: {observed} != {expected}")

    endpoint = data["endpoint_score_rows"]
    _assert_unique(
        endpoint,
        ("cell", "contrast", "dataset", "transform", "detector", "seed"),
        name="endpoint",
    )
    expected_contexts = {
        (cell, seed) for cell, seeds in EXPECTED_SEEDS.items() for seed in seeds
    }
    observed_contexts = {(row["cell"], int(row["seed"])) for row in endpoint}
    if observed_contexts != expected_contexts:
        raise ValueError("endpoint cell/seed coverage mismatch")
    if {row["dataset"] for row in endpoint} != set(DATASETS):
        raise ValueError("endpoint OOD dataset coverage mismatch")

    max_identity_residual = 0.0
    for row in endpoint:
        identity = row.get("identity", {})
        if not identity.get("pass"):
            raise ValueError("endpoint sibling identity mismatch")
        delta_residual = abs(float(row["delta_auroc"]) - (float(row["gain"]) - float(row["loss"])))
        churn_residual = abs(
            float(row["pair_order_churn"]) - (float(row["gain"]) + float(row["loss"]))
        )
        max_identity_residual = max(max_identity_residual, delta_residual, churn_residual)
    if max_identity_residual > 1.0e-12:
        raise ValueError(f"pair-accounting identity mismatch: {max_identity_residual}")

    localization = data["score_localization_rows"]
    _assert_unique(
        localization,
        ("cell", "contrast", "dataset", "transform", "seed"),
        name="localization",
    )
    max_localization_residual = 0.0
    for row in localization:
        replacement = row["rmd_marginal_replacement"]
        residual = abs(
            float(row["delta_auroc"])
            - float(replacement["phi_rmd"])
            - float(replacement["phi_marginal"])
        )
        max_localization_residual = max(max_localization_residual, residual)
    if max_localization_residual > 1.0e-12:
        raise ValueError(f"score-localization identity mismatch: {max_localization_residual}")

    return {
        "status": "PASS",
        "endpoint_rows": len(endpoint),
        "localization_rows": len(localization),
        "formation_rows": len(data["formation_score_rows"]),
        "cell_seed_contexts": len(observed_contexts),
        "max_pair_identity_residual": max_identity_residual,
        "max_localization_identity_residual": max_localization_residual,
    }


def _base_row(
    *,
    scope: str,
    cell: str,
    contrast: str,
    transform: str,
    detector: str,
    axis: str,
    axis_value: Any,
    seed: int,
    region: str,
    dataset_count: int,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "cell": cell,
        "contrast": contrast,
        "transform": transform,
        "detector": detector,
        "axis": axis,
        "axis_value": axis_value,
        "seed": seed,
        "region": region,
        "dataset_count": dataset_count,
    }


def endpoint_macro_seed_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in data["endpoint_score_rows"]:
        key = (
            row["cell"],
            row["contrast"],
            row["transform"],
            row["detector"],
            int(row["seed"]),
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        for region, selected in _macro_groups(rows):
            record = _base_row(
                scope="endpoint",
                cell=str(key[0]),
                contrast=str(key[1]),
                transform=str(key[2]),
                detector=str(key[3]),
                axis="endpoint",
                axis_value=200,
                seed=int(key[4]),
                region=region,
                dataset_count=len(selected),
            )
            record.update(
                {
                    "left_auroc": statistics.fmean(float(row["left"]["auroc"]) for row in selected),
                    "right_auroc": statistics.fmean(float(row["right"]["auroc"]) for row in selected),
                    "delta_auroc": statistics.fmean(float(row["delta_auroc"]) for row in selected),
                    "delta_fpr95": statistics.fmean(float(row["delta_fpr95"]) for row in selected),
                    "gain": statistics.fmean(float(row["gain"]) for row in selected),
                    "loss": statistics.fmean(float(row["loss"]) for row in selected),
                    "pair_order_churn": statistics.fmean(
                        float(row["pair_order_churn"]) for row in selected
                    ),
                }
            )
            output.append(record)
    return output


def localization_macro_seed_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in data["score_localization_rows"]:
        key = (row["cell"], row["contrast"], row["transform"], int(row["seed"]))
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        for region, selected in _macro_groups(rows):
            record = _base_row(
                scope="localization",
                cell=str(key[0]),
                contrast=str(key[1]),
                transform=str(key[2]),
                detector="md_decomposition",
                axis="endpoint",
                axis_value=200,
                seed=int(key[3]),
                region=region,
                dataset_count=len(selected),
            )
            for field in ("delta_auroc",):
                record[field] = statistics.fmean(float(row[field]) for row in selected)
            for field in ("phi_rmd", "phi_marginal"):
                record[field] = statistics.fmean(
                    float(row["rmd_marginal_replacement"][field]) for row in selected
                )
            for field in ("phi_id", "phi_ood"):
                record[field] = statistics.fmean(
                    float(row["id_ood_replacement"][field]) for row in selected
                )
            record["reconstruction_residual"] = abs(
                record["delta_auroc"] - record["phi_rmd"] - record["phi_marginal"]
            )
            output.append(record)
    return output


def formation_macro_seed_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = data["formation_score_rows"]
    contexts: list[tuple[str, Any, list[Mapping[str, Any]]]] = []
    for epoch in (10, 60, 120, 160, 200):
        contexts.append(
            (
                "epoch",
                epoch,
                [
                    row
                    for row in source
                    if int(row["checkpoint_epoch"]) == epoch
                    and row["depth_tap"] == "penultimate"
                ],
            )
        )
    for depth in ("stage1", "stage2", "stage3", "penultimate"):
        contexts.append(
            (
                "depth",
                depth,
                [
                    row
                    for row in source
                    if int(row["checkpoint_epoch"]) == 200 and row["depth_tap"] == depth
                ],
            )
        )

    output: list[dict[str, Any]] = []
    for axis, value, context_rows in contexts:
        groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for row in context_rows:
            groups[(row["transform"], row["detector"], int(row["seed"]))].append(row)
        for key, rows in sorted(groups.items()):
            for region, selected in _macro_groups(rows):
                record = _base_row(
                    scope="formation",
                    cell=PRIMARY_CELL,
                    contrast="C-D",
                    transform=str(key[0]),
                    detector=str(key[1]),
                    axis=axis,
                    axis_value=value,
                    seed=int(key[2]),
                    region=region,
                    dataset_count=len(selected),
                )
                record.update(
                    {
                        "left_auroc": statistics.fmean(float(row["left"]["auroc"]) for row in selected),
                        "right_auroc": statistics.fmean(float(row["right"]["auroc"]) for row in selected),
                        "delta_auroc": statistics.fmean(float(row["delta_auroc"]) for row in selected),
                        "gain": statistics.fmean(float(row["gain"]) for row in selected),
                        "loss": statistics.fmean(float(row["loss"]) for row in selected),
                        "pair_order_churn": statistics.fmean(
                            float(row["pair_order_churn"]) for row in selected
                        ),
                    }
                )
                output.append(record)
    return output


def build_macro_seed_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    return (
        endpoint_macro_seed_rows(data)
        + localization_macro_seed_rows(data)
        + formation_macro_seed_rows(data)
    )


def summarize_macro_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in SUMMARY_KEYS)].append(row)
    output: list[dict[str, Any]] = []
    for key, selected in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        metric_fields = {
            field
            for row in selected
            for field, value in row.items()
            if field not in SUMMARY_KEYS
            and field not in {"seed", "dataset_count", "reconstruction_residual"}
            and isinstance(value, (int, float))
        }
        for metric in sorted(metric_fields):
            if any(metric not in row for row in selected):
                continue
            summary = mean_sd_t90(float(row[metric]) for row in selected)
            output.append(
                {
                    **dict(zip(SUMMARY_KEYS, key)),
                    "metric": metric,
                    **summary,
                }
            )
    return output


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summary_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {
        tuple(row.get(key, "") for key in SUMMARY_KEYS) + (row["metric"],): row
        for row in rows
    }


def table_rows(
    macro_seed_rows: Sequence[Mapping[str, Any]],
    macro_summary_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    summary = summary_index(macro_summary_rows)

    design = []
    for cell in CELLS:
        endpoint_context = [
            row
            for row in macro_seed_rows
            if row["scope"] == "endpoint"
            and row["cell"] == cell
            and row["contrast"] == "C-D"
            and row["transform"] == "raw"
            and row["detector"] == "md"
        ]
        design.append(
            {
                "cell": cell,
                "cell_label": CELL_LABELS[cell],
                "roles": "Adam coupled (C); AdamW decoupled (D)",
                "seeds": ",".join(str(seed) for seed in EXPECTED_SEEDS[cell]),
                "seed_count": len(EXPECTED_SEEDS[cell]),
                "sibling_identity": "PASS"
                if endpoint_context and all(row["dataset_count"] in (2, 4) for row in endpoint_context)
                else "FAIL",
                "id_guardrail": CELL_GUARDRAILS[cell],
                "guardrail_source": "frozen Task F result-analysis record",
            }
        )

    pair_table: list[dict[str, Any]] = []
    for cell in CELLS:
        for contrast in ("C-D", "D-Z"):
            for region in REGION_ORDER:
                selected = [
                    row
                    for row in macro_seed_rows
                    if row["scope"] == "endpoint"
                    and row["cell"] == cell
                    and row["contrast"] == contrast
                    and row["transform"] == "raw"
                    and row["detector"] == "md"
                    and row["region"] == region
                ]
                if not selected:
                    continue
                record = {
                    "cell": cell,
                    "cell_label": CELL_LABELS[cell],
                    "contrast": contrast,
                    "region": region,
                }
                base = ("endpoint", cell, contrast, "raw", "md", "endpoint", 200, region)
                for metric in (
                    "left_auroc",
                    "right_auroc",
                    "delta_auroc",
                    "gain",
                    "loss",
                    "pair_order_churn",
                ):
                    stat = summary[base + (metric,)]
                    record[metric] = stat["mean"]
                    record[f"{metric}_t90_low"] = stat["t90_low"]
                    record[f"{metric}_t90_high"] = stat["t90_high"]
                    record[f"{metric}_n"] = stat["n"]
                pair_table.append(record)

    localization_table: list[dict[str, Any]] = []
    for cell in CELLS:
        for region in REGION_ORDER:
            record = {"cell": cell, "cell_label": CELL_LABELS[cell], "region": region}
            for transform, detector, label in (
                ("raw", "md", "raw_md"),
                ("raw", "rmd", "raw_rmd"),
                ("l2", "md", "l2_md"),
            ):
                key = (
                    "endpoint",
                    cell,
                    "C-D",
                    transform,
                    detector,
                    "endpoint",
                    200,
                    region,
                    "delta_auroc",
                )
                stat = summary[key]
                record[f"delta_{label}"] = stat["mean"]
                record[f"delta_{label}_t90_low"] = stat["t90_low"]
                record[f"delta_{label}_t90_high"] = stat["t90_high"]
            localization_key = (
                "localization",
                cell,
                "C-D",
                "raw",
                "md_decomposition",
                "endpoint",
                200,
                region,
            )
            for metric in ("phi_rmd", "phi_marginal"):
                stat = summary[localization_key + (metric,)]
                record[metric] = stat["mean"]
                record[f"{metric}_t90_low"] = stat["t90_low"]
                record[f"{metric}_t90_high"] = stat["t90_high"]
            record["localization_residual"] = abs(
                record["delta_raw_md"] - record["phi_rmd"] - record["phi_marginal"]
            )
            localization_table.append(record)
    return {
        "table1_design_guardrails.csv": design,
        "table2_pair_multiplicity.csv": pair_table,
        "table3_score_localization.csv": localization_table,
    }


def negative_gate_rows(
    data: Mapping[str, Any], classifier_kill: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    spectral = data["spectral_allocation_rows"]
    counts: dict[str, int] = defaultdict(int)
    for row in spectral:
        counts[str(row["status"])] += 1
    output = [
        {
            "gate": "S_perp spectral allocation applicability",
            "decision": "NOT_APPLICABLE",
            "contexts": len(spectral),
            "support_count": 0,
            "available_one_branch_count": counts.get("PASS", 0),
            "claim_boundary": "no cross-branch spectral attribution",
        }
    ]
    if classifier_kill is not None:
        decision_payload = classifier_kill["decision"]
        decision = (
            decision_payload["decision"]
            if isinstance(decision_payload, Mapping)
            else str(decision_payload)
        )
        for cell in CELLS:
            row = classifier_kill["cell_summary"][cell]
            output.append(
                {
                    "gate": "classifier-insensitive carrier",
                    "cell": cell,
                    "decision": decision,
                    "contexts": row["seed_count"],
                    "support_count": row["rho_above_one_count"],
                    "rho_median": row["rho_median"],
                    "claim_boundary": "negative diagnostic; no rescue threshold",
                }
            )
    return output


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 300
        elif extension == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
        elif extension == "svg":
            kwargs["metadata"] = {"Date": None}
        figure.savefig(path, **kwargs)
        paths.append(path)
    return paths


def _summary_lookup(
    summary: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    scope: str,
    cell: str,
    contrast: str,
    transform: str,
    detector: str,
    axis: str,
    axis_value: Any,
    region: str,
    metric: str,
) -> Mapping[str, Any]:
    return summary[
        (
            scope,
            cell,
            contrast,
            transform,
            detector,
            axis,
            axis_value,
            region,
            metric,
        )
    ]


def _plot_style() -> dict[str, str]:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#3B3B3B",
            "axes.labelcolor": "#252525",
            "text.color": "#252525",
            "xtick.color": "#3B3B3B",
            "ytick.color": "#3B3B3B",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "svg.hashsalt": "task-f-paper-pack-v1",
        }
    )
    return {
        "blue": "#356FA8",
        "orange": "#D97735",
        "gold": "#D4A72C",
        "ink": "#252525",
        "gray": "#7A7A7A",
        "light": "#D9E5F2",
    }


def figure1_design(output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    colors = _plot_style()
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    axis.set_xlim(-0.025, 1.025)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = (
        (0.01, 0.35, 0.18, 0.30, "Shared seed\ninit + data stream", colors["light"], 7.8),
        (0.26, 0.76, 0.21, 0.13, "Zero (Z) · no decay", "#F3F3F3", 7.5),
        (0.26, 0.57, 0.21, 0.13, "AdamW (D) · decoupled", "#EAF1F8", 7.5),
        (0.26, 0.38, 0.21, 0.13, "Midpoint (M) · α = 0.5", "#FFF4D6", 7.5),
        (0.26, 0.19, 0.21, 0.13, "Adam (C) · coupled L2", "#FBEBDD", 7.5),
        (0.56, 0.35, 0.18, 0.30, "Protocol-fixed,\nbranch-refitted\nRaw MD", "#F3F3F3", 7.8),
        (0.81, 0.35, 0.18, 0.30, "Same ID–OOD pairs\nC-D / D-Z / C-Z\nGain · Loss · Churn", "#F3F3F3", 7.2),
    )
    for x, y, width, height, label, facecolor, fontsize in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.015",
                linewidth=0.9,
                edgecolor=colors["ink"],
                facecolor=facecolor,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
        )
    arrows = (
        ((0.19, 0.50), (0.26, 0.825)),
        ((0.19, 0.50), (0.26, 0.635)),
        ((0.19, 0.50), (0.26, 0.445)),
        ((0.19, 0.50), (0.26, 0.255)),
        ((0.47, 0.825), (0.56, 0.58)),
        ((0.47, 0.635), (0.56, 0.54)),
        ((0.47, 0.445), (0.56, 0.46)),
        ((0.47, 0.255), (0.56, 0.42)),
        ((0.74, 0.50), (0.81, 0.50)),
    )
    for start, stop in arrows:
        axis.add_patch(
            FancyArrowPatch(start, stop, arrowstyle="-|>", mutation_scale=10, color=colors["gray"])
        )
    axis.set_title("Parallel sibling design and pair-level readout", loc="left", pad=5)
    return _save_figure(figure, output_dir, "figure1_controlled_design")


def figure2_pair_multiplicity(
    macro_seed_rows: Sequence[Mapping[str, Any]],
    macro_summary_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    source_note: str,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _plot_style()
    summary = summary_index(macro_summary_rows)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.25), gridspec_kw={"width_ratios": [1.15, 1]})

    y = np.arange(len(CELLS), dtype=float)
    for region, offset, color, marker, face in (
        ("Near", -0.12, colors["blue"], "o", colors["blue"]),
        ("Far", 0.12, colors["orange"], "s", "white"),
    ):
        means = []
        low = []
        high = []
        for cell in CELLS:
            stat = _summary_lookup(
                summary,
                scope="endpoint",
                cell=cell,
                contrast="C-D",
                transform="raw",
                detector="md",
                axis="endpoint",
                axis_value=200,
                region=region,
                metric="delta_auroc",
            )
            means.append(stat["mean"])
            low.append(stat["mean"] - stat["t90_low"])
            high.append(stat["t90_high"] - stat["mean"])
        axes[0].errorbar(
            means,
            y + offset,
            xerr=np.asarray([low, high]),
            fmt=marker,
            color=color,
            markerfacecolor=face,
            markeredgecolor=color,
            capsize=2.5,
            linewidth=1.0,
            label=region,
            zorder=3,
        )
        for index, cell in enumerate(CELLS):
            seeds = [
                row["delta_auroc"]
                for row in macro_seed_rows
                if row["scope"] == "endpoint"
                and row["cell"] == cell
                and row["contrast"] == "C-D"
                and row["transform"] == "raw"
                and row["detector"] == "md"
                and row["region"] == region
            ]
            jitter = np.linspace(-0.035, 0.035, len(seeds)) if len(seeds) > 1 else [0.0]
            axes[0].scatter(
                seeds,
                y[index] + offset + jitter,
                s=9,
                facecolors=face,
                edgecolors=color,
                linewidths=0.55,
                alpha=0.7,
                zorder=2,
            )
    axes[0].axvline(0, color=colors["ink"], linewidth=0.8)
    axes[0].set_yticks(y, [CELL_LABELS[cell] for cell in CELLS])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Adam − AdamW Raw-MD AUROC")
    axes[0].set_title("A. Four controlled Adam contexts", loc="left")
    axes[0].grid(axis="x")
    axes[0].legend(frameon=False, loc="lower left")

    categories = []
    gain_values = []
    loss_values = []
    delta_values = []
    churn_values = []
    for contrast in ("D-Z", "C-D"):
        for region in REGION_ORDER:
            base = (
                "endpoint",
                PRIMARY_CELL,
                contrast,
                "raw",
                "md",
                "endpoint",
                200,
                region,
            )
            categories.append(f"{contrast} · {region}")
            gain_values.append(summary[base + ("gain",)]["mean"])
            loss_values.append(summary[base + ("loss",)]["mean"])
            delta_values.append(summary[base + ("delta_auroc",)]["mean"])
            churn_values.append(summary[base + ("pair_order_churn",)]["mean"])
    yy = np.arange(len(categories), dtype=float)
    axes[1].barh(yy, gain_values, color=colors["gold"], edgecolor=colors["ink"], linewidth=0.5, label="Gain")
    axes[1].barh(yy, [-value for value in loss_values], color=colors["light"], edgecolor=colors["blue"], linewidth=0.8, label="−Loss")
    axes[1].scatter(delta_values, yy, marker="D", s=22, color=colors["ink"], label="Net Δ", zorder=3)
    for position, contrast, region in zip(yy, ("D-Z", "D-Z", "C-D", "C-D"), ("Near", "Far", "Near", "Far")):
        base = (
            "endpoint",
            PRIMARY_CELL,
            contrast,
            "raw",
            "md",
            "endpoint",
            200,
            region,
        )
        for metric, sign, color in (("gain", 1.0, colors["gold"]), ("loss", -1.0, colors["blue"]), ("delta_auroc", 1.0, colors["ink"])):
            stat = summary[base + (metric,)]
            center = sign * stat["mean"]
            low = center - sign * (stat["t90_low"] if sign > 0 else stat["t90_high"])
            high = sign * (stat["t90_high"] if sign > 0 else stat["t90_low"]) - center
            axes[1].errorbar(center, position, xerr=[[low], [high]], fmt="none", color=color, capsize=1.8, linewidth=0.7, zorder=4)
        seed_deltas = [
            row["delta_auroc"]
            for row in macro_seed_rows
            if row["scope"] == "endpoint"
            and row["cell"] == PRIMARY_CELL
            and row["contrast"] == contrast
            and row["transform"] == "raw"
            and row["detector"] == "md"
            and row["region"] == region
        ]
        axes[1].scatter(seed_deltas, [position] * len(seed_deltas), s=6, color=colors["ink"], alpha=0.35, zorder=4)
    axes[1].axvline(0, color=colors["ink"], linewidth=0.8)
    axes[1].set_yticks(
        yy,
        [f"{category}\nChurn {churn:.2f}" for category, churn in zip(categories, churn_values)],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Fraction of identical ID–OOD pairs")
    axes[1].set_title("B. Net AUROC can hide pair churn", loc="left")
    axes[1].grid(axis="x")
    axes[1].legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, -0.02))
    figure.suptitle(
        "Raw Mahalanobis pair ordering changes across controlled decay-policy contrasts",
        x=0.06,
        ha="left",
    )
    figure.text(0.01, 0.005, source_note, fontsize=5.7, color=colors["gray"])
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    return _save_figure(figure, output_dir, "figure2_pair_multiplicity")


def figure3_score_localization(
    macro_seed_rows: Sequence[Mapping[str, Any]],
    macro_summary_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    source_note: str,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _plot_style()
    summary = summary_index(macro_summary_rows)
    labels = [f"{CELL_LABELS[cell]} · {region}" for cell in CELLS for region in REGION_ORDER]
    contexts = [(cell, region) for cell in CELLS for region in REGION_ORDER]
    y = np.arange(len(contexts), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.9), sharey=True)

    for transform, detector, label, color, marker, offset in (
        ("raw", "md", "Raw MD", colors["ink"], "D", -0.14),
        ("raw", "rmd", "Raw RMD", colors["blue"], "o", 0.0),
        ("l2", "md", "L2-refit MD", colors["orange"], "s", 0.14),
    ):
        means = []
        lows = []
        highs = []
        for cell, region in contexts:
            stat = _summary_lookup(
                summary,
                scope="endpoint",
                cell=cell,
                contrast="C-D",
                transform=transform,
                detector=detector,
                axis="endpoint",
                axis_value=200,
                region=region,
                metric="delta_auroc",
            )
            means.append(stat["mean"])
            lows.append(stat["mean"] - stat["t90_low"])
            highs.append(stat["t90_high"] - stat["mean"])
        axes[0].errorbar(
            means,
            y + offset,
            xerr=np.asarray([lows, highs]),
            fmt=marker,
            color=color,
            markerfacecolor="white" if label != "Raw MD" else color,
            capsize=2,
            linewidth=0.9,
            label=label,
        )
        for position, (cell, region) in enumerate(contexts):
            seed_values = [
                row["delta_auroc"]
                for row in macro_seed_rows
                if row["scope"] == "endpoint"
                and row["cell"] == cell
                and row["contrast"] == "C-D"
                and row["transform"] == transform
                and row["detector"] == detector
                and row["region"] == region
            ]
            jitter = np.linspace(-0.025, 0.025, len(seed_values)) if len(seed_values) > 1 else [0.0]
            axes[0].scatter(
                seed_values,
                y[position] + offset + jitter,
                s=6,
                facecolors="white" if label != "Raw MD" else color,
                edgecolors=color,
                linewidths=0.45,
                alpha=0.55,
                zorder=2,
            )
    axes[0].axvline(0, color=colors["ink"], linewidth=0.8)
    axes[0].set_xlabel("Adam − AdamW AUROC")
    axes[0].set_title("A. Readout sensitivity", loc="left")
    axes[0].grid(axis="x")
    axes[0].legend(frameon=False, loc="lower left")

    for metric, label, color, marker, offset, face in (
        ("phi_rmd", r"$\phi_{RMD}$", colors["blue"], "o", -0.16, "white"),
        ("phi_marginal", r"$\phi_{Marginal}$", colors["orange"], "s", 0.0, "white"),
        ("delta_auroc", "Total Δ", colors["ink"], "D", 0.16, colors["ink"]),
    ):
        means = []
        lows = []
        highs = []
        for position, (cell, region) in enumerate(contexts):
            base = (
                "localization",
                cell,
                "C-D",
                "raw",
                "md_decomposition",
                "endpoint",
                200,
                region,
            )
            stat = summary[base + (metric,)]
            means.append(stat["mean"])
            lows.append(stat["mean"] - stat["t90_low"])
            highs.append(stat["t90_high"] - stat["mean"])
            seed_values = [
                row[metric]
                for row in macro_seed_rows
                if row["scope"] == "localization"
                and row["cell"] == cell
                and row["contrast"] == "C-D"
                and row["transform"] == "raw"
                and row["region"] == region
            ]
            jitter = np.linspace(-0.022, 0.022, len(seed_values)) if len(seed_values) > 1 else [0.0]
            axes[1].scatter(
                seed_values,
                y[position] + offset + jitter,
                s=6,
                facecolors=face,
                edgecolors=color,
                linewidths=0.45,
                alpha=0.5,
                zorder=2,
            )
        axes[1].errorbar(
            means,
            y + offset,
            xerr=np.asarray([lows, highs]),
            fmt=marker,
            color=color,
            markerfacecolor=face,
            markeredgecolor=color,
            capsize=2,
            linewidth=0.9,
            label=label,
            zorder=3,
        )
    axes[1].axvline(0, color=colors["ink"], linewidth=0.8)
    axes[1].set_xlabel("Exact Shapley AUROC contribution")
    axes[1].set_title(r"B. Exact components ($\phi_{RMD}+\phi_{Marginal}=\Delta$)", loc="left")
    axes[1].grid(axis="x")
    axes[1].legend(frameon=False, loc="lower left")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    figure.suptitle("Sensitivity is readout- and score-structured", x=0.06, ha="left")
    figure.text(0.01, 0.005, source_note, fontsize=5.7, color=colors["gray"])
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    return _save_figure(figure, output_dir, "figure3_score_localization")


def figure4_formation(
    macro_seed_rows: Sequence[Mapping[str, Any]],
    macro_summary_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    source_note: str,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _plot_style()
    summary = summary_index(macro_summary_rows)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    epochs = (10, 60, 120, 160, 200)
    for region, color, marker, offset in (
        ("Near", colors["blue"], "o", -3),
        ("Far", colors["orange"], "s", 3),
    ):
        means = []
        lows = []
        highs = []
        for epoch in epochs:
            stat = _summary_lookup(
                summary,
                scope="formation",
                cell=PRIMARY_CELL,
                contrast="C-D",
                transform="raw",
                detector="md",
                axis="epoch",
                axis_value=epoch,
                region=region,
                metric="delta_auroc",
            )
            means.append(stat["mean"])
            lows.append(stat["mean"] - stat["t90_low"])
            highs.append(stat["t90_high"] - stat["mean"])
        x = np.asarray(epochs, dtype=float) + offset
        axes[0].plot(x, means, color=color, linewidth=0.8, alpha=0.65)
        axes[0].errorbar(
            x,
            means,
            yerr=np.asarray([lows, highs]),
            fmt=marker,
            color=color,
            markerfacecolor="white" if region == "Far" else color,
            capsize=2,
            label=region,
            zorder=3,
        )
        for epoch, x_value in zip(epochs, x):
            seeds = [
                row["delta_auroc"]
                for row in macro_seed_rows
                if row["scope"] == "formation"
                and row["axis"] == "epoch"
                and row["axis_value"] == epoch
                and row["transform"] == "raw"
                and row["detector"] == "md"
                and row["region"] == region
            ]
            axes[0].scatter([x_value] * len(seeds), seeds, s=7, color=color, alpha=0.35, zorder=2)
    axes[0].axhline(0, color=colors["ink"], linewidth=0.8)
    axes[0].set_xticks(epochs)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Adam − AdamW Raw-MD AUROC")
    axes[0].set_title("A. Formation across training", loc="left")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)

    depths = ("stage1", "stage2", "stage3", "penultimate")
    x = np.arange(len(depths), dtype=float)
    for region, color, marker, offset in (
        ("Near", colors["blue"], "o", -0.08),
        ("Far", colors["orange"], "s", 0.08),
    ):
        means = []
        lows = []
        highs = []
        for depth in depths:
            stat = _summary_lookup(
                summary,
                scope="formation",
                cell=PRIMARY_CELL,
                contrast="C-D",
                transform="raw",
                detector="md",
                axis="depth",
                axis_value=depth,
                region=region,
                metric="delta_auroc",
            )
            means.append(stat["mean"])
            lows.append(stat["mean"] - stat["t90_low"])
            highs.append(stat["t90_high"] - stat["mean"])
        axes[1].errorbar(
            x + offset,
            means,
            yerr=np.asarray([lows, highs]),
            fmt=marker,
            color=color,
            markerfacecolor="white" if region == "Far" else color,
            capsize=2,
            label=region,
        )
        for position, depth in enumerate(depths):
            seed_values = [
                row["delta_auroc"]
                for row in macro_seed_rows
                if row["scope"] == "formation"
                and row["axis"] == "depth"
                and row["axis_value"] == depth
                and row["transform"] == "raw"
                and row["detector"] == "md"
                and row["region"] == region
            ]
            axes[1].scatter(
                [x[position] + offset] * len(seed_values),
                seed_values,
                s=6,
                facecolors="white" if region == "Far" else color,
                edgecolors=color,
                linewidths=0.45,
                alpha=0.45,
                zorder=2,
            )
    axes[1].axhline(0, color=colors["ink"], linewidth=0.8)
    axes[1].set_xticks(x, ("Stage 1", "Stage 2", "Stage 3", "Penult."), rotation=20)
    axes[1].set_xlabel("Feature depth (epoch 200)")
    axes[1].set_title("B. Localization across depth", loc="left")
    axes[1].grid(axis="y")
    figure.suptitle("Training-rule sensitivity appears early and grows in deeper features", x=0.07, ha="left")
    figure.text(0.01, 0.005, source_note, fontsize=5.7, color=colors["gray"])
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    return _save_figure(figure, output_dir, "figure4_time_depth_formation")


def appendix_geometry_matrix(data: Mapping[str, Any], output_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _plot_style()
    source = [
        row
        for row in data["geometry_rows"]
        if row["scope"] == "endpoint"
        and row["transform"] == "raw"
        and row["depth_tap"] == "penultimate"
        and row["role"] in ("C", "D")
    ]
    by_key = {(row["cell"], int(row["seed"]), row["role"]): row for row in source}
    definitions = (
        ("log feature norm ratio", lambda c, d: math.log(c["feature_norm"]["mean"] / d["feature_norm"]["mean"])),
        ("Δ global effective rank", lambda c, d: c["global_covariance"]["effective_rank"] - d["global_covariance"]["effective_rank"]),
        ("Δ global top-10 share", lambda c, d: c["global_covariance"]["top_10_trace_share"] - d["global_covariance"]["top_10_trace_share"]),
        ("Δ log10 condition", lambda c, d: math.log10(c["global_covariance"]["retained_condition_number"]) - math.log10(d["global_covariance"]["retained_condition_number"])),
    )
    rows: list[dict[str, Any]] = []
    matrix = np.zeros((len(CELLS), len(definitions)), dtype=float)
    for i, cell in enumerate(CELLS):
        for j, (metric, getter) in enumerate(definitions):
            seed_values = [
                getter(by_key[(cell, seed, "C")], by_key[(cell, seed, "D")])
                for seed in EXPECTED_SEEDS[cell]
            ]
            stat = mean_sd_t90(seed_values)
            matrix[i, j] = stat["mean"]
            rows.append({"cell": cell, "metric": metric, **stat})
    scaled = matrix.copy()
    for column in range(scaled.shape[1]):
        denominator = max(abs(value) for value in scaled[:, column]) or 1.0
        scaled[:, column] /= denominator
    figure, axis = plt.subplots(figsize=(7.2, 2.9))
    image = axis.imshow(scaled, cmap="PuOr_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(definitions)), [name for name, _ in definitions], rotation=17, ha="right")
    axis.set_yticks(np.arange(len(CELLS)), [CELL_LABELS[cell] for cell in CELLS])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:+.3g}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if abs(scaled[i, j]) > 0.58 else colors["ink"],
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.03)
    colorbar.set_label("Column-normalized signed C−D effect")
    axis.set_title("Appendix: endpoint geometry changes are descriptive, not a unique mediator", loc="left")
    figure.tight_layout()
    return _save_figure(figure, output_dir, "appendix_geometry_matrix"), rows


def render_figures(
    data: Mapping[str, Any],
    macro_seed_rows: Sequence[Mapping[str, Any]],
    macro_summary_rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    source_sha256: str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    destination = Path(output_dir)
    source_note = (
        "Dots: training seeds; symbols/whiskers: mean and paired 90% t interval; "
        f"Near/Far averages are formed within seed. Source SHA256 {source_sha256[:12]}."
    )
    paths: list[Path] = []
    paths.extend(figure1_design(destination))
    paths.extend(
        figure2_pair_multiplicity(
            macro_seed_rows, macro_summary_rows, destination, source_note
        )
    )
    paths.extend(
        figure3_score_localization(
            macro_seed_rows, macro_summary_rows, destination, source_note
        )
    )
    paths.extend(
        figure4_formation(
            macro_seed_rows, macro_summary_rows, destination, source_note
        )
    )
    geometry_paths, geometry_rows = appendix_geometry_matrix(data, destination)
    paths.extend(geometry_paths)
    return paths, geometry_rows
