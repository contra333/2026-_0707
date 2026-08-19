"""Build the frozen, seed-first Task F analysis figures and tables.

The builder consumes a previously validated merged analysis JSON plus compact
manifest-only exports.  It does not load examples, feature arrays, score
arrays, checkpoints, or fit a detector.
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

from oge.analysis.task_f_paper_pack import (
    CELLS,
    CELL_LABELS,
    DATASETS,
    DATASET_REGIONS,
    EXPECTED_SEEDS,
    PRIMARY_CELL,
    endpoint_macro_seed_rows,
    formation_macro_seed_rows,
    mean_sd_t90,
    validate_source,
)


SCHEMA_VERSION = "task_f_frozen_paper_pack_v1"
INPUT_SCHEMA_VERSION = "task_f_figure_inputs_v1"
DATASET_LABELS = {
    "cifar100": "CIFAR-100",
    "tin": "Tiny ImageNet",
    "mnist": "MNIST",
    "svhn": "SVHN",
    "texture": "Textures",
    "places365": "Places365",
}
ALPHA_BY_ROLE = {"D": 0.0, "M": 0.5, "C": 1.0}
ALPHA_MARKERS = {"D": "o", "M": "s", "C": "^"}
RAW_SEED_MARKER_SIZE = 11.5
MEAN_MARKER_SIZE = 18.0
CONTEXT_FIGURE_CELLS = (
    "adam_lr3e-4_wd1e-4",
    "adam_lr3e-4_wd1e-3",
    PRIMARY_CELL,
    "adam_lr1e-3_wd1e-3",
)
READOUTS = (
    ("Raw MD", "raw", "md"),
    ("RMD", "raw", "rmd"),
    ("L2-MD", "l2", "md"),
)
GEOMETRY_METRICS = (
    "feature_norm",
    "effective_rank",
    "within_trace",
    "cdnv",
    "nc0",
    "nc1",
    "nc2",
    "nc3",
)
CORE_GEOMETRY_METRICS = ("feature_norm", "effective_rank", "within_trace", "cdnv")
GEOMETRY_LABELS = {
    "feature_norm": "Feature norm\nlog(C/D)",
    "effective_rank": "Total-cov. effective rank\nC − D",
    "within_trace": "Within-class trace\nlog(C/D)",
    "cdnv": "CDNV\nC − D",
    "nc0": "NC0 row-sum raw\nlog(C/D)",
    "nc1": "NC1 pinv\nC − D",
    "nc2": "NC2 ETF raw\nC − D",
    "nc3": "NC3 self-duality raw\nC − D",
    "top10_trace_share": "Total-cov. top-10 trace share\nC − D",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _unique(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, name: str) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate {name} rows")


def combine_manifest_inputs(paths: Sequence[str | Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    score_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    sources = []
    for raw_path in paths:
        path = Path(raw_path)
        data = read_json(path)
        if data.get("schema_version") != INPUT_SCHEMA_VERSION:
            raise ValueError(f"unexpected manifest-input schema: {path}")
        if "manifest-only export" not in data.get("scientific_boundary", ""):
            raise ValueError(f"missing manifest-only boundary: {path}")
        score_rows.extend(data["score_rows"])
        geometry_rows.extend(data["geometry_rows"])
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "host": data["host"],
                "roots": data["roots"],
                "coverage": data["coverage"],
            }
        )
    _unique(
        score_rows,
        ("output_identity_sha256", "transform", "detector", "dataset"),
        name="score",
    )
    _unique(geometry_rows, ("output_identity_sha256",), name="geometry")
    score_contexts = {row["output_identity_sha256"] for row in score_rows}
    if len(score_contexts) != 360 or len(score_rows) != 8640:
        raise ValueError(
            f"score coverage mismatch: {len(score_contexts)} contexts / {len(score_rows)} rows"
        )
    if len(geometry_rows) != 660:
        raise ValueError(f"geometry coverage mismatch: {len(geometry_rows)} contexts")
    return score_rows, geometry_rows, {
        "status": "PASS",
        "score_contexts": len(score_contexts),
        "score_rows": len(score_rows),
        "geometry_contexts": len(geometry_rows),
        "hosts": sorted(source["host"] for source in sources),
        "sources": sources,
    }


def _endpoint(row: Mapping[str, Any]) -> bool:
    return (
        int(row["checkpoint_epoch"]) == 200
        and row["checkpoint_role"] == "last"
        and row["depth_tap"] == "penultimate"
    )


def alpha_seed_rows(score_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [
        {
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "role": row["role"],
            "alpha": ALPHA_BY_ROLE[row["role"]],
            "auroc": float(row["auroc"]),
            "fpr95": float(row["fpr95"]),
            "source_manifest_sha256": row["source_manifest_sha256"],
        }
        for row in score_rows
        if row["cell"] == PRIMARY_CELL
        and row["role"] in ALPHA_BY_ROLE
        and _endpoint(row)
        and row["transform"] == "raw"
        and row["detector"] == "md"
    ]
    _unique(output, ("dataset", "seed", "role"), name="alpha")
    expected = {
        (dataset, seed, role)
        for dataset in DATASETS
        for seed in EXPECTED_SEEDS[PRIMARY_CELL]
        for role in ALPHA_BY_ROLE
    }
    observed = {(row["dataset"], row["seed"], row["role"]) for row in output}
    if observed != expected:
        raise ValueError(f"alpha coverage mismatch: missing={sorted(expected-observed)[:5]}")
    return sorted(output, key=lambda row: (DATASETS.index(row["dataset"]), row["seed"], row["alpha"]))


def zero_reference_rows(merged: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = [
        {
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "role": "Z",
            "auroc": float(row["right"]["auroc"]),
            "fpr95": float(row["right"]["fpr95"]),
        }
        for row in merged["endpoint_score_rows"]
        if row["cell"] == PRIMARY_CELL
        and row["contrast"] == "D-Z"
        and row["transform"] == "raw"
        and row["detector"] == "md"
    ]
    _unique(output, ("dataset", "seed"), name="zero reference")
    if len(output) != len(DATASETS) * len(EXPECTED_SEEDS[PRIMARY_CELL]):
        raise ValueError("zero-reference coverage mismatch")
    return output


def alpha_region_macro_seed_rows(alpha: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role in ("D", "M", "C"):
        for seed in EXPECTED_SEEDS[PRIMARY_CELL]:
            for region in ("Near", "Far"):
                rows = [
                    row
                    for row in alpha
                    if row["role"] == role
                    and row["seed"] == seed
                    and DATASET_REGIONS[row["dataset"]] == region
                ]
                output.append(
                    {
                        "role": role,
                        "alpha": ALPHA_BY_ROLE[role],
                        "seed": seed,
                        "region": region,
                        "dataset_count": len(rows),
                        "auroc": statistics.fmean(row["auroc"] for row in rows),
                        "fpr95": statistics.fmean(row["fpr95"] for row in rows),
                    }
                )
    return output


def heatmap_seed_rows(merged: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = [
        {
            "cell": row["cell"],
            "dataset": row["dataset"],
            "region": DATASET_REGIONS[row["dataset"]],
            "seed": int(row["seed"]),
            "delta_auroc": float(row["delta_auroc"]),
            "delta_fpr95": float(row["delta_fpr95"]),
        }
        for row in merged["endpoint_score_rows"]
        if row["contrast"] == "C-D"
        and row["transform"] == "raw"
        and row["detector"] == "md"
    ]
    _unique(output, ("cell", "dataset", "seed"), name="heatmap")
    if len(output) != len(DATASETS) * sum(len(EXPECTED_SEEDS[cell]) for cell in CELLS):
        raise ValueError("heatmap coverage mismatch")
    return output


def recovery_seed_rows(merged: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    readout_lookup = {(transform, detector): label for label, transform, detector in READOUTS}
    for row in merged["endpoint_score_rows"]:
        key = (row["transform"], row["detector"])
        if row["contrast"] != "C-D" or key not in readout_lookup:
            continue
        for role, side in (("C", "left"), ("D", "right")):
            output.append(
                {
                    "cell": row["cell"],
                    "dataset": row["dataset"],
                    "region": DATASET_REGIONS[row["dataset"]],
                    "seed": int(row["seed"]),
                    "role": role,
                    "readout": readout_lookup[key],
                    "auroc": float(row[side]["auroc"]),
                    "fpr95": float(row[side]["fpr95"]),
                }
            )
    _unique(output, ("cell", "dataset", "seed", "role", "readout"), name="recovery")
    if len(output) != 2 * len(READOUTS) * len(DATASETS) * sum(
        len(EXPECTED_SEEDS[cell]) for cell in CELLS
    ):
        raise ValueError("recovery coverage mismatch")
    return output


def context_absolute_raw_rows(recovery: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [
        dict(row)
        for row in recovery
        if row["cell"] in CONTEXT_FIGURE_CELLS
        and row["role"] in ("D", "C")
        and row["readout"] == "Raw MD"
    ]
    _unique(output, ("cell", "dataset", "seed", "role"), name="context absolute raw")
    expected = {
        (cell, dataset, seed, role)
        for cell in CONTEXT_FIGURE_CELLS
        for dataset in DATASETS
        for seed in EXPECTED_SEEDS[cell]
        for role in ("D", "C")
    }
    observed = {
        (row["cell"], row["dataset"], int(row["seed"]), row["role"])
        for row in output
    }
    if observed != expected:
        raise ValueError(
            "context absolute Raw MD coverage mismatch: "
            f"missing={sorted(expected - observed)[:5]}, "
            f"extra={sorted(observed - expected)[:5]}"
        )
    return sorted(
        output,
        key=lambda row: (
            DATASETS.index(row["dataset"]),
            CONTEXT_FIGURE_CELLS.index(row["cell"]),
            row["role"],
            int(row["seed"]),
        ),
    )


def recovery_gain_seed_rows(recovery: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["cell"], row["dataset"], int(row["seed"]), row["role"], row["readout"]): row
        for row in recovery
    }
    output: list[dict[str, Any]] = []
    for cell in CELLS:
        for dataset in DATASETS:
            for seed in EXPECTED_SEEDS[cell]:
                for role in ("C", "D"):
                    raw = index[(cell, dataset, seed, role, "Raw MD")]
                    for readout in ("RMD", "L2-MD"):
                        candidate = index[(cell, dataset, seed, role, readout)]
                        output.append(
                            {
                                "cell": cell,
                                "dataset": dataset,
                                "region": DATASET_REGIONS[dataset],
                                "seed": seed,
                                "role": role,
                                "readout": readout,
                                "raw_auroc": raw["auroc"],
                                "readout_auroc": candidate["auroc"],
                                "delta_auroc_from_raw": candidate["auroc"] - raw["auroc"],
                                "raw_fpr95": raw["fpr95"],
                                "readout_fpr95": candidate["fpr95"],
                                "delta_fpr95_from_raw": candidate["fpr95"] - raw["fpr95"],
                            }
                        )
    _unique(output, ("cell", "dataset", "seed", "role", "readout"), name="recovery gain")
    return output


def churn_seed_rows(merged: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = [
        {
            "cell": row["cell"],
            "dataset": row["dataset"],
            "region": DATASET_REGIONS[row["dataset"]],
            "seed": int(row["seed"]),
            "gain": float(row["gain"]),
            "loss": float(row["loss"]),
            "delta_auroc": float(row["delta_auroc"]),
            "churn": float(row["pair_order_churn"]),
        }
        for row in merged["endpoint_score_rows"]
        if row["contrast"] == "C-D"
        and row["transform"] == "raw"
        and row["detector"] == "md"
    ]
    for row in output:
        if abs(row["delta_auroc"] - (row["gain"] - row["loss"])) > 1e-12:
            raise ValueError("gain-loss identity failed")
        if abs(row["churn"] - (row["gain"] + row["loss"])) > 1e-12:
            raise ValueError("churn identity failed")
    return output


def endpoint_geometry_rows(geometry_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in geometry_rows if row["cell"] in CELLS and row["role"] in ("C", "D") and _endpoint(row)]
    _unique(output, ("cell", "seed", "role"), name="endpoint geometry")
    expected = {
        (cell, seed, role)
        for cell in CELLS
        for seed in EXPECTED_SEEDS[cell]
        for role in ("C", "D")
    }
    observed = {(row["cell"], int(row["seed"]), row["role"]) for row in output}
    if observed != expected:
        raise ValueError("endpoint geometry coverage mismatch")
    return output


def id_endpoint_rows(geometry_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = [
        {
            "run_id": row["run_id"],
            "cell": row["cell"],
            "role": row["role"],
            "seed": int(row["seed"]),
            "accuracy": float(row["id_accuracy"]),
            "checkpoint_epoch": int(row["checkpoint_epoch"]),
            "checkpoint_role": row["checkpoint_role"],
            "depth_tap": row["depth_tap"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
        for row in geometry_rows
        if _endpoint(row)
    ]
    _unique(output, ("run_id",), name="ID endpoint")
    if len(output) != 50:
        raise ValueError(f"ID endpoint coverage mismatch: {len(output)} != 50")
    return sorted(output, key=lambda row: (row["cell"], row["role"], row["seed"]))


def _paired_effect(metric: str, coupled: float, decoupled: float) -> float:
    if metric in ("feature_norm", "within_trace", "nc0"):
        if coupled <= 0.0 or decoupled <= 0.0:
            raise ValueError(f"nonpositive value for log ratio: {metric}")
        return math.log(coupled / decoupled)
    return coupled - decoupled


def pair_geometry_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["cell"], int(row["seed"]), row["role"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for cell in CELLS:
        for seed in EXPECTED_SEEDS[cell]:
            coupled = index[(cell, seed, "C")]
            decoupled = index[(cell, seed, "D")]
            for metric in GEOMETRY_METRICS + ("top10_trace_share",):
                c_value = coupled.get(metric)
                d_value = decoupled.get(metric)
                if c_value is None or d_value is None:
                    continue
                output.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "metric": metric,
                        "coupled": float(c_value),
                        "decoupled": float(d_value),
                        "effect": _paired_effect(metric, float(c_value), float(d_value)),
                    }
                )
    return output


def formation_geometry_pairs(geometry_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in geometry_rows
        if row["cell"] == PRIMARY_CELL
        and row["role"] in ("C", "D")
        and (
            (row["depth_tap"] == "penultimate" and int(row["checkpoint_epoch"]) in (10, 60, 120, 160, 200))
            or (int(row["checkpoint_epoch"]) == 200 and row["depth_tap"] in ("stage1", "stage2", "stage3"))
        )
    ]
    index = {
        (int(row["seed"]), int(row["checkpoint_epoch"]), row["depth_tap"], row["role"]): row
        for row in selected
    }
    output: list[dict[str, Any]] = []
    contexts = [("epoch", epoch, epoch, "penultimate") for epoch in (10, 60, 120, 160, 200)] + [
        ("depth", depth, 200, depth) for depth in ("stage1", "stage2", "stage3", "penultimate")
    ]
    for axis, axis_value, epoch, depth in contexts:
        for seed in EXPECTED_SEEDS[PRIMARY_CELL]:
            coupled = index[(seed, epoch, depth, "C")]
            decoupled = index[(seed, epoch, depth, "D")]
            for metric in CORE_GEOMETRY_METRICS + ("top10_trace_share",):
                output.append(
                    {
                        "axis": axis,
                        "axis_value": axis_value,
                        "seed": seed,
                        "metric": metric,
                        "effect": _paired_effect(
                            metric, float(coupled[metric]), float(decoupled[metric])
                        ),
                    }
                )
    _unique(output, ("axis", "axis_value", "seed", "metric"), name="formation geometry")
    return output


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    metrics: Sequence[str],
    *,
    paired: bool = False,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, selected in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        base = dict(zip(group_fields, key, strict=True))
        for metric in metrics:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            if not values:
                continue
            summary = mean_sd_t90(values)
            if len(values) > 1:
                summary["interval_definition"] = (
                    "two-sided paired 90% t interval across training seeds"
                    if paired
                    else "two-sided 90% t interval across training seeds"
                )
            output.append({**base, "statistic": metric, **summary})
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (dict, list)):
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _style() -> dict[str, str]:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "task-f-frozen-paper-pack-v1",
        }
    )
    return {
        "blue": "#0072B2",
        "orange": "#D55E00",
        "green": "#009E73",
        "purple": "#CC79A7",
        "gold": "#E69F00",
        "sky": "#56B4E9",
        "ink": "#222222",
        "gray": "#777777",
        "light": "#D9D9D9",
    }


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 300
        elif extension == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None, "Creator": "Matplotlib"}
        else:
            kwargs["metadata"] = {"Date": None, "Creator": "Matplotlib"}
        figure.savefig(path, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def _summary(values: Iterable[float]) -> dict[str, Any]:
    return mean_sd_t90(list(values))


def _errorbar(axis: Any, x: float, stat: Mapping[str, Any], color: str, marker: str = "o") -> None:
    low = float(stat["mean"] - stat["t90_low"])
    high = float(stat["t90_high"] - stat["mean"])
    axis.errorbar(
        [x],
        [stat["mean"]],
        yerr=[[low], [high]],
        fmt=marker,
        markersize=4.2,
        color=color,
        markerfacecolor="white",
        markeredgewidth=1.0,
        capsize=2.2,
        linewidth=1.0,
        zorder=5,
    )


def figure_alpha(alpha: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 5.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    role_colors = {"D": colors["blue"], "M": colors["gold"], "C": colors["orange"]}
    role_x = {"D": 0.0, "M": 1.0, "C": 2.0}
    role_tick_labels = ["AdamW\nα=0", "Mixed\nα=0.5", "Adam\nα=1"]
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        selected = [row for row in alpha if row["dataset"] == dataset]
        for role in ("D", "M", "C"):
            rows = [row for row in selected if row["role"] == role]
            x = role_x[role]
            jitter = np.linspace(-0.065, 0.065, len(rows))
            axis.scatter(
                [x + value for value in jitter],
                [row["auroc"] for row in rows],
                marker=ALPHA_MARKERS[role],
                s=RAW_SEED_MARKER_SIZE,
                color=role_colors[role],
                alpha=0.45,
                linewidths=0,
                zorder=2,
            )
            axis.scatter(
                [x],
                [statistics.fmean(float(row["auroc"]) for row in rows)],
                marker=ALPHA_MARKERS[role],
                s=MEAN_MARKER_SIZE,
                color=role_colors[role],
                edgecolors=colors["ink"],
                linewidths=0.6,
                zorder=4,
            )
        axis.set_title(f"{chr(65+panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.set_xlim(-0.2, 2.2)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([0.0, 1.0, 2.0])
        axis.set_xticklabels(role_tick_labels)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(axis="y")
        axis.tick_params(axis="x", labelbottom=True, labelsize=6.8, pad=2.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Raw MD AUROC")
    figure.suptitle(
        "Raw MD across decay-coupling allocation\n"
        "WRN-28-10 · CIFAR-10 ID · LR=1e−3 · total WD=1e−4 · epoch 200 last · "
        "penultimate · n=5 matched seeds · small translucent markers: seeds · "
        "outlined markers: means",
        x=0.012,
        ha="left",
        fontsize=9.2,
        linespacing=1.4,
    )
    return _save_figure(figure, output_dir, "figure1_alpha_path_raw_md")


def figure_context_absolute_raw(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 5.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    xbase = np.arange(len(CONTEXT_FIGURE_CELLS), dtype=float)
    role_style = {
        "D": (colors["blue"], -0.10, "o", "AdamW"),
        "C": (colors["orange"], 0.10, "^", "Adam"),
    }
    context_labels = [
        "LR 3e−4\nWD 1e−4",
        "LR 3e−4\nWD 1e−3",
        "LR 1e−3\nWD 1e−4",
        "LR 1e−3\nWD 1e−3",
    ]
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        selected = [row for row in rows if row["dataset"] == dataset]
        for role, (color, offset, marker, _) in role_style.items():
            for x, cell in enumerate(CONTEXT_FIGURE_CELLS):
                values = [
                    float(row["auroc"])
                    for row in selected
                    if row["cell"] == cell and row["role"] == role
                ]
                jitter = np.linspace(-0.035, 0.035, len(values))
                axis.scatter(
                    xbase[x] + offset + jitter,
                    values,
                    marker=marker,
                    s=RAW_SEED_MARKER_SIZE,
                    color=color,
                    alpha=0.42,
                    linewidths=0,
                    zorder=2,
                )
                axis.scatter(
                    [xbase[x] + offset],
                    [statistics.fmean(values)],
                    marker=marker,
                    s=MEAN_MARKER_SIZE,
                    color=color,
                    edgecolors=colors["ink"],
                    linewidths=0.6,
                    zorder=4,
                )
        axis.set_title(f"{chr(65+panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.set_xlim(-0.35, len(CONTEXT_FIGURE_CELLS) - 0.65)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(xbase, context_labels)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.grid(axis="y")
        axis.tick_params(axis="x", labelbottom=True, labelsize=6.1, pad=2.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Raw MD AUROC")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor=colors["ink"],
            markeredgewidth=0.6,
            markersize=4.2,
            label=label,
        )
        for color, _, marker, label in role_style.values()
    ]
    figure.legend(handles=handles, frameon=False, ncol=2, loc="outside upper center")
    figure.suptitle(
        "Absolute Raw MD across LR × WD contexts\n"
        "WRN-28-10 · CIFAR-10 ID · epoch 200 last · penultimate · "
        "primary n=5, other contexts n=3 · small translucent markers: seeds · "
        "outlined markers: means",
        x=0.012,
        ha="left",
        fontsize=9.2,
        linespacing=1.4,
    )
    return _save_figure(
        figure, output_dir, "supp_figure1_context_absolute_raw_md"
    )


def figure_heatmap(
    heatmap: Sequence[Mapping[str, Any]],
    macro: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    colors = _style()
    matrix = np.asarray(
        [
            [statistics.fmean(row["delta_auroc"] for row in heatmap if row["dataset"] == dataset and row["cell"] == cell) for cell in CELLS]
            for dataset in DATASETS
        ]
    )
    bound = max(abs(float(matrix.min())), abs(float(matrix.max())))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [1.5, 1.0]}, constrained_layout=True)
    image = axes[0].imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound), aspect="auto")
    axes[0].set_xticks(range(len(CELLS)), [CELL_LABELS[cell].replace(" / ", "\n") for cell in CELLS])
    axes[0].set_yticks(range(len(DATASETS)), [DATASET_LABELS[dataset] for dataset in DATASETS])
    axes[0].set_title("A. Dataset × Adam context", loc="left")
    for i in range(len(DATASETS)):
        for j in range(len(CELLS)):
            axes[0].text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=6.7, color="white" if abs(matrix[i,j]) > bound*0.55 else colors["ink"])
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.045, pad=0.025)
    colorbar.set_label("C − D Raw MD AUROC")

    y = np.arange(len(CELLS), dtype=float)
    for region, offset, color, marker in (("Near", -0.12, colors["blue"], "o"), ("Far", 0.12, colors["orange"], "s")):
        for index, cell in enumerate(CELLS):
            rows = [
                row for row in macro
                if row["scope"] == "endpoint" and row["cell"] == cell and row["contrast"] == "C-D"
                and row["transform"] == "raw" and row["detector"] == "md" and row["region"] == region
            ]
            values = [float(row["delta_auroc"]) for row in rows]
            jitter = np.linspace(-0.025, 0.025, len(values))
            axes[1].scatter(values, y[index] + offset + jitter, s=9, color=color, alpha=0.5)
            stat = _summary(values)
            axes[1].errorbar(
                stat["mean"], y[index] + offset,
                xerr=[[stat["mean"]-stat["t90_low"]], [stat["t90_high"]-stat["mean"]]],
                fmt=marker, color=color, markerfacecolor="white", capsize=2.2, linewidth=1.0,
                label=region if index == 0 else None,
            )
    axes[1].axvline(0, color=colors["ink"], linewidth=0.75)
    axes[1].set_yticks(y, [CELL_LABELS[cell] for cell in CELLS])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("C − D Raw MD AUROC")
    axes[1].set_title("B. Paired seed evidence", loc="left")
    axes[1].grid(axis="x")
    axes[1].legend(frameon=False)
    all_values = [float(row["delta_auroc"]) for row in macro if row["scope"] == "endpoint" and row["contrast"] == "C-D" and row["transform"] == "raw" and row["detector"] == "md"]
    axes[1].set_xlim(min(all_values) - 0.025, max(0.0, max(all_values)) + 0.025)
    return _save_figure(figure, output_dir, "figure2_context_heatmap")


def figure_recovery(recovery: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _style()
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), sharex=True, sharey=True, constrained_layout=True)
    xbase = np.arange(len(READOUTS), dtype=float)
    role_style = {"D": (colors["blue"], -0.08, "o"), "C": (colors["orange"], 0.08, "s")}
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        selected = [row for row in recovery if row["cell"] == PRIMARY_CELL and row["dataset"] == dataset]
        for role, (color, offset, marker) in role_style.items():
            role_rows = [row for row in selected if row["role"] == role]
            for seed in EXPECTED_SEEDS[PRIMARY_CELL]:
                seed_rows = sorted([row for row in role_rows if row["seed"] == seed], key=lambda row: [item[0] for item in READOUTS].index(row["readout"]))
                axis.plot(xbase + offset, [row["auroc"] for row in seed_rows], color=color, linewidth=0.55, alpha=0.25)
            for x, (readout, _, _) in enumerate(READOUTS):
                values = [row["auroc"] for row in role_rows if row["readout"] == readout]
                axis.scatter(np.full(len(values), x+offset), values, s=10, color=color, alpha=0.45, linewidths=0)
                _errorbar(axis, x+offset, _summary(values), color, marker)
        axis.set_title(f"{chr(65+panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks(xbase, [item[0] for item in READOUTS])
        axis.grid(axis="y")
    for axis in axes[:, 0]:
        axis.set_ylabel("AUROC")
    handles = [
        plt.Line2D([0], [0], marker=marker, color=color, markerfacecolor="white", label=role, linestyle="-")
        for role, (color, _, marker) in role_style.items()
    ]
    figure.legend(handles=handles, frameon=False, ncol=2, loc="outside upper center")
    return _save_figure(figure, output_dir, "figure3_absolute_recovery")


def figure_fpr95_recovery(recovery: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _style()
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), sharex=True, sharey=True, constrained_layout=True)
    xbase = np.arange(len(READOUTS), dtype=float)
    role_style = {"D": (colors["blue"], -0.08, "o"), "C": (colors["orange"], 0.08, "s")}
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        selected = [row for row in recovery if row["cell"] == PRIMARY_CELL and row["dataset"] == dataset]
        for role, (color, offset, marker) in role_style.items():
            role_rows = [row for row in selected if row["role"] == role]
            for seed in EXPECTED_SEEDS[PRIMARY_CELL]:
                seed_rows = sorted(
                    [row for row in role_rows if row["seed"] == seed],
                    key=lambda row: [item[0] for item in READOUTS].index(row["readout"]),
                )
                axis.plot(xbase + offset, [row["fpr95"] for row in seed_rows], color=color, linewidth=0.55, alpha=0.25)
            for x, (readout, _, _) in enumerate(READOUTS):
                values = [row["fpr95"] for row in role_rows if row["readout"] == readout]
                axis.scatter(np.full(len(values), x + offset), values, s=10, color=color, alpha=0.45, linewidths=0)
                _errorbar(axis, x + offset, _summary(values), color, marker)
        axis.set_title(f"{chr(65+panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks(xbase, [item[0] for item in READOUTS])
        axis.grid(axis="y")
    for axis in axes[:, 0]:
        axis.set_ylabel("FPR95 (lower is better)")
    handles = [
        plt.Line2D([0], [0], marker=marker, color=color, markerfacecolor="white", label=role, linestyle="-")
        for role, (color, _, marker) in role_style.items()
    ]
    figure.legend(handles=handles, frameon=False, ncol=2, loc="outside upper center")
    return _save_figure(figure, output_dir, "supp_figure_fpr95_recovery")


def figure_churn(churn: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _style()
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), sharex=True, sharey=True, constrained_layout=True)
    fields = ("gain", "loss", "delta_auroc")
    labels = ("Gain", "−Loss", "Net Δ")
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        rows = [row for row in churn if row["cell"] == PRIMARY_CELL and row["dataset"] == dataset]
        values_by_field = {
            "gain": [row["gain"] for row in rows],
            "loss": [-row["loss"] for row in rows],
            "delta_auroc": [row["delta_auroc"] for row in rows],
        }
        means = [statistics.fmean(values_by_field[field]) for field in fields]
        axis.bar(range(3), means, color=[colors["green"], colors["orange"], colors["blue"]], alpha=0.72, width=0.66, edgecolor=colors["ink"], linewidth=0.45)
        for x, field in enumerate(fields):
            vals = values_by_field[field]
            jitter = np.linspace(-0.07, 0.07, len(vals))
            axis.scatter(x+jitter, vals, s=10, color=colors["ink"], alpha=0.5, linewidths=0, zorder=3)
        churn_mean = statistics.fmean(row["churn"] for row in rows)
        axis.text(0.03, 0.05, f"Churn = {churn_mean:.3f}", transform=axis.transAxes, fontsize=7.0)
        axis.axhline(0, color=colors["ink"], linewidth=0.7)
        axis.set_xticks(range(3), labels)
        axis.set_title(f"{chr(65+panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.grid(axis="y")
    for axis in axes[:, 0]:
        axis.set_ylabel("ID–OOD pair fraction")
    return _save_figure(figure, output_dir, "figure4_gain_loss_churn")


def figure_geometry_endpoint(pairs: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _style()
    figure, axes = plt.subplots(4, 2, figsize=(7.2, 8.25), constrained_layout=True)
    x = np.arange(len(CELLS), dtype=float)
    for panel, metric in enumerate(GEOMETRY_METRICS):
        axis = axes.flat[panel]
        for index, cell in enumerate(CELLS):
            values = [row["effect"] for row in pairs if row["metric"] == metric and row["cell"] == cell]
            jitter = np.linspace(-0.055, 0.055, len(values))
            axis.scatter(index+jitter, values, s=11, color=colors["blue"], alpha=0.5, linewidths=0)
            _errorbar(axis, index, _summary(values), colors["orange"], "D")
        axis.axhline(0, color=colors["ink"], linewidth=0.7)
        axis.set_xticks(x, [CELL_LABELS[cell].replace(" / ", "\n") for cell in CELLS])
        axis.set_ylabel(GEOMETRY_LABELS[metric])
        axis.set_title(f"{chr(65+panel)}. {GEOMETRY_LABELS[metric].splitlines()[0]}", loc="left")
        axis.grid(axis="y")
    return _save_figure(figure, output_dir, "figure5_geometry_endpoint")


def _plot_trajectory(axis: Any, rows: Sequence[Mapping[str, Any]], x_values: Sequence[Any], color: str, label: str | None = None) -> None:
    import numpy as np

    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        seed_rows = {row["axis_value"]: row for row in rows if int(row["seed"]) == seed}
        axis.plot(range(len(x_values)), [seed_rows[x]["effect"] for x in x_values], color=color, linewidth=0.55, alpha=0.24)
    means = []
    low = []
    high = []
    for x in x_values:
        stat = _summary(row["effect"] for row in rows if row["axis_value"] == x)
        means.append(stat["mean"])
        low.append(stat["t90_low"])
        high.append(stat["t90_high"])
    xx = np.arange(len(x_values))
    axis.plot(xx, means, color=color, linewidth=1.25, marker="o", markersize=3.5, label=label)
    axis.fill_between(xx, low, high, color=color, alpha=0.13, linewidth=0)


def figure_formation(
    formation_macro: Sequence[Mapping[str, Any]],
    geometry: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = _style()
    figure, axes = plt.subplots(5, 2, figsize=(7.2, 9.0), constrained_layout=True)
    epoch_values = (10, 60, 120, 160, 200)
    depth_values = ("stage1", "stage2", "stage3", "penultimate")
    for column, (axis_name, axis_values) in enumerate((("epoch", epoch_values), ("depth", depth_values))):
        ood_rows = [
            {**row, "effect": row["delta_auroc"]}
            for row in formation_macro
            if row["scope"] == "formation" and row["contrast"] == "C-D"
            and row["transform"] == "raw" and row["detector"] == "md" and row["axis"] == axis_name
        ]
        for region, color in (("Near", colors["blue"]), ("Far", colors["orange"])):
            _plot_trajectory(axes[0, column], [row for row in ood_rows if row["region"] == region], axis_values, color, region)
        axes[0, column].set_ylabel("Raw MD AUROC\nC − D")
        axes[0, column].legend(frameon=False, ncol=2)
        for row_index, metric in enumerate(CORE_GEOMETRY_METRICS, start=1):
            rows = [row for row in geometry if row["axis"] == axis_name and row["metric"] == metric]
            _plot_trajectory(axes[row_index, column], rows, axis_values, colors["green"])
            axes[row_index, column].set_ylabel(GEOMETRY_LABELS[metric])
        for row_index in range(5):
            axes[row_index, column].axhline(0, color=colors["ink"], linewidth=0.65)
            axes[row_index, column].set_xticks(range(len(axis_values)), [str(value).replace("penultimate", "penult.") for value in axis_values])
            axes[row_index, column].grid(axis="y")
        axes[0, column].set_title("A. Training trajectory" if column == 0 else "B. Depth profile", loc="left")
        axes[-1, column].set_xlabel("Epoch" if column == 0 else "Feature tap at epoch 200")
    return _save_figure(figure, output_dir, "figure6_time_depth_concordance")


def figure_top10(pairs: Sequence[Mapping[str, Any]], formation: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = _style()
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    values = [row for row in pairs if row["metric"] == "top10_trace_share"]
    for index, cell in enumerate(CELLS):
        cell_values = [row["effect"] for row in values if row["cell"] == cell]
        jitter = np.linspace(-0.05, 0.05, len(cell_values))
        axes[0].scatter(index+jitter, cell_values, s=10, color=colors["blue"], alpha=0.5)
        _errorbar(axes[0], index, _summary(cell_values), colors["orange"], "D")
    axes[0].set_xticks(range(len(CELLS)), [CELL_LABELS[cell].replace(" / ", "\n") for cell in CELLS])
    axes[0].set_title("A. Endpoint contexts", loc="left")
    for axis, axis_name, axis_values, title in (
        (axes[1], "epoch", (10, 60, 120, 160, 200), "B. Training trajectory"),
        (axes[2], "depth", ("stage1", "stage2", "stage3", "penultimate"), "C. Depth profile"),
    ):
        rows = [row for row in formation if row["metric"] == "top10_trace_share" and row["axis"] == axis_name]
        _plot_trajectory(axis, rows, axis_values, colors["green"])
        axis.set_xticks(range(len(axis_values)), [str(value).replace("penultimate", "penult.") for value in axis_values])
        axis.set_title(title, loc="left")
    for axis in axes:
        axis.axhline(0, color=colors["ink"], linewidth=0.65)
        axis.grid(axis="y")
        axis.set_ylabel("Top-10 trace share C − D")
    return _save_figure(figure, output_dir, "supp_figure_top10_trace_share")


def build_pack(
    *,
    merged_path: str | Path,
    manifest_input_paths: Sequence[str | Path],
    output_dir: str | Path,
    generator_git_sha: str | None = None,
) -> dict[str, Any]:
    merged_path = Path(merged_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    merged = read_json(merged_path)
    merged_validation = validate_source(merged)
    score_rows, geometry_rows, manifest_validation = combine_manifest_inputs(manifest_input_paths)

    alpha = alpha_seed_rows(score_rows)
    alpha_macro = alpha_region_macro_seed_rows(alpha)
    zero = zero_reference_rows(merged)
    heatmap = heatmap_seed_rows(merged)
    recovery = recovery_seed_rows(merged)
    context_absolute = context_absolute_raw_rows(recovery)
    recovery_gain = recovery_gain_seed_rows(recovery)
    churn = churn_seed_rows(merged)
    endpoint_geometry = endpoint_geometry_rows(geometry_rows)
    id_endpoint = id_endpoint_rows(geometry_rows)
    geometry_pairs = pair_geometry_rows(endpoint_geometry)
    formation_geometry = formation_geometry_pairs(geometry_rows)
    endpoint_macro = endpoint_macro_seed_rows(merged)
    formation_macro = formation_macro_seed_rows(merged)

    table_sets: dict[str, list[dict[str, Any]]] = {
        "alpha_seed_rows.csv": alpha,
        "alpha_summary.csv": summarize_rows(alpha, ("dataset", "role", "alpha"), ("auroc", "fpr95")),
        "alpha_region_macro_seed_rows.csv": alpha_macro,
        "alpha_region_macro_summary.csv": summarize_rows(alpha_macro, ("region", "role", "alpha"), ("auroc", "fpr95")),
        "zero_reference_seed_rows.csv": zero,
        "context_heatmap_seed_rows.csv": heatmap,
        "context_heatmap_summary.csv": summarize_rows(
            heatmap,
            ("cell", "dataset", "region"),
            ("delta_auroc", "delta_fpr95"),
            paired=True,
        ),
        "context_absolute_raw_seed_rows.csv": context_absolute,
        "context_absolute_raw_summary.csv": summarize_rows(
            context_absolute,
            ("cell", "dataset", "region", "role", "readout"),
            ("auroc", "fpr95"),
        ),
        "recovery_absolute_seed_rows.csv": recovery,
        "recovery_absolute_summary.csv": summarize_rows(recovery, ("cell", "dataset", "region", "role", "readout"), ("auroc", "fpr95")),
        "recovery_gain_seed_rows.csv": recovery_gain,
        "recovery_gain_summary.csv": summarize_rows(
            recovery_gain,
            ("cell", "dataset", "region", "role", "readout"),
            ("delta_auroc_from_raw", "delta_fpr95_from_raw"),
            paired=True,
        ),
        "churn_seed_rows.csv": churn,
        "churn_summary.csv": summarize_rows(
            churn,
            ("cell", "dataset", "region"),
            ("gain", "loss", "delta_auroc", "churn"),
            paired=True,
        ),
        "geometry_endpoint_seed_rows.csv": endpoint_geometry,
        "id_endpoint_seed_rows.csv": id_endpoint,
        "id_endpoint_summary.csv": summarize_rows(id_endpoint, ("cell", "role"), ("accuracy",)),
        "geometry_paired_effect_seed_rows.csv": geometry_pairs,
        "geometry_paired_effect_summary.csv": summarize_rows(
            geometry_pairs, ("cell", "metric"), ("effect",), paired=True
        ),
        "geometry_formation_seed_rows.csv": formation_geometry,
        "top10_trace_share_table.csv": summarize_rows(
            [row for row in endpoint_geometry], ("cell", "role"), ("top10_trace_share",)
        ),
    }
    table_paths = []
    for name, rows in table_sets.items():
        path = tables_dir / name
        _write_csv(path, rows)
        table_paths.append(path)

    figure_paths = []
    figure_paths += figure_alpha(alpha, figures_dir)
    figure_paths += figure_context_absolute_raw(context_absolute, figures_dir)
    figure_paths += figure_heatmap(heatmap, endpoint_macro, figures_dir)
    figure_paths += figure_recovery(recovery, figures_dir)
    figure_paths += figure_fpr95_recovery(recovery, figures_dir)
    figure_paths += figure_churn(churn, figures_dir)
    figure_paths += figure_geometry_endpoint(geometry_pairs, figures_dir)
    figure_paths += figure_formation(formation_macro, formation_geometry, figures_dir)
    figure_paths += figure_top10(geometry_pairs, formation_geometry, figures_dir)

    declared_outputs = sorted(table_paths + figure_paths)
    checksum_path = output_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir)}\n"
            for path in declared_outputs
        ),
        encoding="utf-8",
    )
    declared_outputs.append(checksum_path)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scientific_boundary": "completed-artifact analysis only; no training, inference, detector refit, or protected-example access",
        "statistical_unit": "training seed",
        "uncertainty": (
            "sample SD and two-sided 90% Student-t intervals across training seeds; "
            "paired label is used only for within-seed contrasts"
        ),
        "endpoint": "epoch 200 / last / penultimate",
        "primary_framing": "Raw MD failure and RMD/L2-MD recovery; decay coupling is the controlled intervention",
        "geometry_main": list(GEOMETRY_METRICS),
        "geometry_supplement": ["top10_trace_share"],
        "validation": {"merged": merged_validation, "manifest_inputs": manifest_validation},
        "inputs": {
            "merged_analysis": {"path": str(merged_path.resolve()), "sha256": sha256_file(merged_path)},
            "manifest_exports": manifest_validation["sources"],
        },
        "generator": {
            "git_sha": generator_git_sha or "WORKTREE_DIRTY",
            "module_path": str(Path(__file__).resolve()),
            "module_sha256": sha256_file(__file__),
        },
        "coverage": {name: len(rows) for name, rows in table_sets.items()},
        "outputs": [],
    }
    for path in declared_outputs:
        manifest["outputs"].append(
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return manifest
