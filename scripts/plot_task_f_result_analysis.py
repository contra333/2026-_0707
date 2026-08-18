#!/usr/bin/env python3
"""Render descriptive Task F figures from the compact result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CELLS = [
    "adam_lr1e-3_wd1e-4_anchor",
    "adam_lr1e-3_wd1e-3",
    "adam_lr3e-4_wd1e-4",
    "adam_lr3e-4_wd1e-3",
]
LABELS = ["1e-3 / 1e-4", "1e-3 / 1e-3", "3e-4 / 1e-4", "3e-4 / 1e-3"]


def endpoint_figure(data: dict, output: Path) -> None:
    rows = {
        (row["cell"], row["region"]): row
        for row in data["endpoint_macros"]
        if row["contrast"] == "C-D" and row["transform"] == "raw" and row["detector"] == "md"
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    x = np.arange(len(CELLS))
    width = 0.36
    for axis, region in zip(axes, ("near", "far")):
        decoupled = [rows[(cell, region)]["right_auroc"] for cell in CELLS]
        coupled = [rows[(cell, region)]["left_auroc"] for cell in CELLS]
        axis.bar(x - width / 2, decoupled, width, label="AdamW (D)", color="#3973ac")
        axis.bar(x + width / 2, coupled, width, label="Adam (C)", color="#c94f3d")
        axis.set_title(region.capitalize())
        axis.set_xticks(x, LABELS, rotation=25, ha="right")
        axis.set_ylim(0.3, 0.85)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Raw MD AUROC")
    axes[1].legend(frameon=False, loc="upper right")
    figure.suptitle("Adam contexts: absolute Raw MD performance")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def formation_figure(data: dict, output: Path) -> None:
    rows = [
        row for row in data["primary_formation_macros"]
        if row["transform"] == "raw" and row["detector"] == "md"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for axis, axis_name, values in (
        (axes[0], "epoch", [10, 60, 120, 160, 200]),
        (axes[1], "depth", ["stage1", "stage2", "stage3", "penultimate"]),
    ):
        for region, color in (("near", "#3973ac"), ("far", "#c94f3d")):
            selected = {
                str(row["value"]): row for row in rows
                if row["axis"] == axis_name and row["region"] == region
            }
            y = [selected[str(value)]["delta_auroc"] for value in values]
            axis.plot(range(len(values)), y, marker="o", label=region.capitalize(), color=color)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xticks(range(len(values)), [str(value) for value in values], rotation=20)
        axis.set_title(axis_name.capitalize())
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Coupled - decoupled Raw MD AUROC")
    axes[1].legend(frameon=False)
    figure.suptitle("Primary formation: early detectable, later/deeper amplified")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def concordance_figure(data: dict, output: Path) -> None:
    endpoint = {
        (row["cell"], row["region"]): row["delta_auroc"]
        for row in data["endpoint_macros"]
        if row["contrast"] == "C-D" and row["transform"] == "raw" and row["detector"] == "md"
    }
    geometry = {
        (row["cell"], row["metric"]): row for row in data["raw_geometry_coupled_minus_decoupled"]
    }
    x = [geometry[(cell, "global_top10_trace_share")]["coupled_minus_decoupled"] for cell in CELLS]
    y = [endpoint[(cell, "near")] for cell in CELLS]
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.scatter(x, y, s=65, color="#3b7d61")
    for x_value, y_value, label in zip(x, y, LABELS):
        axis.annotate(label, (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("C-D global top-10 trace share")
    axis.set_ylabel("C-D Near Raw MD AUROC")
    axis.set_title("Four-cell descriptive concordance (not a regression)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_figure(data, args.output_dir / "adam_context_raw_md.png")
    formation_figure(data, args.output_dir / "primary_formation_raw_md.png")
    concordance_figure(data, args.output_dir / "geometry_effect_concordance.png")


if __name__ == "__main__":
    main()
