#!/usr/bin/env python3
"""Build paper-ready Task F tables and static figures from merged results."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from oge.analysis.task_f_paper_pack import (
    SCHEMA_VERSION,
    build_macro_seed_rows,
    canonical_json_bytes,
    negative_gate_rows,
    render_figures,
    sha256_file,
    summarize_macro_seed_rows,
    table_rows,
    validate_source,
    write_csv,
)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-git-sha")
    parser.add_argument("--classifier-kill-result", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    validation = validate_source(data)
    macro_seed_rows = build_macro_seed_rows(data)
    macro_summary_rows = summarize_macro_seed_rows(macro_seed_rows)
    output_dir = args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    write_csv(table_dir / "paper_macro_seed_rows.csv", macro_seed_rows)
    write_csv(table_dir / "paper_macro_summary.csv", macro_summary_rows)
    for filename, rows in table_rows(macro_seed_rows, macro_summary_rows).items():
        write_csv(table_dir / filename, rows)

    classifier_kill = None
    if args.classifier_kill_result is not None:
        classifier_kill = json.loads(args.classifier_kill_result.read_text(encoding="utf-8"))
    write_csv(table_dir / "appendix_negative_gates.csv", negative_gate_rows(data, classifier_kill))
    source_sha = sha256_file(args.input)
    figure_paths, geometry_rows = render_figures(
        data, macro_seed_rows, macro_summary_rows, figure_dir, source_sha
    )
    write_csv(table_dir / "appendix_geometry_summary.csv", geometry_rows)

    chart_contracts = {
        "figure1_controlled_design": {
            "question": "What is held fixed and what training rule changes?",
            "form": "controlled-design schematic",
        },
        "figure2_pair_multiplicity": {
            "question": "How large is C-D Raw-MD sensitivity, and what pair churn can net AUROC hide?",
            "form": "faceted seed-dot interval plus signed Gain/Loss bridge",
        },
        "figure3_score_localization": {
            "question": "Which readout and exact score component carries the observed sensitivity?",
            "form": "readout dot intervals plus signed Shapley decomposition",
        },
        "figure4_time_depth_formation": {
            "question": "When and where does the primary sensitivity form?",
            "form": "seed-dot interval by discrete epoch and network depth",
        },
        "appendix_geometry_matrix": {
            "question": "Which endpoint geometry diagnostics move with C-D?",
            "form": "column-normalized signed matrix; descriptive only",
        },
        "palette": "blue/orange/gold with marker and fill redundancy; no red/green semantics",
        "statistical_unit": "training seed; OOD datasets are averaged within seed before inference",
    }
    chart_contract_path = output_dir / "chart_contracts.json"
    chart_contract_path.write_bytes(canonical_json_bytes(chart_contracts))

    analysis_sha = args.analysis_git_sha or _git_sha()
    all_outputs = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "paper_pack_manifest.json"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "analysis_git_sha": analysis_sha,
        "source": {
            "path": str(args.input.resolve()),
            "sha256": source_sha,
            "classifier_kill_path": (
                str(args.classifier_kill_result.resolve())
                if args.classifier_kill_result is not None
                else None
            ),
            "classifier_kill_sha256": (
                sha256_file(args.classifier_kill_result)
                if args.classifier_kill_result is not None
                else None
            ),
        },
        "validation": validation,
        "row_counts": {
            "paper_macro_seed_rows": len(macro_seed_rows),
            "paper_macro_summary": len(macro_summary_rows),
        },
        "outputs": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in all_outputs
        ],
        "figure_stems": sorted({path.stem for path in figure_paths}),
        "boundaries": {
            "near_far": "OOD datasets are equally weighted within each seed",
            "interval": "two-sided paired 90% Student-t interval across training seeds",
            "shapley": "exact score accounting, not causal or unique mediation",
            "cross_lr": "descriptive only in the existing WRN study",
        },
    }
    manifest_path = output_dir / "paper_pack_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "source_sha256": source_sha,
                "analysis_git_sha": analysis_sha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
