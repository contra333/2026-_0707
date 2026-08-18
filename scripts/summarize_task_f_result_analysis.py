#!/usr/bin/env python3
"""Create compact Task F results and flat, auditable exports.

This script reads only the merged output of analyze_task_f_existing_artifacts.py.
It does not load checkpoints, fit detectors, or evaluate a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


NEAR = {"cifar100", "tin"}
FAR = {"mnist", "svhn", "texture", "places365"}
PRIMARY = "adam_lr1e-3_wd1e-4_anchor"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def mean(values: Iterable[float]) -> float:
    return float(statistics.mean(list(values)))


def macro_groups(rows: list[Mapping[str, Any]]) -> Iterable[tuple[str, list[Mapping[str, Any]]]]:
    for label, datasets in (("near", NEAR), ("far", FAR)):
        selected = [row for row in rows if row["dataset"] in datasets]
        if selected:
            yield label, selected


def aggregate_mean_rows(rows: list[Mapping[str, Any]], fields: Iterable[str]) -> dict[str, float]:
    return {field: mean(row[field]["mean"] for row in rows) for field in fields}


def endpoint_macros(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in data["endpoint_aggregates"]:
        groups[(row["cell"], row["contrast"], row["transform"], row["detector"])].append(row)
    output = []
    fields = (
        "left_auroc", "right_auroc", "delta_auroc", "left_fpr95", "right_fpr95",
        "delta_fpr95", "gain", "loss", "pair_order_churn",
    )
    for key, rows in sorted(groups.items()):
        for region, selected in macro_groups(rows):
            output.append(
                {
                    "cell": key[0], "contrast": key[1], "transform": key[2],
                    "detector": key[3], "region": region,
                    **aggregate_mean_rows(selected, fields),
                    "seeds": rows[0]["seeds"],
                }
            )
    return output


def localization_macros(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in data["score_localization_aggregates"]:
        groups[(row["cell"], row["contrast"], row["transform"])].append(row)
    output = []
    fields = ("delta_auroc", "phi_rmd", "phi_marginal", "phi_id", "phi_ood")
    for key, rows in sorted(groups.items()):
        for region, selected in macro_groups(rows):
            values = aggregate_mean_rows(selected, fields)
            delta = values["delta_auroc"]
            output.append(
                {
                    "cell": key[0], "contrast": key[1], "transform": key[2],
                    "region": region, **values,
                    "marginal_share_ratio_of_means": (
                        abs(values["phi_marginal"]) / abs(delta) if delta else None
                    ),
                    "boundary": "score accounting; not causal or unique mediation",
                }
            )
    return output


def within_lr_wd_macros(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in data["within_lr_wd_aggregates"]:
        groups[(row["cell"], row["contrast"], row["transform"], row["detector"])].append(row)
    output = []
    fields = ("delta_auroc", "delta_fpr95", "gain", "loss", "pair_order_churn")
    for key, rows in sorted(groups.items()):
        for region, selected in macro_groups(rows):
            output.append(
                {
                    "cell": key[0], "contrast": key[1], "transform": key[2],
                    "detector": key[3], "region": region,
                    **aggregate_mean_rows(selected, fields),
                    "boundary": "controlled within fixed LR; no cross-LR causal interpretation",
                }
            )
    return output


def did_macros(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in data["coupling_by_wd_difference_in_differences"]:
        groups[(row["lr_context"], row["transform"], row["detector"])].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        for region, selected in macro_groups(rows):
            output.append(
                {
                    "lr_context": key[0], "transform": key[1], "detector": key[2],
                    "region": region,
                    "did_delta_auroc": mean(row["did_delta_auroc"]["mean"] for row in selected),
                    "definition": "(C-D at WD=1e-3) - (C-D at WD=1e-4)",
                }
            )
    return output


GEOMETRY_FIELDS: dict[str, Callable[[Mapping[str, Any]], float]] = {
    "feature_norm_mean": lambda row: row["feature_norm"]["mean"],
    "feature_norm_cv": lambda row: row["feature_norm"]["cv"],
    "within_rank": lambda row: row["within_covariance"]["numerical_rank"],
    "global_rank": lambda row: row["global_covariance"]["numerical_rank"],
    "within_condition": lambda row: row["within_covariance"]["retained_condition_number"],
    "global_condition": lambda row: row["global_covariance"]["retained_condition_number"],
    "within_effective_rank": lambda row: row["within_covariance"]["effective_rank"],
    "global_effective_rank": lambda row: row["global_covariance"]["effective_rank"],
    "within_top10_trace_share": lambda row: row["within_covariance"]["top_10_trace_share"],
    "global_top10_trace_share": lambda row: row["global_covariance"]["top_10_trace_share"],
    "within_trace": lambda row: row["within_covariance"]["trace"],
    "global_trace": lambda row: row["global_covariance"]["trace"],
    "rankme": lambda row: row["rankme"],
}


def geometry_contrasts(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row for row in data["geometry_rows"]
        if row["scope"] == "endpoint" and row["transform"] == "raw" and row["role"] in ("C", "D")
    ]
    output = []
    for cell in sorted({row["cell"] for row in rows}):
        selected = [row for row in rows if row["cell"] == cell]
        for name, getter in GEOMETRY_FIELDS.items():
            by_role = {
                role: mean(getter(row) for row in selected if row["role"] == role)
                for role in ("D", "C")
            }
            output.append(
                {
                    "cell": cell, "metric": name, "decoupled_mean": by_role["D"],
                    "coupled_mean": by_role["C"], "coupled_minus_decoupled": by_role["C"] - by_role["D"],
                }
            )
    return output


def formation_macros(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    contexts = [("epoch", value) for value in (10, 60, 120, 160, 200)]
    contexts += [("depth", value) for value in ("stage1", "stage2", "stage3", "penultimate")]
    for axis, value in contexts:
        rows = [
            row for row in data["formation_score_rows"]
            if (
                row["checkpoint_epoch"] == value and row["depth_tap"] == "penultimate"
                if axis == "epoch" else row["checkpoint_epoch"] == 200 and row["depth_tap"] == value
            )
        ]
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row["transform"], row["detector"])].append(row)
        for key, grouped in sorted(groups.items()):
            for region, selected in macro_groups(grouped):
                output.append(
                    {
                        "axis": axis, "value": value, "transform": key[0], "detector": key[1],
                        "region": region,
                        "left_auroc": mean(row["left"]["auroc"] for row in selected),
                        "right_auroc": mean(row["right"]["auroc"] for row in selected),
                        "delta_auroc": mean(row["delta_auroc"] for row in selected),
                        "pair_order_churn": mean(row["pair_order_churn"] for row in selected),
                    }
                )
    return output


def burden_summary(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    consistency = {row["dataset"]: row for row in data["pair_burden_rank_consistency"]}
    for dataset in sorted({row["dataset"] for row in data["pair_burden_rows"]}):
        rows = [row for row in data["pair_burden_rows"] if row["dataset"] == dataset]
        output.append(
            {
                "dataset": dataset,
                "ood_churn_median": mean(row["ood_sample"]["churn"]["median"] for row in rows),
                "ood_churn_top10_share": mean(row["ood_sample"]["churn"]["concentration"]["top_10pct"] for row in rows),
                "ood_loss_top10_share": mean(row["ood_sample"]["loss"]["concentration"]["top_10pct"] for row in rows),
                "cross_seed_flip_burden_spearman": consistency[dataset]["pairwise_spearman"]["mean"],
                "cross_seed_top10_jaccard": consistency[dataset]["top_10pct_jaccard"]["mean"],
                "statistical_unit": "seed; samples and pairs are descriptive",
            }
        )
    return output


def spectral_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    rows = data["spectral_allocation_rows"]
    counts = Counter(row["status"] for row in rows)
    return {
        "status_counts": dict(sorted(counts.items())),
        "decoupled_max_reconstruction_residual": max(
            row["max_abs_stored_marginal_reconstruction_residual"]
            for row in rows if row["status"] == "PASS"
        ),
        "coupled_result": "NOT AVAILABLE: raw fit NOT_APPLICABLE for all five seeds",
        "claim_boundary": "prior spectrum-allocation lens; no cross-branch spectral attribution",
    }


def flatten(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--compact-output", type=Path, required=True)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--artifact-uri")
    parser.add_argument("--artifact-manifest-sha256")
    parser.add_argument("--artifact-checksums-sha256")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    input_provenance = {
        "extraction_host_path": str(args.input.resolve()),
        "sha256": sha256_file(args.input),
    }
    if args.artifact_uri:
        input_provenance["artifact_uri"] = args.artifact_uri
    if args.artifact_manifest_sha256:
        input_provenance["artifact_manifest_sha256"] = args.artifact_manifest_sha256
    if args.artifact_checksums_sha256:
        input_provenance["artifact_checksums_sha256"] = args.artifact_checksums_sha256
    compact = {
        "schema_version": "task_f_result_analysis_v1",
        "git_sha": args.git_sha,
        "input": input_provenance,
        "coverage": data["coverage"],
        "validation": data["validation"],
        "endpoint_macros": endpoint_macros(data),
        "score_localization_macros": localization_macros(data),
        "within_lr_wd_macros": within_lr_wd_macros(data),
        "coupling_by_wd_did_macros": did_macros(data),
        "raw_geometry_coupled_minus_decoupled": geometry_contrasts(data),
        "primary_formation_macros": formation_macros(data),
        "primary_pair_burden": burden_summary(data),
        "spectral_allocation": spectral_summary(data),
        "boundaries": {
            "cross_lr": "descriptive only",
            "score_accounting": "not causal or unique mediation",
            "s_perp": "primary raw/L2 component theorem NOT_APPLICABLE",
            "optimizer_origin": "research-program question; Task F is a controlled coupling/WD case study",
        },
    }
    args.compact_output.parent.mkdir(parents=True, exist_ok=True)
    args.compact_output.write_bytes(canonical_bytes(compact))
    args.detail_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "endpoint_aggregates.csv": data["endpoint_aggregates"],
        "endpoint_seed_rows.csv": data["endpoint_score_rows"],
        "score_localization_aggregates.csv": data["score_localization_aggregates"],
        "score_localization_seed_rows.csv": data["score_localization_rows"],
        "within_lr_wd_aggregates.csv": data["within_lr_wd_aggregates"],
        "coupling_by_wd_did.csv": data["coupling_by_wd_difference_in_differences"],
        "geometry_rows.csv": data["geometry_rows"],
        "formation_seed_rows.csv": data["formation_score_rows"],
        "pair_burden_rows.csv": data["pair_burden_rows"],
        "alignment_rows.csv": data["alignment_rows"],
        "telemetry_rows.csv": data["telemetry_rows"],
    }
    for filename, rows in tables.items():
        write_csv(args.detail_dir / filename, rows)
    manifest = {
        "schema_version": "task_f_result_analysis_export_v1",
        "git_sha": args.git_sha,
        "input": compact["input"],
        "compact_output": {
            "path": str(args.compact_output.resolve()), "sha256": sha256_file(args.compact_output)
        },
        "outputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(args.detail_dir.iterdir())
            if path.is_file() and path.name != "extraction_manifest.json"
        ],
        "commands": [
            (
                "python3 scripts/summarize_task_f_result_analysis.py --input <merged.json> "
                "--compact-output results/task_f_result_analysis_v1.json --detail-dir <external-dir> "
                f"--git-sha {args.git_sha} --artifact-uri <hf-uri> "
                "--artifact-manifest-sha256 <sha256> --artifact-checksums-sha256 <sha256>"
            ),
            (
                "python3 scripts/plot_task_f_result_analysis.py "
                "--input results/task_f_result_analysis_v1.json --output-dir <external-dir>"
            ),
        ],
    }
    manifest_path = args.detail_dir / "extraction_manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    print(json.dumps({"compact": manifest["compact_output"], "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
