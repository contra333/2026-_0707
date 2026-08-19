"""Export manifest-only inputs for the frozen Task F paper figures.

This module deliberately reads only JSON manifests.  It never opens feature
arrays, score arrays, checkpoints, or protected examples.  The compact output
is suitable for transferring completed per-seed metrics from the three worker
hosts to the analysis workstation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "task_f_figure_inputs_v1"
DATASETS = ("cifar100", "tin", "mnist", "svhn", "texture", "places365")
TRANSFORMS = ("raw", "l2")
DETECTORS = ("md", "rmd")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_role(manifest: Mapping[str, Any]) -> str | None:
    role = str(manifest.get("sibling_role", ""))
    policy = str(manifest.get("branch_policy", ""))
    if role == "zero" or policy == "zero_decay":
        return "Z"
    if role == "alpha_1" or "_adam_coupled" in role or "_sgdm_coupled" in role:
        return "C"
    if role == "alpha_0" or "_adamw_decoupled" in role or "_sgdw_decoupled" in role:
        return "D"
    if role == "alpha_0_5":
        return "M"
    return None


def _value(record: Any) -> float | None:
    if isinstance(record, Mapping):
        value = record.get("value")
    else:
        value = record
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _identity(manifest: Mapping[str, Any], *, source_path: Path, host: str) -> dict[str, Any]:
    return {
        "host": host,
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "output_identity_sha256": str(manifest["output_identity_sha256"]),
        "run_id": str(manifest["run_id"]),
        "cell": str(manifest["cell_id"]),
        "role": normalized_role(manifest),
        "seed": int(manifest["training_seed"]),
        "checkpoint_epoch": int(manifest["checkpoint_epoch"]),
        "checkpoint_role": str(manifest["checkpoint_role"]),
        "depth_tap": str(manifest["depth_tap"]),
        "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
        "initialization_sha256": str(manifest["initialization_sha256"]),
        "data_stream_sha256": str(manifest["data_stream_sha256"]),
        "sibling_group_id": str(manifest["sibling_group_id"]),
        "protected_data_access": bool(manifest["protected_data_access"]),
    }


def score_rows(manifest: Mapping[str, Any], *, source_path: Path, host: str) -> list[dict[str, Any]]:
    identity = _identity(manifest, source_path=source_path, host=host)
    if identity["role"] is None:
        raise ValueError(f"unrecognized sibling role in {source_path}")
    if not identity["protected_data_access"]:
        raise ValueError(f"score manifest does not declare protected access: {source_path}")
    output: list[dict[str, Any]] = []
    metrics = manifest["ood_metrics"]
    id_accuracy = _value(manifest.get("id_utility", {}).get("accuracy"))
    for transform in TRANSFORMS:
        for detector in DETECTORS:
            per_dataset = metrics.get(transform, {}).get(detector, {}).get("per_dataset", {})
            for dataset in DATASETS:
                record = per_dataset.get(dataset)
                if record is None:
                    continue
                output.append(
                    {
                        **identity,
                        "transform": transform,
                        "detector": detector,
                        "dataset": dataset,
                        "auroc": float(record["auroc"]),
                        "fpr95": float(record["fpr95_id_tpr"]),
                        "fpr95_achieved_id_tpr": float(record["fpr95_achieved_id_tpr"]),
                        "id_accuracy": id_accuracy,
                    }
                )
    return output


def _top_trace_share(spectrum: Mapping[str, Any], count: int) -> float | None:
    values = [max(float(value), 0.0) for value in spectrum.get("raw_eigenvalues", [])]
    trace = float(spectrum.get("trace", sum(values)))
    if not values or not math.isfinite(trace) or trace <= 0.0:
        return None
    return sum(values[:count]) / trace


def geometry_row(manifest: Mapping[str, Any], *, source_path: Path, host: str) -> dict[str, Any]:
    identity = _identity(manifest, source_path=source_path, host=host)
    if identity["role"] is None:
        raise ValueError(f"unrecognized sibling role in {source_path}")
    if identity["protected_data_access"]:
        raise ValueError(f"geometry manifest unexpectedly declares protected access: {source_path}")
    raw = manifest["summary"]["transforms"]["raw"]
    collapse = raw["neural_collapse"]
    total_spectrum = raw["covariance_spectra"]["total"]
    within_spectrum = raw["covariance_spectra"]["sw"]
    return {
        **identity,
        "feature_norm": float(raw["norm"]["global"]["mean"]),
        "effective_rank": _value(total_spectrum["entropy_rank"]),
        "top10_trace_share": _top_trace_share(total_spectrum, 10),
        "within_trace": float(within_spectrum["trace"]),
        "cdnv": _value(raw["cdnv"]["metric"]),
        "nc0": _value(collapse["nc0_row_sum_raw"]),
        "nc1": _value(collapse["nc1_pinv"]),
        "nc2": _value(collapse["nc2_etf_raw"]),
        "nc3": _value(collapse["nc3_self_duality_raw"]),
        "id_accuracy": _value(raw.get("id_utility", {}).get("accuracy")),
        "metric_definitions": {
            "feature_norm": "mean raw penultimate feature L2 norm",
            "effective_rank": "entropy effective rank of raw total covariance",
            "top10_trace_share": "top-10 nonnegative eigenvalue share of raw total covariance trace",
            "within_trace": "trace of raw pooled within-class covariance Sw",
            "cdnv": "mean class-distance-normalized variance",
            "nc0": "nc0_row_sum_raw",
            "nc1": "nc1_pinv",
            "nc2": "nc2_etf_raw",
            "nc3": "nc3_self_duality_raw",
        },
    }


def _read_manifests(root: Path) -> list[tuple[dict[str, Any], Path]]:
    output = []
    for path in sorted(root.glob("*/manifest.json")):
        output.append((json.loads(path.read_text(encoding="utf-8")), path))
    return output


def export_inputs(
    *,
    host: str,
    score_root: Path | None = None,
    geometry_root: Path | None = None,
) -> dict[str, Any]:
    score_entries = _read_manifests(score_root) if score_root is not None else []
    geometry_entries = _read_manifests(geometry_root) if geometry_root is not None else []
    scores = [
        row
        for manifest, path in score_entries
        for row in score_rows(manifest, source_path=path, host=host)
    ]
    geometry = [
        geometry_row(manifest, source_path=path, host=host)
        for manifest, path in geometry_entries
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "host": host,
        "roots": {
            "score": str(score_root) if score_root is not None else None,
            "geometry": str(geometry_root) if geometry_root is not None else None,
        },
        "coverage": {
            "score_manifests": len(score_entries),
            "score_metric_rows": len(scores),
            "geometry_manifests": len(geometry_entries),
        },
        "score_rows": scores,
        "geometry_rows": geometry,
        "scientific_boundary": (
            "manifest-only export; no checkpoint, feature array, score array, or protected example opened"
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--score-root", type=Path)
    parser.add_argument("--geometry-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.score_root is None and args.geometry_root is None:
        raise SystemExit("at least one of --score-root or --geometry-root is required")
    json.dump(
        export_inputs(host=args.host, score_root=args.score_root, geometry_root=args.geometry_root),
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
