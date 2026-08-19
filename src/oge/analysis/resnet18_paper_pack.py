"""Seed-first ResNet-18 endpoint analysis and PARTIAL-boundary paper pack.

The two entry points intentionally operate on completed artifacts only:

``build_host_geometry`` reads host-local ID feature artifacts and computes
post-result endpoint geometry on ``id_train``.  ``build_paper_pack`` reads the
already collected pair-score artifacts, host geometry summaries, and frozen
WRN reader pack.  Neither function loads a checkpoint, refits a detector, or
performs protected inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oge.analysis.fixed_readout_component_attribution import (
    pair_outcome_summary,
    pair_transition_summary,
    paired_component_attribution,
)
from oge.analysis.task_f_paper_pack import mean_sd_t90


SCHEMA_VERSION = "resnet18_seed_first_paper_pack_v1"
GEOMETRY_SCHEMA_VERSION = "resnet18_id_endpoint_geometry_host_v1"
WRN_MANIFEST_SHA256 = (
    "29de87077cedd732b1ffc9b5827664e4e7b96d1bc75442103021a1b7a3acbf0e"
)
PAIR_PLAN_SHA256 = (
    "3bc0348684702973a16d3b6f0c8fe84ee24a565b425e1149ff214d19af090df2"
)
TRAINING_TERMINAL_SHA256 = (
    "780dcf602a955c8936d3c901a2d473fe2c510145f5520886887b0a5dd52b99b5"
)
RESNET_TERMINAL_SHA256 = (
    "f0492271d31d13a1d8b4774303ba22485c701844e46bf37a2947743b5343721f"
)

DATASETS = ("cifar100", "tin", "mnist", "svhn", "texture", "places365")
DATASET_LABELS = {
    "cifar100": "CIFAR-100",
    "tin": "Tiny ImageNet",
    "mnist": "MNIST",
    "svhn": "SVHN",
    "texture": "Textures",
    "places365": "Places365",
}
DATASET_REGIONS = {
    "cifar100": "Near",
    "tin": "Near",
    "mnist": "Far",
    "svhn": "Far",
    "texture": "Far",
    "places365": "Far",
}
REGIONS = ("Near", "Far")
CELLS = (
    "resnet18_c10_lr1e-03_wd1e-4",
    "resnet18_c10_lr3e-04_wd1e-4",
)
CELL_CONTEXT = {
    CELLS[0]: "large",
    CELLS[1]: "small",
}
CONTEXT_LABELS = {
    "large": "Large: LR 1e-3",
    "small": "Small: LR 3e-4",
}
WRN_CELLS = {
    "large": "adam_lr1e-3_wd1e-4_anchor",
    "small": "adam_lr3e-4_wd1e-4",
}
ROLE_LABELS = {"D": "AdamW (D)", "C": "Adam (C)"}
READOUTS = (
    ("Raw MD", "raw", "md", "md"),
    ("RMD", "raw", "rmd", "rmd"),
    ("L2-MD", "l2", "md", "l2_md"),
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
GEOMETRY_LABELS = {
    "feature_norm": "Feature norm\nlog(C/D)",
    "effective_rank": "Total-cov. effective rank\nC - D",
    "within_trace": "Within-class trace\nlog(C/D)",
    "cdnv": "CDNV\nC - D",
    "nc0": "NC0 row-sum raw\nlog(C/D)",
    "nc1": "NC1 pinv\nC - D",
    "nc2": "NC2 ETF raw\nC - D",
    "nc3": "NC3 self-duality raw\nC - D",
}
EXPECTED_COUNTS = {
    "id_test": 10000,
    "cifar100": 9000,
    "tin": 7793,
    "mnist": 70000,
    "svhn": 26032,
    "texture": 5640,
    "places365": 35195,
}
RUN_PATTERN = re.compile(
    r"^resnet18-c10-rep-v3-lr(?P<lr>1e-03|3e-04)-wd1e-04-seed"
    r"(?P<seed>[0-4])-(?P<policy>adam-coupled|adamw-decoupled)$"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_value(payload: Mapping[str, Any], *, name: str) -> float:
    if payload.get("status") != "success" or payload.get("value") is None:
        raise ValueError(f"geometry metric {name} is unavailable: {payload}")
    value = float(payload["value"])
    if not math.isfinite(value):
        raise ValueError(f"geometry metric {name} is not finite")
    return value


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _verify_checksum_catalog(root: Path) -> dict[str, str]:
    catalog = root / "checksums.sha256"
    if not catalog.is_file():
        raise ValueError(f"missing checksum catalog: {catalog}")
    verified: dict[str, str] = {}
    for line in catalog.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = root / relative
        if relative in verified or not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"checksum mismatch: {root} / {relative}")
        verified[relative] = digest
    return verified


def _parse_run_id(run_id: str) -> dict[str, Any]:
    match = RUN_PATTERN.fullmatch(run_id)
    if match is None:
        raise ValueError(f"unexpected ResNet run id: {run_id}")
    lr_text = match.group("lr")
    role = "C" if match.group("policy") == "adam-coupled" else "D"
    cell = CELLS[0] if lr_text == "1e-03" else CELLS[1]
    return {
        "run_id": run_id,
        "cell": cell,
        "context": CELL_CONTEXT[cell],
        "seed": int(match.group("seed")),
        "role": role,
        "branch_policy": "adam_coupled" if role == "C" else "adamw_decoupled",
        "learning_rate": 1.0e-3 if lr_text == "1e-03" else 3.0e-4,
        "weight_decay": 1.0e-4,
    }


def _load_feature_array(
    root: Path, manifest: Mapping[str, Any], split: str, name: str
) -> np.ndarray:
    split_meta = manifest["dataset"]["splits"][split]
    path = root / split_meta["relative_directory"] / f"{name}.npy"
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != split_meta["array_shapes"][name]:
        raise ValueError(f"feature array shape mismatch: {path}")
    if str(value.dtype) != split_meta["array_dtypes"][name]:
        raise ValueError(f"feature array dtype mismatch: {path}")
    return value


def _validate_id_manifest(
    manifest: Mapping[str, Any], *, terminal_row: Mapping[str, Any]
) -> dict[str, Any]:
    parsed = _parse_run_id(str(terminal_row["run_id"]))
    checkpoint = manifest.get("checkpoint", {})
    model = manifest.get("model", {})
    dataset = manifest.get("dataset", {})
    evaluation = manifest.get("evaluation", {})
    if (
        manifest.get("artifact_role") != "raw_checkpoint_feature_cache"
        or manifest.get("schema_version") != "1.0"
        or checkpoint.get("training_run_id") != terminal_row["run_id"]
        or checkpoint.get("sha256") != terminal_row["checkpoint_sha256"]
        or checkpoint.get("role") != "last"
        or int(checkpoint.get("completed_epoch", -1)) != 200
        or model.get("name") != "resnet18"
        or int(model.get("feature_dim", -1)) != 512
        or int(model.get("class_count", -1)) != 10
        or dataset.get("protocol") != "oge_cifar10_holdout_v1"
        or evaluation.get("protected_split_authorization") is not None
        or evaluation.get("smoke_only") is not False
    ):
        raise ValueError(f"ID feature manifest protocol mismatch: {terminal_row['run_id']}")
    splits = dataset.get("splits", {})
    for split, expected in (("id_train", 45000), ("id_validation", 5000)):
        row = splits.get(split, {})
        if (
            int(row.get("sample_count", -1)) != expected
            or row.get("group") != "id"
            or row.get("is_id") is not True
            or row.get("array_shapes", {}).get("features") != [expected, 512]
        ):
            raise ValueError(f"ID split contract mismatch: {terminal_row['run_id']} / {split}")
    return parsed


def build_host_geometry(
    *, id_terminal_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Compute endpoint geometry from existing host-local ID artifacts."""

    # Geometry depends on the training runtime (including torch through the
    # evaluation package). Keep it host-local; the central score-only builder
    # must not acquire a checkpoint or training-runtime dependency.
    from oge.evaluation.geometry import (
        cdnv,
        covariance_spectrum,
        feature_norm_distribution,
        fit_geometry_statistics,
        neural_collapse_metrics,
    )

    terminal_path = Path(id_terminal_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite geometry output: {output}")
    terminal = read_json(terminal_path)
    if (
        terminal.get("status") != "PASS"
        or terminal.get("phase") != "ID_ONLY_EXPORT_AND_FIT"
        or terminal.get("protected_data_access") is not False
        or terminal.get("plan_sha256") != PAIR_PLAN_SHA256
        or int(terminal.get("run_count", -1)) != len(terminal.get("runs", []))
    ):
        raise ValueError("host ID terminal contract mismatch")

    rows: list[dict[str, Any]] = []
    for terminal_row in terminal["runs"]:
        root = Path(terminal_row["id_artifact"])
        verified = _verify_checksum_catalog(root)
        manifest = read_json(root / "manifest.json")
        if canonical_sha256(manifest) != terminal_row["id_artifact_manifest_sha256"]:
            raise ValueError(f"ID manifest identity mismatch: {terminal_row['run_id']}")
        parsed = _validate_id_manifest(manifest, terminal_row=terminal_row)
        train_meta = manifest["dataset"]["splits"]["id_train"]
        validation_meta = manifest["dataset"]["splits"]["id_validation"]
        if (
            train_meta["ordered_sample_id_sha256"]
            != terminal["sample_order_sha256"]["id_train"]
            or validation_meta["ordered_sample_id_sha256"]
            != terminal["sample_order_sha256"]["id_validation"]
        ):
            raise ValueError(f"ID sample ordering mismatch: {terminal_row['run_id']}")

        features = _load_feature_array(root, manifest, "id_train", "features")
        labels = _load_feature_array(root, manifest, "id_train", "class_labels")
        classifier_weight = np.load(root / "classifier_weight.npy", allow_pickle=False)
        classifier_bias = np.load(root / "classifier_bias.npy", allow_pickle=False)
        if classifier_weight.shape != (10, 512) or classifier_bias.shape != (10,):
            raise ValueError(f"classifier shape mismatch: {terminal_row['run_id']}")

        statistics_payload = fit_geometry_statistics(features, labels, num_classes=10)
        norm = feature_norm_distribution(statistics_payload)["global"]["mean"]
        cdnv_payload = cdnv(statistics_payload)["metric"]
        nc = neural_collapse_metrics(
            statistics_payload, classifier_weight, classifier_bias
        )
        spectrum = covariance_spectrum(statistics_payload.total_covariance)
        if spectrum.get("status") != "success":
            raise ValueError(f"total covariance spectrum failed: {terminal_row['run_id']}")
        eigenvalues = np.asarray(spectrum["eigenvalues"], dtype=np.float64)
        trace = float(spectrum["trace"])
        metrics = {
            "feature_norm": float(norm),
            "effective_rank": _metric_value(spectrum["entropy_rank"], name="effective_rank"),
            "within_trace": float(np.trace(statistics_payload.within_covariance)),
            "cdnv": _metric_value(cdnv_payload, name="cdnv"),
            "nc0": _metric_value(nc["nc0_row_sum_raw"], name="nc0"),
            "nc1": _metric_value(nc["nc1_pinv"], name="nc1"),
            "nc2": _metric_value(nc["nc2_etf_raw"], name="nc2"),
            "nc3": _metric_value(nc["nc3_self_duality_raw"], name="nc3"),
            "top10_trace_share": float(np.sum(eigenvalues[:10]) / trace),
        }
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"non-finite endpoint geometry: {terminal_row['run_id']}")
        rows.append(
            {
                **parsed,
                **metrics,
                "checkpoint_role": "last",
                "checkpoint_epoch": 200,
                "feature_tap": "penultimate",
                "geometry_fit_split": "id_train",
                "geometry_fit_count": 45000,
                "held_out_geometry_split": "id_validation",
                "held_out_geometry_count": 5000,
                "classifier_shape": [10, 512],
                "total_covariance_rank": int(
                    _metric_value(spectrum["numerical_rank"], name="numerical_rank")
                ),
                "total_covariance_dimension": 512,
                "total_covariance_condition": _metric_value(
                    spectrum["condition_number_retained"], name="condition"
                ),
                "source_artifact": str(root),
                "source_manifest_canonical_sha256": canonical_sha256(manifest),
                "source_manifest_file_sha256": sha256_file(root / "manifest.json"),
                "source_checksum_catalog_sha256": sha256_file(root / "checksums.sha256"),
                "verified_file_count": len(verified),
                "id_train_order_sha256": train_meta["ordered_sample_id_sha256"],
                "id_validation_order_sha256": validation_meta["ordered_sample_id_sha256"],
            }
        )

    expected = {
        (CELL_CONTEXT[cell], seed, role)
        for cell in CELLS
        for seed in range(5)
        for role in ("C", "D")
    }
    observed = {(row["context"], row["seed"], row["role"]) for row in rows}
    host_seeds = {row["seed"] for row in rows}
    if not observed or not observed.issubset(expected):
        raise ValueError("host geometry contains an unexpected run")
    for seed in host_seeds:
        for context in ("large", "small"):
            if {(context, seed, role) for role in ("C", "D")} - observed:
                raise ValueError("host geometry is missing a C/D sibling")

    payload: dict[str, Any] = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "status": "PASS",
        "scientific_boundary": (
            "post-result ID-only exploratory endpoint geometry; not a replication "
            "gate and cannot change PARTIAL"
        ),
        "protected_data_access": False,
        "checkpoint_inference": False,
        "detector_refit": False,
        "host": terminal["host_id"],
        "source_terminal": {
            "path": str(terminal_path.resolve()),
            "sha256": sha256_file(terminal_path),
            "terminal_sha256": terminal["terminal_sha256"],
        },
        "split_roles": {
            "primary_geometry_fit": {"split": "id_train", "count": 45000},
            "held_out_geometry_control": {"split": "id_validation", "count": 5000},
            "id_utility": {"split": "id_test", "count": 10000},
        },
        "rows": sorted(rows, key=lambda row: (row["context"], row["seed"], row["role"])),
    }
    payload["output_identity_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return payload


def verify_host_geometry(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    identity = payload.pop("output_identity_sha256", None)
    if (
        payload.get("schema_version") != GEOMETRY_SCHEMA_VERSION
        or payload.get("status") != "PASS"
        or payload.get("protected_data_access") is not False
        or identity != canonical_sha256(payload)
    ):
        raise ValueError(f"host geometry identity mismatch: {path}")
    payload["output_identity_sha256"] = identity
    return payload


def _ordinary_summary(values: Iterable[float], *, paired: bool) -> dict[str, Any]:
    summary = mean_sd_t90(values)
    if summary["n"] > 1:
        summary["interval_definition"] = (
            "two-sided paired 90% Student-t interval across training seeds"
            if paired
            else "two-sided 90% Student-t interval across training seeds"
        )
    return summary


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    metrics: Sequence[str],
    *,
    paired: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        base = dict(zip(group_fields, key, strict=True))
        for metric in metrics:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            if values:
                output.append(
                    {**base, "statistic": metric, **_ordinary_summary(values, paired=paired)}
                )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _transition_rates(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    utility = {"incorrect": 0.0, "tie": 0.5, "correct": 1.0}
    pair_count = int(summary["pair_count"])
    gain = 0.0
    loss = 0.0
    for source, targets in summary["transitions"].items():
        for target, number in targets.items():
            delta = utility[target] - utility[source]
            gain += max(delta, 0.0) * int(number)
            loss += max(-delta, 0.0) * int(number)
    return gain / pair_count, loss / pair_count, (gain + loss) / pair_count


def _verify_pair_artifact(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record = read_json(root / "record.json")
    if (
        record.get("status") != "PASS"
        or record.get("research_evidence") is not True
        or record.get("protected_data_access") is not True
        or record.get("plan_sha256") != PAIR_PLAN_SHA256
        or root.name != record.get("output_identity_sha256")
    ):
        raise ValueError(f"pair record protocol mismatch: {root}")
    catalog = _verify_checksum_catalog(root)
    if set(catalog) != {"record.json", "scores.npz"}:
        raise ValueError(f"pair artifact catalog mismatch: {root}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(root / "scores.npz", allow_pickle=False) as source:
        if len(source.files) != 84 or set(source.files) != set(record["score_arrays"]):
            raise ValueError(f"pair score-array coverage mismatch: {root}")
        for name in source.files:
            value = np.asarray(source[name])
            metadata = record["score_arrays"][name]
            if (
                list(value.shape) != metadata["shape"]
                or _array_sha256(value) != metadata["array_sha256"]
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"pair score-array identity mismatch: {root} / {name}")
            arrays[name] = value
    return record, arrays


def _score_key(role: str, transform: str, split: str, detector: str) -> str:
    prefix = "coupled" if role == "C" else "decoupled"
    return f"{prefix}__{transform}__{split}__{detector}"


def _float_close(left: float, right: float, *, tolerance: float = 1.0e-12) -> None:
    if abs(float(left) - float(right)) > tolerance:
        raise ValueError(f"stored/recomputed metric mismatch: {left} != {right}")


def _compute_primary_ood_metrics(
    id_like_scores: np.ndarray, ood_like_scores: np.ndarray
) -> dict[str, float]:
    """Recompute the frozen ID-positive AUROC and project FPR95 estimands."""

    id_scores = np.asarray(id_like_scores, dtype=np.float64).reshape(-1)
    ood_scores = np.asarray(ood_like_scores, dtype=np.float64).reshape(-1)
    if (
        id_scores.size == 0
        or ood_scores.size == 0
        or not np.isfinite(id_scores).all()
        or not np.isfinite(ood_scores).all()
    ):
        raise ValueError("score arrays must be non-empty and finite")
    threshold = float(np.quantile(id_scores, 0.05, method="linear"))
    return {
        "auroc": float(
            pair_outcome_summary(id_scores, ood_scores)["auroc_id_positive"]
        ),
        "fpr95_id_tpr": float(np.mean(ood_scores >= threshold)),
        "fpr95_threshold": threshold,
        "fpr95_achieved_id_tpr": float(np.mean(id_scores >= threshold)),
    }


def _pair_rows(
    pair_roots: Sequence[Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    absolute: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    localization: list[dict[str, Any]] = []
    id_utility: list[dict[str, Any]] = []
    numerical: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    run_ids: set[str] = set()
    sample_hashes: dict[str, set[str]] = defaultdict(set)
    maximum_identity_residual = 0.0

    for root in sorted(pair_roots, key=str):
        record, arrays = _verify_pair_artifact(root)
        cell = str(record["cell_id"])
        seed = int(record["training_seed"])
        if cell not in CELLS or (cell, seed) in identities:
            raise ValueError("duplicate or unexpected pair identity")
        identities.add((cell, seed))
        context = CELL_CONTEXT[cell]
        run_ids.update(record["source_run_ids"].values())
        if (
            record["direction"] != "coupled_minus_decoupled"
            or record["checkpoint_role"] != "last"
            or int(record["checkpoint_epoch"]) != 200
            or record["depth_tap"] != "penultimate"
        ):
            raise ValueError("pair endpoint contract mismatch")
        for split, digest in record["sample_order_sha256"].items():
            sample_hashes[split].add(str(digest))

        for role in ("D", "C"):
            utility = record["id_utility"]["coupled" if role == "C" else "decoupled"]
            id_utility.append(
                {
                    "cell": cell,
                    "context": context,
                    "seed": seed,
                    "role": role,
                    "accuracy": float(utility["accuracy"]),
                    "nll": float(utility["nll"]),
                    "ece": float(utility["ece"]),
                    "split": "id_test",
                    "count": EXPECTED_COUNTS["id_test"],
                }
            )
            fit = record["fit_diagnostics"]["coupled" if role == "C" else "decoupled"]
            for transform in ("raw", "l2"):
                diagnostics = fit[transform]
                spectrum = diagnostics["numerical"]["global_spectrum"]
                condition = spectrum["condition_number"]
                numerical.append(
                    {
                        "cell": cell,
                        "context": context,
                        "seed": seed,
                        "role": role,
                        "transform": transform,
                        "applicable": bool(diagnostics["applicable"]),
                        "ridge": float(diagnostics["ridge"]),
                        "dimension": int(spectrum["dimension"]),
                        "rank": int(spectrum["rank"]),
                        "condition_number": None
                        if condition == "infinity"
                        else float(condition),
                        "condition_is_infinite": condition == "infinity",
                        "finite_score_arrays": True,
                    }
                )

        for dataset in DATASETS:
            for readout, transform, detector, record_target in READOUTS:
                role_metrics: dict[str, dict[str, float]] = {}
                for role in ("D", "C"):
                    id_scores = arrays[_score_key(role, transform, "id_test", detector)]
                    ood_scores = arrays[_score_key(role, transform, dataset, detector)]
                    if id_scores.shape != (EXPECTED_COUNTS["id_test"],) or ood_scores.shape != (
                        EXPECTED_COUNTS[dataset],
                    ):
                        raise ValueError("score sample-count mismatch")
                    metrics = _compute_primary_ood_metrics(id_scores, ood_scores)
                    role_metrics[role] = {
                        "auroc": float(metrics["auroc"]),
                        "fpr95": float(metrics["fpr95_id_tpr"]),
                    }
                    stored = record["datasets"][dataset][record_target]
                    stored_prefix = "coupled" if role == "C" else "decoupled"
                    _float_close(metrics["auroc"], stored[f"{stored_prefix}_auroc"])
                    _float_close(
                        metrics["fpr95_id_tpr"], stored[f"{stored_prefix}_fpr95"]
                    )
                    absolute.append(
                        {
                            "cell": cell,
                            "context": context,
                            "dataset": dataset,
                            "region": DATASET_REGIONS[dataset],
                            "seed": seed,
                            "role": role,
                            "readout": readout,
                            "auroc": float(metrics["auroc"]),
                            "fpr95": float(metrics["fpr95_id_tpr"]),
                            "score_direction": "higher_is_more_id_like",
                            "fpr95_convention": "linear_id_5th_percentile_ood_acceptance",
                        }
                    )

                transition = pair_transition_summary(
                    arrays[_score_key("D", transform, "id_test", detector)],
                    arrays[_score_key("D", transform, dataset, detector)],
                    arrays[_score_key("C", transform, "id_test", detector)],
                    arrays[_score_key("C", transform, dataset, detector)],
                )
                gain, loss, churn = _transition_rates(transition)
                delta_auroc = role_metrics["C"]["auroc"] - role_metrics["D"]["auroc"]
                delta_fpr95 = role_metrics["C"]["fpr95"] - role_metrics["D"]["fpr95"]
                maximum_identity_residual = max(
                    maximum_identity_residual,
                    abs(delta_auroc - (gain - loss)),
                    abs(churn - (gain + loss)),
                )
                stored = record["datasets"][dataset][record_target]
                for name, value in (
                    ("delta_auroc", delta_auroc),
                    ("gain", gain),
                    ("loss", loss),
                    ("pair_order_churn", churn),
                ):
                    _float_close(value, stored[name])
                paired.append(
                    {
                        "cell": cell,
                        "context": context,
                        "dataset": dataset,
                        "region": DATASET_REGIONS[dataset],
                        "seed": seed,
                        "readout": readout,
                        "delta_auroc": delta_auroc,
                        "delta_fpr95": delta_fpr95,
                        "gain": gain,
                        "loss": loss,
                        "churn": churn,
                        "pair_count": int(transition["pair_count"]),
                        "gain_loss_residual": delta_auroc - (gain - loss),
                        "churn_residual": churn - (gain + loss),
                    }
                )

            attribution = paired_component_attribution(
                {
                    split_name: {
                        detector: arrays[_score_key("D", "raw", split, detector)]
                        for detector in ("md", "rmd", "marginal")
                    }
                    for split_name, split in (("id", "id_test"), ("ood", dataset))
                },
                {
                    split_name: {
                        detector: arrays[_score_key("C", "raw", split, detector)]
                        for detector in ("md", "rmd", "marginal")
                    }
                    for split_name, split in (("id", "id_test"), ("ood", dataset))
                },
            )
            if not attribution["pass"]:
                raise ValueError("Raw-MD Shapley reconstruction failed")
            phi = attribution["component_auroc_attribution"]
            _float_close(attribution["auroc_delta"], record["datasets"][dataset]["md"]["delta_auroc"])
            _float_close(phi["rmd"], record["datasets"][dataset]["md"]["component_attribution"]["component_auroc_attribution"]["rmd"])
            _float_close(phi["marginal"], record["datasets"][dataset]["md"]["component_attribution"]["component_auroc_attribution"]["marginal"])
            total = float(attribution["auroc_delta"])
            localization.append(
                {
                    "cell": cell,
                    "context": context,
                    "dataset": dataset,
                    "region": DATASET_REGIONS[dataset],
                    "seed": seed,
                    "delta_raw_md": total,
                    "phi_rmd": float(phi["rmd"]),
                    "phi_marginal": float(phi["marginal"]),
                    "reconstruction_residual": float(phi["reconstruction_residual"]),
                    "marginal_absolute_share": None
                    if total == 0.0
                    else abs(float(phi["marginal"])) / abs(total),
                    "marginal_signed_adverse_share": None
                    if total == 0.0
                    else math.copysign(
                        abs(float(phi["marginal"])) / abs(total),
                        float(phi["marginal"]) * total,
                    ),
                    "interpretation_boundary": "exact score accounting; not causal mediation",
                }
            )

    expected_identities = {(cell, seed) for cell in CELLS for seed in range(5)}
    if identities != expected_identities or len(run_ids) != 20:
        raise ValueError("pair/run coverage mismatch")
    if set(sample_hashes) != set(EXPECTED_COUNTS) or any(
        len(values) != 1 for values in sample_hashes.values()
    ):
        raise ValueError("C/D or cross-pair sample ordering mismatch")
    if maximum_identity_residual > 1.0e-12:
        raise ValueError("pair identity residual exceeds 1e-12")
    return (
        absolute,
        paired,
        localization,
        id_utility,
        numerical,
        {
            "pair_count": len(identities),
            "run_count": len(run_ids),
            "score_arrays_per_pair": 84,
            "maximum_pair_identity_residual": maximum_identity_residual,
            "sample_order_sha256": {
                split: next(iter(values)) for split, values in sorted(sample_hashes.items())
            },
        },
    )


def _audit_training_bundle(training_root: Path, production_plan: Mapping[str, Any]) -> dict[str, Any]:
    import yaml

    terminal_path = training_root / "collected" / "terminal_training_validation.json"
    run_dir = training_root / "runs"
    if sha256_file(terminal_path) != TRAINING_TERMINAL_SHA256:
        raise ValueError("training terminal SHA mismatch")
    terminal = read_json(terminal_path)
    if (
        terminal.get("status") != "PASS"
        or terminal.get("execution_sha") != production_plan["source_training_sha"]
        or int(terminal.get("run_count", -1)) != 20
        or int(terminal.get("seed_count", -1)) != 5
    ):
        raise ValueError("training terminal protocol mismatch")
    terminal_rows = {row["run_id"]: row for row in terminal["runs"]}
    if len(terminal_rows) != 20:
        raise ValueError("training terminal has duplicate run ids")

    config_paths = sorted(run_dir.glob("*.yaml"))
    if len(config_paths) != 20:
        raise ValueError("training config coverage is not 20/20")
    config_rows: list[dict[str, Any]] = []
    for path in config_paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        replica = data.get("resnet18_replication", {})
        run_id = str(replica.get("run_id"))
        parsed = _parse_run_id(run_id)
        optimizer = data.get("optimizer", {})
        model = data.get("model", {})
        dataset = data.get("dataset", {})
        training = data.get("training", {})
        if (
            path.stem != run_id
            or model != {"name": "resnet18", "variant": "cifar", "num_classes": 10}
            or dataset.get("protocol") != "oge_cifar10_holdout_v1"
            or dataset.get("train_split") != "id_train"
            or dataset.get("validation_split") != "id_validation"
            or dataset.get("test_split") != "id_test"
            or float(optimizer.get("lr", -1.0)) != parsed["learning_rate"]
            or float(optimizer.get("weight_decay", -1.0)) != 1.0e-4
            or replica.get("branch_policy") != parsed["branch_policy"]
            or optimizer.get("name") != ("adam" if parsed["role"] == "C" else "adamw")
            or int(training.get("seed", -1)) != parsed["seed"]
            or int(training.get("max_epochs", -1)) != 200
            or training.get("precision") != "fp32"
            or training.get("deterministic") is not True
            or replica.get("study_id") != "resnet18_cifar10_replication_v3"
            or replica.get("artifact_namespace") != "resnet18_cifar10_replication_v3"
        ):
            raise ValueError(f"training config contract mismatch: {path}")
        terminal_row = terminal_rows.get(run_id)
        if terminal_row is None or terminal_row["branch_policy"] != parsed["branch_policy"]:
            raise ValueError(f"training terminal/config mismatch: {run_id}")
        config_rows.append(
            {
                **parsed,
                "model": "resnet18",
                "model_variant": "cifar",
                "dataset": "cifar10",
                "optimizer": optimizer["name"],
                "checkpoint_role": "last",
                "checkpoint_epoch": 200,
                "config_path": str(path.resolve()),
                "config_sha256": sha256_file(path),
                "initialization_sha256": terminal_row["initialization_sha256"],
                "data_stream_id": terminal_row["data_stream_id"],
                "sibling_group_id": terminal_row["sibling_group_id"],
                "cross_lr_pairing_block_id": terminal_row["cross_lr_pairing_block_id"],
            }
        )

    by_pair: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in config_rows:
        by_pair[(row["cell"], row["seed"])].append(row)
        by_seed[row["seed"]].append(row)
    for key, rows in by_pair.items():
        if len(rows) != 2 or {row["role"] for row in rows} != {"C", "D"}:
            raise ValueError(f"incomplete training sibling pair: {key}")
        for field in ("initialization_sha256", "data_stream_id", "sibling_group_id"):
            if len({row[field] for row in rows}) != 1:
                raise ValueError(f"same-LR sibling identity mismatch: {key} / {field}")
    for seed, rows in by_seed.items():
        if len(rows) != 4 or len({row["cross_lr_pairing_block_id"] for row in rows}) != 1:
            raise ValueError(f"cross-LR pairing-block mismatch: seed {seed}")
    planned_ids = {row["run_id"] for row in production_plan["records"]}
    if planned_ids != set(terminal_rows) or planned_ids != {row["run_id"] for row in config_rows}:
        raise ValueError("training/production run coverage mismatch")
    return {
        "status": "PASS",
        "run_count": 20,
        "seed_count": 5,
        "architecture": "resnet18",
        "architecture_variant": "cifar",
        "dataset": "cifar10",
        "roles": ["adam_coupled", "adamw_decoupled"],
        "learning_rates": [1.0e-3, 3.0e-4],
        "weight_decay": 1.0e-4,
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "feature_tap": "penultimate",
        "terminal_path": str(terminal_path.resolve()),
        "terminal_sha256": sha256_file(terminal_path),
        "config_rows": config_rows,
    }


def _validate_production_plan(evaluation_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = evaluation_root / "production_plan.json"
    terminal_path = evaluation_root / "collected" / "resnet18_terminal.json"
    plan = read_json(plan_path)
    terminal = read_json(terminal_path)
    if (
        plan.get("plan_sha256") != PAIR_PLAN_SHA256
        or canonical_sha256({key: value for key, value in plan.items() if key != "plan_sha256"})
        != PAIR_PLAN_SHA256
        or plan.get("study_id") != "resnet18_cifar10_replication_v3"
        or plan.get("checkpoint_role") != "last"
        or int(plan.get("checkpoint_epoch", -1)) != 200
        or plan.get("depth_tap") != "penultimate"
        or int(plan.get("feature_dim", -1)) != 512
        or plan.get("id_fit_split") != "id_train"
        or plan.get("id_validation_split") != "id_validation"
        or plan.get("protected_splits")
        != ["id_test", "cifar100", "tin", "mnist", "svhn", "texture", "places365"]
        or len(plan.get("records", [])) != 20
    ):
        raise ValueError("production plan contract mismatch")
    if (
        terminal.get("status") != "PASS"
        or terminal.get("terminal_sha256") != RESNET_TERMINAL_SHA256
        or terminal.get("scientific_verdict") != "PARTIAL"
        or int(terminal.get("pair_record_count", -1)) != 10
        or terminal.get("plan_sha256") != PAIR_PLAN_SHA256
    ):
        raise ValueError("ResNet endpoint terminal contract mismatch")
    return plan, {
        "production_plan_path": str(plan_path.resolve()),
        "production_plan_file_sha256": sha256_file(plan_path),
        "production_plan_identity": plan["plan_sha256"],
        "terminal_path": str(terminal_path.resolve()),
        "terminal_file_sha256": sha256_file(terminal_path),
        "terminal_identity": terminal["terminal_sha256"],
        "technical_status": terminal["status"],
        "scientific_verdict": terminal["scientific_verdict"],
    }


def _macro_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_fields: Sequence[str],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        datasets = {row["dataset"] for row in selected}
        region = str(selected[0]["region"])
        if datasets != {dataset for dataset in DATASETS if DATASET_REGIONS[dataset] == region}:
            raise ValueError(f"seed-first region macro coverage mismatch: {key}")
        output.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                **{
                    field: statistics.fmean(float(row[field]) for row in selected)
                    for field in value_fields
                },
                "dataset_count": len(datasets),
            }
        )
    return output


def _build_recovery_rows(
    absolute: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    absolute_index = {
        (row["cell"], row["dataset"], row["seed"], row["role"], row["readout"]): row
        for row in absolute
    }
    paired_index = {
        (row["cell"], row["dataset"], row["seed"], row["readout"]): row
        for row in paired
    }
    recovery: list[dict[str, Any]] = []
    attenuation: list[dict[str, Any]] = []
    for cell in CELLS:
        for dataset in DATASETS:
            for seed in range(5):
                for role in ("D", "C"):
                    raw = absolute_index[(cell, dataset, seed, role, "Raw MD")]
                    for readout in ("RMD", "L2-MD"):
                        alternative = absolute_index[(cell, dataset, seed, role, readout)]
                        recovery.append(
                            {
                                "cell": cell,
                                "context": CELL_CONTEXT[cell],
                                "dataset": dataset,
                                "region": DATASET_REGIONS[dataset],
                                "seed": seed,
                                "role": role,
                                "readout": readout,
                                "raw_auroc": raw["auroc"],
                                "alternative_auroc": alternative["auroc"],
                                "auroc_recovery": alternative["auroc"] - raw["auroc"],
                                "raw_fpr95": raw["fpr95"],
                                "alternative_fpr95": alternative["fpr95"],
                                "fpr95_recovery": raw["fpr95"] - alternative["fpr95"],
                            }
                        )
                raw_pair = paired_index[(cell, dataset, seed, "Raw MD")]
                for readout in ("RMD", "L2-MD"):
                    alternative_pair = paired_index[(cell, dataset, seed, readout)]
                    attenuation.append(
                        {
                            "cell": cell,
                            "context": CELL_CONTEXT[cell],
                            "dataset": dataset,
                            "region": DATASET_REGIONS[dataset],
                            "seed": seed,
                            "readout": readout,
                            "delta_raw_md": raw_pair["delta_auroc"],
                            "delta_alternative": alternative_pair["delta_auroc"],
                            "contrast_attenuation": abs(raw_pair["delta_auroc"])
                            - abs(alternative_pair["delta_auroc"]),
                        }
                    )
    return recovery, attenuation


def _combine_geometry(
    paths: Sequence[str | Path], expected_run_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payloads = [verify_host_geometry(path) for path in paths]
    if {payload["host"] for payload in payloads} != {
        "curie",
        "lise",
        "precision_medicine",
    }:
        raise ValueError("geometry host coverage mismatch")
    rows = [row for payload in payloads for row in payload["rows"]]
    if len(rows) != 20 or {row["run_id"] for row in rows} != expected_run_ids:
        raise ValueError("geometry run coverage mismatch")
    if any(
        row["geometry_fit_split"] != "id_train"
        or int(row["geometry_fit_count"]) != 45000
        or row["held_out_geometry_split"] != "id_validation"
        or int(row["held_out_geometry_count"]) != 5000
        for row in rows
    ):
        raise ValueError("geometry split roles drifted")
    index = {(row["cell"], int(row["seed"]), row["role"]): row for row in rows}
    effects: list[dict[str, Any]] = []
    for cell in CELLS:
        for seed in range(5):
            coupled = index[(cell, seed, "C")]
            decoupled = index[(cell, seed, "D")]
            for metric in GEOMETRY_METRICS:
                left = float(coupled[metric])
                right = float(decoupled[metric])
                if metric in {"feature_norm", "within_trace", "nc0"}:
                    if left <= 0.0 or right <= 0.0:
                        raise ValueError(f"log-ratio geometry metric is non-positive: {metric}")
                    effect = math.log(left / right)
                    effect_scale = "log(C/D)"
                else:
                    effect = left - right
                    effect_scale = "C-D"
                effects.append(
                    {
                        "cell": cell,
                        "context": CELL_CONTEXT[cell],
                        "seed": seed,
                        "metric": metric,
                        "effect": effect,
                        "effect_scale": effect_scale,
                    }
                )
            effects.append(
                {
                    "cell": cell,
                    "context": CELL_CONTEXT[cell],
                    "seed": seed,
                    "metric": "top10_trace_share",
                    "effect": float(coupled["top10_trace_share"])
                    - float(decoupled["top10_trace_share"]),
                    "effect_scale": "C-D",
                }
            )
    return rows, effects, {
        "status": "PASS",
        "host_count": 3,
        "run_count": len(rows),
        "source_files": [
            {
                "path": str(Path(path).resolve()),
                "sha256": sha256_file(path),
                "output_identity_sha256": payload["output_identity_sha256"],
            }
            for path, payload in zip(paths, payloads, strict=True)
        ],
    }


def _verify_wrn_pack(wrn_root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    manifest_path = wrn_root / "manifest.json"
    if sha256_file(manifest_path) != WRN_MANIFEST_SHA256:
        raise ValueError("WRN pack is not the frozen clean-main manifest")
    manifest = read_json(manifest_path)
    for output in manifest["outputs"]:
        path = wrn_root / output["path"]
        if sha256_file(path) != output["sha256"] or path.stat().st_size != output["bytes"]:
            raise ValueError(f"WRN pack output mismatch: {path}")
    required = (
        "context_heatmap_seed_rows.csv",
        "churn_seed_rows.csv",
        "recovery_absolute_seed_rows.csv",
    )
    tables = {name: _read_csv(wrn_root / "tables" / name) for name in required}
    merged_meta = manifest["inputs"]["merged_analysis"]
    merged_path = Path(merged_meta["path"])
    if not merged_path.is_file() or sha256_file(merged_path) != merged_meta["sha256"]:
        raise ValueError("WRN merged-analysis dependency is unavailable or changed")
    tables["score_localization_rows"] = read_json(merged_path)["score_localization_rows"]
    return manifest, tables


def _architecture_boundary_rows(
    *,
    resnet_paired_macro: Sequence[Mapping[str, Any]],
    resnet_localization_macro: Sequence[Mapping[str, Any]],
    wrn_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wrn_manifest, tables = _verify_wrn_pack(wrn_root)
    output: list[dict[str, Any]] = []

    resnet_pair_index = {
        (row["context"], row["seed"], row["region"], row["readout"]): row
        for row in resnet_paired_macro
    }
    resnet_loc_index = {
        (row["context"], row["seed"], row["region"]): row
        for row in resnet_localization_macro
    }
    for context in ("large", "small"):
        for seed in range(5):
            for region in REGIONS:
                raw = resnet_pair_index[(context, seed, region, "Raw MD")]
                rmd = resnet_pair_index[(context, seed, region, "RMD")]
                loc = resnet_loc_index[(context, seed, region)]
                output.append(
                    {
                        "architecture": "ResNet-18",
                        "context": context,
                        "seed": seed,
                        "region": region,
                        "raw_delta": raw["delta_auroc"],
                        "churn": raw["churn"],
                        "rmd_attenuation": abs(raw["delta_auroc"])
                        - abs(rmd["delta_auroc"]),
                        "marginal_signed_adverse_share": loc[
                            "marginal_signed_adverse_share"
                        ],
                    }
                )

    heatmap = tables["context_heatmap_seed_rows.csv"]
    churn = tables["churn_seed_rows.csv"]
    recovery = tables["recovery_absolute_seed_rows.csv"]
    localization = tables["score_localization_rows"]
    for context, cell in WRN_CELLS.items():
        seeds = sorted(
            {
                int(row["seed"])
                for row in heatmap
                if row["cell"] == cell
            }
        )
        for seed in seeds:
            for region in REGIONS:
                region_datasets = {
                    dataset for dataset in DATASETS if DATASET_REGIONS[dataset] == region
                }
                raw_rows = [
                    row
                    for row in heatmap
                    if row["cell"] == cell
                    and int(row["seed"]) == seed
                    and row["dataset"] in region_datasets
                ]
                churn_rows = [
                    row
                    for row in churn
                    if row["cell"] == cell
                    and int(row["seed"]) == seed
                    and row["dataset"] in region_datasets
                ]
                recovery_rows = [
                    row
                    for row in recovery
                    if row["cell"] == cell
                    and int(row["seed"]) == seed
                    and row["dataset"] in region_datasets
                    and row["readout"] == "RMD"
                ]
                localization_rows = [
                    row
                    for row in localization
                    if row["cell"] == cell
                    and row["contrast"] == "C-D"
                    and row["transform"] == "raw"
                    and row["scope"] == "endpoint"
                    and int(row["checkpoint_epoch"]) == 200
                    and row["checkpoint_role"] == "last"
                    and row["depth_tap"] == "penultimate"
                    and int(row["seed"]) == seed
                    and row["dataset"] in region_datasets
                ]
                if not (
                    len(raw_rows)
                    == len(churn_rows)
                    == len(localization_rows)
                    == len(region_datasets)
                    and len(recovery_rows) == 2 * len(region_datasets)
                ):
                    raise ValueError("WRN architecture-boundary coverage mismatch")
                raw_delta = statistics.fmean(float(row["delta_auroc"]) for row in raw_rows)
                raw_churn = statistics.fmean(float(row["churn"]) for row in churn_rows)
                by_role = {
                    role: statistics.fmean(
                        float(row["auroc"]) for row in recovery_rows if row["role"] == role
                    )
                    for role in ("C", "D")
                }
                rmd_delta = by_role["C"] - by_role["D"]
                phi_marginal = statistics.fmean(
                    float(row["rmd_marginal_replacement"]["phi_marginal"])
                    for row in localization_rows
                )
                total = statistics.fmean(float(row["delta_auroc"]) for row in localization_rows)
                signed_share = None if total == 0.0 else math.copysign(
                    abs(phi_marginal) / abs(total), phi_marginal * total
                )
                output.append(
                    {
                        "architecture": "WRN-28-10",
                        "context": context,
                        "seed": seed,
                        "region": region,
                        "raw_delta": raw_delta,
                        "churn": raw_churn,
                        "rmd_attenuation": abs(raw_delta) - abs(rmd_delta),
                        "marginal_signed_adverse_share": signed_share,
                    }
                )
    return output, {
        "manifest_path": str((wrn_root / "manifest.json").resolve()),
        "manifest_sha256": WRN_MANIFEST_SHA256,
        "merged_analysis_sha256": wrn_manifest["inputs"]["merged_analysis"]["sha256"],
    }


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
            "svg.hashsalt": "resnet18-seed-first-partial-boundary-v1",
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
    paths: list[Path] = []
    for extension in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 300
        elif extension == "pdf":
            kwargs["metadata"] = {
                "CreationDate": None,
                "ModDate": None,
                "Creator": "Matplotlib",
            }
        else:
            kwargs["metadata"] = {"Date": None, "Creator": "Matplotlib"}
        figure.savefig(path, **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def _mean_sd(axis: Any, x: float, values: Sequence[float], *, color: str, marker: str) -> None:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    axis.errorbar(
        [x],
        [mean],
        yerr=[[sd], [sd]],
        fmt=marker,
        markersize=4.5,
        color=color,
        markerfacecolor="white",
        markeredgewidth=1.0,
        capsize=2.3,
        linewidth=1.0,
        zorder=5,
    )


def _paired_ci(axis: Any, x: float, values: Sequence[float], *, color: str, marker: str = "D") -> None:
    summary = _ordinary_summary(values, paired=True)
    low = float(summary["mean"] - summary["t90_low"])
    high = float(summary["t90_high"] - summary["mean"])
    axis.errorbar(
        [x],
        [summary["mean"]],
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


def _r1_rows(
    absolute: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    absolute_index = {
        (row["cell"], row["dataset"], row["seed"], row["role"]): row
        for row in absolute
        if row["readout"] == "Raw MD"
    }
    return [
        {
            "cell": row["cell"],
            "context": row["context"],
            "dataset": row["dataset"],
            "region": row["region"],
            "seed": row["seed"],
            "decoupled_auroc": absolute_index[
                (row["cell"], row["dataset"], row["seed"], "D")
            ]["auroc"],
            "coupled_auroc": absolute_index[
                (row["cell"], row["dataset"], row["seed"], "C")
            ]["auroc"],
            "delta_auroc": row["delta_auroc"],
        }
        for row in paired
        if row["readout"] == "Raw MD"
    ]


def figure_r1(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = _style()
    figure, axes = plt.subplots(
        2, 3, figsize=(7.2, 4.7), sharey=True, constrained_layout=True
    )
    role_colors = {"D": colors["blue"], "C": colors["orange"]}
    x_positions = {"large": {"D": 0.0, "C": 1.0}, "small": {"D": 3.0, "C": 4.0}}
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        selected = [row for row in rows if row["dataset"] == dataset]
        for context in ("large", "small"):
            context_rows = [row for row in selected if row["context"] == context]
            for row in context_rows:
                axis.plot(
                    [x_positions[context]["D"], x_positions[context]["C"]],
                    [row["decoupled_auroc"], row["coupled_auroc"]],
                    color=colors["gray"],
                    alpha=0.38,
                    linewidth=0.65,
                    zorder=1,
                )
            for role, field in (("D", "decoupled_auroc"), ("C", "coupled_auroc")):
                values = [float(row[field]) for row in context_rows]
                x = x_positions[context][role]
                jitter = np.linspace(-0.04, 0.04, len(values))
                axis.scatter(
                    x + jitter,
                    values,
                    s=11,
                    color=role_colors[role],
                    alpha=0.58,
                    linewidths=0,
                    zorder=3,
                )
                _mean_sd(
                    axis,
                    x,
                    values,
                    color=role_colors[role],
                    marker="o" if role == "D" else "s",
                )
            delta = _ordinary_summary(
                [float(row["delta_auroc"]) for row in context_rows], paired=True
            )
            axis.text(
                statistics.fmean(x_positions[context].values()),
                0.025 if context == "large" else 0.105,
                f"Delta={delta['mean']:+.3f}\n90% CI [{delta['t90_low']:+.3f}, {delta['t90_high']:+.3f}]",
                ha="center",
                va="bottom",
                fontsize=5.8,
                transform=axis.get_xaxis_transform(),
            )
        axis.set_title(f"{chr(65 + panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.set_xlim(-0.35, 4.35)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([0.5, 3.5])
        axis.set_xticklabels(["Large LR", "Small LR"])
        axis.grid(axis="y")
        if panel % 3 == 0:
            axis.set_ylabel("Raw MD AUROC")
    handles = [
        plt.Line2D([], [], color=role_colors["D"], marker="o", markerfacecolor="white", label="D: AdamW"),
        plt.Line2D([], [], color=role_colors["C"], marker="s", markerfacecolor="white", label="C: Adam"),
    ]
    figure.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    return _save_figure(figure, output_dir, "R1_paired_raw_md")


def figure_r2(
    absolute: Sequence[Mapping[str, Any]], *, context: str, output_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = _style()
    figure, axes = plt.subplots(
        2, 3, figsize=(7.2, 4.55), sharex=True, sharey=True, constrained_layout=True
    )
    role_colors = {"D": colors["blue"], "C": colors["orange"]}
    readout_order = [item[0] for item in READOUTS]
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        selected = [
            row for row in absolute if row["context"] == context and row["dataset"] == dataset
        ]
        for role, offset in (("D", -0.09), ("C", 0.09)):
            role_rows = [row for row in selected if row["role"] == role]
            for seed in range(5):
                seed_rows = {
                    row["readout"]: row for row in role_rows if int(row["seed"]) == seed
                }
                axis.plot(
                    [index + offset for index in range(3)],
                    [seed_rows[readout]["auroc"] for readout in readout_order],
                    color=role_colors[role],
                    linewidth=0.55,
                    alpha=0.24,
                    zorder=1,
                )
            for index, readout in enumerate(readout_order):
                values = [float(row["auroc"]) for row in role_rows if row["readout"] == readout]
                jitter = np.linspace(-0.025, 0.025, len(values))
                axis.scatter(
                    index + offset + jitter,
                    values,
                    s=10,
                    color=role_colors[role],
                    alpha=0.57,
                    linewidths=0,
                )
                _mean_sd(
                    axis,
                    index + offset,
                    values,
                    color=role_colors[role],
                    marker="o" if role == "D" else "s",
                )
        axis.set_title(f"{chr(65 + panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.set_xticks(range(3))
        axis.set_xticklabels(readout_order)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y")
        if panel % 3 == 0:
            axis.set_ylabel("AUROC")
    figure.legend(
        handles=[
            plt.Line2D([], [], color=role_colors["D"], marker="o", markerfacecolor="white", label="D"),
            plt.Line2D([], [], color=role_colors["C"], marker="s", markerfacecolor="white", label="C"),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    suffix = "A_large" if context == "large" else "B_small"
    return _save_figure(figure, output_dir, f"R2{suffix}_absolute_auroc")


def figure_r3(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = _style()
    figure, axes = plt.subplots(
        2, 3, figsize=(7.2, 4.65), sharey=True, constrained_layout=True
    )
    context_colors = {"large": colors["purple"], "small": colors["sky"]}
    labels = (("gain", 1.0, "Gain"), ("loss", -1.0, "-Loss"), ("delta_auroc", 1.0, "Net Delta"))
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        selected = [row for row in rows if row["dataset"] == dataset and row["readout"] == "Raw MD"]
        for context, offset in (("large", -0.12), ("small", 0.12)):
            context_rows = [row for row in selected if row["context"] == context]
            for index, (field, sign, _) in enumerate(labels):
                values = [sign * float(row[field]) for row in context_rows]
                jitter = np.linspace(-0.025, 0.025, len(values))
                axis.scatter(
                    index + offset + jitter,
                    values,
                    s=10,
                    color=context_colors[context],
                    alpha=0.55,
                    linewidths=0,
                )
                _paired_ci(
                    axis,
                    index + offset,
                    values,
                    color=context_colors[context],
                    marker="D" if field == "delta_auroc" else "o",
                )
            churn = statistics.fmean(float(row["churn"]) for row in context_rows)
            axis.text(
                0.03 if context == "large" else 0.97,
                0.04 if context == "large" else 0.12,
                f"{context[0].upper()} churn={churn:.3f}",
                transform=axis.transAxes,
                ha="left" if context == "large" else "right",
                fontsize=6.2,
                color=context_colors[context],
            )
        axis.axhline(0.0, color=colors["ink"], linewidth=0.7)
        axis.set_title(f"{chr(65 + panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.set_xticks(range(3))
        axis.set_xticklabels([item[2] for item in labels])
        axis.grid(axis="y")
        if panel % 3 == 0:
            axis.set_ylabel("ID-OOD pair fraction")
    figure.legend(
        handles=[
            plt.Line2D([], [], color=context_colors["large"], marker="o", linestyle="", label="Large LR"),
            plt.Line2D([], [], color=context_colors["small"], marker="s", linestyle="", label="Small LR"),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    return _save_figure(figure, output_dir, "R3_gain_loss_churn")


def figure_r4(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = _style()
    figure, axes = plt.subplots(
        2, 3, figsize=(7.2, 4.65), sharey=True, constrained_layout=True
    )
    context_colors = {"large": colors["purple"], "small": colors["sky"]}
    fields = (("phi_rmd", "phi RMD"), ("phi_marginal", "phi Marginal"), ("delta_raw_md", "Total Delta"))
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        selected = [row for row in rows if row["dataset"] == dataset]
        for context, offset in (("large", -0.12), ("small", 0.12)):
            context_rows = [row for row in selected if row["context"] == context]
            for index, (field, _) in enumerate(fields):
                values = [float(row[field]) for row in context_rows]
                jitter = np.linspace(-0.025, 0.025, len(values))
                axis.scatter(
                    index + offset + jitter,
                    values,
                    s=10,
                    color=context_colors[context],
                    alpha=0.55,
                    linewidths=0,
                )
                _paired_ci(axis, index + offset, values, color=context_colors[context])
        axis.axhline(0.0, color=colors["ink"], linewidth=0.7)
        axis.set_title(f"{chr(65 + panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.set_xticks(range(3))
        axis.set_xticklabels([label for _, label in fields], rotation=10)
        axis.grid(axis="y")
        if panel % 3 == 0:
            axis.set_ylabel("AUROC contribution")
    figure.legend(
        handles=[
            plt.Line2D([], [], color=context_colors["large"], marker="o", linestyle="", label="Large LR"),
            plt.Line2D([], [], color=context_colors["small"], marker="s", linestyle="", label="Small LR"),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    return _save_figure(figure, output_dir, "R4_score_localization")


def figure_r5(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = _style()
    figure, axes = plt.subplots(4, 2, figsize=(7.2, 7.7), constrained_layout=True)
    for panel, metric in enumerate(GEOMETRY_METRICS):
        axis = axes.flat[panel]
        selected = [row for row in rows if row["metric"] == metric]
        for index, context in enumerate(("large", "small")):
            values = [float(row["effect"]) for row in selected if row["context"] == context]
            jitter = np.linspace(-0.04, 0.04, len(values))
            axis.scatter(
                index + jitter,
                values,
                s=12,
                color=colors["sky"],
                edgecolor=colors["blue"],
                linewidth=0.35,
                alpha=0.7,
            )
            _paired_ci(axis, index, values, color=colors["orange"])
        axis.axhline(0.0, color=colors["ink"], linewidth=0.7)
        axis.set_title(f"{chr(65 + panel)}. {GEOMETRY_LABELS[metric]}", loc="left")
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Large LR", "Small LR"])
        axis.grid(axis="y")
    figure.suptitle(
        "Post-result, ID-only exploratory endpoint geometry (does not change PARTIAL)",
        fontsize=9,
    )
    return _save_figure(figure, output_dir, "R5_exploratory_endpoint_geometry")


def figure_r6(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = _style()
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 4.25), constrained_layout=True)
    metrics = (
        ("raw_delta", "Raw MD C-D"),
        ("churn", "Churn"),
        ("rmd_attenuation", "RMD attenuation"),
        ("marginal_signed_adverse_share", "Marginal signed share"),
    )
    region_colors = {"Near": colors["blue"], "Far": colors["orange"]}
    for row_index, architecture in enumerate(("WRN-28-10", "ResNet-18")):
        architecture_rows = [row for row in rows if row["architecture"] == architecture]
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column]
            for context_index, context in enumerate(("large", "small")):
                for region, offset, marker in (("Near", -0.10, "o"), ("Far", 0.10, "s")):
                    values = [
                        float(row[metric])
                        for row in architecture_rows
                        if row["context"] == context
                        and row["region"] == region
                        and row.get(metric) is not None
                    ]
                    jitter = np.linspace(-0.025, 0.025, len(values))
                    axis.scatter(
                        context_index + offset + jitter,
                        values,
                        s=9,
                        color=region_colors[region],
                        alpha=0.5,
                        linewidths=0,
                    )
                    _paired_ci(
                        axis,
                        context_index + offset,
                        values,
                        color=region_colors[region],
                        marker=marker,
                    )
            axis.axhline(0.0, color=colors["ink"], linewidth=0.7)
            axis.set_xticks([0, 1])
            axis.set_xticklabels(["Large", "Small"])
            axis.grid(axis="y")
            if row_index == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(architecture)
    figure.legend(
        handles=[
            plt.Line2D([], [], color=region_colors["Near"], marker="o", linestyle="", label="Near"),
            plt.Line2D([], [], color=region_colors["Far"], marker="s", linestyle="", label="Far"),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    return _save_figure(figure, output_dir, "R6_architecture_boundary")


def figure_s1(absolute: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = _style()
    figure, axes = plt.subplots(
        2, 3, figsize=(7.2, 4.55), sharex=True, sharey=True, constrained_layout=True
    )
    context_styles = {
        "large": (colors["purple"], "-"),
        "small": (colors["sky"], "--"),
    }
    readout_order = [item[0] for item in READOUTS]
    for panel, dataset in enumerate(DATASETS):
        axis = axes.flat[panel]
        selected = [row for row in absolute if row["dataset"] == dataset]
        for context, context_offset in (("large", -0.14), ("small", 0.14)):
            color, line_style = context_styles[context]
            for role, role_offset, marker in (("D", -0.035, "o"), ("C", 0.035, "s")):
                role_rows = [
                    row for row in selected if row["context"] == context and row["role"] == role
                ]
                for seed in range(5):
                    seed_rows = {
                        row["readout"]: row for row in role_rows if int(row["seed"]) == seed
                    }
                    axis.plot(
                        [index + context_offset + role_offset for index in range(3)],
                        [seed_rows[readout]["fpr95"] for readout in readout_order],
                        color=color,
                        linestyle=line_style,
                        linewidth=0.45,
                        alpha=0.18,
                    )
                for index, readout in enumerate(readout_order):
                    values = [
                        float(row["fpr95"]) for row in role_rows if row["readout"] == readout
                    ]
                    _mean_sd(
                        axis,
                        index + context_offset + role_offset,
                        values,
                        color=color,
                        marker=marker,
                    )
        axis.set_title(f"{chr(65 + panel)}. {DATASET_LABELS[dataset]}", loc="left")
        axis.set_xticks(range(3))
        axis.set_xticklabels(readout_order)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y")
        if panel % 3 == 0:
            axis.set_ylabel("FPR95 (lower is better)")
    return _save_figure(figure, output_dir, "S1_absolute_fpr95_recovery")


def figure_s2(
    geometry_effects: Sequence[Mapping[str, Any]],
    numerical: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    colors = _style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    top10 = [row for row in geometry_effects if row["metric"] == "top10_trace_share"]
    for index, context in enumerate(("large", "small")):
        values = [float(row["effect"]) for row in top10 if row["context"] == context]
        jitter = np.linspace(-0.04, 0.04, len(values))
        axes[0, 0].scatter(index + jitter, values, color=colors["sky"], s=12, alpha=0.65)
        _paired_ci(axes[0, 0], index, values, color=colors["orange"])
    axes[0, 0].axhline(0.0, color=colors["ink"], linewidth=0.7)
    axes[0, 0].set_title("A. Top-10 trace share C-D", loc="left")

    raw_rows = [row for row in numerical if row["transform"] == "raw"]
    for context_index, context in enumerate(("large", "small")):
        for role, offset, color, marker in (
            ("D", -0.10, colors["blue"], "o"),
            ("C", 0.10, colors["orange"], "s"),
        ):
            selected = [row for row in raw_rows if row["context"] == context and row["role"] == role]
            ranks = [float(row["rank"]) / float(row["dimension"]) for row in selected]
            axes[0, 1].scatter(context_index + offset + np.linspace(-0.02, 0.02, len(ranks)), ranks, color=color, s=10, alpha=0.55)
            _mean_sd(axes[0, 1], context_index + offset, ranks, color=color, marker=marker)
            finite_conditions = [
                math.log10(float(row["condition_number"]))
                for row in selected
                if row["condition_number"] is not None
            ]
            if finite_conditions:
                axes[1, 0].scatter(context_index + offset + np.linspace(-0.02, 0.02, len(finite_conditions)), finite_conditions, color=color, s=10, alpha=0.55)
                _mean_sd(axes[1, 0], context_index + offset, finite_conditions, color=color, marker=marker)
            infinite_count = sum(bool(row["condition_is_infinite"]) for row in selected)
            if infinite_count:
                axes[1, 0].text(
                    context_index + offset,
                    0.96,
                    f"inf x{infinite_count}",
                    transform=axes[1, 0].get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=5.8,
                    color=color,
                )
            applicability = [1.0 if row["applicable"] else 0.0 for row in selected]
            axes[1, 1].scatter(context_index + offset + np.linspace(-0.02, 0.02, len(applicability)), applicability, color=color, s=10, alpha=0.55)
            _mean_sd(axes[1, 1], context_index + offset, applicability, color=color, marker=marker)
    axes[0, 1].set_title("B. Raw total-cov. numerical rank / 512", loc="left")
    axes[1, 0].set_title("C. Raw finite log10(condition); infinity annotated", loc="left")
    axes[1, 1].set_title("D. Raw theorem applicability", loc="left")
    for axis in axes.flat:
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Large LR", "Small LR"])
        axis.grid(axis="y")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_ylim(-0.05, 1.05)
    figure.legend(
        handles=[
            plt.Line2D([], [], color=colors["blue"], marker="o", linestyle="", label="D"),
            plt.Line2D([], [], color=colors["orange"], marker="s", linestyle="", label="C"),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    return _save_figure(figure, output_dir, "S2_spectrum_numerical_audit")


def _summary_for_figure(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    metrics: Sequence[str],
    *,
    paired: bool,
) -> list[dict[str, Any]]:
    return summarize_rows(rows, group_fields, metrics, paired=paired)


def _join_r1_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    absolute = _summary_for_figure(
        rows,
        ("context", "dataset", "region"),
        ("decoupled_auroc", "coupled_auroc"),
        paired=False,
    )
    paired = _summary_for_figure(
        rows,
        ("context", "dataset", "region"),
        ("delta_auroc",),
        paired=True,
    )
    return absolute + paired


def _protocol_audit_rows(
    *,
    training_audit: Mapping[str, Any],
    production_audit: Mapping[str, Any],
    pair_audit: Mapping[str, Any],
    geometry_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values = {
        "architecture": training_audit["architecture"],
        "architecture_variant": training_audit["architecture_variant"],
        "dataset": training_audit["dataset"],
        "training_seed_count": training_audit["seed_count"],
        "roles": "adam_coupled,adamw_decoupled",
        "learning_rates": "1e-3,3e-4",
        "weight_decay": "1e-4",
        "checkpoint": "epoch-200 last",
        "feature_tap": "penultimate (512 dimensions)",
        "detector_fit_split": "id_train (45,000)",
        "held_out_geometry_control": "id_validation (5,000; excluded from fit)",
        "id_utility_split": "id_test (10,000)",
        "ood_datasets": ",".join(DATASETS),
        "score_direction": "higher is more ID-like",
        "fpr95_convention": "linear ID 5th percentile; OOD acceptance; lower is better",
        "pair_record_coverage": pair_audit["pair_count"],
        "score_arrays_per_pair": pair_audit["score_arrays_per_pair"],
        "geometry_run_coverage": geometry_audit["run_count"],
        "technical_status": production_audit["technical_status"],
        "scientific_verdict": production_audit["scientific_verdict"],
        "formation_analysis": "NOT_AVAILABLE_BY_DESIGN",
    }
    return [
        {"audit_item": key, "observed": str(value), "status": "PASS"}
        for key, value in values.items()
    ]


def _id_delta_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {(row["cell"], row["seed"], row["role"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for cell in CELLS:
        for seed in range(5):
            coupled = index[(cell, seed, "C")]
            decoupled = index[(cell, seed, "D")]
            output.append(
                {
                    "cell": cell,
                    "context": CELL_CONTEXT[cell],
                    "seed": seed,
                    "delta_accuracy": coupled["accuracy"] - decoupled["accuracy"],
                    "delta_nll": coupled["nll"] - decoupled["nll"],
                    "delta_ece": coupled["ece"] - decoupled["ece"],
                }
            )
    return output


def _reader_summary(
    *,
    paired_macro: Sequence[Mapping[str, Any]],
    localization_macro: Sequence[Mapping[str, Any]],
    id_delta: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pair_summary = summarize_rows(
        paired_macro,
        ("context", "region", "readout"),
        ("delta_auroc", "delta_fpr95", "gain", "loss", "churn"),
        paired=True,
    )
    localization_summary = summarize_rows(
        localization_macro,
        ("context", "region"),
        (
            "delta_raw_md",
            "phi_rmd",
            "phi_marginal",
            "marginal_signed_adverse_share",
        ),
        paired=True,
    )
    id_summary = summarize_rows(
        id_delta,
        ("context",),
        ("delta_accuracy", "delta_nll", "delta_ece"),
        paired=True,
    )

    def value(
        rows: Sequence[Mapping[str, Any]], context: str, region: str, statistic: str,
        *, readout: str | None = None,
    ) -> float:
        selected = [
            row
            for row in rows
            if row.get("context") == context
            and row.get("region") == region
            and row.get("statistic") == statistic
            and (readout is None or row.get("readout") == readout)
        ]
        if len(selected) != 1:
            raise ValueError("reader-summary lookup is not unique")
        return float(selected[0]["mean"])

    return {
        "technical_status": "PASS",
        "scientific_verdict": "PARTIAL",
        "rescue_experiment": False,
        "partial_reason": (
            "large-context Near abs(C-D RMD) exceeds abs(C-D Raw MD); "
            "the preregistered RMD-attenuation gate fails"
        ),
        "formation_analysis": "NOT_AVAILABLE_BY_DESIGN",
        "macro": {
            context: {
                region: {
                    "raw_delta_auroc": value(
                        pair_summary, context, region, "delta_auroc", readout="Raw MD"
                    ),
                    "raw_churn": value(
                        pair_summary, context, region, "churn", readout="Raw MD"
                    ),
                    "rmd_delta_auroc": value(
                        pair_summary, context, region, "delta_auroc", readout="RMD"
                    ),
                    "phi_rmd": value(localization_summary, context, region, "phi_rmd"),
                    "phi_marginal": value(
                        localization_summary, context, region, "phi_marginal"
                    ),
                }
                for region in REGIONS
            }
            for context in ("large", "small")
        },
        "id_delta_summary": id_summary,
        "claim_boundary": (
            "architecture replication evidence is PARTIAL; endpoint geometry is "
            "post-result ID-only exploratory and cannot change the verdict"
        ),
    }


def _write_reader_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    macro = summary["macro"]
    lines = [
        "# ResNet-18 Replication Analysis — PARTIAL Architecture Boundary",
        "",
        "> **Technical PASS / Scientific PARTIAL / rescue experiment 없음.**",
        "",
        "## 한 화면 요약",
        "",
        "| Context | Region | Raw C-D | Churn | RMD C-D | phi_RMD | phi_Marginal |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for context in ("large", "small"):
        for region in REGIONS:
            row = macro[context][region]
            lines.append(
                f"| {context} | {region} | {row['raw_delta_auroc']:+.4f} | "
                f"{row['raw_churn']:.4f} | {row['rmd_delta_auroc']:+.4f} | "
                f"{row['phi_rmd']:+.4f} | {row['phi_marginal']:+.4f} |"
            )
    lines += [
        "",
        "## 판정",
        "",
        summary["partial_reason"] + ".",
        "",
        "R1–R6와 S1–S2는 이 판정을 바꾸는 분석이 아니라 사람이 PARTIAL의 "
        "구조와 architecture boundary를 읽을 수 있게 펼친 reader pack이다.",
        "",
        "## 금지 주장",
        "",
        "- ResNet 결과를 alpha trajectory로 부르지 않는다.",
        "- endpoint geometry를 preregistered replication gate나 causal mediator로 쓰지 않는다.",
        "- WRN과 ResNet의 같은 seed 번호를 paired sample로 취급하지 않는다.",
        "- `applicable=false`를 finite Mahalanobis score의 부재로 해석하지 않는다.",
        "- ResNet epoch/depth formation은 `NOT_AVAILABLE_BY_DESIGN`이다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figure_source_record(
    *,
    root: Path,
    seed_path: Path,
    summary_path: Path,
    figure_paths: Sequence[Path],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(seed_path.read_bytes())
    digest.update(summary_path.read_bytes())
    return {
        "seed_csv": str(seed_path.relative_to(root)),
        "summary_csv": str(summary_path.relative_to(root)),
        "input_sha256": digest.hexdigest(),
        "output_sha256": {
            path.suffix.removeprefix("."): sha256_file(path) for path in figure_paths
        },
    }


def _build_pack_to(
    *,
    evaluation_root: Path,
    training_root: Path,
    geometry_paths: Sequence[str | Path],
    wrn_root: Path,
    output_root: Path,
    analysis_git_sha: str,
) -> dict[str, Any]:
    plan, production_audit = _validate_production_plan(evaluation_root)
    training_audit = _audit_training_bundle(training_root, plan)
    pair_roots = sorted(
        path.parent
        for path in (evaluation_root / "collected" / "pair_artifacts").rglob("record.json")
    )
    if len(pair_roots) != 10:
        raise ValueError("central pair artifact coverage is not 10/10")
    absolute, paired, localization, id_utility, numerical, pair_audit = _pair_rows(
        pair_roots
    )
    expected_run_ids = {row["run_id"] for row in plan["records"]}
    geometry, geometry_effects, geometry_audit = _combine_geometry(
        geometry_paths, expected_run_ids
    )

    recovery, attenuation = _build_recovery_rows(absolute, paired)
    id_delta = _id_delta_rows(id_utility)
    absolute_macro = _macro_rows(
        absolute,
        value_fields=("auroc", "fpr95"),
        group_fields=("cell", "context", "seed", "role", "readout", "region"),
    )
    paired_macro = _macro_rows(
        paired,
        value_fields=("delta_auroc", "delta_fpr95", "gain", "loss", "churn"),
        group_fields=("cell", "context", "seed", "readout", "region"),
    )
    localization_macro = _macro_rows(
        localization,
        value_fields=(
            "delta_raw_md",
            "phi_rmd",
            "phi_marginal",
            "reconstruction_residual",
            "marginal_absolute_share",
            "marginal_signed_adverse_share",
        ),
        group_fields=("cell", "context", "seed", "region"),
    )
    architecture_boundary, wrn_audit = _architecture_boundary_rows(
        resnet_paired_macro=paired_macro,
        resnet_localization_macro=localization_macro,
        wrn_root=wrn_root,
    )
    protocol_audit = _protocol_audit_rows(
        training_audit=training_audit,
        production_audit=production_audit,
        pair_audit=pair_audit,
        geometry_audit=geometry_audit,
    )

    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    figure_data_dir = output_root / "figure_data"
    table_sets: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = {
        "protocol_audit.csv": (protocol_audit, None),
        "id_utility_seed.csv": (id_utility, None),
        "id_utility_summary.csv": (
            summarize_rows(
                id_utility,
                ("cell", "context", "role"),
                ("accuracy", "nll", "ece"),
                paired=False,
            ),
            None,
        ),
        "id_utility_paired_effect_seed.csv": (id_delta, None),
        "id_utility_paired_effect_summary.csv": (
            summarize_rows(
                id_delta,
                ("cell", "context"),
                ("delta_accuracy", "delta_nll", "delta_ece"),
                paired=True,
            ),
            None,
        ),
        "absolute_ood_seed.csv": (absolute, None),
        "absolute_ood_summary.csv": (
            summarize_rows(
                absolute,
                ("cell", "context", "dataset", "region", "role", "readout"),
                ("auroc", "fpr95"),
                paired=False,
            ),
            None,
        ),
        "paired_ood_effect_seed.csv": (paired, None),
        "paired_ood_effect_summary.csv": (
            summarize_rows(
                paired,
                ("cell", "context", "dataset", "region", "readout"),
                ("delta_auroc", "delta_fpr95", "gain", "loss", "churn"),
                paired=True,
            ),
            None,
        ),
        "seed_first_absolute_macro.csv": (absolute_macro, None),
        "seed_first_paired_macro.csv": (paired_macro, None),
        "seed_first_paired_macro_summary.csv": (
            summarize_rows(
                paired_macro,
                ("cell", "context", "region", "readout"),
                ("delta_auroc", "delta_fpr95", "gain", "loss", "churn"),
                paired=True,
            ),
            None,
        ),
        "absolute_recovery_seed.csv": (recovery, None),
        "absolute_recovery_summary.csv": (
            summarize_rows(
                recovery,
                ("cell", "context", "dataset", "region", "role", "readout"),
                ("auroc_recovery", "fpr95_recovery"),
                paired=True,
            ),
            None,
        ),
        "contrast_attenuation_seed.csv": (attenuation, None),
        "contrast_attenuation_summary.csv": (
            summarize_rows(
                attenuation,
                ("cell", "context", "dataset", "region", "readout"),
                ("contrast_attenuation",),
                paired=True,
            ),
            None,
        ),
        "score_localization_seed.csv": (localization, None),
        "score_localization_summary.csv": (
            summarize_rows(
                localization,
                ("cell", "context", "dataset", "region"),
                ("delta_raw_md", "phi_rmd", "phi_marginal"),
                paired=True,
            ),
            None,
        ),
        "seed_first_localization_macro.csv": (localization_macro, None),
        "endpoint_geometry_seed.csv": (geometry, None),
        "endpoint_geometry_summary.csv": (
            summarize_rows(
                geometry,
                ("cell", "context", "role"),
                GEOMETRY_METRICS + ("top10_trace_share",),
                paired=False,
            ),
            None,
        ),
        "endpoint_geometry_effect_seed.csv": (geometry_effects, None),
        "endpoint_geometry_effect_summary.csv": (
            summarize_rows(
                geometry_effects,
                ("cell", "context", "metric", "effect_scale"),
                ("effect",),
                paired=True,
            ),
            None,
        ),
        "numerical_fit_audit.csv": (numerical, None),
        "architecture_boundary_seed.csv": (architecture_boundary, None),
        "architecture_boundary_summary.csv": (
            summarize_rows(
                architecture_boundary,
                ("architecture", "context", "region"),
                (
                    "raw_delta",
                    "churn",
                    "rmd_attenuation",
                    "marginal_signed_adverse_share",
                ),
                paired=True,
            ),
            None,
        ),
    }
    table_paths: list[Path] = []
    for name, (rows, _) in table_sets.items():
        path = tables_dir / name
        _write_csv(path, rows)
        table_paths.append(path)

    r1_seed = _r1_rows(absolute, paired)
    r1_summary = _join_r1_summary(r1_seed)
    r2a_seed = [row for row in absolute if row["context"] == "large"]
    r2b_seed = [row for row in absolute if row["context"] == "small"]
    r2a_summary = summarize_rows(
        r2a_seed,
        ("dataset", "region", "role", "readout"),
        ("auroc",),
        paired=False,
    )
    r2b_summary = summarize_rows(
        r2b_seed,
        ("dataset", "region", "role", "readout"),
        ("auroc",),
        paired=False,
    )
    r3_seed = [row for row in paired if row["readout"] == "Raw MD"]
    r3_summary = summarize_rows(
        r3_seed,
        ("context", "dataset", "region"),
        ("gain", "loss", "delta_auroc", "churn"),
        paired=True,
    )
    r4_seed = localization
    r4_summary = summarize_rows(
        r4_seed,
        ("context", "dataset", "region"),
        ("phi_rmd", "phi_marginal", "delta_raw_md"),
        paired=True,
    )
    r5_seed = [row for row in geometry_effects if row["metric"] in GEOMETRY_METRICS]
    r5_summary = summarize_rows(
        r5_seed, ("context", "metric", "effect_scale"), ("effect",), paired=True
    )
    r6_seed = architecture_boundary
    r6_summary = summarize_rows(
        r6_seed,
        ("architecture", "context", "region"),
        ("raw_delta", "churn", "rmd_attenuation", "marginal_signed_adverse_share"),
        paired=True,
    )
    s1_seed = absolute
    s1_summary = summarize_rows(
        s1_seed,
        ("context", "dataset", "region", "role", "readout"),
        ("fpr95",),
        paired=False,
    )
    s2_seed = [
        {
            "audit_type": "top10_trace_share_effect",
            "context": row["context"],
            "seed": row["seed"],
            "role": "C-D",
            "value": row["effect"],
        }
        for row in geometry_effects
        if row["metric"] == "top10_trace_share"
    ] + [
        {
            "audit_type": "raw_numerical",
            "context": row["context"],
            "seed": row["seed"],
            "role": row["role"],
            "rank_fraction": float(row["rank"]) / float(row["dimension"]),
            "condition_number": row["condition_number"],
            "condition_is_infinite": row["condition_is_infinite"],
            "applicable": row["applicable"],
            "finite_score_arrays": row["finite_score_arrays"],
        }
        for row in numerical
        if row["transform"] == "raw"
    ]
    s2_summary = summarize_rows(
        [row for row in s2_seed if row["audit_type"] == "top10_trace_share_effect"],
        ("audit_type", "context", "role"),
        ("value",),
        paired=True,
    ) + summarize_rows(
        [row for row in s2_seed if row["audit_type"] == "raw_numerical"],
        ("audit_type", "context", "role"),
        ("rank_fraction", "applicable"),
        paired=False,
    )

    figure_inputs = {
        "R1": (r1_seed, r1_summary),
        "R2A": (r2a_seed, r2a_summary),
        "R2B": (r2b_seed, r2b_summary),
        "R3": (r3_seed, r3_summary),
        "R4": (r4_seed, r4_summary),
        "R5": (r5_seed, r5_summary),
        "R6": (r6_seed, r6_summary),
        "S1": (s1_seed, s1_summary),
        "S2": (s2_seed, s2_summary),
    }
    figure_csv_paths: dict[str, tuple[Path, Path]] = {}
    for figure_id, (seed_rows, summary_rows) in figure_inputs.items():
        seed_path = figure_data_dir / f"{figure_id}_seed.csv"
        summary_path = figure_data_dir / f"{figure_id}_summary.csv"
        _write_csv(seed_path, seed_rows)
        _write_csv(summary_path, summary_rows)
        figure_csv_paths[figure_id] = (seed_path, summary_path)

    figure_outputs = {
        "R1": figure_r1(r1_seed, figures_dir),
        "R2A": figure_r2(absolute, context="large", output_dir=figures_dir),
        "R2B": figure_r2(absolute, context="small", output_dir=figures_dir),
        "R3": figure_r3(paired, figures_dir),
        "R4": figure_r4(localization, figures_dir),
        "R5": figure_r5(r5_seed, figures_dir),
        "R6": figure_r6(architecture_boundary, figures_dir),
        "S1": figure_s1(absolute, figures_dir),
        "S2": figure_s2(geometry_effects, numerical, figures_dir),
    }

    reader_summary = _reader_summary(
        paired_macro=paired_macro,
        localization_macro=localization_macro,
        id_delta=id_delta,
    )
    summary_json_path = output_root / "analysis_summary.json"
    summary_json_path.write_text(
        json.dumps(reader_summary, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    reader_markdown_path = output_root / "reader_summary.md"
    _write_reader_markdown(reader_markdown_path, reader_summary)

    figure_sources = {
        figure_id: _figure_source_record(
            root=output_root,
            seed_path=figure_csv_paths[figure_id][0],
            summary_path=figure_csv_paths[figure_id][1],
            figure_paths=figure_outputs[figure_id],
        )
        for figure_id in figure_inputs
    }
    declared = sorted(
        table_paths
        + [path for pair in figure_csv_paths.values() for path in pair]
        + [path for paths in figure_outputs.values() for path in paths]
        + [summary_json_path, reader_markdown_path]
    )
    checksum_path = output_root / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n"
            for path in declared
        ),
        encoding="utf-8",
    )
    declared.append(checksum_path)

    import matplotlib
    import scipy
    import sklearn

    module_path = Path(__file__)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "technical_status": "PASS",
        "scientific_verdict": "PARTIAL",
        "rescue_experiment": False,
        "formation_analysis": "NOT_AVAILABLE_BY_DESIGN",
        "statistical_unit": "training seed",
        "uncertainty": {
            "absolute": "mean, sample SD, ordinary two-sided 90% Student-t interval",
            "paired": "within-seed difference, then two-sided paired 90% Student-t interval",
            "near_far": "equal-weight datasets within seed, then summarize seeds",
        },
        "recovery_definitions": {
            "auroc_recovery": "alternative - Raw MD",
            "fpr95_recovery": "Raw MD - alternative",
            "contrast_attenuation": "abs(delta Raw MD) - abs(delta alternative)",
        },
        "scientific_boundary": (
            "ResNet endpoint geometry is post-result ID-only exploratory; the frozen "
            "PARTIAL verdict is unchanged and no architecture-general claim opens"
        ),
        "audits": {
            "training": {key: value for key, value in training_audit.items() if key != "config_rows"},
            "production": production_audit,
            "pairs": pair_audit,
            "geometry": geometry_audit,
            "wrn": wrn_audit,
        },
        "inputs": {
            "evaluation_root": str(evaluation_root.resolve()),
            "training_root": str(training_root.resolve()),
            "pair_artifacts": [
                {
                    "path": str(path.resolve()),
                    "record_sha256": sha256_file(path / "record.json"),
                    "scores_sha256": sha256_file(path / "scores.npz"),
                }
                for path in pair_roots
            ],
            "geometry_host_outputs": geometry_audit["source_files"],
            "wrn_manifest": wrn_audit,
        },
        "generator": {
            "analysis_git_sha": analysis_git_sha,
            "module": "src/oge/analysis/resnet18_paper_pack.py",
            "module_sha256": sha256_file(module_path),
        },
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "coverage": {
            "runs": 20,
            "pairs": 10,
            "datasets": 6,
            "score_arrays_per_pair": 84,
            "geometry_runs": 20,
            "figures": len(figure_outputs),
        },
        "figure_sources": figure_sources,
        "outputs": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in declared
        ],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return manifest


def build_paper_pack(
    *,
    evaluation_root: str | Path,
    training_root: str | Path,
    geometry_paths: Sequence[str | Path],
    wrn_root: str | Path,
    output_root: str | Path,
    analysis_git_sha: str,
) -> dict[str, Any]:
    """Build atomically or emit only a BLOCKED report on audit failure."""

    destination = Path(output_root)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite paper pack: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        manifest = _build_pack_to(
            evaluation_root=Path(evaluation_root),
            training_root=Path(training_root),
            geometry_paths=geometry_paths,
            wrn_root=Path(wrn_root),
            output_root=temporary,
            analysis_git_sha=analysis_git_sha,
        )
        os.replace(temporary, destination)
        return manifest
    except BaseException as error:
        shutil.rmtree(temporary, ignore_errors=True)
        destination.mkdir()
        blocked = {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED",
            "reason": f"{type(error).__name__}: {error}",
            "partial_figures_written": False,
            "scientific_evidence": False,
        }
        (destination / "BLOCKED.json").write_text(
            json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


def manifest_sha256(pack_root: str | Path) -> str:
    return sha256_file(Path(pack_root) / "manifest.json")
