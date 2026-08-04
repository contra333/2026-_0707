"""Protected production execution for the frozen Metric Contract v1.2 grid."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
import weakref
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from oge.data.openood_cifar10 import load_dataset_config
from oge.studies.hashing import scientific_config_hash

from .analysis import restoration_gaps
from .common import DEFINITION_VERSION, REFERENCE_ATOL, REFERENCE_RTOL
from .extraction import (
    extract_checkpoint_artifact,
    load_checkpoint_model,
    sha256_file,
    verify_raw_feature_artifact,
)
from .planning import validate_checkpoint_inventory
from .registry import METRIC_REGISTRY, MetricDefinition
from .runtime import FittedMetricRuntime

PRODUCTION_SCHEMA_VERSION = "1.0"
PROTECTED_AUTHORIZATION = "issue-57-protected-metric-v1.2"
EXECUTABLE_SPLITS = (
    "id_train",
    "id_validation",
    "id_test",
    "cifar100",
    "tin",
    "mnist",
    "svhn",
    "texture",
    "places365",
)
OOD_SPLITS = ("cifar100", "tin", "mnist", "svhn", "texture", "places365")

FORMULA_SOURCES = {
    "ID-4": "gpleiss/temperature_scaling@ce1154ec7d5235eee046f28875c3dd8421f11d45",
    "ID-7": "IML-DKFZ/fd-shifts@c4467aec134e99691359da209f811d91283fc1e3",
    "GAUSS-4": "mueller-mp/maha-norm@53de550be7f88acad959f5fdd8f2430fb9a8b6ea",
    "GAUSS-5": "mueller-mp/maha-norm@53de550be7f88acad959f5fdd8f2430fb9a8b6ea",
    "GAUSS-6": "omegafragger/DDU@f597744c65df4ff51615ace5e86e82ffefe1cd0f",
    "SUB-1": "deeplearning-wisc/knn-ood@2afb2bbed60a8d69384dc9b28e5637711345222b",
    "SUB-3": "Fsoft-AIC/CTM-OOD@3587259bd6a69abd6b4103cb7311ffaa0857d60f",
    "SUB-5": "haoqiwang/vim@dabf9e5b242dbd31c15e09ff12af3d11f009f79c",
    "SUB-6": "drti/neco@6a55640669f0aad3e23f45ce2f6a8e6400c929ba",
    "GEO-5": "jydzhao/neural_collapse_optimizer@7cab4a59bc28da6e356cee1e793ec67a694933b9",
    "GEO-6": "jydzhao/neural_collapse_optimizer@7cab4a59bc28da6e356cee1e793ec67a694933b9",
    "GEO-7": "jydzhao/neural_collapse_optimizer@7cab4a59bc28da6e356cee1e793ec67a694933b9",
    "GEO-8": "jydzhao/neural_collapse_optimizer@7cab4a59bc28da6e356cee1e793ec67a694933b9",
    "GEO-9": "jydzhao/neural_collapse_optimizer@7cab4a59bc28da6e356cee1e793ec67a694933b9",
    "GEO-15": "sissa-data-science/DADApy@c37e52ca3cbe00107696720e5e3efa09f8370bbb",
}


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_state(repository_root: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, dirty


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_inventory(path: str | Path) -> dict[str, Any]:
    inventory_path = Path(path)
    sidecar = inventory_path.with_suffix(inventory_path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_file(inventory_path) != expected:
        raise ValueError("checkpoint inventory file SHA256 sidecar mismatch")
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    validate_checkpoint_inventory(payload)
    return payload


def inventory_job(
    inventory: Mapping[str, Any], *, job_id: str, host_id: str
) -> dict[str, Any]:
    matches = [job for job in inventory["checkpoint_jobs"] if job["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"inventory must contain exactly one job {job_id!r}")
    job = dict(matches[0])
    if job["assigned_host"] != host_id:
        raise ValueError(f"job {job_id!r} is assigned to {job['assigned_host']!r}")
    return job


def validate_dataset_policy(
    inventory: Mapping[str, Any],
    *,
    dataset_config_path: str | Path,
) -> None:
    config_path = Path(dataset_config_path)
    policy = inventory["dataset_policy"]
    if sha256_file(config_path) != policy["dataset_config_sha256"]:
        raise ValueError("dataset config SHA256 differs from the frozen inventory")
    config = load_dataset_config(config_path)
    membership = config["membership_manifest"]
    if membership["sha256"] != policy["membership_manifest"]["sha256"]:
        raise ValueError("dataset membership manifest identity drift")
    if int(membership["row_count"]) != int(policy["membership_manifest"]["row_count"]):
        raise ValueError("dataset membership manifest count drift")
    expected: dict[str, Mapping[str, Any]] = {
        **policy["id_splits"],
        **policy["ood_splits"],
    }
    if set(expected) != set(EXECUTABLE_SPLITS):
        raise ValueError("frozen executable split population drift")
    for split_key in EXECUTABLE_SPLITS:
        item = config["datasets"][split_key]
        planned = expected[split_key]
        if int(item["expected_count"]) != int(planned["count"]):
            raise ValueError(f"{split_key} count differs from the frozen inventory")
        if item["expected_sha256"] != planned["imglist_sha256"]:
            raise ValueError(f"{split_key} imglist differs from the frozen inventory")
        if item["dataset_name"] != planned["dataset_name"]:
            raise ValueError(f"{split_key} dataset identity drift")


def validate_job_checkpoint(
    job: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
) -> None:
    checkpoint = Path(checkpoint_path)
    if sha256_file(checkpoint) != job["checkpoint_sha256"]:
        raise ValueError("source checkpoint SHA256 differs from the frozen inventory")
    _, payload, observed_sha = load_checkpoint_model(
        checkpoint,
        expected_role=str(job["checkpoint_role"]),
        device="cpu",
    )
    resolved = payload["resolved_config"]
    checks = {
        "checkpoint_sha256": observed_sha == job["checkpoint_sha256"],
        "completed_epoch": int(payload["completed_epoch"]) == int(job["completed_epoch"]),
        "training_seed": int(resolved["training"]["seed"]) == int(job["training_seed"]),
        "config_hash": scientific_config_hash(resolved) == job["config_hash"],
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"checkpoint identity gate failed: {failed}")


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_.")
    return normalized or "value"


class ArrayStore:
    """Externalize NumPy arrays while retaining deterministic semantic references."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._seen: dict[
            int, tuple[weakref.ReferenceType[np.ndarray], dict[str, Any]]
        ] = {}

    def store(self, array: np.ndarray, semantic_path: Sequence[str]) -> dict[str, Any]:
        seen = self._seen.get(id(array))
        if seen is not None and seen[0]() is array:
            return dict(seen[1])
        relative = Path("arrays").joinpath(
            *(_safe_component(str(part)) for part in semantic_path[:-1]),
            f"{_safe_component(str(semantic_path[-1]))}.npy",
        )
        path = self.root / relative
        if path.exists():
            suffix = hashlib.sha256("/".join(semantic_path).encode("utf-8")).hexdigest()[:12]
            relative = relative.with_name(f"{relative.stem}__{suffix}.npy")
            path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
        reference = {
            "array_path": relative.as_posix(),
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        self._seen[id(array)] = (weakref.ref(array), reference)
        return dict(reference)

    def externalize(self, value: Any, semantic_path: Sequence[str]) -> Any:
        if isinstance(value, np.ndarray):
            return self.store(value, semantic_path)
        if isinstance(value, np.generic):
            return value.item()
        if is_dataclass(value):
            return {
                field.name: self.externalize(
                    getattr(value, field.name), (*semantic_path, field.name)
                )
                for field in fields(value)
            }
        if isinstance(value, Mapping):
            return {
                str(key): self.externalize(item, (*semantic_path, str(key)))
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self.externalize(item, (*semantic_path, str(index)))
                for index, item in enumerate(value)
            ]
        if isinstance(value, Path):
            return str(value)
        return value


def _definition_for_key(artifact_key: str) -> MetricDefinition:
    for definition in METRIC_REGISTRY:
        if definition.artifact_key == artifact_key:
            return definition
    spectrum = re.sub(
        r"^geometry/spectrum/(sw|total|between)_",
        "geometry/spectrum/{matrix}_",
        artifact_key,
    )
    for definition in METRIC_REGISTRY:
        if definition.artifact_key == spectrum:
            return definition
    if artifact_key.startswith("ood_metric/") and artifact_key.endswith("_macro_mean"):
        return next(item for item in METRIC_REGISTRY if item.formula_id == "OOD-5")
    raise KeyError(f"artifact key is absent from the frozen registry: {artifact_key}")


def _metric_record(
    *,
    job: Mapping[str, Any],
    evaluation_git_sha: str,
    artifact_key: str,
    metric: Mapping[str, Any] | float | int | None,
    fit_split: str | None,
    query_split: str,
    transform: str,
    sample_count: int,
    class_counts: Sequence[int] | None = None,
    dimensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    definition = _definition_for_key(artifact_key)
    if isinstance(metric, Mapping):
        value = metric.get("value")
        status = str(metric.get("status", "success"))
        reason_codes = list(metric.get("reason_codes", []))
        diagnostics = dict(metric.get("diagnostics", {}))
    else:
        value = metric
        status = "success" if metric is not None else "failed"
        reason_codes = [] if metric is not None else ["missing_scalar"]
        diagnostics = {}
    if status not in {"success", "degenerate", "failed"}:
        raise ValueError(f"invalid metric status {status!r}")
    if value is not None and not np.isfinite(value):
        raise ValueError(f"metric {artifact_key} is non-finite without failure status")
    return {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "definition_version": DEFINITION_VERSION,
        "formula_id": definition.formula_id,
        "artifact_key": artifact_key,
        "reporting_tier": definition.reporting_tier,
        "direction": definition.direction,
        "checkpoint_role": job["checkpoint_role"],
        "reporting_role": job["reporting_role"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "completed_epoch": int(job["completed_epoch"]),
        "config_hash": job["config_hash"],
        "training_seed": int(job["training_seed"]),
        "optimizer_family": job["optimizer_family"],
        "labels": list(job["labels"]),
        "evaluation_git_sha": evaluation_git_sha,
        "fit_split": fit_split,
        "query_split": query_split,
        "transform": transform,
        "fit_dtype": "float64",
        "evaluation_dtype": "float64",
        "sample_count": int(sample_count),
        "class_counts": None if class_counts is None else [int(value) for value in class_counts],
        "status": status,
        "reason_codes": reason_codes,
        "value": None if value is None else float(value),
        "runtime_diagnostics": diagnostics,
        "source_commit": FORMULA_SOURCES.get(
            definition.formula_id, "project_metric_contract_v1.2"
        ),
        "tolerance": {"atol": REFERENCE_ATOL, "rtol": REFERENCE_RTOL},
        **dict(dimensions or {}),
    }


def _distribution_records(
    records: list[dict[str, Any]],
    *,
    job: Mapping[str, Any],
    evaluation_git_sha: str,
    artifact_key: str,
    distribution: Mapping[str, Any],
    query_split: str,
    sample_count: int,
    dimensions: Mapping[str, Any] | None = None,
) -> None:
    for statistic in ("mean", "population_std", "cv", "minimum", "maximum"):
        value = distribution.get(statistic)
        records.append(
            _metric_record(
                job=job,
                evaluation_git_sha=evaluation_git_sha,
                artifact_key=artifact_key,
                metric=value,
                fit_split=query_split,
                query_split=query_split,
                transform="raw_feature",
                sample_count=sample_count,
                dimensions={**dict(dimensions or {}), "statistic": statistic},
            )
        )
    for probability, value in distribution.get("quantiles", {}).items():
        records.append(
            _metric_record(
                job=job,
                evaluation_git_sha=evaluation_git_sha,
                artifact_key=artifact_key,
                metric=value,
                fit_split=query_split,
                query_split=query_split,
                transform="raw_feature",
                sample_count=sample_count,
                dimensions={
                    **dict(dimensions or {}),
                    "statistic": "linear_quantile",
                    "quantile_probability": probability,
                },
            )
        )


def _scalar_records(
    *,
    job: Mapping[str, Any],
    evaluation_git_sha: str,
    splits: Mapping[str, Mapping[str, np.ndarray]],
    id_metrics: Mapping[str, Any],
    temperature: Mapping[str, Any],
    geometry: Mapping[str, Any],
    validation_geometry: Mapping[str, Any],
    test_geometry: Mapping[str, Any],
    ood_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    id_count = len(splits["id_test"]["features"])
    id_class_counts = np.bincount(splits["id_test"]["class_labels"], minlength=10)
    for artifact_key, payload in id_metrics.items():
        records.append(
            _metric_record(
                job=job,
                evaluation_git_sha=evaluation_git_sha,
                artifact_key=artifact_key,
                metric=payload["metric"],
                fit_split=None,
                query_split="id_test",
                transform="raw_logits",
                sample_count=id_count,
                class_counts=id_class_counts,
            )
        )
    for artifact_key, payload in (
        ("id/temperature", temperature["temperature"]),
        ("id/nll_ts", temperature["nll"]["metric"]),
        ("id/ece_ts_m15", temperature["ece_m15"]["metric"]),
    ):
        records.append(
            _metric_record(
                job=job,
                evaluation_git_sha=evaluation_git_sha,
                artifact_key=artifact_key,
                metric=payload,
                fit_split="id_validation",
                query_split="id_test",
                transform="temperature_scaled_logits",
                sample_count=id_count,
                class_counts=id_class_counts,
            )
        )

    train_count = len(splits["id_train"]["features"])
    train_class_counts = np.bincount(splits["id_train"]["class_labels"], minlength=10)
    _distribution_records(
        records,
        job=job,
        evaluation_git_sha=evaluation_git_sha,
        artifact_key="geometry/feature_norm/global",
        distribution=geometry["feature_norm"]["global"],
        query_split="id_train",
        sample_count=train_count,
    )
    for class_index, distribution in geometry["feature_norm"]["classwise"].items():
        _distribution_records(
            records,
            job=job,
            evaluation_git_sha=evaluation_git_sha,
            artifact_key="geometry/feature_norm/classwise",
            distribution=distribution,
            query_split="id_train",
            sample_count=int(distribution["count"]),
            dimensions={"class_index": int(class_index)},
        )
    records.append(
        _metric_record(
            job=job,
            evaluation_git_sha=evaluation_git_sha,
            artifact_key="geometry/cdnv/mean",
            metric=geometry["cdnv"]["metric"],
            fit_split="id_train",
            query_split="id_train",
            transform="raw_feature",
            sample_count=train_count,
            class_counts=train_class_counts,
        )
    )
    nc = geometry["neural_collapse"]
    for name in (
        "nc0_row_sum_raw",
        "nc0_eq12_per_dim",
        "nc0_theory_squared",
        "nc1_pinv",
        "nc1_svd_diagnostic",
        "nc1_trace_quotient_diagnostic",
        "nc2_equinorm",
        "nc2_equiangular",
        "nc2_etf_raw",
        "nc2_etf_eq5_scaled",
        "nc2w_etf_raw",
        "nc2w_equinorm",
        "nc2w_equiangular",
        "nc3_self_duality_raw",
        "nc3_eq10_scaled",
    ):
        records.append(
            _metric_record(
                job=job,
                evaluation_git_sha=evaluation_git_sha,
                artifact_key=f"geometry/{name}",
                metric=nc[name],
                fit_split="id_train",
                query_split="id_train",
                transform="raw_feature",
                sample_count=train_count,
                class_counts=train_class_counts,
            )
        )
    records.append(
        _metric_record(
            job=job,
            evaluation_git_sha=evaluation_git_sha,
            artifact_key="geometry/rankme_uncentered",
            metric=geometry["rankme"]["metric"],
            fit_split="id_train",
            query_split="id_train",
            transform="raw_feature_uncentered",
            sample_count=train_count,
            class_counts=train_class_counts,
        )
    )
    for matrix_name, spectrum in geometry["covariance_spectra"].items():
        for statistic in (
            "entropy_rank",
            "trace_top_rank",
            "participation_ratio",
            "numerical_rank",
            "condition_number_retained",
        ):
            records.append(
                _metric_record(
                    job=job,
                    evaluation_git_sha=evaluation_git_sha,
                    artifact_key=f"geometry/spectrum/{matrix_name}_{statistic}",
                    metric=spectrum.get(
                        statistic,
                        {
                            "value": None,
                            "status": spectrum.get("status", "failed"),
                            "reason_codes": spectrum.get("reason_codes", ["missing_spectrum_scalar"]),
                        },
                    ),
                    fit_split="id_train",
                    query_split="id_train",
                    transform="raw_feature_covariance",
                    sample_count=train_count,
                    class_counts=train_class_counts,
                )
            )
    for statistic in ("log_slope", "top20_mean_log_decay"):
        records.append(
            _metric_record(
                job=job,
                evaluation_git_sha=evaluation_git_sha,
                artifact_key=f"geometry/spectrum/sw_{statistic}",
                metric=geometry["covariance_spectra"]["sw"][statistic],
                fit_split="id_train",
                query_split="id_train",
                transform="raw_feature_covariance",
                sample_count=train_count,
                class_counts=train_class_counts,
            )
        )
    for k in (10, 25, 50, 100):
        records.append(
            _metric_record(
                job=job,
                evaluation_git_sha=evaluation_git_sha,
                artifact_key=f"geometry/lid_k{k}",
                metric=geometry["lid"][str(k)]["metric"],
                fit_split="id_train",
                query_split="id_train",
                transform="raw_feature_exact_neighbors",
                sample_count=train_count,
                class_counts=train_class_counts,
            )
        )
    for artifact_key, payload in (
        ("geometry/twonn_base_mu_fraction_09", geometry["twonn"]["metric"]),
        (
            "geometry/hypersphere/class_alignment_exact",
            geometry["hypersphere"]["class_alignment"],
        ),
        ("geometry/hypersphere/uniformity_t2", geometry["hypersphere"]["uniformity"]),
    ):
        records.append(
            _metric_record(
                job=job,
                evaluation_git_sha=evaluation_git_sha,
                artifact_key=artifact_key,
                metric=payload,
                fit_split="id_train",
                query_split="id_train",
                transform="raw_feature",
                sample_count=train_count,
                class_counts=train_class_counts,
            )
        )

    _distribution_records(
        records,
        job=job,
        evaluation_git_sha=evaluation_git_sha,
        artifact_key="geometry/feature_norm/heldout_validation",
        distribution=validation_geometry["feature_norm"]["global"],
        query_split="id_validation",
        sample_count=len(splits["id_validation"]["features"]),
    )
    for query_split, heldout in (
        ("id_validation", validation_geometry),
        ("id_test", test_geometry),
    ):
        for suffix in ("agreement_with_bias", "agreement_without_bias"):
            records.append(
                _metric_record(
                    job=job,
                    evaluation_git_sha=evaluation_git_sha,
                    artifact_key=f"geometry/nc4_{suffix}",
                    metric=heldout["nc4"][suffix],
                    fit_split="id_train",
                    query_split=query_split,
                    transform="raw_feature_and_logits",
                    sample_count=len(splits[query_split]["features"]),
                    class_counts=np.bincount(
                        splits[query_split]["class_labels"], minlength=10
                    ),
                )
            )

    metric_key_map = {
        "auroc": "ood_metric/auroc_id_positive",
        "ap_in": "ood_metric/ap_in",
        "aupr_in_openood_auc": "ood_metric/aupr_in_openood_auc",
        "ap_out": "ood_metric/ap_out",
        "aupr_out_openood_auc": "ood_metric/aupr_out_openood_auc",
        "fpr95_id_tpr": "ood_metric/fpr95_id_tpr",
        "fpr95_threshold": "ood_metric/fpr95_threshold",
        "fpr95_achieved_id_tpr": "ood_metric/fpr95_achieved_id_tpr",
    }
    for detector_name, detector_payload in ood_metrics.items():
        for dataset_name, dataset_metrics in detector_payload["per_dataset"].items():
            for metric_name, value in dataset_metrics.items():
                records.append(
                    _metric_record(
                        job=job,
                        evaluation_git_sha=evaluation_git_sha,
                        artifact_key=metric_key_map[metric_name],
                        metric=value,
                        fit_split="id_train" if detector_name.startswith("detector/") else None,
                        query_split=f"id_test_vs_{dataset_name}",
                        transform="detector_declared",
                        sample_count=id_count + len(splits[dataset_name]["features"]),
                        dimensions={
                            "detector": detector_name,
                            "ood_dataset": dataset_name,
                            "ood_group": "near" if dataset_name in {"cifar100", "tin"} else "far",
                            "num_id": id_count,
                            "num_ood": len(splits[dataset_name]["features"]),
                        },
                    )
                )
        for group_name in ("near", "far", "overall"):
            group_payload = detector_payload[f"{group_name}_macro_mean"]
            for metric_name, value in group_payload.items():
                records.append(
                    _metric_record(
                        job=job,
                        evaluation_git_sha=evaluation_git_sha,
                        artifact_key=f"ood_metric/{metric_name}/{group_name}_macro_mean",
                        metric=value,
                        fit_split="id_train" if detector_name.startswith("detector/") else None,
                        query_split=f"id_test_vs_{group_name}_ood",
                        transform="per_dataset_arithmetic_macro_mean",
                        sample_count=id_count,
                        dimensions={"detector": detector_name, "ood_group": group_name},
                    )
                )

    for dataset_name in OOD_SPLITS:
        per_detector = {
            name: payload["per_dataset"][dataset_name]
            for name, payload in ood_metrics.items()
        }
        identity = {
            "checkpoint_sha256": job["checkpoint_sha256"],
            "training_seed": int(job["training_seed"]),
            "ood_dataset": dataset_name,
        }
        gaps = restoration_gaps(
            mahalanobis_raw={**identity, **per_detector["detector/mahalanobis_raw"]},
            mahalanobis_pp={**identity, **per_detector["detector/mahalanobis_pp"]},
            relative_raw={**identity, **per_detector["detector/relative_mahalanobis_raw"]},
            relative_pp={**identity, **per_detector["detector/relative_mahalanobis_pp"]},
            ncp={**identity, **per_detector["detector/ncp_euclidean_raw"]},
            ctm={**identity, **per_detector["detector/ctm_prototype_cosine"]},
        )
        for gap_name, value in gaps.items():
            records.append(
                _metric_record(
                    job=job,
                    evaluation_git_sha=evaluation_git_sha,
                    artifact_key=f"analysis/{gap_name}",
                    metric=value,
                    fit_split="id_train",
                    query_split=f"id_test_vs_{dataset_name}",
                    transform="paired_auroc_difference",
                    sample_count=id_count + len(splits[dataset_name]["features"]),
                    dimensions={"ood_dataset": dataset_name},
                )
            )
    return records


def _write_checksums(root: Path, filename: str) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != filename
    ]
    (root / filename).write_text("\n".join(rows) + "\n", encoding="utf-8")


def _verify_checksums(root: Path, filename: str) -> dict[str, str]:
    checksum_path = root / filename
    observed: dict[str, str] = {}
    for row in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = row.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"checksum mismatch for {relative}")
        observed[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != filename
    }
    if actual != set(observed):
        raise ValueError("checksum manifest does not enumerate the complete artifact tree")
    return observed


def verify_production_bundle(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    verified = _verify_checksums(root, "bundle_checksums.sha256")
    artifact_manifest = json.loads((root / "artifact_manifest.json").read_text())
    completion = json.loads((root / "JOB_COMPLETE.json").read_text())
    if artifact_manifest["status"] != "LOCAL_COMPLETE" or completion["status"] != "LOCAL_COMPLETE":
        raise ValueError("production bundle lacks a terminal local-complete state")
    if artifact_manifest["checkpoint_sha256"] != root.name:
        raise ValueError("production bundle path/checkpoint identity mismatch")
    return {
        "artifact_manifest": artifact_manifest,
        "completion": completion,
        "verified_files": verified,
    }


def evaluate_raw_feature_cache_production(
    *,
    raw_artifact_path: str | Path,
    job: Mapping[str, Any],
    inventory: Mapping[str, Any],
    evaluation_git_sha: str,
    sources_lock_path: str | Path,
) -> Path:
    root = Path(raw_artifact_path)
    if (root / "JOB_COMPLETE.json").is_file():
        verify_production_bundle(root)
        return root
    manifest, splits, runtime = FittedMetricRuntime.fit(
        root, include_gda=True, include_neco=True
    )
    if tuple(splits) != EXECUTABLE_SPLITS:
        raise ValueError("raw cache does not contain the exact executable split order")
    expected = {
        **inventory["dataset_policy"]["id_splits"],
        **inventory["dataset_policy"]["ood_splits"],
    }
    for split_key in EXECUTABLE_SPLITS:
        if len(splits[split_key]["features"]) != int(expected[split_key]["count"]):
            raise ValueError(f"{split_key} raw cache count differs from the frozen policy")
    if manifest["evaluation"]["protected_split_authorization"] != PROTECTED_AUTHORIZATION:
        raise ValueError("raw cache protected authorization mismatch")
    checkpoint = manifest["checkpoint"]
    identity_checks = {
        "sha256": checkpoint["sha256"] == job["checkpoint_sha256"],
        "role": checkpoint["role"] == job["checkpoint_role"],
        "completed_epoch": int(checkpoint["completed_epoch"]) == int(job["completed_epoch"]),
        "training_seed": int(checkpoint["training_seed"]) == int(job["training_seed"]),
        "config_hash": checkpoint["config_hash"] == job["config_hash"],
        "evaluation_git_sha": manifest["evaluation"]["oge_git_sha"] == evaluation_git_sha,
        "git_clean": manifest["evaluation"]["git_dirty"] is False,
    }
    failed = sorted(name for name, passed in identity_checks.items() if not passed)
    if failed:
        raise ValueError(f"raw cache identity gate failed: {failed}")

    metrics_destination = root / "metrics"
    if metrics_destination.exists():
        _verify_checksums(metrics_destination, "checksums.sha256")
    else:
        temporary = root.parent / f".{root.name}.metrics.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=True)
        array_store = ArrayStore(temporary)
        score_documents: dict[str, Any] = {}
        detector_scores: dict[str, dict[str, np.ndarray]] = {}
        for split_key in ("id_test", *OOD_SPLITS):
            scored = runtime.score(splits[split_key])
            detector_scores[split_key] = {
                key: value for key, value in scored.items() if key != "diagnostics"
            }
            score_documents[split_key] = array_store.externalize(
                scored, ("scores", split_key)
            )
        id_metrics = runtime.id_metrics(splits["id_test"])
        temperature = runtime.temperature_control(
            splits["id_validation"], splits["id_test"]
        )
        geometry = runtime.geometry(include_neighbors=True, uniformity_pair_count=100_000)
        validation_geometry = runtime.heldout_geometry(splits["id_validation"])
        test_geometry = runtime.heldout_geometry(splits["id_test"])
        ood_metrics = runtime.ood_metrics(
            detector_scores["id_test"],
            {name: detector_scores[name] for name in OOD_SPLITS},
        )
        records = _scalar_records(
            job=job,
            evaluation_git_sha=evaluation_git_sha,
            splits=splits,
            id_metrics=id_metrics,
            temperature=temperature,
            geometry=geometry,
            validation_geometry=validation_geometry,
            test_geometry=test_geometry,
            ood_metrics=ood_metrics,
        )
        fit_state = {
            "raw_gaussian": runtime.raw_gaussian,
            "normalized_gaussian": runtime.normalized_gaussian,
            "gda": runtime.gda,
            "vim_author": runtime.vim,
            "vim_sensitivity": runtime.vim_sensitivity,
            "neco": runtime.neco,
            "geometry_statistics": runtime.statistics,
        }
        payload = {
            "schema_version": PRODUCTION_SCHEMA_VERSION,
            "definition_version": DEFINITION_VERSION,
            "protected_result": True,
            "smoke_only": False,
            "authorization": PROTECTED_AUTHORIZATION,
            "job": dict(job),
            "inventory_hash": inventory["inventory_hash"],
            "dataset_policy_hash": inventory["dataset_policy_hash"],
            "evaluation_git_sha": evaluation_git_sha,
            "sources_lock": {
                "path": "docs/sources.lock.yaml",
                "sha256": _sha256_text(Path(sources_lock_path)),
            },
            "scores": score_documents,
            "id_metrics": array_store.externalize(id_metrics, ("id_metrics",)),
            "temperature": array_store.externalize(temperature, ("temperature",)),
            "geometry": array_store.externalize(geometry, ("geometry", "train")),
            "validation_geometry": array_store.externalize(
                validation_geometry, ("geometry", "validation")
            ),
            "test_geometry": array_store.externalize(
                test_geometry, ("geometry", "test")
            ),
            "ood_metrics": array_store.externalize(ood_metrics, ("ood_metrics",)),
            "fit_state": array_store.externalize(fit_state, ("fit_state",)),
            "scalar_record_count": len(records),
        }
        _json_dump(temporary / "result.json", payload)
        with (temporary / "scalar_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        _json_dump(
            temporary / "METRICS_COMPLETE.json",
            {
                "schema_version": PRODUCTION_SCHEMA_VERSION,
                "status": "METRICS_COMPLETE",
                "checkpoint_sha256": job["checkpoint_sha256"],
                "scalar_record_count": len(records),
                "split_order": list(EXECUTABLE_SPLITS),
            },
        )
        _write_checksums(temporary, "checksums.sha256")
        _verify_checksums(temporary, "checksums.sha256")
        os.replace(temporary, metrics_destination)

    metrics_verified = _verify_checksums(metrics_destination, "checksums.sha256")
    scalar_count = sum(
        1 for _ in (metrics_destination / "scalar_records.jsonl").open(encoding="utf-8")
    )
    split_counts = {
        key: int(manifest["dataset"]["splits"][key]["sample_count"])
        for key in EXECUTABLE_SPLITS
    }
    completion = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "definition_version": DEFINITION_VERSION,
        "status": "LOCAL_COMPLETE",
        "authorization": PROTECTED_AUTHORIZATION,
        "job_id": job["job_id"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "checkpoint_role": job["checkpoint_role"],
        "evaluation_git_sha": evaluation_git_sha,
        "inventory_hash": inventory["inventory_hash"],
        "dataset_policy_hash": inventory["dataset_policy_hash"],
        "split_counts": split_counts,
        "scalar_record_count": scalar_count,
        "metric_file_count": len(metrics_verified),
    }
    _json_dump(root / "JOB_COMPLETE.json", completion)
    artifact_manifest = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "definition_version": DEFINITION_VERSION,
        "artifact_role": "checkpoint_metric_contract_v1.2_bundle",
        "status": "LOCAL_COMPLETE",
        "checkpoint_sha256": job["checkpoint_sha256"],
        "checkpoint_role": job["checkpoint_role"],
        "reporting_role": job["reporting_role"],
        "config_hash": job["config_hash"],
        "training_seed": int(job["training_seed"]),
        "evaluation_git_sha": evaluation_git_sha,
        "inventory_hash": inventory["inventory_hash"],
        "dataset_policy_hash": inventory["dataset_policy_hash"],
        "protected_authorization": PROTECTED_AUTHORIZATION,
        "split_order": list(EXECUTABLE_SPLITS),
        "split_counts": split_counts,
        "raw_manifest_sha256": sha256_file(root / "manifest.json"),
        "metric_result_sha256": sha256_file(metrics_destination / "result.json"),
        "scalar_records_sha256": sha256_file(
            metrics_destination / "scalar_records.jsonl"
        ),
    }
    _json_dump(root / "artifact_manifest.json", artifact_manifest)
    _write_checksums(root, "bundle_checksums.sha256")
    verify_production_bundle(root)
    return root


def run_production_job(
    *,
    inventory_path: str | Path,
    job_id: str,
    host_id: str,
    checkpoint_path: str | Path,
    dataset_config_path: str | Path,
    data_root: str | Path,
    artifact_root: str | Path,
    repository_root: str | Path,
    device: str,
    batch_size: int = 128,
    num_workers: int = 4,
    protected_authorization: str,
    expected_evaluation_git_sha: str | None = None,
) -> Path:
    if protected_authorization != PROTECTED_AUTHORIZATION:
        raise PermissionError("production evaluation authorization is absent or incorrect")
    repository = Path(repository_root).resolve()
    evaluation_git_sha, dirty = _git_state(repository)
    if dirty:
        raise ValueError("production evaluation requires a clean Git worktree")
    if expected_evaluation_git_sha is not None and evaluation_git_sha != expected_evaluation_git_sha:
        raise ValueError("evaluation Git SHA differs from the supervisor gate")
    inventory = load_frozen_inventory(inventory_path)
    job = inventory_job(inventory, job_id=job_id, host_id=host_id)
    validate_job_checkpoint(job, checkpoint_path=checkpoint_path)
    validate_dataset_policy(inventory, dataset_config_path=dataset_config_path)
    checkpoint_root = Path(artifact_root) / job["checkpoint_sha256"]
    if checkpoint_root.is_dir():
        if (checkpoint_root / "JOB_COMPLETE.json").is_file():
            verify_production_bundle(checkpoint_root)
            return checkpoint_root
        verified = verify_raw_feature_artifact(checkpoint_root)
        if verified["manifest"]["checkpoint"]["sha256"] != job["checkpoint_sha256"]:
            raise ValueError("resumed raw cache checkpoint identity mismatch")
    else:
        extracted = extract_checkpoint_artifact(
            checkpoint_path=checkpoint_path,
            dataset_config_path=dataset_config_path,
            data_root=data_root,
            artifact_root=artifact_root,
            dataset_keys=list(EXECUTABLE_SPLITS),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            protected_authorization=protected_authorization,
            smoke_only=False,
            extraction_command=f"production-job:{job_id}",
            repository_root=repository,
        )
        if extracted != checkpoint_root:
            raise ValueError("extraction destination differs from checkpoint identity")
    return evaluate_raw_feature_cache_production(
        raw_artifact_path=checkpoint_root,
        job=job,
        inventory=inventory,
        evaluation_git_sha=evaluation_git_sha,
        sources_lock_path=repository / "docs/sources.lock.yaml",
    )
