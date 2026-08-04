"""Fit-once checkpoint metric runtime over verified raw feature caches."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .classification import (
    augrc,
    aurc_tie_grouped,
    expected_calibration_error,
    evaluate_temperature_scaled,
    fit_temperature,
    logit_ood_scores,
    misclassification_auroc,
    negative_log_likelihood,
    top1_accuracy,
)
from .detectors import (
    GDAClassDensityFit,
    NECOFit,
    TiedGaussianFit,
    ViMFit,
    exact_knn_l2_scores,
    fit_gda_class_density,
    fit_neco,
    fit_tied_gaussians,
    fit_vim,
    prototype_scores,
    score_gda_class_density,
    score_neco,
    score_tied_gaussians,
    score_vim,
)
from .extraction import sha256_file, verify_raw_feature_artifact
from .metrics import compute_ood_metrics, macro_average_ood_metrics
from .geometry import (
    GeometryStatistics,
    all_covariance_spectra,
    cdnv,
    class_pair_geometry,
    exact_self_neighbor_distances,
    feature_norm_distribution,
    fit_geometry_statistics,
    hypersphere_alignment_uniformity,
    local_intrinsic_dimensionality,
    neural_collapse_metrics,
    rankme,
    twonn,
)


def _load_split(root: Path, split_record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    directory = root / split_record["relative_directory"]
    arrays = {
        name: np.load(directory / f"{name}.npy", allow_pickle=False)
        for name in (
            "features",
            "logits",
            "class_labels",
            "predictions",
            "is_id",
            "sample_ids",
        )
    }
    expected_count = int(split_record["sample_count"])
    if any(len(array) != expected_count for array in arrays.values()):
        raise ValueError("raw cache array counts do not match manifest")
    return arrays


def load_raw_feature_cache(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    root = Path(path)
    verified = verify_raw_feature_artifact(root)
    manifest = verified["manifest"]
    splits = {
        key: _load_split(root, record)
        for key, record in manifest["dataset"]["splits"].items()
    }
    return manifest, splits


@dataclass
class FittedMetricRuntime:
    train: dict[str, np.ndarray]
    statistics: GeometryStatistics
    classifier_weight: np.ndarray
    classifier_bias: np.ndarray | None
    raw_gaussian: TiedGaussianFit
    normalized_gaussian: TiedGaussianFit
    vim: ViMFit
    vim_sensitivity: dict[int, ViMFit]
    neco: NECOFit | None
    gda: GDAClassDensityFit | None

    @classmethod
    def fit(
        cls,
        raw_artifact_path: str | Path,
        *,
        include_gda: bool = True,
        include_neco: bool = True,
    ) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], "FittedMetricRuntime"]:
        root = Path(raw_artifact_path)
        manifest, splits = load_raw_feature_cache(root)
        if "id_train" not in splits:
            raise ValueError("metric fitting requires an id_train raw cache")
        train = splits["id_train"]
        weight = np.load(root / "classifier_weight.npy", allow_pickle=False).astype(np.float64)
        bias_path = root / "classifier_bias.npy"
        bias = np.load(bias_path, allow_pickle=False).astype(np.float64) if bias_path.is_file() else None
        labels = train["class_labels"]
        features = train["features"]
        logits = train["logits"]
        statistics = fit_geometry_statistics(
            features, labels, num_classes=weight.shape[0]
        )
        raw_gaussian = fit_tied_gaussians(
            features, labels, num_classes=weight.shape[0], normalize=False
        )
        normalized_gaussian = fit_tied_gaussians(
            features, labels, num_classes=weight.shape[0], normalize=True
        )
        vim = fit_vim(features, logits, weight, bias)
        vim_sensitivity = {vim.dim: vim}
        if features.shape[1] == 640:
            for sensitivity_dim in (64, 128, 256, 320):
                if sensitivity_dim not in vim_sensitivity:
                    vim_sensitivity[sensitivity_dim] = fit_vim(
                        features, logits, weight, bias, dim=sensitivity_dim
                    )
        neco = fit_neco(features, dim=100) if include_neco else None
        gda = (
            fit_gda_class_density(features, labels, num_classes=weight.shape[0])
            if include_gda
            else None
        )
        runtime = cls(
            train=train,
            statistics=statistics,
            classifier_weight=weight,
            classifier_bias=bias,
            raw_gaussian=raw_gaussian,
            normalized_gaussian=normalized_gaussian,
            vim=vim,
            vim_sensitivity=vim_sensitivity,
            neco=neco,
            gda=gda,
        )
        return manifest, splits, runtime

    def score(self, query: Mapping[str, np.ndarray], *, knn_k: int = 50) -> dict[str, Any]:
        features = query["features"]
        logits = query["logits"]
        raw = score_tied_gaussians(self.raw_gaussian, features)
        normalized = score_tied_gaussians(self.normalized_gaussian, features)
        prototypes = prototype_scores(
            self.train["features"],
            self.train["class_labels"],
            features,
            classifier_weight=self.classifier_weight,
            num_classes=self.classifier_weight.shape[0],
        )
        knn = exact_knn_l2_scores(
            self.train["features"],
            features,
            fit_sample_ids=self.train["sample_ids"],
            k=knn_k,
        )
        vim = score_vim(self.vim, features, logits)
        result: dict[str, Any] = {
            "ood_score/msp": logit_ood_scores(logits)["msp"],
            "ood_score/max_logit": logit_ood_scores(logits)["max_logit"],
            "ood_score/energy_t1": logit_ood_scores(logits)["energy_t1"],
            "detector/mahalanobis_raw": raw["mahalanobis"],
            "detector/marginal_mahalanobis_raw": raw["marginal_mahalanobis"],
            "detector/relative_mahalanobis_raw": raw["relative_mahalanobis"],
            "detector/mahalanobis_pp": normalized["mahalanobis"],
            "detector/relative_mahalanobis_pp": normalized["relative_mahalanobis"],
            "detector/knn_l2_k50": knn["score"],
            "detector/ncp_euclidean_raw": prototypes["ncp_score"],
            "detector/ctm_prototype_cosine": prototypes["ctm_score"],
            "detector/weight_cosine_appendix": prototypes["weight_cosine_score"],
            "detector/vim_author_dim": vim["score"],
            "diagnostics": {
                "raw_gaussian": raw,
                "normalized_gaussian": normalized,
                "knn": knn,
                "prototype": prototypes,
                "vim": vim,
            },
        }
        if self.classifier_weight.shape[1] == 640:
            for sensitivity_dim, sensitivity_fit in sorted(self.vim_sensitivity.items()):
                sensitivity = score_vim(sensitivity_fit, features, logits)
                result[f"detector/vim_dim{sensitivity_dim}"] = sensitivity["score"]
                result["diagnostics"][f"vim_dim{sensitivity_dim}"] = sensitivity
        if self.neco is not None:
            neco = score_neco(self.neco, features)
            result["detector/neco_author_std_dim100"] = neco["score"]
            result["diagnostics"]["neco"] = neco
        if self.gda is not None:
            gda = score_gda_class_density(self.gda, features)
            result["detector/gda_class_density"] = gda["score"]
            result["diagnostics"]["gda"] = gda
        return result

    def id_metrics(self, query: Mapping[str, np.ndarray]) -> dict[str, Any]:
        logits = query["logits"]
        labels = query["class_labels"]
        return {
            "id/accuracy": top1_accuracy(logits, labels),
            "id/nll_raw": negative_log_likelihood(logits, labels),
            "id/ece_raw_m10": expected_calibration_error(logits, labels, bins=10),
            "id/ece_raw_m15": expected_calibration_error(logits, labels, bins=15),
            "id/ece_raw_m30": expected_calibration_error(logits, labels, bins=30),
            "id/misclassification_auroc_msp_raw": misclassification_auroc(logits, labels),
            "id/augrc_msp_raw": augrc(logits, labels),
            "id/aurc_msp_raw": aurc_tie_grouped(logits, labels),
        }

    def temperature_control(
        self,
        validation: Mapping[str, np.ndarray],
        query: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        fit = fit_temperature(validation["logits"], validation["class_labels"])
        return evaluate_temperature_scaled(query["logits"], query["class_labels"], fit)

    def heldout_geometry(self, query: Mapping[str, np.ndarray]) -> dict[str, Any]:
        heldout_statistics = fit_geometry_statistics(
            query["features"],
            query["class_labels"],
            num_classes=self.classifier_weight.shape[0],
        )
        nc = neural_collapse_metrics(
            self.statistics,
            self.classifier_weight,
            self.classifier_bias,
            query_features=query["features"],
            query_logits=query["logits"],
            query_labels=query["class_labels"],
        )
        return {
            "feature_norm": feature_norm_distribution(heldout_statistics),
            "nc4": nc["nc4"],
        }

    @staticmethod
    def ood_metrics(
        id_detector_scores: Mapping[str, np.ndarray],
        ood_detector_scores: Mapping[str, Mapping[str, np.ndarray]],
        *,
        near_datasets: tuple[str, ...] = ("cifar100", "tin"),
        far_datasets: tuple[str, ...] = ("mnist", "svhn", "texture", "places365"),
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        detector_names = sorted(
            key for key in id_detector_scores if key != "diagnostics"
        )
        for detector_name in detector_names:
            per_dataset = {
                dataset_name: compute_ood_metrics(
                    id_detector_scores[detector_name], scores[detector_name]
                )
                for dataset_name, scores in ood_detector_scores.items()
            }
            result[detector_name] = {
                "per_dataset": per_dataset,
                **macro_average_ood_metrics(
                    per_dataset,
                    near_datasets=near_datasets,
                    far_datasets=far_datasets,
                ),
            }
        return result

    def geometry(
        self,
        *,
        include_neighbors: bool = True,
        uniformity_pair_count: int = 100_000,
    ) -> dict[str, Any]:
        result = {
            "feature_norm": feature_norm_distribution(self.statistics),
            "cdnv": cdnv(self.statistics),
            "class_pair": class_pair_geometry(self.statistics),
            "neural_collapse": neural_collapse_metrics(
                self.statistics, self.classifier_weight, self.classifier_bias
            ),
            "rankme": rankme(self.train["features"]),
            "covariance_spectra": all_covariance_spectra(self.statistics),
            "hypersphere": hypersphere_alignment_uniformity(
                self.statistics, pair_count=uniformity_pair_count
            ),
        }
        if include_neighbors:
            neighbor_distances, neighbor_ids = exact_self_neighbor_distances(
                self.train["features"],
                k=100,
                sample_ids=self.train["sample_ids"],
                include_neighbor_ids=False,
            )
            result["lid"] = local_intrinsic_dimensionality(
                self.train["features"],
                sample_ids=self.train["sample_ids"],
                precomputed_neighbors=(neighbor_distances, neighbor_ids),
            )
            result["twonn"] = twonn(
                self.train["features"],
                sample_ids=self.train["sample_ids"],
                precomputed_neighbors=(neighbor_distances, neighbor_ids),
            )
        return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size > 10_000:
            return {
                "stored_separately": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_bounded_smoke_result(
    *,
    raw_artifact_path: str | Path,
    output_dir: str | Path,
    query_split: str = "id_validation",
) -> Path:
    manifest, splits, runtime = FittedMetricRuntime.fit(
        raw_artifact_path, include_gda=False, include_neco=True
    )
    if query_split not in splits:
        raise ValueError(f"raw cache does not contain {query_split!r}")
    query = splits[query_split]
    scores = runtime.score(query)
    geometry = runtime.geometry(include_neighbors=True, uniformity_pair_count=1_000)
    id_metrics = runtime.id_metrics(query)
    heldout_geometry = runtime.heldout_geometry(query)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"smoke output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        score_dir = temporary / "scores"
        score_dir.mkdir()
        score_index = {}
        for artifact_key, values in scores.items():
            if artifact_key == "diagnostics":
                continue
            filename = artifact_key.replace("/", "__") + ".npy"
            path = score_dir / filename
            np.save(path, values, allow_pickle=False)
            score_index[artifact_key] = {
                "path": path.relative_to(temporary).as_posix(),
                "sha256": sha256_file(path),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
        payload = {
            "schema_version": "1.0",
            "definition_version": "metric_contract_v1.2",
            "smoke_only": True,
            "protected_result": False,
            "checkpoint": manifest["checkpoint"],
            "query_split": query_split,
            "id_metrics": _json_safe(id_metrics),
            "geometry": _json_safe(geometry),
            "heldout_geometry": _json_safe(heldout_geometry),
            "score_index": score_index,
            "fit_diagnostics": {
                "gda_executed": False,
                "gda_reason": "bounded WRN smoke excludes costly singular small-N covariance ladder; synthetic parity covers GAUSS-6",
                "neco_executed": True,
                "vim_dim": runtime.vim.dim,
                "knn_k": 50,
            },
        }
        (temporary / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        checksum_rows = [
            f"{sha256_file(path)}  {path.relative_to(temporary).as_posix()}"
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        ]
        (temporary / "checksums.sha256").write_text(
            "\n".join(checksum_rows) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
