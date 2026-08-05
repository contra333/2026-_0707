"""C1-C4 analysis for the completed Metric Contract v1.2 aggregate.

This module deliberately consumes only the checksum-verified central scalar
aggregate.  It never opens a checkpoint, dataset, raw score, or feature cache.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


DEFINITION_VERSION = "metric_contract_v1.2"
EXPECTED_EVALUATION_SHA = "c38b09694be88aa74de0741b39e9d3ba0d6ff61a"
EXPECTED_INVENTORY_HASH = (
    "35a1a66f78dfd6255c4d804586ee6863969be78d21acec5cd0ccf00476ddaa5c"
)
EXPECTED_DATASET_POLICY_HASH = (
    "f80984d28ef9a7550633fb3aea514f783f3db1fa655a6d3f49850b10857e6afc"
)
ROLE_ORDER = ("C1", "C2", "C3", "C4")
OPTIMIZER_ORDER = ("sgd", "adam", "adamw")
CHECKPOINT_ORDER = ("last", "best_val")
OOD_DATASETS = ("cifar100", "tin", "mnist", "svhn", "texture", "places365")
OOD_DISPLAY = {
    "cifar100": "CIFAR-100",
    "tin": "TinyImageNet",
    "mnist": "MNIST",
    "svhn": "SVHN",
    "texture": "Textures",
    "places365": "Places365",
}
OPTIMIZER_DISPLAY = {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW"}

CANONICAL_DETECTORS = (
    "detector/ctm_prototype_cosine",
    "detector/knn_l2_k50",
    "detector/mahalanobis_pp",
    "detector/mahalanobis_raw",
    "detector/neco_author_std_dim100",
    "detector/relative_mahalanobis_pp",
    "detector/relative_mahalanobis_raw",
    "detector/vim_author_dim",
    "ood_score/energy_t1",
    "ood_score/max_logit",
    "ood_score/msp",
)

ALL_DETECTORS = (
    "detector/ctm_prototype_cosine",
    "detector/gda_class_density",
    "detector/knn_l2_k50",
    "detector/mahalanobis_pp",
    "detector/mahalanobis_raw",
    "detector/marginal_mahalanobis_raw",
    "detector/ncp_euclidean_raw",
    "detector/neco_author_std_dim100",
    "detector/relative_mahalanobis_pp",
    "detector/relative_mahalanobis_raw",
    "detector/vim_author_dim",
    "detector/vim_dim128",
    "detector/vim_dim256",
    "detector/vim_dim320",
    "detector/vim_dim64",
    "detector/weight_cosine_appendix",
    "ood_score/energy_t1",
    "ood_score/max_logit",
    "ood_score/msp",
)

ASSOCIATION_ENDPOINTS = {
    "auroc": ("ood_metric/auroc_id_positive", 1.0),
    "fpr95_performance": ("ood_metric/fpr95_id_tpr", -1.0),
    "aupr_in_appendix": ("ood_metric/aupr_in_openood_auc", 1.0),
    "aupr_out_appendix": ("ood_metric/aupr_out_openood_auc", 1.0),
}
HEATMAP_ENDPOINTS = ("auroc", "fpr95_performance")
OOD_TABLE_METRICS = (
    ("auroc", "ood_metric/auroc_id_positive"),
    ("fpr95", "ood_metric/fpr95_id_tpr"),
    ("aupr_in", "ood_metric/aupr_in_openood_auc"),
    ("aupr_out", "ood_metric/aupr_out_openood_auc"),
)

NOTION_ID_PREDICTIVE_METRICS = (
    ("Accuracy ↑", "id/accuracy", "up"),
    ("NLL ↓", "id/nll_raw", "down"),
    ("ECE-M15 ↓", "id/ece_raw_m15", "down"),
    ("TS NLL ↓", "id/nll_ts", "down"),
    ("TS ECE-M15 ↓", "id/ece_ts_m15", "down"),
)
NOTION_ID_FAILURE_METRICS = (
    ("Misclassification AUROC ↑", "id/misclassification_auroc_msp_raw", "up"),
    ("AUGRC ↓", "id/augrc_msp_raw", "down"),
    ("AURC ↓", "id/aurc_msp_raw", "down"),
)
NOTION_DETECTOR_GROUPS = (
    (
        "Distance·prototype detectors",
        (
            ("CTM", "detector/ctm_prototype_cosine"),
            ("kNN-L2", "detector/knn_l2_k50"),
            ("Mahalanobis++", "detector/mahalanobis_pp"),
            ("Mahalanobis raw", "detector/mahalanobis_raw"),
            ("Relative Mahalanobis++", "detector/relative_mahalanobis_pp"),
            ("Relative Mahalanobis raw", "detector/relative_mahalanobis_raw"),
        ),
    ),
    (
        "Subspace·logit detectors",
        (
            ("NECO", "detector/neco_author_std_dim100"),
            ("ViM", "detector/vim_author_dim"),
            ("Energy-T1", "ood_score/energy_t1"),
            ("Maximum Logit", "ood_score/max_logit"),
            ("MSP", "ood_score/msp"),
        ),
    ),
)


@dataclass(frozen=True)
class GeometrySpec:
    name: str
    artifact_key: str
    statistic: str | None = None
    query_split: str | None = None


NOTION_GEOMETRY_COLLAPSE = (
    (
        "Feature norm mean",
        GeometrySpec("Feature norm mean", "geometry/feature_norm/global", "mean"),
        None,
    ),
    (
        "Feature norm CV",
        GeometrySpec("Feature norm CV", "geometry/feature_norm/global", "cv"),
        None,
    ),
    ("CDNV ↓", GeometrySpec("CDNV", "geometry/cdnv/mean"), "down"),
    ("NC1 ↓", GeometrySpec("NC1", "geometry/nc1_pinv"), "down"),
    (
        "NC2 equinorm ↓",
        GeometrySpec("NC2 equinorm", "geometry/nc2_equinorm"),
        "down",
    ),
    (
        "NC2 equiangular ↓",
        GeometrySpec("NC2 equiangular", "geometry/nc2_equiangular"),
        "down",
    ),
    ("NC2 ETF ↓", GeometrySpec("NC2 ETF", "geometry/nc2_etf_raw"), "down"),
    ("NC3 ↓", GeometrySpec("NC3", "geometry/nc3_self_duality_raw"), "down"),
    (
        "NC4 ↑",
        GeometrySpec(
            "NC4",
            "geometry/nc4_agreement_with_bias",
            query_split="id_test",
        ),
        "up",
    ),
)
NOTION_GEOMETRY_SPECTRUM = (
    ("RankMe", GeometrySpec("RankMe", "geometry/rankme_uncentered"), None),
    (
        "SW entropy rank",
        GeometrySpec("SW entropy rank", "geometry/spectrum/sw_entropy_rank"),
        None,
    ),
    (
        "SW trace/top rank",
        GeometrySpec("SW trace/top rank", "geometry/spectrum/sw_trace_top_rank"),
        None,
    ),
    (
        "SW participation ratio",
        GeometrySpec(
            "SW participation ratio",
            "geometry/spectrum/sw_participation_ratio",
        ),
        None,
    ),
    (
        "SW numerical rank",
        GeometrySpec("SW numerical rank", "geometry/spectrum/sw_numerical_rank"),
        None,
    ),
    (
        "SW condition number",
        GeometrySpec(
            "SW condition number",
            "geometry/spectrum/sw_condition_number_retained",
        ),
        None,
    ),
    ("LID-50", GeometrySpec("LID-50", "geometry/lid_k50"), None),
)


GEOMETRY_PANEL = (
    GeometrySpec("Feature norm mean", "geometry/feature_norm/global", "mean"),
    GeometrySpec("Feature norm CV", "geometry/feature_norm/global", "cv"),
    GeometrySpec("CDNV", "geometry/cdnv/mean"),
    GeometrySpec("NC1", "geometry/nc1_pinv"),
    GeometrySpec("NC2 equinorm", "geometry/nc2_equinorm"),
    GeometrySpec("NC2 equiangular", "geometry/nc2_equiangular"),
    GeometrySpec("NC2 ETF", "geometry/nc2_etf_raw"),
    GeometrySpec("NC3", "geometry/nc3_self_duality_raw"),
    GeometrySpec(
        "NC4", "geometry/nc4_agreement_with_bias", query_split="id_test"
    ),
    GeometrySpec("RankMe", "geometry/rankme_uncentered"),
    GeometrySpec("SW entropy rank", "geometry/spectrum/sw_entropy_rank"),
    GeometrySpec("SW trace/top rank", "geometry/spectrum/sw_trace_top_rank"),
    GeometrySpec(
        "SW participation ratio", "geometry/spectrum/sw_participation_ratio"
    ),
    GeometrySpec("SW numerical rank", "geometry/spectrum/sw_numerical_rank"),
    GeometrySpec(
        "SW condition number", "geometry/spectrum/sw_condition_number_retained"
    ),
    GeometrySpec("LID-50", "geometry/lid_k50"),
)

ID_PAPER_METRICS = (
    ("Accuracy", "id/accuracy"),
    ("NLL", "id/nll_raw"),
    ("ECE-M15", "id/ece_raw_m15"),
    ("Misclass. AUROC", "id/misclassification_auroc_msp_raw"),
    ("AUGRC", "id/augrc_msp_raw"),
)

METHOD_SENTENCES = {
    "ID-1": "We report top-1 accuracy on the frozen 10,000-image CIFAR-10 test split.",
    "ID-2": "NLL is the mean negative log-probability assigned to the true class using unscaled logits.",
    "ID-3": "ECE uses 15 fixed equal-width confidence bins; 10- and 30-bin values are prespecified sensitivity analyses.",
    "ID-4": "A single positive temperature was fitted on ID validation by float64 NLL minimization and applied once to the frozen ID test logits.",
    "ID-5": "Misclassification AUROC treats correct predictions as the positive class and ranks them using raw maximum softmax probability.",
    "ID-6": "AUGRC combines residual error prevalence and raw-MSP ranking quality using its closed-form binary-error expression.",
    "ID-7": "AURC is the unscaled trapezoidal area under the tie-grouped raw-MSP risk-coverage curve.",
    "OOD-1": "OOD AUROC treats ID as positive and uses the declared ID-oriented detector score.",
    "OOD-2": "AUPR-In uses ID as positive; trapezoidal PR-AUC is primary and Average Precision is reported separately.",
    "OOD-3": "AUPR-Out treats OOD as positive and ranks examples with the negated ID-oriented score.",
    "OOD-4": "FPR@95 is the fraction of OOD scores above the linear 5th percentile of ID scores, using an inclusive threshold.",
    "OOD-5": "Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates.",
    "OOD-7": "Restoration gaps are paired within-checkpoint AUROC differences between normalized and raw Gaussian readouts.",
    "GEO-1": "We characterize radial feature geometry using the full ID-training norm distribution and class-wise summaries.",
    "GEO-2": "CDNV averages the pairwise ratio of within-class variance to squared class-mean separation.",
    "GEO-4": "NC0 is the Euclidean norm of the sum of classifier-weight rows; scaled expressions are reported only as audits.",
    "GEO-5": "We report NC1 as Tr(Sigma_W Sigma_B^dagger)/K, using biased within- and between-class covariance estimators computed from raw ID-training features.",
    "GEO-6": "NC2n is the sample coefficient of variation of centered class-mean norms, while NC2a and NC2-ETF quantify simplex equiangularity.",
    "GEO-7": "NC2W applies the corresponding ETF geometry diagnostics to classifier-weight rows.",
    "GEO-8": "NC3 is the Frobenius distance between globally normalized classifier weights and transposed centered class means.",
    "GEO-9": "NC4 is the agreement rate between the affine classifier and nearest ID-training class-mean predictions.",
    "GEO-10": "RankMe is spectral entropy computed from singular values of the uncentered ID-training feature matrix.",
    "GEO-11": "We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum.",
    "GEO-12": "Numerical rank retains covariance eigenvalues above lambda_max p eps64; conditioning uses only that retained spectrum.",
    "GEO-13": "The exploratory spectral slope is an OLS fit of natural log within-class eigenvalues against one-based rank.",
    "GEO-14": "LID uses the maximum-likelihood k-neighbor distance-ratio estimator on exact raw ID-training neighbors, with k=50 primary.",
    "GEO-15": "TwoNN is the DADApy base estimator fitted through the origin after retaining the lowest 90% of log second-to-first-neighbor ratios.",
    "GEO-16": "Class alignment is the class-macro mean pairwise squared distance between L2-normalized same-class features.",
    "GEO-17": "Uniformity is the log mean exponential pair potential at t=2 over three prespecified 100,000-pair Monte Carlo repeats.",
}

IDENTITY_FIELDS = (
    "artifact_key",
    "checkpoint_role",
    "config_hash",
    "detector",
    "ood_dataset",
    "ood_group",
    "class_index",
    "quantile_probability",
    "statistic",
    "query_split",
)

EXPORT_FIELDS = (
    "definition_version",
    "evaluation_git_sha",
    "inventory_hash",
    "dataset_policy_hash",
    "role",
    "optimizer",
    "lr",
    "weight_decay",
    "config_hash",
    "checkpoint_role",
    "artifact_key",
    "formula_id",
    "reporting_tier",
    "direction",
    "fit_split",
    "query_split",
    "transform",
    "detector",
    "ood_dataset",
    "ood_group",
    "class_index",
    "statistic",
    "quantile_probability",
    "sample_count",
)


@dataclass(frozen=True)
class ConfigInfo:
    config_hash: str
    optimizer: str
    lr: float
    weight_decay: float
    labels: tuple[str, ...]
    roles: tuple[str, ...]


class IndexedAggregateRows(list[dict[str, Any]]):
    """List-compatible aggregate rows with a narrow selection index."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        super().__init__(rows)
        self._selection_index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self:
            self._selection_index[
                (row["config_hash"], row["checkpoint_role"], row["artifact_key"])
            ].append(row)

    def selection_candidates(
        self,
        config_hash: str,
        checkpoint_role: str,
        artifact_key: str,
    ) -> Sequence[Mapping[str, Any]]:
        return self._selection_index.get(
            (config_hash, checkpoint_role, artifact_key),
            (),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in IDENTITY_FIELDS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_aggregate(aggregate_dir: Path) -> dict[str, Any]:
    manifest_path = aggregate_dir / "artifact_manifest.json"
    sidecar_path = aggregate_dir / "artifact_manifest.json.sha256"
    _require(manifest_path.is_file(), f"missing {manifest_path}")
    _require(sidecar_path.is_file(), f"missing {sidecar_path}")
    sidecar_parts = sidecar_path.read_text(encoding="utf-8").strip().split()
    _require(len(sidecar_parts) >= 1, "empty artifact manifest sidecar")
    _require(sidecar_parts[0] == _sha256(manifest_path), "manifest sidecar mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest["file_count"] == 4, "aggregate manifest must contain 4 payloads")
    for entry in manifest["files"]:
        path = aggregate_dir / entry["path"]
        _require(path.is_file(), f"missing aggregate payload {entry['path']}")
        _require(path.stat().st_size == entry["size"], f"size mismatch: {entry['path']}")
        _require(_sha256(path) == entry["sha256"], f"hash mismatch: {entry['path']}")

    summary = json.loads((aggregate_dir / "aggregate.json").read_text(encoding="utf-8"))
    expected = {
        "status": "AGGREGATED",
        "definition_version": DEFINITION_VERSION,
        "evaluation_git_sha": EXPECTED_EVALUATION_SHA,
        "inventory_hash": EXPECTED_INVENTORY_HASH,
        "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
        "checkpoint_job_count": 60,
        "training_identity_count": 30,
        "configuration_count": 10,
        "per_checkpoint_scalar_record_count": 95_160,
        "seed_aggregate_record_count": 31_720,
        "successful_aggregate_count": 31_720,
        "non_success_aggregate_count": 0,
        "detector_rank_concordance_record_count": 40,
    }
    for key, value in expected.items():
        _require(summary.get(key) == value, f"aggregate summary mismatch: {key}")
    _require(summary["seed_set"] == [0, 1, 2], "unexpected aggregate seed set")
    _require(summary["checkpoint_roles"] == ["last", "best_val"], "checkpoint roles changed")
    return {"manifest": manifest, "summary": summary}


def load_authorized_configs(inventory_path: Path) -> dict[str, ConfigInfo]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    _require(inventory["inventory_hash"] == EXPECTED_INVENTORY_HASH, "inventory hash changed")
    _require(
        inventory["dataset_policy_hash"] == EXPECTED_DATASET_POLICY_HASH,
        "dataset policy hash changed",
    )
    grouped: dict[str, dict[str, Any]] = {}
    identities = [
        row for row in inventory["training_identities"] if row["disposition"] == "authorized_role"
    ]
    _require(len(identities) == 30, "authorized training identity count must be 30")
    for row in identities:
        config_hash = row["config_hash"]
        optimizer = row["optimizer"]
        candidate = {
            "optimizer": row["optimizer_family"],
            "lr": float(optimizer["lr"]),
            "weight_decay": float(optimizer["weight_decay"]),
            "labels": tuple(sorted(row["labels"])),
            "seeds": set(),
        }
        if config_hash in grouped:
            previous = grouped[config_hash]
            for key in ("optimizer", "lr", "weight_decay", "labels"):
                _require(previous[key] == candidate[key], f"config metadata drift: {config_hash}")
        else:
            grouped[config_hash] = candidate
        grouped[config_hash]["seeds"].add(int(row["training_seed"]))
    _require(len(grouped) == 10, "authorized configuration count must be 10")

    result: dict[str, ConfigInfo] = {}
    observed_roles: dict[tuple[str, str], str] = {}
    for config_hash, row in grouped.items():
        _require(row["seeds"] == {0, 1, 2}, f"seed coverage changed: {config_hash}")
        roles = []
        for label in row["labels"]:
            if label.startswith("role:"):
                _, label_optimizer, role = label.split(":")
                _require(label_optimizer == row["optimizer"], f"role optimizer mismatch: {label}")
                roles.append(role)
                key = (role, label_optimizer)
                _require(key not in observed_roles, f"duplicate role assignment: {key}")
                observed_roles[key] = config_hash
        _require(roles, f"authorized config has no role: {config_hash}")
        result[config_hash] = ConfigInfo(
            config_hash=config_hash,
            optimizer=row["optimizer"],
            lr=row["lr"],
            weight_decay=row["weight_decay"],
            labels=row["labels"],
            roles=tuple(sorted(roles, key=ROLE_ORDER.index)),
        )

    expected_present = {
        (role, optimizer)
        for role in ROLE_ORDER
        for optimizer in OPTIMIZER_ORDER
        if not (role == "C2" and optimizer == "adamw")
    }
    _require(set(observed_roles) == expected_present, "frozen C1-C4 role matrix changed")
    expected_hparams = {
        ("C1", "sgd"): (0.1, 5e-4),
        ("C1", "adam"): (3e-4, 0.0),
        ("C1", "adamw"): (3e-3, 0.1),
        ("C2", "sgd"): (0.3, 1e-4),
        ("C2", "adam"): (3e-4, 1e-3),
        ("C3", "sgd"): (0.1, 0.0),
        ("C3", "adam"): (3e-4, 0.0),
        ("C3", "adamw"): (1e-3, 1e-4),
        ("C4", "sgd"): (0.3, 0.0),
        ("C4", "adam"): (3e-4, 1e-4),
        ("C4", "adamw"): (3e-3, 1e-3),
    }
    for key, expected_pair in expected_hparams.items():
        config = result[observed_roles[key]]
        _require(
            (config.lr, config.weight_decay) == expected_pair,
            f"frozen hyperparameter mismatch: {key}",
        )
    _require(
        observed_roles[("C1", "adam")] == observed_roles[("C3", "adam")],
        "Adam C1/C3 reuse identity changed",
    )
    return result


def validate_records(
    aggregate_dir: Path,
    configs: Mapping[str, ConfigInfo],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    aggregates = _read_jsonl(aggregate_dir / "seed_aggregates.jsonl")
    _require(len(aggregates) == 31_720, "seed aggregate row count changed")
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    detector_set: set[str] = set()
    for row in aggregates:
        _require(row["aggregate_status"] == "success", "non-success seed aggregate")
        _require(row["definition_version"] == DEFINITION_VERSION, "definition version drift")
        _require(row["evaluation_git_sha"] == EXPECTED_EVALUATION_SHA, "evaluation SHA drift")
        _require(row["inventory_hash"] == EXPECTED_INVENTORY_HASH, "inventory identity drift")
        _require(row["dataset_policy_hash"] == EXPECTED_DATASET_POLICY_HASH, "dataset identity drift")
        _require(row["config_hash"] in configs, "pair-only or unknown config entered aggregate")
        _require(row["seed_count"] == 3, "seed aggregate must contain three seeds")
        _require(set(row["seed_values"]) == {"0", "1", "2"}, "seed values changed")
        values = []
        for seed in ("0", "1", "2"):
            seed_row = row["seed_values"][seed]
            _require(seed_row["status"] == "success", "non-success seed value")
            values.append(float(seed_row["value"]))
        expected_mean = float(np.mean(values))
        expected_sd = float(np.std(values, ddof=1))
        _require(math.isclose(row["mean"], expected_mean, rel_tol=1e-12, abs_tol=1e-12), "mean mismatch")
        _require(
            math.isclose(row["sample_sd_ddof1"], expected_sd, rel_tol=1e-12, abs_tol=1e-12),
            "sample SD mismatch",
        )
        identity = _identity(row)
        _require(identity not in lookup, "duplicate seed aggregate identity")
        lookup[identity] = row
        if row.get("detector"):
            detector_set.add(row["detector"])
    _require(detector_set == set(ALL_DETECTORS), "19-detector inventory changed")

    seed0_records: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    seen_seed0: set[tuple[Any, ...]] = set()
    with (aggregate_dir / "per_checkpoint_records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            counts["per_checkpoint"] += 1
            counts[f"checkpoint:{row['checkpoint_role']}"] += 1
            counts[f"seed:{row['training_seed']}"] += 1
            _require(row["status"] == "success", "non-success per-checkpoint record")
            _require(row["config_hash"] in configs, "unknown per-checkpoint config")
            if row["training_seed"] == 0:
                identity = _identity(row)
                _require(identity not in seen_seed0, "duplicate seed-0 scalar identity")
                seen_seed0.add(identity)
                aggregate = lookup.get(identity)
                _require(aggregate is not None, "seed-0 scalar missing aggregate row")
                seed_value = aggregate["seed_values"]["0"]
                _require(
                    math.isclose(float(row["value"]), float(seed_value["value"]), rel_tol=1e-12, abs_tol=1e-12),
                    "seed-0 value mismatch",
                )
                _require(row["checkpoint_sha256"] == seed_value["checkpoint_sha256"], "seed-0 checkpoint mismatch")
                seed0_records.append(row)
    _require(counts["per_checkpoint"] == 95_160, "per-checkpoint row count changed")
    _require(counts["checkpoint:last"] == 47_580, "last row count changed")
    _require(counts["checkpoint:best_val"] == 47_580, "best_val row count changed")
    for seed in (0, 1, 2):
        _require(counts[f"seed:{seed}"] == 31_720, f"seed {seed} row count changed")
    _require(len(seed0_records) == 31_720, "seed-0 scalar count changed")
    _require(len(seen_seed0) == len(lookup), "seed-0 and aggregate identities differ")
    return aggregates, seed0_records, dict(counts)


def _config_sort_key(config: ConfigInfo) -> tuple[int, float, float, str]:
    return (
        OPTIMIZER_ORDER.index(config.optimizer),
        config.lr,
        config.weight_decay,
        config.config_hash,
    )


def _role_sort_key(role: str, optimizer: str) -> tuple[int, int]:
    return ROLE_ORDER.index(role), OPTIMIZER_ORDER.index(optimizer)


def _role_configs(configs: Mapping[str, ConfigInfo]) -> dict[tuple[str, str], ConfigInfo]:
    result = {}
    for config in configs.values():
        for role in config.roles:
            result[(role, config.optimizer)] = config
    return result


def _expanded_rows(
    rows: Iterable[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
    *,
    aggregate: bool,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        config = configs[row["config_hash"]]
        for role in config.roles:
            output = {field: row.get(field) for field in EXPORT_FIELDS if field not in {"role", "optimizer", "lr", "weight_decay"}}
            output.update(
                {
                    "role": role,
                    "optimizer": config.optimizer,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                }
            )
            if aggregate:
                output.update(
                    {
                        "mean": row["mean"],
                        "sample_sd_ddof1": row["sample_sd_ddof1"],
                        "formatted": _format_mean_sd(row["mean"], row["sample_sd_ddof1"]),
                        "seed0": row["seed_values"]["0"]["value"],
                        "seed1": row["seed_values"]["1"]["value"],
                        "seed2": row["seed_values"]["2"]["value"],
                    }
                )
            else:
                output.update(
                    {
                        "value": row["value"],
                        "training_seed": row["training_seed"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "completed_epoch": row["completed_epoch"],
                    }
                )
            expanded.append(output)
    expanded.sort(
        key=lambda row: (
            *_role_sort_key(row["role"], row["optimizer"]),
            row["artifact_key"],
            str(row.get("detector") or ""),
            str(row.get("ood_dataset") or ""),
            str(row.get("query_split") or ""),
            str(row.get("class_index") if row.get("class_index") is not None else ""),
            str(row.get("statistic") or ""),
            str(row.get("quantile_probability") or ""),
        )
    )
    return expanded


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )


def _format_mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.6f} ± {sd:.6f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "±": r"$\pm$",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _write_latex_longtable(path: Path, caption: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = "l" * len(headers)
    lines = [
        rf"\begin{{longtable}}{{{columns}}}",
        rf"\caption{{{_latex_escape(caption)}}}\\",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        r"\hline",
        r"\endfirsthead",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        r"\hline",
        r"\endhead",
    ]
    lines.extend(" & ".join(_latex_escape(value) for value in row) + r" \\" for row in rows)
    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_lookup(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {_identity(row): row for row in rows}


def _select(
    rows: Iterable[Mapping[str, Any]],
    *,
    config_hash: str,
    checkpoint_role: str,
    artifact_key: str,
    detector: str | None = None,
    ood_dataset: str | None = None,
    statistic: str | None = None,
    query_split: str | None = None,
) -> Mapping[str, Any]:
    matches = []
    candidates = (
        rows.selection_candidates(config_hash, checkpoint_role, artifact_key)
        if isinstance(rows, IndexedAggregateRows)
        else rows
    )
    for row in candidates:
        if row["config_hash"] != config_hash or row["checkpoint_role"] != checkpoint_role:
            continue
        if row["artifact_key"] != artifact_key:
            continue
        if detector is not None and row.get("detector") != detector:
            continue
        if ood_dataset is not None and row.get("ood_dataset") != ood_dataset:
            continue
        if statistic is not None and row.get("statistic") != statistic:
            continue
        if query_split is not None and row.get("query_split") != query_split:
            continue
        matches.append(row)
    _require(len(matches) == 1, f"expected one row, found {len(matches)} for {artifact_key}")
    return matches[0]


def _notion_number(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude >= 1_000_000 or (0.0 < magnitude < 0.0001):
        return f"{value:.6e}"
    return f"{value:.6f}"


def _notion_mean_sd(row: Mapping[str, Any], *, bold: bool = False) -> str:
    text = (
        f"{_notion_number(float(row['mean']))} ± "
        f"{_notion_number(float(row['sample_sd_ddof1']))}"
    )
    return f"**{text}**" if bold else text


def _notion_best_optimizers(
    rows: Mapping[str, Mapping[str, Any]],
    direction: str | None,
) -> set[str]:
    if direction is None or not rows:
        return set()
    values = {optimizer: float(row["mean"]) for optimizer, row in rows.items()}
    target = max(values.values()) if direction == "up" else min(values.values())
    return {
        optimizer
        for optimizer, value in values.items()
        if math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12)
    }


def _notion_id_table(
    aggregates: Sequence[Mapping[str, Any]],
    role_configs: Mapping[tuple[str, str], ConfigInfo],
    role: str,
    metric_specs: Sequence[tuple[str, str, str]],
    *,
    include_hyperparameters: bool,
) -> str:
    headers = ["Optimizer"]
    if include_hyperparameters:
        headers.extend(["LR", "WD"])
    headers.extend(name for name, _, _ in metric_specs)

    metric_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    metric_best: dict[str, set[str]] = {}
    for _, artifact_key, direction in metric_specs:
        by_optimizer = {}
        for optimizer in OPTIMIZER_ORDER:
            config = role_configs.get((role, optimizer))
            if config is not None:
                by_optimizer[optimizer] = _select(
                    aggregates,
                    config_hash=config.config_hash,
                    checkpoint_role="last",
                    artifact_key=artifact_key,
                )
        metric_rows[artifact_key] = by_optimizer
        metric_best[artifact_key] = _notion_best_optimizers(
            by_optimizer,
            direction,
        )

    rows = []
    for optimizer in OPTIMIZER_ORDER:
        config = role_configs.get((role, optimizer))
        output = [OPTIMIZER_DISPLAY[optimizer]]
        if include_hyperparameters:
            output.extend(
                ["NA", "NA"]
                if config is None
                else [f"{config.lr:g}", f"{config.weight_decay:g}"]
            )
        if config is None:
            output.extend("NA — protocol absent" for _ in metric_specs)
        else:
            for _, artifact_key, _ in metric_specs:
                output.append(
                    _notion_mean_sd(
                        metric_rows[artifact_key][optimizer],
                        bold=optimizer in metric_best[artifact_key],
                    )
                )
        rows.append(output)
    return _markdown_table(headers, rows).rstrip()


def _notion_geometry_table(
    aggregates: Sequence[Mapping[str, Any]],
    role_configs: Mapping[tuple[str, str], ConfigInfo],
    role: str,
    metric_specs: Sequence[tuple[str, GeometrySpec, str | None]],
) -> str:
    headers = ["Optimizer", *(name for name, _, _ in metric_specs)]
    metric_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    metric_best: dict[str, set[str]] = {}
    for name, spec, direction in metric_specs:
        by_optimizer = {}
        for optimizer in OPTIMIZER_ORDER:
            config = role_configs.get((role, optimizer))
            if config is not None:
                by_optimizer[optimizer] = _select(
                    aggregates,
                    config_hash=config.config_hash,
                    checkpoint_role="last",
                    artifact_key=spec.artifact_key,
                    statistic=spec.statistic,
                    query_split=spec.query_split,
                )
        metric_rows[name] = by_optimizer
        metric_best[name] = _notion_best_optimizers(by_optimizer, direction)

    rows = []
    for optimizer in OPTIMIZER_ORDER:
        config = role_configs.get((role, optimizer))
        output = [OPTIMIZER_DISPLAY[optimizer]]
        if config is None:
            output.extend("NA — protocol absent" for _ in metric_specs)
        else:
            for name, _, _ in metric_specs:
                output.append(
                    _notion_mean_sd(
                        metric_rows[name][optimizer],
                        bold=optimizer in metric_best[name],
                    )
                )
        rows.append(output)
    return _markdown_table(headers, rows).rstrip()


def _notion_ood_table(
    aggregates: Sequence[Mapping[str, Any]],
    role_configs: Mapping[tuple[str, str], ConfigInfo],
    role: str,
    dataset: str,
    detector_specs: Sequence[tuple[str, str]],
) -> str:
    headers = ["Optimizer", *(name for name, _ in detector_specs)]
    detector_rows: dict[str, dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    detector_best: dict[str, tuple[set[str], set[str]]] = {}
    for _, detector in detector_specs:
        by_optimizer = {}
        auroc_rows = {}
        fpr_rows = {}
        for optimizer in OPTIMIZER_ORDER:
            config = role_configs.get((role, optimizer))
            if config is None:
                continue
            auroc = _select(
                aggregates,
                config_hash=config.config_hash,
                checkpoint_role="last",
                artifact_key="ood_metric/auroc_id_positive",
                detector=detector,
                ood_dataset=dataset,
            )
            fpr = _select(
                aggregates,
                config_hash=config.config_hash,
                checkpoint_role="last",
                artifact_key="ood_metric/fpr95_id_tpr",
                detector=detector,
                ood_dataset=dataset,
            )
            by_optimizer[optimizer] = (auroc, fpr)
            auroc_rows[optimizer] = auroc
            fpr_rows[optimizer] = fpr
        detector_rows[detector] = by_optimizer
        detector_best[detector] = (
            _notion_best_optimizers(auroc_rows, "up"),
            _notion_best_optimizers(fpr_rows, "down"),
        )

    rows = []
    for optimizer in OPTIMIZER_ORDER:
        config = role_configs.get((role, optimizer))
        output = [OPTIMIZER_DISPLAY[optimizer]]
        if config is None:
            output.extend("NA — protocol absent" for _ in detector_specs)
        else:
            for _, detector in detector_specs:
                auroc, fpr = detector_rows[detector][optimizer]
                best_auroc, best_fpr = detector_best[detector]
                output.append(
                    f"{_notion_mean_sd(auroc, bold=optimizer in best_auroc)} / "
                    f"{_notion_mean_sd(fpr, bold=optimizer in best_fpr)}"
                )
        rows.append(output)
    return _markdown_table(headers, rows).rstrip()


def build_notion_primary_markdown(
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
) -> str:
    role_configs = _role_configs(configs)
    lines = [
        "# C1–C4 optimizer 비교 — Notion용 primary 결과",
        "",
        "## 문서 범위와 읽는 법",
        "",
        "이 문서는 WRN-28-10 × CIFAR-10 Metric Contract v1.2의 3-seed 결과를 C1부터 C4까지 같은 형식으로 비교한다. 단일 optimizer 우승을 주장하기보다 ID 성능, representation geometry, OOD detector 성능이 함께 어떻게 달라지는지 확인하기 위한 표 중심 문서다.",
        "",
        "- 모든 값은 seed 0, 1, 2의 산술평균 ± sample standard deviation이다.",
        "- confirmatory primary인 last.pt만 사용한다. best_val.pt control은 섞지 않는다.",
        "- ↑는 클수록, ↓는 작을수록 해당 지표상 유리하다는 뜻이다.",
        "- 굵은 값은 같은 C 안에서 방향 기준으로 가장 좋은 optimizer의 기술통계다. 통계적 유의성을 의미하지 않는다.",
        "- 방향을 지정하지 않은 geometry 지표는 우열을 굵게 표시하지 않는다.",
        "- OOD cell은 AUROC ↑ / FPR@95 ↓ 순서다. ID는 positive class이며 FPR@95는 ID score의 linear 5th percentile threshold를 사용한다.",
        "- OOD에는 canonical 11 detectors만 포함한다. Appendix detector variants와 AUPR-In/Out은 이 문서에서 제외한다.",
        "- C2 AdamW는 protocol-defined absent이므로 임의로 채우지 않는다.",
        "- Adam C1과 C3은 동일한 scientific configuration이다. 역할별 비교를 위해 두 section에 모두 표시한다.",
        "",
        "## Metric 범위",
        "",
        "ID 성능은 frozen CIFAR-10 id_test 10,000장에 대해 계산했다. Geometry는 NC4를 제외하고 raw id_train feature를 사용하며, NC4는 id_test에서 affine classifier와 nearest class mean의 agreement를 측정한다. Temperature scaling은 id_validation에서 fit한 뒤 id_test에 적용한다. Temperature-scaled logits는 OOD score에 사용하지 않는다.",
        "",
    ]

    for role in ROLE_ORDER:
        lines.extend(
            [
                f"## {role}",
                "",
                "### 학습 설정과 ID 예측 성능",
                "",
                _notion_id_table(
                    aggregates,
                    role_configs,
                    role,
                    NOTION_ID_PREDICTIVE_METRICS,
                    include_hyperparameters=True,
                ),
                "",
                "### ID failure ranking",
                "",
                _notion_id_table(
                    aggregates,
                    role_configs,
                    role,
                    NOTION_ID_FAILURE_METRICS,
                    include_hyperparameters=False,
                ),
                "",
                "### Representation geometry — collapse·prototype",
                "",
                _notion_geometry_table(
                    aggregates,
                    role_configs,
                    role,
                    NOTION_GEOMETRY_COLLAPSE,
                ),
                "",
                "### Representation geometry — spectrum·dimension",
                "",
                _notion_geometry_table(
                    aggregates,
                    role_configs,
                    role,
                    NOTION_GEOMETRY_SPECTRUM,
                ),
                "",
                "### OOD dataset별 canonical detector 성능",
                "",
                "아래 모든 cell은 AUROC ↑ / FPR@95 ↓ 순서다.",
                "",
            ]
        )
        for dataset in OOD_DATASETS:
            lines.extend([f"#### {OOD_DISPLAY[dataset]}", ""])
            for group_name, detector_specs in NOTION_DETECTOR_GROUPS:
                lines.extend(
                    [
                        f"**{group_name}**",
                        "",
                        _notion_ood_table(
                            aggregates,
                            role_configs,
                            role,
                            dataset,
                            detector_specs,
                        ),
                        "",
                    ]
                )

    lines.extend(
        [
            "## 해석 경계와 다음 단계",
            "",
            "이 표들은 n=3 descriptive comparison이다. Optimizer causality, 통계적 우월성, universal best detector를 확립하지 않는다. OOD dataset 간 값은 합치지 않았고 각 dataset 안에서만 읽어야 한다.",
            "",
            "다음 단계에서는 이 primary 3-seed 비교와 분리하여 seed=0 전체 결과를 selection audit, per-checkpoint diagnostic, appendix lookup 중 어떤 구조로 읽을지 결정한다.",
            "",
            "## 재현성 정보",
            "",
            f"- Definition version: {DEFINITION_VERSION}",
            f"- Scientific evaluator SHA: {EXPECTED_EVALUATION_SHA}",
            f"- Inventory hash: {EXPECTED_INVENTORY_HASH}",
            f"- Dataset-policy hash: {EXPECTED_DATASET_POLICY_HASH}",
            "- Checkpoint role: last only",
            "- Seed aggregation: arithmetic mean and sample SD with ddof=1",
            "",
        ]
    )
    return "\n".join(lines)


def write_notion_primary_comparison(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
) -> None:
    path = output_dir / "notion_c1_c4_primary_comparison.md"
    path.write_text(
        build_notion_primary_markdown(aggregates, configs),
        encoding="utf-8",
    )


def _paper_role_rows(
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
    checkpoint_role: str,
    metric_specs: Sequence[tuple[str, str]],
) -> list[list[str]]:
    role_configs = _role_configs(configs)
    rows: list[list[str]] = []
    for role in ROLE_ORDER:
        for optimizer in OPTIMIZER_ORDER:
            config = role_configs.get((role, optimizer))
            if config is None:
                rows.append([role, optimizer.upper(), "NA — protocol absent", *("—" for _ in metric_specs)])
                continue
            values = []
            for _, artifact_key in metric_specs:
                row = _select(
                    aggregates,
                    config_hash=config.config_hash,
                    checkpoint_role=checkpoint_role,
                    artifact_key=artifact_key,
                )
                values.append(_format_mean_sd(row["mean"], row["sample_sd_ddof1"]))
            rows.append([role, optimizer.upper(), f"{config.lr:g} / {config.weight_decay:g}", *values])
    return rows


def write_role_and_id_tables(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
) -> None:
    role_configs = _role_configs(configs)
    role_rows = []
    for role in ROLE_ORDER:
        for optimizer in OPTIMIZER_ORDER:
            config = role_configs.get((role, optimizer))
            if config is None:
                role_rows.append(
                    {
                        "role": role,
                        "optimizer": optimizer,
                        "lr": None,
                        "weight_decay": None,
                        "config_hash": None,
                        "status": "protocol_defined_absent",
                    }
                )
            else:
                role_rows.append(
                    {
                        "role": role,
                        "optimizer": optimizer,
                        "lr": config.lr,
                        "weight_decay": config.weight_decay,
                        "config_hash": config.config_hash,
                        "status": "selected",
                    }
                )
    _write_csv(output_dir / "tables/role_hyperparameters.csv", role_rows)
    role_md = _markdown_table(
        ["Role", "Optimizer", "LR", "WD", "Status"],
        [
            [
                row["role"],
                row["optimizer"].upper(),
                "—" if row["lr"] is None else f"{row['lr']:g}",
                "—" if row["weight_decay"] is None else f"{row['weight_decay']:g}",
                row["status"],
            ]
            for row in role_rows
        ],
    )
    (output_dir / "tables/role_hyperparameters.md").write_text(role_md, encoding="utf-8")

    for checkpoint_role in CHECKPOINT_ORDER:
        paper_rows = _paper_role_rows(aggregates, configs, checkpoint_role, ID_PAPER_METRICS)
        headers = ["Role", "Optimizer", "LR / WD", *(name for name, _ in ID_PAPER_METRICS)]
        stem = output_dir / f"tables/id/paper_{checkpoint_role}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".md").write_text(
            _markdown_table(headers, paper_rows)
            + "\nCrosswalk: [`methods_crosswalk.md`](../methods_crosswalk.md); "
            + f"checkpoint=`{checkpoint_role}`, query split=`id_test`.\n",
            encoding="utf-8",
        )
        _write_latex_longtable(
            stem.with_suffix(".tex"),
            f"C1-C4 ID metrics ({checkpoint_role}); values are three-seed mean and sample SD.",
            headers,
            paper_rows,
        )


def write_geometry_paper_tables(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
) -> None:
    role_configs = _role_configs(configs)
    reference_config = sorted(configs.values(), key=_config_sort_key)[0]
    headers = [
        "Role",
        "Optimizer",
        "LR / WD",
        "Geometry metric",
        "Formula",
        "Artifact key",
        "Direction",
        "Mean ± sample SD",
    ]
    for checkpoint_role in CHECKPOINT_ORDER:
        rows: list[list[str]] = []
        for role in ROLE_ORDER:
            for optimizer in OPTIMIZER_ORDER:
                config = role_configs.get((role, optimizer))
                for spec in GEOMETRY_PANEL:
                    reference = _select(
                        aggregates,
                        config_hash=reference_config.config_hash,
                        checkpoint_role=checkpoint_role,
                        artifact_key=spec.artifact_key,
                        statistic=spec.statistic,
                        query_split=spec.query_split,
                    )
                    if config is None:
                        lr_wd = "NA — protocol absent"
                        value = "—"
                    else:
                        metric = _select(
                            aggregates,
                            config_hash=config.config_hash,
                            checkpoint_role=checkpoint_role,
                            artifact_key=spec.artifact_key,
                            statistic=spec.statistic,
                            query_split=spec.query_split,
                        )
                        lr_wd = f"{config.lr:g} / {config.weight_decay:g}"
                        value = _format_mean_sd(
                            metric["mean"], metric["sample_sd_ddof1"]
                        )
                    rows.append(
                        [
                            role,
                            optimizer.upper(),
                            lr_wd,
                            spec.name,
                            reference["formula_id"],
                            reference["artifact_key"],
                            reference["direction"],
                            value,
                        ]
                    )
        stem = output_dir / f"tables/geometry/paper_{checkpoint_role}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        crosswalk_note = (
            "\nCrosswalk: [`methods_crosswalk.md`](../methods_crosswalk.md); "
            f"checkpoint=`{checkpoint_role}`, geometry fit/query splits follow each artifact row.\n"
        )
        stem.with_suffix(".md").write_text(
            _markdown_table(headers, rows) + crosswalk_note,
            encoding="utf-8",
        )
        _write_latex_longtable(
            stem.with_suffix(".tex"),
            f"C1-C4 geometry panel ({checkpoint_role}); three-seed mean and sample SD. Formula and artifact columns cross-reference Card 11.",
            headers,
            rows,
        )


def write_detector_panel(output_dir: Path) -> None:
    rows = [
        {
            "detector": detector,
            "paper_panel": "canonical_main" if detector in CANONICAL_DETECTORS else "exhaustive_appendix",
            "association_panel": detector in CANONICAL_DETECTORS,
        }
        for detector in ALL_DETECTORS
    ]
    _write_csv(output_dir / "tables/detector_panel.csv", rows)
    markdown = _markdown_table(
        ["Detector", "Paper panel", "Association panel"],
        [
            [f"`{row['detector']}`", row["paper_panel"], row["association_panel"]]
            for row in rows
        ],
    )
    (output_dir / "tables/detector_panel.md").write_text(markdown, encoding="utf-8")


def _paired_delta(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, float]:
    deltas = np.asarray(
        [
            float(left["seed_values"][str(seed)]["value"])
            - float(right["seed_values"][str(seed)]["value"])
            for seed in range(3)
        ],
        dtype=np.float64,
    )
    mean = float(np.mean(deltas))
    sample_sd = float(np.std(deltas, ddof=1))
    _require(
        math.isclose(
            mean,
            float(left["mean"]) - float(right["mean"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "paired delta mean does not match aggregate mean difference",
    )
    return {
        "seed0_delta": float(deltas[0]),
        "seed1_delta": float(deltas[1]),
        "seed2_delta": float(deltas[2]),
        "mean_delta": mean,
        "sample_sd_delta_ddof1": sample_sd,
        "formatted_delta": _format_mean_sd(mean, sample_sd),
    }


def write_paired_delta_tables(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
) -> int:
    role_configs = _role_configs(configs)
    optimizer_pairs = (("sgd", "adam"), ("sgd", "adamw"), ("adam", "adamw"))
    total = 0

    def base_row(
        *,
        role: str,
        optimizer_a: str,
        optimizer_b: str,
        config_a: ConfigInfo,
        config_b: ConfigInfo,
        checkpoint_role: str,
        metric_a: Mapping[str, Any],
        metric_name: str,
    ) -> dict[str, Any]:
        return {
            "role": role,
            "optimizer_a": optimizer_a,
            "optimizer_b": optimizer_b,
            "delta_definition": "optimizer_a_minus_optimizer_b_seed_matched",
            "config_hash_a": config_a.config_hash,
            "config_hash_b": config_b.config_hash,
            "lr_a": config_a.lr,
            "weight_decay_a": config_a.weight_decay,
            "lr_b": config_b.lr,
            "weight_decay_b": config_b.weight_decay,
            "checkpoint_role": checkpoint_role,
            "definition_version": DEFINITION_VERSION,
            "evaluation_git_sha": EXPECTED_EVALUATION_SHA,
            "inventory_hash": EXPECTED_INVENTORY_HASH,
            "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
            "analysis_dtype": "float64",
            "metric_name": metric_name,
            "artifact_key": metric_a["artifact_key"],
            "formula_id": metric_a["formula_id"],
            "reporting_tier": metric_a["reporting_tier"],
            "direction": metric_a["direction"],
            "fit_split": metric_a.get("fit_split"),
            "query_split": metric_a.get("query_split"),
            "transform": metric_a.get("transform"),
        }

    for checkpoint_role in CHECKPOINT_ORDER:
        id_rows: list[dict[str, Any]] = []
        geometry_rows: list[dict[str, Any]] = []
        for role in ROLE_ORDER:
            for optimizer_a, optimizer_b in optimizer_pairs:
                config_a = role_configs.get((role, optimizer_a))
                config_b = role_configs.get((role, optimizer_b))
                if config_a is None or config_b is None:
                    continue
                for metric_name, artifact_key in ID_PAPER_METRICS:
                    left = _select(
                        aggregates,
                        config_hash=config_a.config_hash,
                        checkpoint_role=checkpoint_role,
                        artifact_key=artifact_key,
                    )
                    right = _select(
                        aggregates,
                        config_hash=config_b.config_hash,
                        checkpoint_role=checkpoint_role,
                        artifact_key=artifact_key,
                    )
                    id_rows.append(
                        {
                            **base_row(
                                role=role,
                                optimizer_a=optimizer_a,
                                optimizer_b=optimizer_b,
                                config_a=config_a,
                                config_b=config_b,
                                checkpoint_role=checkpoint_role,
                                metric_a=left,
                                metric_name=metric_name,
                            ),
                            **_paired_delta(left, right),
                        }
                    )
                for spec in GEOMETRY_PANEL:
                    left = _select(
                        aggregates,
                        config_hash=config_a.config_hash,
                        checkpoint_role=checkpoint_role,
                        artifact_key=spec.artifact_key,
                        statistic=spec.statistic,
                        query_split=spec.query_split,
                    )
                    right = _select(
                        aggregates,
                        config_hash=config_b.config_hash,
                        checkpoint_role=checkpoint_role,
                        artifact_key=spec.artifact_key,
                        statistic=spec.statistic,
                        query_split=spec.query_split,
                    )
                    geometry_rows.append(
                        {
                            **base_row(
                                role=role,
                                optimizer_a=optimizer_a,
                                optimizer_b=optimizer_b,
                                config_a=config_a,
                                config_b=config_b,
                                checkpoint_role=checkpoint_role,
                                metric_a=left,
                                metric_name=spec.name,
                            ),
                            "statistic": spec.statistic,
                            **_paired_delta(left, right),
                        }
                    )
        _write_csv(output_dir / f"tables/deltas/id/{checkpoint_role}.csv", id_rows)
        _write_csv(
            output_dir / f"tables/deltas/geometry/{checkpoint_role}.csv",
            geometry_rows,
        )
        total += len(id_rows) + len(geometry_rows)

        for dataset in OOD_DATASETS:
            ood_rows: list[dict[str, Any]] = []
            for role in ROLE_ORDER:
                for optimizer_a, optimizer_b in optimizer_pairs:
                    config_a = role_configs.get((role, optimizer_a))
                    config_b = role_configs.get((role, optimizer_b))
                    if config_a is None or config_b is None:
                        continue
                    for detector in CANONICAL_DETECTORS:
                        for metric_name, artifact_key in OOD_TABLE_METRICS:
                            left = _select(
                                aggregates,
                                config_hash=config_a.config_hash,
                                checkpoint_role=checkpoint_role,
                                artifact_key=artifact_key,
                                detector=detector,
                                ood_dataset=dataset,
                            )
                            right = _select(
                                aggregates,
                                config_hash=config_b.config_hash,
                                checkpoint_role=checkpoint_role,
                                artifact_key=artifact_key,
                                detector=detector,
                                ood_dataset=dataset,
                            )
                            ood_rows.append(
                                {
                                    **base_row(
                                        role=role,
                                        optimizer_a=optimizer_a,
                                        optimizer_b=optimizer_b,
                                        config_a=config_a,
                                        config_b=config_b,
                                        checkpoint_role=checkpoint_role,
                                        metric_a=left,
                                        metric_name=metric_name,
                                    ),
                                    "detector": detector,
                                    "ood_dataset": dataset,
                                    **_paired_delta(left, right),
                                }
                            )
            _write_csv(
                output_dir / f"tables/deltas/ood/{dataset}/{checkpoint_role}.csv",
                ood_rows,
            )
            total += len(ood_rows)
    return total


def write_exhaustive_tables(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    seed0_records: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    families = {
        "id": lambda row: row["artifact_key"].startswith("id/"),
        "geometry": lambda row: row["artifact_key"].startswith("geometry/"),
    }
    for family, predicate in families.items():
        for checkpoint_role in CHECKPOINT_ORDER:
            aggregate_rows = [row for row in aggregates if row["checkpoint_role"] == checkpoint_role and predicate(row)]
            seed_rows = [row for row in seed0_records if row["checkpoint_role"] == checkpoint_role and predicate(row)]
            expanded_aggregate = _expanded_rows(aggregate_rows, configs, aggregate=True)
            expanded_seed0 = _expanded_rows(seed_rows, configs, aggregate=False)
            _write_csv(output_dir / f"tables/{family}/all_three_seed_{checkpoint_role}.csv", expanded_aggregate)
            _write_csv(output_dir / f"tables/{family}/all_seed0_{checkpoint_role}.csv", expanded_seed0)
            counts[f"{family}:three_seed:{checkpoint_role}"] = len(expanded_aggregate)
            counts[f"{family}:seed0:{checkpoint_role}"] = len(expanded_seed0)

    for dataset in OOD_DATASETS:
        for checkpoint_role in CHECKPOINT_ORDER:
            aggregate_rows = [
                row
                for row in aggregates
                if row["checkpoint_role"] == checkpoint_role and row.get("ood_dataset") == dataset
            ]
            seed_rows = [
                row
                for row in seed0_records
                if row["checkpoint_role"] == checkpoint_role and row.get("ood_dataset") == dataset
            ]
            expanded_aggregate = _expanded_rows(aggregate_rows, configs, aggregate=True)
            expanded_seed0 = _expanded_rows(seed_rows, configs, aggregate=False)
            base = output_dir / f"tables/ood/{dataset}"
            _write_csv(base / f"all_three_seed_{checkpoint_role}.csv", expanded_aggregate)
            _write_csv(base / f"all_seed0_{checkpoint_role}.csv", expanded_seed0)
            counts[f"ood:{dataset}:three_seed:{checkpoint_role}"] = len(expanded_aggregate)
            counts[f"ood:{dataset}:seed0:{checkpoint_role}"] = len(expanded_seed0)

    macro_aggregates = [
        row
        for row in aggregates
        if row["artifact_key"].startswith("ood_metric/") and row.get("ood_dataset") is None
    ]
    macro_seed0 = [
        row
        for row in seed0_records
        if row["artifact_key"].startswith("ood_metric/") and row.get("ood_dataset") is None
    ]
    _write_csv(
        output_dir / "tables/cross_dataset_macro_audit/all_three_seed.csv",
        _expanded_rows(macro_aggregates, configs, aggregate=True),
    )
    _write_csv(
        output_dir / "tables/cross_dataset_macro_audit/all_seed0.csv",
        _expanded_rows(macro_seed0, configs, aggregate=False),
    )
    counts["macro:three_seed"] = len(macro_aggregates)
    counts["macro:seed0"] = len(macro_seed0)
    return counts


def write_ood_paper_tables(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
) -> None:
    role_configs = _role_configs(configs)
    for dataset in OOD_DATASETS:
        for checkpoint_role in CHECKPOINT_ORDER:
            rows: list[dict[str, Any]] = []
            latex_rows: list[list[str]] = []
            for role in ROLE_ORDER:
                for optimizer in OPTIMIZER_ORDER:
                    config = role_configs.get((role, optimizer))
                    if config is None:
                        continue
                    for detector in CANONICAL_DETECTORS:
                        output = {
                            "definition_version": DEFINITION_VERSION,
                            "evaluation_git_sha": EXPECTED_EVALUATION_SHA,
                            "inventory_hash": EXPECTED_INVENTORY_HASH,
                            "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
                            "role": role,
                            "optimizer": optimizer,
                            "lr": config.lr,
                            "weight_decay": config.weight_decay,
                            "config_hash": config.config_hash,
                            "detector": detector,
                            "ood_dataset": dataset,
                            "checkpoint_role": checkpoint_role,
                        }
                        formatted = []
                        for name, artifact_key in OOD_TABLE_METRICS:
                            metric = _select(
                                aggregates,
                                config_hash=config.config_hash,
                                checkpoint_role=checkpoint_role,
                                artifact_key=artifact_key,
                                detector=detector,
                                ood_dataset=dataset,
                            )
                            output[f"{name}_mean"] = metric["mean"]
                            output[f"{name}_sample_sd"] = metric["sample_sd_ddof1"]
                            output[f"{name}_artifact_key"] = metric["artifact_key"]
                            output[f"{name}_formula_id"] = metric["formula_id"]
                            output[f"{name}_direction"] = metric["direction"]
                            output[f"{name}_fit_split"] = metric.get("fit_split")
                            output[f"{name}_query_split"] = metric.get("query_split")
                            output[f"{name}_transform"] = metric.get("transform")
                            formatted.append(_format_mean_sd(metric["mean"], metric["sample_sd_ddof1"]))
                        rows.append(output)
                        latex_rows.append([role, optimizer.upper(), detector, *formatted])
            base = output_dir / f"tables/ood/{dataset}/paper_{checkpoint_role}"
            _write_csv(base.with_suffix(".csv"), rows)
            headers = ["Role", "Optimizer", "Detector", "AUROC", "FPR@95", "AUPR-In", "AUPR-Out"]
            base.with_suffix(".md").write_text(
                _markdown_table(headers, latex_rows)
                + "\nCrosswalk: [`methods_crosswalk.md`](../../methods_crosswalk.md); "
                + f"checkpoint=`{checkpoint_role}`, OOD dataset=`{dataset}`, ID side=`id_test`.\n",
                encoding="utf-8",
            )
            _write_latex_longtable(
                base.with_suffix(".tex"),
                f"{OOD_DISPLAY[dataset]} OOD results ({checkpoint_role}); each value is three-seed mean and sample SD.",
                headers,
                latex_rows,
            )


def _rank_columns(values: np.ndarray) -> np.ndarray:
    return np.asarray(rankdata(values, axis=0, method="average"), dtype=np.float64)


def _corr_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - np.mean(left, axis=0, keepdims=True)
    right_centered = right - np.mean(right, axis=0, keepdims=True)
    numerator = left_centered.T @ right_centered
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=0)[:, None]
        * np.sum(right_centered**2, axis=0)[None, :]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    flat = np.asarray(pvalues, dtype=np.float64).ravel()
    order = np.argsort(flat, kind="mergesort")
    ranked = flat[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.reshape(pvalues.shape)


def _bootstrap_correlations(
    x_seed: np.ndarray,
    y_seed: np.ndarray,
    bootstrap_configs: np.ndarray,
    *,
    batch_size: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rank = _rank_columns(x_seed)
    y_rank = _rank_columns(y_seed)
    observed = _corr_columns(x_rank, y_rank)
    seed_offsets = np.arange(3, dtype=np.int64)
    boot_indices = (bootstrap_configs[:, :, None] * 3 + seed_offsets).reshape(len(bootstrap_configs), -1)
    samples = np.empty(
        (len(bootstrap_configs), x_seed.shape[1], y_seed.shape[1]),
        dtype=np.float64,
    )
    for start in range(0, len(bootstrap_configs), batch_size):
        stop = min(start + batch_size, len(bootstrap_configs))
        indices = boot_indices[start:stop]
        # A block-bootstrap replicate can repeat configurations. Re-rank the
        # resampled raw values inside each replicate so this remains an exact
        # Spearman statistic, including the ties created by repeated blocks.
        left = np.asarray(rankdata(x_seed[indices], axis=1, method="average"), dtype=np.float64)
        right = np.asarray(rankdata(y_seed[indices], axis=1, method="average"), dtype=np.float64)
        left -= np.mean(left, axis=1, keepdims=True)
        right -= np.mean(right, axis=1, keepdims=True)
        numerator = np.einsum("bni,bnj->bij", left, right, optimize=True)
        denominator = np.sqrt(
            np.sum(left**2, axis=1)[:, :, None]
            * np.sum(right**2, axis=1)[:, None, :]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            samples[start:stop] = numerator / denominator
    low = np.nanquantile(samples, 0.025, axis=0)
    high = np.nanquantile(samples, 0.975, axis=0)
    return observed, low, high


def _permutation_pvalues(
    x_config: np.ndarray,
    y_config: np.ndarray,
    permutations: np.ndarray,
    observed: np.ndarray,
    *,
    batch_size: int = 1000,
) -> np.ndarray:
    x_rank = _rank_columns(x_config)
    y_rank = _rank_columns(y_config)
    x_rank -= np.mean(x_rank, axis=0, keepdims=True)
    y_rank -= np.mean(y_rank, axis=0, keepdims=True)
    x_norm = np.sqrt(np.sum(x_rank**2, axis=0))
    y_norm = np.sqrt(np.sum(y_rank**2, axis=0))
    extreme = np.zeros_like(observed, dtype=np.int64)
    for start in range(0, len(permutations), batch_size):
        stop = min(start + batch_size, len(permutations))
        permuted = y_rank[permutations[start:stop]]
        correlations = np.einsum("ni,bnj->bij", x_rank, permuted, optimize=True)
        correlations /= x_norm[None, :, None] * y_norm[None, None, :]
        extreme += np.sum(np.abs(correlations) >= np.abs(observed)[None, :, :] - 1e-15, axis=0)
    return (extreme + 1.0) / (len(permutations) + 1.0)


def _geometry_matrices(
    aggregates: Sequence[Mapping[str, Any]],
    configs: Sequence[ConfigInfo],
    checkpoint_role: str,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty((len(configs), len(GEOMETRY_PANEL)), dtype=np.float64)
    seeds = np.empty((len(configs) * 3, len(GEOMETRY_PANEL)), dtype=np.float64)
    for config_index, config in enumerate(configs):
        for metric_index, spec in enumerate(GEOMETRY_PANEL):
            row = _select(
                aggregates,
                config_hash=config.config_hash,
                checkpoint_role=checkpoint_role,
                artifact_key=spec.artifact_key,
                statistic=spec.statistic,
                query_split=spec.query_split,
            )
            means[config_index, metric_index] = row["mean"]
            for seed in range(3):
                seeds[config_index * 3 + seed, metric_index] = row["seed_values"][str(seed)]["value"]
    return means, seeds


def _ood_matrices(
    aggregates: Sequence[Mapping[str, Any]],
    configs: Sequence[ConfigInfo],
    checkpoint_role: str,
    dataset: str,
    artifact_key: str,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty((len(configs), len(CANONICAL_DETECTORS)), dtype=np.float64)
    seeds = np.empty((len(configs) * 3, len(CANONICAL_DETECTORS)), dtype=np.float64)
    for config_index, config in enumerate(configs):
        for detector_index, detector in enumerate(CANONICAL_DETECTORS):
            row = _select(
                aggregates,
                config_hash=config.config_hash,
                checkpoint_role=checkpoint_role,
                artifact_key=artifact_key,
                detector=detector,
                ood_dataset=dataset,
            )
            means[config_index, detector_index] = sign * row["mean"]
            for seed in range(3):
                seeds[config_index * 3 + seed, detector_index] = sign * row["seed_values"][str(seed)]["value"]
    return means, seeds


def compute_associations(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
    *,
    resamples: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray]]:
    _require(resamples >= 100, "resamples must be at least 100")
    config_order = sorted(configs.values(), key=_config_sort_key)
    rng = np.random.default_rng(random_seed)
    bootstrap_configs = rng.integers(0, len(config_order), size=(resamples, len(config_order)), endpoint=False)
    permutations = np.empty((resamples, len(config_order)), dtype=np.int64)
    for index in range(resamples):
        permutations[index] = rng.permutation(len(config_order))

    records: list[dict[str, Any]] = []
    heatmaps: dict[tuple[str, str, str], np.ndarray] = {}
    for checkpoint_role in CHECKPOINT_ORDER:
        geometry_means, geometry_seeds = _geometry_matrices(aggregates, config_order, checkpoint_role)
        geometry_rank = _rank_columns(geometry_means)
        for dataset in OOD_DATASETS:
            for endpoint, (artifact_key, sign) in ASSOCIATION_ENDPOINTS.items():
                ood_means, ood_seeds = _ood_matrices(
                    aggregates,
                    config_order,
                    checkpoint_role,
                    dataset,
                    artifact_key,
                    sign,
                )
                config_rho = _corr_columns(geometry_rank, _rank_columns(ood_means))
                seed_rho, ci_low, ci_high = _bootstrap_correlations(
                    geometry_seeds,
                    ood_seeds,
                    bootstrap_configs,
                )
                pvalue = _permutation_pvalues(
                    geometry_means,
                    ood_means,
                    permutations,
                    config_rho,
                )
                qvalue = _bh_adjust(pvalue)
                heatmaps[(checkpoint_role, dataset, endpoint)] = config_rho
                block_rows = []
                for geometry_index, geometry in enumerate(GEOMETRY_PANEL):
                    geometry_reference = _select(
                        aggregates,
                        config_hash=config_order[0].config_hash,
                        checkpoint_role=checkpoint_role,
                        artifact_key=geometry.artifact_key,
                        statistic=geometry.statistic,
                        query_split=geometry.query_split,
                    )
                    for detector_index, detector in enumerate(CANONICAL_DETECTORS):
                        ood_reference = _select(
                            aggregates,
                            config_hash=config_order[0].config_hash,
                            checkpoint_role=checkpoint_role,
                            artifact_key=artifact_key,
                            detector=detector,
                            ood_dataset=dataset,
                        )
                        row = {
                            "definition_version": DEFINITION_VERSION,
                            "evaluation_git_sha": EXPECTED_EVALUATION_SHA,
                            "inventory_hash": EXPECTED_INVENTORY_HASH,
                            "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
                            "checkpoint_role": checkpoint_role,
                            "ood_dataset": dataset,
                            "endpoint": endpoint,
                            "ood_artifact_key": artifact_key,
                            "performance_transform": "identity" if sign > 0 else "negate",
                            "geometry_name": geometry.name,
                            "geometry_artifact_key": geometry.artifact_key,
                            "geometry_formula_id": geometry_reference["formula_id"],
                            "geometry_direction": geometry_reference["direction"],
                            "geometry_fit_split": geometry_reference.get("fit_split"),
                            "geometry_transform": geometry_reference.get("transform"),
                            "geometry_statistic": geometry.statistic,
                            "geometry_query_split": geometry.query_split,
                            "detector": detector,
                            "ood_formula_id": ood_reference["formula_id"],
                            "ood_direction": ood_reference["direction"],
                            "ood_fit_split": ood_reference.get("fit_split"),
                            "ood_query_split": ood_reference.get("query_split"),
                            "ood_transform": ood_reference.get("transform"),
                            "config_mean_spearman_rho": config_rho[geometry_index, detector_index],
                            "seed_level_clustered_rho": seed_rho[geometry_index, detector_index],
                            "cluster_bootstrap_ci_low": ci_low[geometry_index, detector_index],
                            "cluster_bootstrap_ci_high": ci_high[geometry_index, detector_index],
                            "permutation_pvalue": pvalue[geometry_index, detector_index],
                            "bh_fdr_qvalue": qvalue[geometry_index, detector_index],
                            "config_count": 10,
                            "seed_observation_count": 30,
                            "resamples": resamples,
                            "random_seed": random_seed,
                            "analysis_dtype": "float64",
                            "rank_tie_method": "average",
                            "bootstrap_ci_method": "percentile_2.5_97.5",
                            "permutation_pvalue_rule": "(extreme+1)/(resamples+1)",
                            "bh_fdr_scope": "dataset_checkpoint_endpoint_16x11",
                            "interpretation": "exploratory_association_not_causal",
                        }
                        records.append(row)
                        block_rows.append(row)
                _write_csv(
                    output_dir / f"tables/associations/{checkpoint_role}/{dataset}_{endpoint}.csv",
                    block_rows,
                )
    _require(
        len(records)
        == 2 * 6 * len(ASSOCIATION_ENDPOINTS) * len(GEOMETRY_PANEL) * len(CANONICAL_DETECTORS),
        "association count changed",
    )
    return records, heatmaps


def write_methods_crosswalk(output_dir: Path, aggregates: Sequence[Mapping[str, Any]]) -> None:
    grouped: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in aggregates:
        key = (row["artifact_key"], row["formula_id"])
        target = grouped.setdefault(
            key,
            {
                "reporting_tier": set(),
                "direction": set(),
                "fit_split": set(),
                "query_split": set(),
                "transform": set(),
            },
        )
        for field in target:
            value = row.get(field)
            if value is not None:
                target[field].add(str(value))
    rows = []
    for (artifact_key, formula_id), values in sorted(grouped.items()):
        rows.append(
            {
                "formula_id": formula_id,
                "artifact_key": artifact_key,
                **{field: ";".join(sorted(entries)) for field, entries in values.items()},
                "methods_sentence": METHOD_SENTENCES.get(
                    formula_id, f"See Reference Card 11, formula {formula_id}."
                ),
                "authoritative_source": "docs/reference_cards/11_metric_contract_v1_2.md",
            }
        )
    _write_csv(output_dir / "tables/methods_crosswalk.csv", rows)
    markdown = _markdown_table(
        ["Formula", "Artifact key", "Direction", "Tier", "Methods wording"],
        [
            [
                row["formula_id"],
                f"`{row['artifact_key']}`",
                row["direction"],
                row["reporting_tier"],
                row["methods_sentence"],
            ]
            for row in rows
        ],
    )
    (output_dir / "tables/methods_crosswalk.md").write_text(markdown, encoding="utf-8")


def write_methods_ready_english(output_dir: Path, *, resamples: int, random_seed: int) -> None:
    text = f"""# Methods-ready English text

## Experimental units and reporting

We analyzed ten unique WRN-28-10 configurations trained on CIFAR-10 with three seeds per configuration. The terminal `last.pt` checkpoint was the confirmatory primary checkpoint, whereas the validation-selected `best_val.pt` checkpoint was analyzed separately as a deployment control. We report arithmetic means and sample standard deviations (`ddof=1`) across seeds 0, 1, and 2. Adam C1 and C3 denote the same scientific configuration and were therefore counted once in cross-configuration association analyses. AdamW C2 was absent by protocol and was not imputed.

## ID performance and representation geometry

ID metrics were evaluated on the frozen 10,000-image CIFAR-10 test split. We report top-1 accuracy, raw-logit negative log-likelihood, 15-bin expected calibration error, raw-MSP misclassification AUROC, and AUGRC in the main descriptive panel. Geometry statistics were fitted on raw ID-training features unless the corresponding Card 11 definition declares another transform. In particular, we report NC1 as $\\operatorname{{Tr}}(\\Sigma_W\\Sigma_B^{{\\dagger}})/K$, using biased within-class and between-class covariance estimators computed from raw ID-training features.

## OOD evaluation

OOD performance was evaluated separately on CIFAR-100, TinyImageNet, MNIST, SVHN, Textures, and Places365; samples were never pooled across datasets. AUROC treated ID as the positive class. FPR@95 used the inclusive threshold $\\tau_{{95}}=\\operatorname{{quantile}}(s_{{ID}},0.05)$ with linear interpolation and was computed as $\\Pr(s_{{OOD}}\\geq\\tau_{{95}})$. Temperature-scaled logits were not used for primary OOD scoring. The main panel contained the 11 prespecified canonical detectors, while complete 19-detector results were retained in the appendix tables.

## Geometry–OOD association analysis

Within each OOD dataset, checkpoint role, endpoint, geometry metric, and canonical detector, we computed Spearman's rank correlation across the ten unique three-seed configuration means. AUROC was the primary association endpoint; FPR@95 was negated so that larger values consistently represented better OOD performance. As a robustness analysis, we computed a separate seed-level Spearman coefficient from 30 observations and obtained a 95% percentile interval using {resamples:,} configuration-block bootstrap replicates. Each bootstrap replicate resampled ten configurations with replacement, retained all three seeds in every selected block, and reranked the resampled raw observations before computing Spearman's coefficient. Two-sided permutation p-values used {resamples:,} permutations of the ten configuration-mean blocks. Benjamini–Hochberg correction was applied separately within each dataset, checkpoint, and endpoint across the 16-by-11 geometry–detector panel. All stochastic analysis used the fixed random seed `{random_seed}`.

AUPR-In and AUPR-Out associations were computed by the same procedure and retained as appendix CSVs; they were not promoted to the AUROC/FPR heatmap panel.

## Interpretation boundary

All geometry–OOD results are exploratory associations rather than causal effects. The small number of unique configurations, selection of the role grid using seed-0 validation results, multiplicity, and common checkpoint provenance limit inferential interpretation. Seed-0 tables are provided for auditing and are not treated as independent confirmatory evidence.

Metric-level canonical sentences and their formula, artifact, split, transform, direction, and reporting-tier identities are provided in [`tables/methods_crosswalk.md`](tables/methods_crosswalk.md).
"""
    (output_dir / "METHODS_READY_EN.md").write_text(text, encoding="utf-8")


def _figure_runtime() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "metric-contract-v1.2-c1-c4"
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required unless --skip-figures is used") from exc
    return plt, LinearSegmentedColormap


def _save_figure(fig: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    common = {"bbox_inches": "tight", "facecolor": "white"}
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", metadata={"Date": None}, **common)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        format="pdf",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "oge metric analysis", "Producer": "Matplotlib"},
        **common,
    )


def write_association_figures(
    output_dir: Path,
    association_rows: Sequence[Mapping[str, Any]],
    heatmaps: Mapping[tuple[str, str, str], np.ndarray],
) -> None:
    plt, LinearSegmentedColormap = _figure_runtime()
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_orange", ["#2F6B8A", "#F7F7F5", "#C7772D"]
    )
    detector_labels = [detector.replace("detector/", "").replace("ood_score/", "") for detector in CANONICAL_DETECTORS]
    for checkpoint_role in CHECKPOINT_ORDER:
        for dataset in OOD_DATASETS:
            for endpoint in HEATMAP_ENDPOINTS:
                matrix = heatmaps[(checkpoint_role, dataset, endpoint)]
                qmatrix = np.empty_like(matrix)
                block = [
                    row
                    for row in association_rows
                    if row["checkpoint_role"] == checkpoint_role
                    and row["ood_dataset"] == dataset
                    and row["endpoint"] == endpoint
                ]
                for i, geometry in enumerate(GEOMETRY_PANEL):
                    for j, detector in enumerate(CANONICAL_DETECTORS):
                        match = [
                            row
                            for row in block
                            if row["geometry_name"] == geometry.name and row["detector"] == detector
                        ]
                        _require(len(match) == 1, "association heatmap row mismatch")
                        qmatrix[i, j] = match[0]["bh_fdr_qvalue"]
                fig, ax = plt.subplots(figsize=(12.5, 9.2))
                fig.subplots_adjust(left=0.18, right=0.87, bottom=0.24, top=0.93)
                image = ax.imshow(matrix, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
                ax.set_xticks(range(len(detector_labels)), detector_labels, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(len(GEOMETRY_PANEL)), [spec.name for spec in GEOMETRY_PANEL], fontsize=8)
                ax.set_title(
                    f"{OOD_DISPLAY[dataset]} geometry–OOD association ({checkpoint_role}, {endpoint})",
                    loc="left",
                    fontsize=13,
                    color="#222222",
                )
                ax.set_xlabel("Canonical detector")
                ax.set_ylabel("Geometry metric")
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        label = f"{matrix[i, j]:.2f}"
                        if qmatrix[i, j] < 0.10:
                            label += "*"
                        ax.text(j, i, label, ha="center", va="center", fontsize=6, color="#202020")
                colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
                colorbar.set_label("Spearman ρ across 10 config means")
                fig.text(
                    0.01,
                    0.015,
                    "Dataset-specific; * BH-FDR q<0.10 within this dataset/endpoint. Exploratory, not causal. FPR uses −FPR@95.",
                    fontsize=8,
                    color="#444444",
                )
                _save_figure(
                    fig,
                    output_dir / f"figures/associations/{checkpoint_role}/{dataset}_{endpoint}",
                )
                plt.close(fig)


def write_role_figures(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
) -> None:
    plt, _ = _figure_runtime()
    colors = {"sgd": "#2F6B8A", "adam": "#C7772D", "adamw": "#92704A"}
    markers = {"sgd": "o", "adam": "s", "adamw": "^"}
    metrics = (
        ("ID accuracy", "id/accuracy", None, None),
        ("ID NLL", "id/nll_raw", None, None),
        ("NC1", "geometry/nc1_pinv", None, None),
        ("RankMe", "geometry/rankme_uncentered", None, None),
    )
    role_configs = _role_configs(configs)
    offsets = {"sgd": -0.2, "adam": 0.0, "adamw": 0.2}
    for checkpoint_role in CHECKPOINT_ORDER:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for ax, (title, artifact_key, statistic, query_split) in zip(axes.ravel(), metrics, strict=True):
            for optimizer in OPTIMIZER_ORDER:
                xs, means, sds = [], [], []
                for role_index, role in enumerate(ROLE_ORDER):
                    config = role_configs.get((role, optimizer))
                    if config is None:
                        continue
                    row = _select(
                        aggregates,
                        config_hash=config.config_hash,
                        checkpoint_role=checkpoint_role,
                        artifact_key=artifact_key,
                        statistic=statistic,
                        query_split=query_split,
                    )
                    x = role_index + offsets[optimizer]
                    xs.append(x)
                    means.append(row["mean"])
                    sds.append(row["sample_sd_ddof1"])
                    seed_values = [row["seed_values"][str(seed)]["value"] for seed in range(3)]
                    ax.scatter(
                        [x - 0.035, x, x + 0.035],
                        seed_values,
                        color=colors[optimizer],
                        marker=markers[optimizer],
                        s=16,
                        alpha=0.45,
                        linewidths=0,
                    )
                ax.errorbar(
                    xs,
                    means,
                    yerr=sds,
                    color=colors[optimizer],
                    marker=markers[optimizer],
                    linestyle="none",
                    capsize=3,
                    label=optimizer.upper(),
                )
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_xticks(range(4), ROLE_ORDER)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(frameon=False, ncol=3, loc="best")
        fig.suptitle(
            f"C1–C4 three-seed comparison ({checkpoint_role})",
            x=0.01,
            ha="left",
            fontsize=14,
        )
        fig.text(
            0.01,
            0.005,
            "Markers show seed values; center and whiskers show mean ± sample SD (n=3), not a confidence interval. AdamW C2 is absent by protocol.",
            fontsize=8,
            color="#444444",
        )
        _save_figure(fig, output_dir / f"figures/role_summary_{checkpoint_role}")
        plt.close(fig)


def write_chart_map(output_dir: Path) -> None:
    lines = [
        "# Chart map",
        "",
        "| Artifact | Formula and artifact crosswalk | Checkpoint / split / direction | Form | Claim boundary |",
        "| --- | --- | --- | --- | --- |",
        "| `role_summary_{last,best_val}` | ID-1 `id/accuracy`; ID-2 `id/nll_raw`; GEO-5 `geometry/nc1_pinv`; GEO-10 `geometry/rankme_uncentered` | file-specific checkpoint; `id_test` or `id_train`; see Card 11 direction | Faceted seed dots and mean ± sample-SD interval | Descriptive n=3; no significance claim |",
        "| `associations/<checkpoint>/<dataset>_<endpoint>` | geometry formula/artifact × OOD-1 AUROC or OOD-4 FPR; exact row mapping in the same-stem CSV | file-specific checkpoint and OOD dataset; AUROC up, FPR negated to performance-up | Dataset-specific 10-config Spearman heatmap | Exploratory association; no dataset pooling or causal claim |",
        "",
        "Card 11 Methods wording for every formula/artifact pair is in [`tables/methods_crosswalk.md`](tables/methods_crosswalk.md).",
        "",
    ]
    (output_dir / "chart_map.md").write_text("\n".join(lines), encoding="utf-8")


def _find_record(
    aggregates: Sequence[Mapping[str, Any]],
    config: ConfigInfo,
    artifact_key: str,
    checkpoint_role: str = "last",
) -> Mapping[str, Any]:
    return _select(
        aggregates,
        config_hash=config.config_hash,
        checkpoint_role=checkpoint_role,
        artifact_key=artifact_key,
    )


def write_report(
    output_dir: Path,
    aggregates: Sequence[Mapping[str, Any]],
    configs: Mapping[str, ConfigInfo],
    associations: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    random_seed: int,
) -> None:
    role_configs = _role_configs(configs)
    c1_accuracy = []
    for optimizer in OPTIMIZER_ORDER:
        config = role_configs[("C1", optimizer)]
        row = _find_record(aggregates, config, "id/accuracy")
        c1_accuracy.append((optimizer.upper(), row["mean"], row["sample_sd_ddof1"]))

    dataset_summary = []
    for dataset in OOD_DATASETS:
        candidates = []
        for optimizer in OPTIMIZER_ORDER:
            config = role_configs[("C1", optimizer)]
            for detector in CANONICAL_DETECTORS:
                row = _select(
                    aggregates,
                    config_hash=config.config_hash,
                    checkpoint_role="last",
                    artifact_key="ood_metric/auroc_id_positive",
                    detector=detector,
                    ood_dataset=dataset,
                )
                candidates.append((row["mean"], optimizer, detector, row["sample_sd_ddof1"]))
        best = max(candidates)
        dataset_summary.append((OOD_DISPLAY[dataset], best[1].upper(), best[2], best[0], best[3]))

    top_associations = []
    for dataset in OOD_DATASETS:
        rows = [
            row
            for row in associations
            if row["checkpoint_role"] == "last"
            and row["ood_dataset"] == dataset
            and row["endpoint"] == "auroc"
        ]
        top = max(rows, key=lambda row: (abs(row["config_mean_spearman_rho"]), -row["bh_fdr_qvalue"]))
        top_associations.append(top)

    role_table = (output_dir / "tables/role_hyperparameters.md").read_text(encoding="utf-8").strip()
    accuracy_text = ", ".join(
        f"{optimizer} {mean:.4f} ± {sd:.4f}" for optimizer, mean, sd in c1_accuracy
    )
    dataset_md = _markdown_table(
        ["OOD dataset", "C1 optimizer", "Detector", "AUROC mean ± SD"],
        [
            [dataset, optimizer, f"`{detector}`", _format_mean_sd(mean, sd)]
            for dataset, optimizer, detector, mean, sd in dataset_summary
        ],
    ).strip()
    association_md = _markdown_table(
        [
            "OOD dataset",
            "Geometry",
            "Detector",
            "config-mean ρ",
            "seed-level ρ",
            "seed-level cluster 95% CI",
            "q",
        ],
        [
            [
                OOD_DISPLAY[row["ood_dataset"]],
                row["geometry_name"],
                f"`{row['detector']}`",
                f"{row['config_mean_spearman_rho']:.3f}",
                f"{row['seed_level_clustered_rho']:.3f}",
                f"[{row['cluster_bootstrap_ci_low']:.3f}, {row['cluster_bootstrap_ci_high']:.3f}]",
                f"{row['bh_fdr_qvalue']:.3f}",
            ]
            for row in top_associations
        ],
    ).strip()

    report = f"""# C1–C4 Metric Contract v1.2 로컬 분석

## 기술 요약

checksum 검증을 통과한 Metric Contract v1.2 중앙 aggregate만 사용하여 C1–C4의 10개 unique configuration과 각 3개 training seed를 정리했다. Confirmatory primary인 `last.pt`에서 C1의 ID-test accuracy는 {accuracy_text}이다. 모든 `mean ± SD`는 seed 0·1·2의 산술평균과 sample standard deviation (`ddof=1`)이며 confidence interval이 아니다.

OOD 결과는 CIFAR-100, TinyImageNet, MNIST, SVHN, Textures, Places365를 끝까지 분리했다. 아래의 dataset별 최고 AUROC는 C1 canonical detector panel을 기술적으로 훑은 결과이며, 새로운 primary detector를 사후 선택한 것이 아니다. Geometry–OOD 상관은 unique config가 10개이고 seed 0가 role 선택에 사용되었으므로 모두 exploratory association으로만 해석한다.

## 고정된 role configuration

{role_table}

Adam C1과 C3은 같은 scientific configuration이다. 역할 해석을 위해 두 위치에 모두 표시하지만 association 분석에는 한 번만 포함한다. AdamW C2는 `NA — protocol absent`이며 어떤 값도 대입하지 않았다.

## C1–C4 ID·geometry 비교

논문용 ID 표는 [`last` primary](tables/id/paper_last.md)와 [`best_val` control](tables/id/paper_best_val.md), geometry 표는 [`last` primary](tables/geometry/paper_last.md)와 [`best_val` control](tables/geometry/paper_best_val.md)이다. 그림은 각 seed 값과 mean ± sample SD를 함께 보인다. Notion의 `Text & Markdown` 가져오기용 primary 비교 문서는 [`notion_c1_c4_primary_comparison.md`](notion_c1_c4_primary_comparison.md)이다.

- [`last.pt` role comparison](figures/role_summary_last.svg)
- [`best_val.pt` role comparison](figures/role_summary_best_val.svg)

calibration sensitivity, feature-norm quantile, classwise summary, Neural Collapse diagnostic, spectrum, LID, TwoNN을 포함한 중앙 aggregate의 모든 scalar는 `tables/id/`와 `tables/geometry/`에 저장했다. Main/context 배치는 Card 11을 따르며 원본 aggregate의 tier 값은 수정하지 않았다. Seed-matched optimizer delta는 추론검정 없이 `tables/deltas/`에 별도로 제공한다.

## OOD dataset별 결과

{dataset_md}

`tables/ood/` 아래 각 dataset 디렉터리에는 `last`와 `best_val`을 분리한 canonical 11-detector 논문 표, 19-detector 전수 3-seed scalar 표, seed=0 표가 있다. AUROC, FPR@95, AUPR-In, AUPR-Out을 raw sample 수준에서 dataset 간 pooling하지 않았다. Near/far/overall macro는 `tables/cross_dataset_macro_audit/`에만 격리했으며 위 dataset별 결론에는 사용하지 않았다. Detector의 본문·부록 배치는 [`detector_panel.md`](tables/detector_panel.md)에 고정했다.

## Geometry–OOD association은 exploratory

각 OOD dataset과 checkpoint에서 primary coefficient는 10개 unique configuration mean의 Spearman correlation이다. 강건성 계수와 95% CI는 30개 seed-level 관측을 사용한다. {resamples:,}회 configuration-block bootstrap에서는 config 10개를 복원추출하고 선택된 각 config의 seed 3개를 함께 유지한 뒤, replicate 안에서 raw value를 다시 rank 변환해 Spearman coefficient를 계산했다. Two-sided permutation은 10개 config-mean block을 {resamples:,}회 섞었다. Benjamini–Hochberg correction은 dataset·checkpoint·endpoint별 16×11 가설군에 적용했다. Fixed random seed는 `{random_seed}`이다. FPR association은 `−FPR@95`를 사용하므로 양의 상관은 항상 더 좋은 OOD 성능을 뜻한다.

아래에는 dataset별 `last.pt` AUROC matrix에서 absolute config-mean correlation이 가장 큰 한 셀을 위치 확인용으로 표시한다. `seed-level cluster 95% CI`는 바로 앞의 seed-level coefficient에 대한 구간이며 config-mean coefficient의 구간이 아니다. 이 표는 confirmatory evidence나 causal result가 아니다.

{association_md}

각 dataset은 `figures/associations/` 아래 독립된 AUROC와 FPR heatmap을 갖는다. AUPR-In/Out association은 같은 절차로 계산해 `tables/associations/`의 appendix CSV에 보존했으며 heatmap panel로 승격하지 않았다. Dataset을 합친 correlation은 없다. Config 수, seed-0 selection bias, multiplicity, 공통 checkpoint provenance 때문에 q-value가 작더라도 인과나 일반화 주장을 하지 않는다.

## Metric 정의와 Methods 연결

[`methods_crosswalk.csv`](tables/methods_crosswalk.csv)와 [`methods_crosswalk.md`](tables/methods_crosswalk.md)는 formula ID, artifact key, checkpoint/split, direction, reporting tier, transform, Card 11 문장을 연결한다. 논문에 바로 옮길 English 문단은 [`METHODS_READY_EN.md`](METHODS_READY_EN.md)에 있다. 핵심 규칙은 다음과 같다.

- `last.pt`: confirmatory primary; `best_val.pt`: 별도 deployment control.
- 모든 3-seed 값: seed 0·1·2 산술평균 ± sample SD (`ddof=1`).
- AUROC: ID positive; FPR@95: inclusive linear 5th-percentile ID threshold.
- Geometry: 정의가 별도 transform을 명시하지 않으면 raw ID-training feature.
- seed=0 표: audit layer이며 독립적인 confirmatory replication이 아님.

## 검증·한계·결과 경계

분석 시작 시 모든 input hash, 60 checkpoint job, 30 training identity, 10 configuration, 31,720개 successful seed aggregate, 95,160개 successful per-checkpoint scalar, seed=0 exact parity를 다시 검증했다. 전수 정리는 중앙 aggregate에 있는 scalar만 대상으로 한다. Raw score, feature array, 전체 spectrum, class-pair matrix와 그 밖의 checkpoint-bundle intermediate는 내려받거나 재계산하지 않았으며 **미포함**이다.

따라서 이 보고서는 재현 가능한 descriptive result와 exploratory association을 제공한다. Optimizer causality, `n=3`에 기반한 통계적 우월성, universal best detector, cross-dataset generalization을 확립하지 않는다.

## 다음 사용 단계

Metric 정의를 바꾸지 않은 채 dataset별 표에서 논문 본문과 부록의 최종 배치를 선택할 수 있다. Confirmatory inference, 새 detector selection, DDU/SN ablation, 추가 protected population은 별도로 사전 명시해야 한다.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def write_output_manifest(
    output_dir: Path,
    source_manifest: Mapping[str, Any],
    *,
    resamples: int,
    random_seed: int,
    figures_generated: bool,
    analysis_summary: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = output_dir / "output_manifest.json"
    sidecar_path = output_dir / "output_manifest.json.sha256"
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path in {manifest_path, sidecar_path}:
            continue
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "analysis_version": "metric_contract_v1.2_c1_c4_analysis_v1",
        "definition_version": DEFINITION_VERSION,
        "evaluation_git_sha": EXPECTED_EVALUATION_SHA,
        "inventory_hash": EXPECTED_INVENTORY_HASH,
        "dataset_policy_hash": EXPECTED_DATASET_POLICY_HASH,
        "source_payloads": source_manifest["files"],
        "resamples": resamples,
        "random_seed": random_seed,
        "analysis_dtype": "float64",
        "comparison_tolerance": {"atol": 1e-12, "rtol": 1e-12},
        "figures_generated": figures_generated,
        "analysis_summary": dict(analysis_summary),
        "file_count": len(files),
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sidecar_path.write_text(f"{_sha256(manifest_path)}  output_manifest.json\n", encoding="utf-8")
    return manifest


def run_analysis(
    *,
    aggregate_dir: Path,
    inventory_path: Path,
    output_dir: Path,
    resamples: int = 10_000,
    random_seed: int = 20_260_805,
    make_figures: bool = True,
) -> dict[str, Any]:
    aggregate_dir = aggregate_dir.resolve()
    inventory_path = inventory_path.resolve()
    _require(
        not output_dir.exists() or not any(output_dir.iterdir()),
        "output directory must be absent or empty",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    verified = verify_aggregate(aggregate_dir)
    configs = load_authorized_configs(inventory_path)
    aggregate_rows, seed0_records, record_counts = validate_records(aggregate_dir, configs)
    aggregates = IndexedAggregateRows(aggregate_rows)

    write_role_and_id_tables(output_dir, aggregates, configs)
    write_geometry_paper_tables(output_dir, aggregates, configs)
    table_counts = write_exhaustive_tables(output_dir, aggregates, seed0_records, configs)
    write_ood_paper_tables(output_dir, aggregates, configs)
    write_detector_panel(output_dir)
    paired_delta_count = write_paired_delta_tables(output_dir, aggregates, configs)
    write_notion_primary_comparison(output_dir, aggregates, configs)
    associations, heatmaps = compute_associations(
        output_dir,
        aggregates,
        configs,
        resamples=resamples,
        random_seed=random_seed,
    )
    write_methods_crosswalk(output_dir, aggregates)
    write_methods_ready_english(output_dir, resamples=resamples, random_seed=random_seed)
    write_chart_map(output_dir)
    if make_figures:
        write_role_figures(output_dir, aggregates, configs)
        write_association_figures(output_dir, associations, heatmaps)
    write_report(
        output_dir,
        aggregates,
        configs,
        associations,
        resamples=resamples,
        random_seed=random_seed,
    )

    summary = {
        "configuration_count": len(configs),
        "seed_aggregate_count": len(aggregates),
        "seed0_record_count": len(seed0_records),
        "association_record_count": len(associations),
        "paired_delta_record_count": paired_delta_count,
        "per_checkpoint_record_count": record_counts["per_checkpoint"],
        "table_partition_count": len(table_counts),
        "all_status": "success",
    }
    write_output_manifest(
        output_dir,
        verified["manifest"],
        resamples=resamples,
        random_seed=random_seed,
        figures_generated=make_figures,
        analysis_summary=summary,
    )
    return summary
