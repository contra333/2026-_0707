"""Prespecified exploratory comparisons over completed metric records."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import kendalltau, rankdata

from .common import MetricValue


def detector_rank_concordance(
    optimizer_detector_auroc: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    optimizers = sorted(optimizer_detector_auroc)
    if len(optimizers) < 2:
        return {
            "status": "degenerate",
            "reason_codes": ["fewer_than_two_optimizers"],
            "pairs": {},
        }
    detector_sets = [set(optimizer_detector_auroc[name]) for name in optimizers]
    if not detector_sets or any(detectors != detector_sets[0] for detectors in detector_sets):
        raise ValueError("optimizer detector panels do not match")
    detectors = sorted(detector_sets[0])
    if len(detectors) < 2:
        return {
            "status": "degenerate",
            "reason_codes": ["fewer_than_two_detectors"],
            "pairs": {},
        }
    ranks = {
        optimizer: rankdata(
            [-float(optimizer_detector_auroc[optimizer][detector]) for detector in detectors],
            method="average",
        )
        for optimizer in optimizers
    }
    pairs: dict[str, Any] = {}
    for left_index, left in enumerate(optimizers):
        for right in optimizers[left_index + 1 :]:
            result = kendalltau(ranks[left], ranks[right], variant="b")
            value = float(result.statistic)
            pairs[f"{left}__{right}"] = {
                "metric": MetricValue(value).to_dict(),
                "pvalue": float(result.pvalue),
                "left_ranks": ranks[left],
                "right_ranks": ranks[right],
            }
    return {"status": "success", "reason_codes": [], "detectors": detectors, "pairs": pairs}


def restoration_gaps(
    *,
    mahalanobis_raw: Mapping[str, Any],
    mahalanobis_pp: Mapping[str, Any],
    relative_raw: Mapping[str, Any],
    relative_pp: Mapping[str, Any],
    ncp: Mapping[str, Any],
    ctm: Mapping[str, Any],
) -> dict[str, float]:
    rows = (mahalanobis_raw, mahalanobis_pp, relative_raw, relative_pp, ncp, ctm)
    identity_fields = ("checkpoint_sha256", "training_seed", "ood_dataset")
    expected = tuple(rows[0].get(field) for field in identity_fields)
    if any(tuple(row.get(field) for field in identity_fields) != expected for row in rows[1:]):
        raise ValueError("restoration gaps require identical checkpoint/seed/OOD identity")
    return {
        "restoration_gap_mahalanobis": float(mahalanobis_pp["auroc"] - mahalanobis_raw["auroc"]),
        "restoration_gap_relative_mahalanobis": float(relative_pp["auroc"] - relative_raw["auroc"]),
        "angular_control_gap_ctm_ncp": float(ctm["auroc"] - ncp["auroc"]),
    }
