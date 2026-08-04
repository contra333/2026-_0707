"""Metric Contract v1.2 evaluation utilities."""

from .classification import (
    augrc,
    aurc_tie_grouped,
    evaluate_temperature_scaled,
    expected_calibration_error,
    fit_temperature,
    logit_ood_scores,
    misclassification_auroc,
    negative_log_likelihood,
    top1_accuracy,
)
from .extraction import extract_checkpoint_artifact, verify_raw_feature_artifact
from .metrics import compute_ood_metrics, macro_average_ood_metrics
from .msp import evaluate_msp_protocol, infer_msp
from .runtime import FittedMetricRuntime, load_raw_feature_cache, write_bounded_smoke_result

__all__ = [
    "augrc",
    "aurc_tie_grouped",
    "compute_ood_metrics",
    "evaluate_msp_protocol",
    "evaluate_temperature_scaled",
    "expected_calibration_error",
    "extract_checkpoint_artifact",
    "fit_temperature",
    "FittedMetricRuntime",
    "infer_msp",
    "logit_ood_scores",
    "load_raw_feature_cache",
    "macro_average_ood_metrics",
    "misclassification_auroc",
    "negative_log_likelihood",
    "top1_accuracy",
    "verify_raw_feature_artifact",
    "write_bounded_smoke_result",
]
