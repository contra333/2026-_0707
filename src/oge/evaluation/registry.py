"""Canonical formula, artifact-key, tier, and implementation registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    formula_id: str
    artifact_key: str | None
    reporting_tier: str
    entrypoint: str
    direction: str


METRIC_REGISTRY = (
    MetricDefinition("ID-1", "id/accuracy", "primary", "classification.top1_accuracy", "up"),
    MetricDefinition("ID-2", "id/nll_raw", "primary", "classification.negative_log_likelihood", "down"),
    MetricDefinition("ID-3", "id/ece_raw_m15", "primary", "classification.expected_calibration_error", "down"),
    MetricDefinition("ID-3", "id/ece_raw_m10", "appendix", "classification.expected_calibration_error", "down"),
    MetricDefinition("ID-3", "id/ece_raw_m30", "appendix", "classification.expected_calibration_error", "down"),
    MetricDefinition("ID-4", "id/temperature", "control", "classification.fit_temperature", "descriptive"),
    MetricDefinition("ID-4", "id/nll_ts", "control", "classification.evaluate_temperature_scaled", "down"),
    MetricDefinition("ID-4", "id/ece_ts_m15", "control", "classification.evaluate_temperature_scaled", "down"),
    MetricDefinition("ID-5", "id/misclassification_auroc_msp_raw", "primary", "classification.misclassification_auroc", "up"),
    MetricDefinition("ID-6", "id/augrc_msp_raw", "primary", "classification.augrc", "down"),
    MetricDefinition("ID-7", "id/aurc_msp_raw", "appendix", "classification.aurc_tie_grouped", "down"),
    MetricDefinition("LOGIT-1", "ood_score/msp", "primary", "classification.logit_ood_scores", "up"),
    MetricDefinition("LOGIT-2", "ood_score/max_logit", "primary", "classification.logit_ood_scores", "up"),
    MetricDefinition("LOGIT-3", "ood_score/energy_t1", "primary", "classification.logit_ood_scores", "up"),
    MetricDefinition("LOGIT-X", None, "excluded", "classification.softmax_probabilities", "ood_like"),
    MetricDefinition("GAUSS-0", None, "appendix", "detectors.fit_tied_gaussians", "descriptive"),
    MetricDefinition("GAUSS-1", "detector/mahalanobis_raw", "primary", "detectors.score_tied_gaussians", "up"),
    MetricDefinition("GAUSS-2", "detector/marginal_mahalanobis_raw", "appendix", "detectors.score_tied_gaussians", "up"),
    MetricDefinition("GAUSS-3", "detector/relative_mahalanobis_raw", "primary", "detectors.score_tied_gaussians", "up"),
    MetricDefinition("GAUSS-4", "detector/mahalanobis_pp", "primary", "detectors.score_tied_gaussians", "up"),
    MetricDefinition("GAUSS-5", "detector/relative_mahalanobis_pp", "primary", "detectors.score_tied_gaussians", "up"),
    MetricDefinition("GAUSS-6", "detector/gda_class_density", "appendix", "detectors.score_gda_class_density", "up"),
    MetricDefinition("SUB-1", "detector/knn_l2_k50", "primary", "detectors.exact_knn_l2_scores", "up"),
    MetricDefinition("SUB-2", "detector/ncp_euclidean_raw", "appendix", "detectors.prototype_scores", "up"),
    MetricDefinition("SUB-3", "detector/ctm_prototype_cosine", "primary", "detectors.prototype_scores", "up"),
    MetricDefinition("SUB-4", "detector/weight_cosine_appendix", "appendix", "detectors.prototype_scores", "up"),
    MetricDefinition("SUB-5", "detector/vim_author_dim", "primary", "detectors.score_vim", "up"),
    MetricDefinition("SUB-5", "detector/vim_dim64", "appendix", "detectors.score_vim", "up"),
    MetricDefinition("SUB-5", "detector/vim_dim128", "appendix", "detectors.score_vim", "up"),
    MetricDefinition("SUB-5", "detector/vim_dim256", "appendix", "detectors.score_vim", "up"),
    MetricDefinition("SUB-5", "detector/vim_dim320", "appendix", "detectors.score_vim", "up"),
    MetricDefinition("SUB-6", "detector/neco_author_std_dim100", "primary", "detectors.score_neco", "up"),
    MetricDefinition("OOD-1", "ood_metric/auroc_id_positive", "primary", "metrics.compute_ood_metrics", "up"),
    MetricDefinition("OOD-2", "ood_metric/aupr_in_openood_auc", "primary", "metrics.compute_ood_metrics", "up"),
    MetricDefinition("OOD-2", "ood_metric/ap_in", "appendix", "metrics.compute_ood_metrics", "up"),
    MetricDefinition("OOD-3", "ood_metric/aupr_out_openood_auc", "primary", "metrics.compute_ood_metrics", "up"),
    MetricDefinition("OOD-3", "ood_metric/ap_out", "appendix", "metrics.compute_ood_metrics", "up"),
    MetricDefinition("OOD-4", "ood_metric/fpr95_id_tpr", "primary", "metrics.fpr95_id_tpr", "down"),
    MetricDefinition("OOD-4", "ood_metric/fpr95_threshold", "appendix", "metrics.fpr95_id_tpr", "descriptive"),
    MetricDefinition("OOD-4", "ood_metric/fpr95_achieved_id_tpr", "appendix", "metrics.fpr95_id_tpr", "descriptive"),
    MetricDefinition("OOD-5", "ood_metric/{metric}/{group}_macro_mean", "primary", "metrics.macro_average_ood_metrics", "metric_specific"),
    MetricDefinition("OOD-6", "analysis/detector_rank_concordance_tau_b", "exploratory", "analysis.detector_rank_concordance", "descriptive"),
    MetricDefinition("OOD-7", "analysis/restoration_gap_mahalanobis", "exploratory", "analysis.restoration_gaps", "descriptive"),
    MetricDefinition("OOD-7", "analysis/restoration_gap_relative_mahalanobis", "exploratory", "analysis.restoration_gaps", "descriptive"),
    MetricDefinition("OOD-7", "analysis/angular_control_gap_ctm_ncp", "exploratory", "analysis.restoration_gaps", "descriptive"),
    MetricDefinition("GEO-1", "geometry/feature_norm/global", "primary", "geometry.feature_norm_distribution", "descriptive"),
    MetricDefinition("GEO-1", "geometry/feature_norm/classwise", "primary", "geometry.feature_norm_distribution", "descriptive"),
    MetricDefinition("GEO-1", "geometry/feature_norm/heldout_validation", "control", "geometry.feature_norm_distribution", "descriptive"),
    MetricDefinition("GEO-2", "geometry/cdnv/mean", "primary", "geometry.cdnv", "down"),
    MetricDefinition("GEO-2", "geometry/cdnv/pair_matrix", "primary", "geometry.cdnv", "down"),
    MetricDefinition("GEO-3", "geometry/class_pair/distance", "primary", "geometry.class_pair_geometry", "descriptive"),
    MetricDefinition("GEO-3", "geometry/class_pair/centered_cosine", "primary", "geometry.class_pair_geometry", "descriptive"),
    MetricDefinition("GEO-3", "geometry/class_pair/raw_origin_cosine_appendix", "appendix", "geometry.class_pair_geometry", "descriptive"),
    MetricDefinition("GEO-4", "geometry/nc0_row_sum_raw", "primary", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-4", "geometry/nc0_eq12_per_dim", "appendix", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-4", "geometry/nc0_theory_squared", "appendix", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-5", "geometry/nc1_pinv", "primary", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-5", "geometry/nc1_svd_diagnostic", "appendix", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-5", "geometry/nc1_trace_quotient_diagnostic", "appendix", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-6", "geometry/nc2_equinorm", "primary", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-6", "geometry/nc2_equiangular", "primary", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-6", "geometry/nc2_etf_raw", "primary", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-6", "geometry/nc2_etf_eq5_scaled", "appendix", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-7", "geometry/nc2w_etf_raw", "appendix", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-7", "geometry/nc2w_equinorm", "appendix", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-7", "geometry/nc2w_equiangular", "appendix", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-8", "geometry/nc3_self_duality_raw", "primary", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-8", "geometry/nc3_eq10_scaled", "appendix", "geometry.neural_collapse_metrics", "down"),
    MetricDefinition("GEO-9", "geometry/nc4_agreement_with_bias", "primary", "geometry.neural_collapse_metrics", "up"),
    MetricDefinition("GEO-9", "geometry/nc4_agreement_without_bias", "appendix", "geometry.neural_collapse_metrics", "up"),
    MetricDefinition("GEO-10", "geometry/rankme_uncentered", "primary", "geometry.rankme", "descriptive"),
    MetricDefinition("GEO-11", "geometry/spectrum/{matrix}_entropy_rank", "primary", "geometry.covariance_spectrum", "descriptive"),
    MetricDefinition("GEO-11", "geometry/spectrum/{matrix}_trace_top_rank", "primary", "geometry.covariance_spectrum", "descriptive"),
    MetricDefinition("GEO-11", "geometry/spectrum/{matrix}_participation_ratio", "primary", "geometry.covariance_spectrum", "descriptive"),
    MetricDefinition("GEO-12", "geometry/spectrum/{matrix}_numerical_rank", "primary", "geometry.covariance_spectrum", "descriptive"),
    MetricDefinition("GEO-12", "geometry/spectrum/{matrix}_condition_number_retained", "primary", "geometry.covariance_spectrum", "descriptive"),
    MetricDefinition("GEO-13", "geometry/spectrum/sw_log_slope", "exploratory", "geometry.covariance_spectrum", "descriptive"),
    MetricDefinition("GEO-13", "geometry/spectrum/sw_top20_mean_log_decay", "exploratory", "geometry.covariance_spectrum", "descriptive"),
    MetricDefinition("GEO-14", "geometry/lid_k10", "appendix", "geometry.local_intrinsic_dimensionality", "descriptive"),
    MetricDefinition("GEO-14", "geometry/lid_k25", "appendix", "geometry.local_intrinsic_dimensionality", "descriptive"),
    MetricDefinition("GEO-14", "geometry/lid_k50", "primary", "geometry.local_intrinsic_dimensionality", "descriptive"),
    MetricDefinition("GEO-14", "geometry/lid_k100", "appendix", "geometry.local_intrinsic_dimensionality", "descriptive"),
    MetricDefinition("GEO-15", "geometry/twonn_base_mu_fraction_09", "appendix", "geometry.twonn", "descriptive"),
    MetricDefinition("GEO-16", "geometry/hypersphere/class_alignment_exact", "exploratory", "geometry.hypersphere_alignment_uniformity", "down"),
    MetricDefinition("GEO-17", "geometry/hypersphere/uniformity_t2", "exploratory", "geometry.hypersphere_alignment_uniformity", "down"),
)


def registry_by_artifact_key() -> dict[str, MetricDefinition]:
    definitions = [item for item in METRIC_REGISTRY if item.artifact_key is not None]
    keys = [item.artifact_key for item in definitions]
    if len(keys) != len(set(keys)):
        raise ValueError("metric registry contains duplicate artifact keys")
    return {item.artifact_key: item for item in definitions}
