| Formula | Artifact key | Direction | Tier | Methods wording |
| --- | --- | --- | --- | --- |
| OOD-7 | `analysis/angular_control_gap_ctm_ncp` | descriptive | exploratory | Restoration gaps are paired within-checkpoint AUROC differences between normalized and raw Gaussian readouts. |
| OOD-7 | `analysis/restoration_gap_mahalanobis` | descriptive | exploratory | Restoration gaps are paired within-checkpoint AUROC differences between normalized and raw Gaussian readouts. |
| OOD-7 | `analysis/restoration_gap_relative_mahalanobis` | descriptive | exploratory | Restoration gaps are paired within-checkpoint AUROC differences between normalized and raw Gaussian readouts. |
| GEO-2 | `geometry/cdnv/mean` | down | primary | CDNV averages the pairwise ratio of within-class variance to squared class-mean separation. |
| GEO-1 | `geometry/feature_norm/classwise` | descriptive | primary | We characterize radial feature geometry using the full ID-training norm distribution and class-wise summaries. |
| GEO-1 | `geometry/feature_norm/global` | descriptive | primary | We characterize radial feature geometry using the full ID-training norm distribution and class-wise summaries. |
| GEO-1 | `geometry/feature_norm/heldout_validation` | descriptive | control | We characterize radial feature geometry using the full ID-training norm distribution and class-wise summaries. |
| GEO-16 | `geometry/hypersphere/class_alignment_exact` | down | exploratory | Class alignment is the class-macro mean pairwise squared distance between L2-normalized same-class features. |
| GEO-17 | `geometry/hypersphere/uniformity_t2` | down | exploratory | Uniformity is the log mean exponential pair potential at t=2 over three prespecified 100,000-pair Monte Carlo repeats. |
| GEO-14 | `geometry/lid_k10` | descriptive | appendix | LID uses the maximum-likelihood k-neighbor distance-ratio estimator on exact raw ID-training neighbors, with k=50 primary. |
| GEO-14 | `geometry/lid_k100` | descriptive | appendix | LID uses the maximum-likelihood k-neighbor distance-ratio estimator on exact raw ID-training neighbors, with k=50 primary. |
| GEO-14 | `geometry/lid_k25` | descriptive | appendix | LID uses the maximum-likelihood k-neighbor distance-ratio estimator on exact raw ID-training neighbors, with k=50 primary. |
| GEO-14 | `geometry/lid_k50` | descriptive | primary | LID uses the maximum-likelihood k-neighbor distance-ratio estimator on exact raw ID-training neighbors, with k=50 primary. |
| GEO-4 | `geometry/nc0_eq12_per_dim` | down | appendix | NC0 is the Euclidean norm of the sum of classifier-weight rows; scaled expressions are reported only as audits. |
| GEO-4 | `geometry/nc0_row_sum_raw` | down | primary | NC0 is the Euclidean norm of the sum of classifier-weight rows; scaled expressions are reported only as audits. |
| GEO-4 | `geometry/nc0_theory_squared` | down | appendix | NC0 is the Euclidean norm of the sum of classifier-weight rows; scaled expressions are reported only as audits. |
| GEO-5 | `geometry/nc1_pinv` | down | primary | We report NC1 as Tr(Sigma_W Sigma_B^dagger)/K, using biased within- and between-class covariance estimators computed from raw ID-training features. |
| GEO-5 | `geometry/nc1_svd_diagnostic` | down | appendix | We report NC1 as Tr(Sigma_W Sigma_B^dagger)/K, using biased within- and between-class covariance estimators computed from raw ID-training features. |
| GEO-5 | `geometry/nc1_trace_quotient_diagnostic` | down | appendix | We report NC1 as Tr(Sigma_W Sigma_B^dagger)/K, using biased within- and between-class covariance estimators computed from raw ID-training features. |
| GEO-6 | `geometry/nc2_equiangular` | down | primary | NC2n is the sample coefficient of variation of centered class-mean norms, while NC2a and NC2-ETF quantify simplex equiangularity. |
| GEO-6 | `geometry/nc2_equinorm` | down | primary | NC2n is the sample coefficient of variation of centered class-mean norms, while NC2a and NC2-ETF quantify simplex equiangularity. |
| GEO-6 | `geometry/nc2_etf_eq5_scaled` | down | appendix | NC2n is the sample coefficient of variation of centered class-mean norms, while NC2a and NC2-ETF quantify simplex equiangularity. |
| GEO-6 | `geometry/nc2_etf_raw` | down | primary | NC2n is the sample coefficient of variation of centered class-mean norms, while NC2a and NC2-ETF quantify simplex equiangularity. |
| GEO-7 | `geometry/nc2w_equiangular` | down | appendix | NC2W applies the corresponding ETF geometry diagnostics to classifier-weight rows. |
| GEO-7 | `geometry/nc2w_equinorm` | down | appendix | NC2W applies the corresponding ETF geometry diagnostics to classifier-weight rows. |
| GEO-7 | `geometry/nc2w_etf_raw` | down | appendix | NC2W applies the corresponding ETF geometry diagnostics to classifier-weight rows. |
| GEO-8 | `geometry/nc3_eq10_scaled` | down | appendix | NC3 is the Frobenius distance between globally normalized classifier weights and transposed centered class means. |
| GEO-8 | `geometry/nc3_self_duality_raw` | down | primary | NC3 is the Frobenius distance between globally normalized classifier weights and transposed centered class means. |
| GEO-9 | `geometry/nc4_agreement_with_bias` | up | primary | NC4 is the agreement rate between the affine classifier and nearest ID-training class-mean predictions. |
| GEO-9 | `geometry/nc4_agreement_without_bias` | up | appendix | NC4 is the agreement rate between the affine classifier and nearest ID-training class-mean predictions. |
| GEO-10 | `geometry/rankme_uncentered` | descriptive | primary | RankMe is spectral entropy computed from singular values of the uncentered ID-training feature matrix. |
| GEO-12 | `geometry/spectrum/between_condition_number_retained` | descriptive | primary | Numerical rank retains covariance eigenvalues above lambda_max p eps64; conditioning uses only that retained spectrum. |
| GEO-11 | `geometry/spectrum/between_entropy_rank` | descriptive | primary | We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum. |
| GEO-12 | `geometry/spectrum/between_numerical_rank` | descriptive | primary | Numerical rank retains covariance eigenvalues above lambda_max p eps64; conditioning uses only that retained spectrum. |
| GEO-11 | `geometry/spectrum/between_participation_ratio` | descriptive | primary | We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum. |
| GEO-11 | `geometry/spectrum/between_trace_top_rank` | descriptive | primary | We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum. |
| GEO-12 | `geometry/spectrum/sw_condition_number_retained` | descriptive | primary | Numerical rank retains covariance eigenvalues above lambda_max p eps64; conditioning uses only that retained spectrum. |
| GEO-11 | `geometry/spectrum/sw_entropy_rank` | descriptive | primary | We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum. |
| GEO-13 | `geometry/spectrum/sw_log_slope` | descriptive | exploratory | The exploratory spectral slope is an OLS fit of natural log within-class eigenvalues against one-based rank. |
| GEO-12 | `geometry/spectrum/sw_numerical_rank` | descriptive | primary | Numerical rank retains covariance eigenvalues above lambda_max p eps64; conditioning uses only that retained spectrum. |
| GEO-11 | `geometry/spectrum/sw_participation_ratio` | descriptive | primary | We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum. |
| GEO-13 | `geometry/spectrum/sw_top20_mean_log_decay` | descriptive | exploratory | The exploratory spectral slope is an OLS fit of natural log within-class eigenvalues against one-based rank. |
| GEO-11 | `geometry/spectrum/sw_trace_top_rank` | descriptive | primary | We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum. |
| GEO-12 | `geometry/spectrum/total_condition_number_retained` | descriptive | primary | Numerical rank retains covariance eigenvalues above lambda_max p eps64; conditioning uses only that retained spectrum. |
| GEO-11 | `geometry/spectrum/total_entropy_rank` | descriptive | primary | We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum. |
| GEO-12 | `geometry/spectrum/total_numerical_rank` | descriptive | primary | Numerical rank retains covariance eigenvalues above lambda_max p eps64; conditioning uses only that retained spectrum. |
| GEO-11 | `geometry/spectrum/total_participation_ratio` | descriptive | primary | We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum. |
| GEO-11 | `geometry/spectrum/total_trace_top_rank` | descriptive | primary | We report covariance entropy rank, trace-to-top rank, and participation ratio as distinct functions of the same nonnegative covariance eigenvalue spectrum. |
| GEO-15 | `geometry/twonn_base_mu_fraction_09` | descriptive | appendix | TwoNN is the DADApy base estimator fitted through the origin after retaining the lowest 90% of log second-to-first-neighbor ratios. |
| ID-1 | `id/accuracy` | up | primary | We report top-1 accuracy on the frozen 10,000-image CIFAR-10 test split. |
| ID-6 | `id/augrc_msp_raw` | down | primary | AUGRC combines residual error prevalence and raw-MSP ranking quality using its closed-form binary-error expression. |
| ID-7 | `id/aurc_msp_raw` | down | appendix | AURC is the unscaled trapezoidal area under the tie-grouped raw-MSP risk-coverage curve. |
| ID-3 | `id/ece_raw_m10` | down | appendix | ECE uses 15 fixed equal-width confidence bins; 10- and 30-bin values are prespecified sensitivity analyses. |
| ID-3 | `id/ece_raw_m15` | down | primary | ECE uses 15 fixed equal-width confidence bins; 10- and 30-bin values are prespecified sensitivity analyses. |
| ID-3 | `id/ece_raw_m30` | down | appendix | ECE uses 15 fixed equal-width confidence bins; 10- and 30-bin values are prespecified sensitivity analyses. |
| ID-4 | `id/ece_ts_m15` | down | control | A single positive temperature was fitted on ID validation by float64 NLL minimization and applied once to the frozen ID test logits. |
| ID-5 | `id/misclassification_auroc_msp_raw` | up | primary | Misclassification AUROC treats correct predictions as the positive class and ranks them using raw maximum softmax probability. |
| ID-2 | `id/nll_raw` | down | primary | NLL is the mean negative log-probability assigned to the true class using unscaled logits. |
| ID-4 | `id/nll_ts` | down | control | A single positive temperature was fitted on ID validation by float64 NLL minimization and applied once to the frozen ID test logits. |
| ID-4 | `id/temperature` | descriptive | control | A single positive temperature was fitted on ID validation by float64 NLL minimization and applied once to the frozen ID test logits. |
| OOD-2 | `ood_metric/ap_in` | up | appendix | AUPR-In uses ID as positive; trapezoidal PR-AUC is primary and Average Precision is reported separately. |
| OOD-5 | `ood_metric/ap_in/far_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/ap_in/near_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/ap_in/overall_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-3 | `ood_metric/ap_out` | up | appendix | AUPR-Out treats OOD as positive and ranks examples with the negated ID-oriented score. |
| OOD-5 | `ood_metric/ap_out/far_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/ap_out/near_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/ap_out/overall_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-2 | `ood_metric/aupr_in_openood_auc` | up | primary | AUPR-In uses ID as positive; trapezoidal PR-AUC is primary and Average Precision is reported separately. |
| OOD-5 | `ood_metric/aupr_in_openood_auc/far_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/aupr_in_openood_auc/near_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/aupr_in_openood_auc/overall_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-3 | `ood_metric/aupr_out_openood_auc` | up | primary | AUPR-Out treats OOD as positive and ranks examples with the negated ID-oriented score. |
| OOD-5 | `ood_metric/aupr_out_openood_auc/far_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/aupr_out_openood_auc/near_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/aupr_out_openood_auc/overall_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/auroc/far_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/auroc/near_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/auroc/overall_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-1 | `ood_metric/auroc_id_positive` | up | primary | OOD AUROC treats ID as positive and uses the declared ID-oriented detector score. |
| OOD-4 | `ood_metric/fpr95_achieved_id_tpr` | descriptive | appendix | FPR@95 is the fraction of OOD scores above the linear 5th percentile of ID scores, using an inclusive threshold. |
| OOD-5 | `ood_metric/fpr95_achieved_id_tpr/far_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/fpr95_achieved_id_tpr/near_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/fpr95_achieved_id_tpr/overall_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-4 | `ood_metric/fpr95_id_tpr` | down | primary | FPR@95 is the fraction of OOD scores above the linear 5th percentile of ID scores, using an inclusive threshold. |
| OOD-5 | `ood_metric/fpr95_id_tpr/far_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/fpr95_id_tpr/near_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/fpr95_id_tpr/overall_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-4 | `ood_metric/fpr95_threshold` | descriptive | appendix | FPR@95 is the fraction of OOD scores above the linear 5th percentile of ID scores, using an inclusive threshold. |
| OOD-5 | `ood_metric/fpr95_threshold/far_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/fpr95_threshold/near_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
| OOD-5 | `ood_metric/fpr95_threshold/overall_macro_mean` | metric_specific | primary | Near-, far-, and overall OOD results are arithmetic means of per-dataset metrics, not pooled-sample estimates. |
