# Methods-ready English text

## Experimental units and reporting

We analyzed ten unique WRN-28-10 configurations trained on CIFAR-10 with three seeds per configuration. The terminal `last.pt` checkpoint was the confirmatory primary checkpoint, whereas the validation-selected `best_val.pt` checkpoint was analyzed separately as a deployment control. We report arithmetic means and sample standard deviations (`ddof=1`) across seeds 0, 1, and 2. Adam C1 and C3 denote the same scientific configuration and were therefore counted once in cross-configuration association analyses. AdamW C2 was absent by protocol and was not imputed.

## ID performance and representation geometry

ID metrics were evaluated on the frozen 10,000-image CIFAR-10 test split. We report top-1 accuracy, raw-logit negative log-likelihood, 15-bin expected calibration error, raw-MSP misclassification AUROC, and AUGRC in the main descriptive panel. Geometry statistics were fitted on raw ID-training features unless the corresponding Card 11 definition declares another transform. In particular, we report NC1 as $\operatorname{Tr}(\Sigma_W\Sigma_B^{\dagger})/K$, using biased within-class and between-class covariance estimators computed from raw ID-training features.

## OOD evaluation

OOD performance was evaluated separately on CIFAR-100, TinyImageNet, MNIST, SVHN, Textures, and Places365; samples were never pooled across datasets. AUROC treated ID as the positive class. FPR@95 used the inclusive threshold $\tau_{95}=\operatorname{quantile}(s_{ID},0.05)$ with linear interpolation and was computed as $\Pr(s_{OOD}\geq\tau_{95})$. Temperature-scaled logits were not used for primary OOD scoring. The main panel contained the 11 prespecified canonical detectors, while complete 19-detector results were retained in the appendix tables.

## Geometry–OOD association analysis

Within each OOD dataset, checkpoint role, endpoint, geometry metric, and canonical detector, we computed Spearman's rank correlation across the ten unique three-seed configuration means. AUROC was the primary association endpoint; FPR@95 was negated so that larger values consistently represented better OOD performance. As a robustness analysis, we computed a separate seed-level Spearman coefficient from 30 observations and obtained a 95% percentile interval using 10,000 configuration-block bootstrap replicates. Each bootstrap replicate resampled ten configurations with replacement, retained all three seeds in every selected block, and reranked the resampled raw observations before computing Spearman's coefficient. Two-sided permutation p-values used 10,000 permutations of the ten configuration-mean blocks. Benjamini–Hochberg correction was applied separately within each dataset, checkpoint, and endpoint across the 16-by-11 geometry–detector panel. All stochastic analysis used the fixed random seed `20260805`.

AUPR-In and AUPR-Out associations were computed by the same procedure and retained as appendix CSVs; they were not promoted to the AUROC/FPR heatmap panel.

## Interpretation boundary

All geometry–OOD results are exploratory associations rather than causal effects. The small number of unique configurations, selection of the role grid using seed-0 validation results, multiplicity, and common checkpoint provenance limit inferential interpretation. Seed-0 tables are provided for auditing and are not treated as independent confirmatory evidence.

Metric-level canonical sentences and their formula, artifact, split, transform, direction, and reporting-tier identities are provided in [`tables/methods_crosswalk.md`](tables/methods_crosswalk.md).
