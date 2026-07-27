# Reference Card 09: Core Representation Metrics

## Purpose and authority boundary

This card freezes the confirmatory representation and low-complexity detector
panel used to study:

```text
optimizer -> penultimate geometry -> detector score -> ID/OOD reliability
```

The panel is fixed before comparative results are inspected. It uses raw
artifacts defined by
[`08_raw_feature_artifact_contract.md`](08_raw_feature_artifact_contract.md),
the penultimate endpoint from [`02_architectures.md`](02_architectures.md), and
the OOD score/aggregation rules from
[`04_openood_v1_5_protocol.md`](04_openood_v1_5_protocol.md).

The audit evidence paths below refer to the reviewed
`OOD_metric_audit_handoff.zip` supplied for Issue #24. This card, not that local
ZIP, is the repository authority after merge. A later implementation must
verify formulas against the pinned upstream commits in
`docs/sources.lock.yaml`.

## Common rules

- Fit class means, covariance, reference banks, and representation statistics
  on deterministic `id_train` raw artifacts only.
- Do not use ID test or any OOD sample to fit a detector or select a
  hyperparameter.
- Every detector score is ID-like: larger means more ID-like.
- Reuse `src/oge/evaluation/metrics.py::compute_ood_metrics`; do not introduce a
  second AUROC/AUPR/FPR95 implementation or add a sign flip to an already
  ID-like score.
- Reject empty classes, zero-norm inputs where normalization is required,
  non-finite arrays, non-finite fitted statistics, and non-finite outputs.
- Store the metric name, fit/query splits, normalization, dtype, estimator,
  inverse or decomposition method, and runtime decisions with every result.

## Frozen geometry panel

### `NC0-classifier-row-sum`

```text
NC0 = || sum_c w_c ||_2
```

Use the rows of `model.classifier.weight`; no dataset split is needed. Smaller
values are closer to the audited collapse condition. The value is
CODE-VERIFIED, but interpreting it as a sufficient condition is not
code-verified and must not be claimed. Evidence:
`evidence/A1-A6_neural_collapse_optimizer.md`, audited commit
`7cab4a59bc28da6e356cee1e793ec67a694933b9`.

### `NC1-SVD`

Using raw `id_train` features, compute class means, the class-balanced global
mean, biased within-class covariance, and biased between-class covariance:

```text
mu_G    = (1 / K) sum_c mu_c
Sigma_W = (1 / N) sum_c sum_{i in c} (z_i - mu_c)(z_i - mu_c)^T
Sigma_B = (1 / K) sum_c (mu_c - mu_G)(mu_c - mu_G)^T
```

The primary NC1 value is the audited `svd` variant using the top `K - 1`
singular directions of `Sigma_B` to evaluate the within/between variability
ratio. The Moore-Penrose and trace-quotient forms are not primary panel values.
Smaller is closer to collapse. Record `K`, `N`, covariance denominators, SVD
rank, and tolerance. Evidence:
`evidence/A1-A6_neural_collapse_optimizer.md`, audited commit `7cab4a5`.

### NC2 family

Let `m_c = mu_c - mu_G` from raw `id_train` features.

- `NC2-equinorm` is
  `std_c(||m_c||_2) / mean_c(||m_c||_2)`.
- `NC2-equiangular` is the mean over `c != c'` of
  `|cos(m_c, m_c') + 1/(K-1)|`, with the audited cosine epsilon `1e-9`.
- `NC2-ETF` is the audited normalized simplex-ETF structure error.

Smaller values are closer to collapse. Store the per-class norms and the
off-diagonal cosine matrix in addition to aggregate scalars. Evidence:
`evidence/A1-A6_neural_collapse_optimizer.md`, audited commit `7cab4a5`.

### `NC3-self-duality`

With `M` containing the centered class means as columns and `W` the classifier
weight matrix:

```text
NC3 = || W / ||W||_F - M^T / ||M||_F ||_F
```

Use raw `id_train` means and the matching checkpoint classifier. Smaller is
closer to self-duality. Evidence:
`evidence/A1-A6_neural_collapse_optimizer.md`, audited commit `7cab4a5`.

### `feature-norm-IDTest`

After protected-split authorization, compute `||z_i||_2` on raw `id_test`
features and report count, mean, sample standard deviation, CV
`std / mean`, minimum, quantiles, maximum, a histogram, and the same summaries
per class. Do not replace the distribution with only one aggregate. This is a
project measurement protocol rather than a single official method. Evidence:
`evidence/A9_feature_norm_cosine.md` and `output/groupA_summary.md`.

### `RankMe`

Use the uncentered raw `id_train` feature matrix `Z`. If `sigma` contains its
singular values:

```text
p_k = sigma_k / sum_j sigma_j + 1e-7
RankMe = exp(-sum_k p_k log(p_k))
```

The epsilon is added after normalization. Do not center `Z` and do not move the
epsilon into the denominator. This definition is PAPER-verified; no official
standalone implementation was found. Evidence:
`evidence/A7_effective_rank_rankme.md`, RankMe arXiv:2210.02885.

### `covariance-effective-rank`: `UNSPECIFIED`

Issue #24 requests covariance spectrum and covariance effective rank, but the
audit only freezes singular-value effective rank/RankMe on uncentered `Z`. It
does not fix whether a covariance version is centered, which covariance is
used, or whether probabilities derive from eigenvalues or singular values.
A later literature-backed Issue must resolve those fields before implementation.
Do not alias this name to RankMe or invent a formula.

## Frozen logit controls

### `MSP`

```text
MSP(z) = max_c softmax(logits(z))_c
```

Use query logits only, with no fit split, temperature, or transformation. The
score is already ID-like. The existing project implementation remains
authoritative. Evidence: `evidence/B9_MSP.md`, OpenOOD commit `3c35632`, and
`hendrycks/error-detection` commit `276d605bfa9a9bd7701bd88937c537c3fcab94cf`.

### `Energy-T1`

```text
Energy-T1(z) = logsumexp_c(logits_c(z))
```

This is the detection score `-E(z)` at fixed `T = 1`, so it is already
ID-like. Do not negate it or tune temperature. Evidence:
`evidence/B10_Energy.md`, OpenOOD commit `3c35632`, and
`wetliu/energy_ood` commit `77f3c09b788bb5a7bfde6fd3671228320ea0949c`.

## Frozen distance and angle panel

All class statistics and reference banks below use deterministic `id_train`.
Queries are authorized ID-test and per-dataset OOD-test artifacts.

### `Mahalanobis-Penultimate-Raw`

- input: raw penultimate features;
- means: sample-weighted class means;
- covariance: one tied within-class covariance, biased denominator `1/N`;
- estimator/inverse: `sklearn.covariance.EmpiricalCovariance.precision_`
  (`pinvh` behavior), never `torch.inverse`;
- stabilization: no explicit jitter;
- score: `max_c -(z-mu_c)^T Precision (z-mu_c)`.

The score is ID-like. The panel deliberately omits the original method's input
perturbation, multi-layer ensemble, and logistic-regression combination.
Evidence: `evidence/B1_openood_mds.md`, OpenOOD commit `3c35632`.

### `Mahalanobis-Penultimate-L2`

Apply `z_hat = z / ||z||_2` to both fit and query features, failing on zero or
non-finite norms, then use exactly the raw Mahalanobis mean, tied-covariance,
precision, and score contract above. This is an explicitly named project
variant used to isolate feature-norm effects; it is not the audited OpenOOD
base detector and must not overwrite the raw result.

### `kNN-Penultimate-L2`

- normalize every fit/query feature as
  `z / (||z||_2 + 1e-10)`;
- use the complete normalized `id_train` bank and exact squared-L2 distance;
- set `K = 50` with no sweep and no OOD-validation APS;
- return the negative 50th-neighbor squared distance, not a sum or mean.

The score is ID-like. This is the only project kNN definition; a raw/L2-off
kNN variant is forbidden. Evidence: `evidence/B3_knn-ood.md`, audited commit
`2afb2bbed60a8d69384dc9b28e5637711345222b`. The project follows the official
denominator-inside epsilon placement rather than OpenOOD's audited
denominator-outside port difference.

### `NCC-Distance`

Use raw `id_train` class means and query raw features:

```text
NCC-Distance(z) = -min_c ||z - mu_c||_2
```

The score is ID-like. Store the nearest class and distance. This detector score
is not NC4.

### `Prototype-Cosine`

Normalize raw `id_train` class means and every query feature, then compute:

```text
Prototype-Cosine(z) = max_c cos(z, mu_c)
```

The score is ID-like. Store the matched class. `CTM` is the attributed detector
configuration that uses this primitive with both class-mean and query
normalization explicitly enabled. It receives a distinct score name and
metadata even when its numeric primitive is identical. Evidence:
`evidence/B8_CTM.md`, audited commit
`3587259bd6a69abd6b4103cb7311ffaa0857d60f`.

## Required name boundaries

1. `NC4` is the classifier-vs-nearest-class-center **agreement rate**;
   `NCC-Distance` is a per-sample detector score. They are not aliases.
2. `CTM` uses cosine to an ID-train **class mean**, never cosine to a
   classifier-weight row.
3. `RankMe` is singular-value spectral entropy on uncentered features; it is
   not covariance effective rank.
4. Covariance effective rank is not participation ratio. Neither may be
   emitted until its own estimator and spectrum contract are fixed.

Base DDU also remains distinct from Mahalanobis, tied GMM, diagonal GMM, and
shrinkage GMM. [`06_feature_ood_detectors.md`](06_feature_ood_detectors.md) is
the sole authority for DDU and its variants; this card does not redefine them.

## Artifact requirements

Every metric result records:

```text
metric_name and definition_version
checkpoint and raw-artifact identities
fit and query splits
feature transform and normalization
fit/evaluation dtype
class/sample counts
covariance denominator and precision/decomposition method when applicable
fixed and runtime hyperparameters
score direction
per-sample score artifact when applicable
aggregate OOD metric artifact from compute_ood_metrics
smoke_only
```

Detector outputs follow the per-dataset score and near/far arithmetic-mean
contract in reference card 04. Geometry outputs retain per-class, spectrum, or
distribution details needed to reproduce aggregates.

## Freeze and exclusions

The confirmatory panel is frozen before comparative results are inspected.
Result-dependent method addition, sign changes, normalization switches, or
hyperparameter sweeps require a new Issue and explicitly exploratory result
names.

ViM, NeCo, RMD, TwoNN, advanced calibration, DDU variants, and any unresolved
covariance-rank statistic are outside this core card. Their audited definitions
may be promoted by later reference cards, but they are not silently added to
the confirmatory panel.

This card is documentation only. It does not claim that extraction, geometry,
Energy, distance/angle detectors, or DDU are implemented or validated.
