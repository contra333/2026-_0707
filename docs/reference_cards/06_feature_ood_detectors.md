# Reference Card 06: Feature-Based OOD Detector Semantics

## Purpose and authority boundary

This card defines durable naming, fitting, scoring, and artifact semantics for
feature-based OOD detectors in this repository. The first detector fixed here is
DDU. The implementation status remains **planned** until a later bounded Issue
adds code, tests, configuration, and real-environment validation.

The following cards remain authoritative for adjacent concerns:

- [`02_architectures.md`](02_architectures.md) defines the penultimate feature
  exposed immediately before `model.classifier`.
- [`04_openood_v1_5_protocol.md`](04_openood_v1_5_protocol.md) defines fixed
  CIFAR-10 ID/OOD membership, preprocessing, ID-like score orientation, metrics,
  and per-dataset aggregation.
- [`05_training_protocol.md`](05_training_protocol.md) defines checkpoint and
  run metadata. Whether spectral normalization was used is a training variable,
  not a detector-name change.

OpenOOD is not the authority for the DDU estimator. For DDU, the pinned paper
and official `omegafragger/DDU` implementation in `docs/sources.lock.yaml` are
the external references, while this card fixes the project-facing contract.

## Common detector rules

- Fit detector statistics using `id_train` features only.
- Never fit detector parameters on `id_validation`, `id_test`, OOD validation,
  or OOD test samples.
- Main-protocol detector and variant hyperparameters must be fixed a priori or
  selected using ID-only information. OpenOOD's compatibility-only OOD
  validation split is not used.
- Every project-facing detector score is ID-like: larger means more ID-like.
- Fit and query features must come from the same documented model state and the
  same penultimate-feature endpoint.
- A transformation applied to fitting features must also be applied identically
  to query features.
- Base detectors and post-hoc variants must produce distinct score names and
  artifacts. A variant must never overwrite or be reported as the base result.
- Reject non-finite features, non-finite fitted parameters, empty classes, and
  non-finite scores with a clear error.

## DDU naming and experimental boundary

The project-facing detector name is **DDU**, including when the evaluated
classifier was trained without spectral normalization (SN).

SN on/off is a separately recorded training ablation. It changes the learned
representation supplied to the detector, but it does not rename the detector to
`DDU-style` or another alias. Every DDU result must retain the checkpoint's
training metadata so the paper can distinguish main SN-off runs from the SN
ablation.

The base detector name is `DDU`. Explicit post-hoc variants use names such as:

- `DDU-PCA`
- `DDU-Diag`
- `DDU-L2`
- `DDU-Shrinkage`

Compositions must be named compositionally, for example `DDU-L2-PCA`, and must
not be pooled with a one-axis ablation.

## Base DDU definition

Let the ID-train penultimate feature of sample `i` be
`z_i = f_theta(x_i) in R^d`, and let `I_c = {i : y_i = c}` with
`n_c = |I_c|`.

For each class `c`, fit the empirical mean

```text
mu_c = (1 / n_c) * sum_{i in I_c} z_i
```

and the unbiased full covariance

```text
Sigma_c = (1 / (n_c - 1))
          * sum_{i in I_c} (z_i - mu_c) (z_i - mu_c)^T.
```

The base DDU fit uses:

- raw penultimate features;
- one full covariance matrix per class;
- denominator `n_c - 1`;
- float64 fitting and density evaluation unless a later implementation Issue
  provides verified parity for another precision;
- no global mean centering;
- no feature-wise standardization;
- no L2 normalization;
- no PCA;
- no diagonal approximation;
- no statistical covariance shrinkage;
- no explicit matrix inverse or Moore-Penrose pseudo-inverse.

The paper describes the class-conditional GDA density. The pinned official code
is authoritative for the numerical covariance and score convention below.

## Official adaptive jitter contract

The official implementation constructs this ordered candidate sequence:

```text
0
finfo(float64).tiny
10^-308, 10^-307, ..., 10^-1
```

For one candidate `epsilon`, add the same diagonal jitter to every class
covariance:

```text
Sigma_c(epsilon) = Sigma_c + epsilon * I_d.
```

Then attempt to construct one class-batched
`torch.distributions.MultivariateNormal` with all class means and covariance
matrices. Select the first candidate for which construction, including its
positive-definiteness/Cholesky checks, succeeds.

The later implementation must preserve these semantics:

- one shared candidate `epsilon` per fit attempt, not a separately selected
  jitter for each class;
- candidates tried in the exact order above;
- the first successful candidate is selected;
- the selected value is stored in artifacts;
- if every candidate fails, fail loudly rather than returning a partially
  initialized detector.

Jitter is a numerical positive-definiteness repair. It is not the same as a
statistical shrinkage estimator and must not be labeled `shrinkage`.

## Base DDU score

For a query feature `z`, compute every class log density

```text
ell_c(z) = log Normal(z; mu_c, Sigma_c(epsilon)).
```

Following the pinned official repository, aggregate with

```text
DDU(z) = logsumexp_c ell_c(z).
```

This is already ID-like: a larger feature-space log density means more ID-like.
Do not negate it at the project boundary.

The official code does not add an explicit `log pi_c` class-prior term before
`logsumexp`. The balanced CIFAR-10 setting therefore follows the exact official
code convention. Adding a uniform prior would only subtract the constant
`log K`, leaving rank metrics unchanged, but it is still a different stored
score. A future class-imbalanced protocol must make its prior convention
explicit rather than silently changing this detector.

The base DDU score is not any of the following:

- maximum class density instead of `logsumexp`;
- negative nearest-class Mahalanobis distance;
- a determinant-free quadratic distance;
- a tied-covariance Gaussian score;
- a diagonal or shrinkage covariance score.

## Rank and conditioning facts

For centered class features, the empirical covariance satisfies

```text
rank(Sigma_c) <= min(d, n_c - 1).
```

It is therefore guaranteed singular when `n_c - 1 < d`. The converse is false:
when `n_c - 1 >= d`, collapse, duplicated directions, low effective rank, or
finite-precision effects can still make the covariance singular or severely
ill-conditioned.

For the current WRN-28-10 CIFAR-10 main path, `d = 640` and the detector fits on
`id_train`, which has about 5,000 samples per balanced class. Sample count alone
does not guarantee singularity in that setting, but the official adaptive jitter
is still mandatory because representation geometry can be low-rank or
ill-conditioned. The 1,000-sample ID validation split and 9,000-sample ID test
split are not covariance-fitting data.

## Relationship to neighboring detector families

DDU must remain distinct from generic GMM and distance baselines:

- Mahalanobis commonly uses a shared covariance and a nearest-class quadratic
  distance, often omitting the Gaussian log-determinant and mixture aggregation.
- `GMM-Tied` uses one covariance shared across classes.
- `GMM-Diag` uses class-wise diagonal covariance.
- `GMM-Shrinkage` uses an explicitly named statistical shrinkage rule.
- Base DDU uses class-wise full unbiased covariance, official adaptive jitter,
  full Gaussian log density, and class `logsumexp` aggregation.

A later experiment may compare these methods, but no implementation may alias
one score to another merely because some terms coincide in a special case.

## Planned post-hoc DDU variants

### `DDU-PCA`

Fit PCA using ID-train features only, transform both fitting and query features,
and then apply the base DDU mean, covariance, jitter, and score contract in the
projected space.

The configuration must record the PCA centering rule and either an explicit
component count or an explicit ID-only component-selection rule. The canonical
component-selection rule is not fixed by this card and requires a later
literature-backed implementation Issue.

### `DDU-Diag`

Replace each class full covariance with the diagonal matrix formed from that
class's unbiased per-dimension variances. Preserve class-wise fitting, Gaussian
log-density terms, `logsumexp` aggregation, ID-like orientation, and explicit
jitter metadata.

An implementation may use an algebraically equivalent diagonal Gaussian path
instead of materializing dense diagonal matrices only if parity is tested.

### `DDU-L2`

Normalize every fitting and query feature before any class statistic is fitted:

```text
z_hat = z / ||z||_2.
```

A zero-norm or non-finite feature must fail clearly; it must not be silently
replaced by an arbitrary direction. Fit the base class-wise full covariance and
apply the official jitter and score contract on normalized features.

### `DDU-Shrinkage`

Replace each class empirical covariance with an explicitly identified
statistical shrinkage estimator or an explicitly configured shrinkage target and
coefficient. The estimator name, target, coefficient-selection rule, and any
remaining numerical jitter must be stored.

Diagonal loading used only to make Cholesky succeed is numerical jitter, not
`DDU-Shrinkage`. This card intentionally does not select Ledoit-Wolf, OAS, or a
fixed coefficient; that decision requires a cited later Issue.

## Required configuration and artifact metadata

A later implementation must record at least:

```text
score_name
base_detector = DDU
variant_operations in application order
fit_split = id_train
checkpoint identifier and completed epoch
model name and feature endpoint
spectral-normalization training flag or equivalent checkpoint metadata
feature_dim_before_transform
feature_dim_after_transform
class counts n_c
fit dtype
covariance type
covariance denominator
class-prior convention
jitter candidate policy
selected jitter epsilon
PCA configuration when applicable
L2 normalization flag when applicable
shrinkage estimator/target/coefficient when applicable
implementation source commit
```

Per-sample artifacts use the common `id_like_score` field from the OpenOOD
protocol card. Aggregate metrics remain the already pinned project metrics and
per-dataset near/far arithmetic-mean policy.

## Later implementation validation requirements

The future code Issue must include focused tests for:

- class means and unbiased covariance against a direct float64 calculation;
- the exact official jitter candidate order;
- first-success shared jitter selection on singular synthetic covariance;
- a loud failure when all candidates are exhausted;
- per-class log-density and `logsumexp` parity against the pinned official DDU
  code on synthetic features;
- ID-like score orientation;
- separation of base and variant score names/artifacts;
- PCA, diagonal, L2, and shrinkage transformation consistency between fitting
  and query features;
- complete finite-score and metadata validation.

A bounded actual-checkpoint CUDA or CPU evaluation is also required in that
later Issue. This documentation card alone is not implementation or detector
performance evidence.
