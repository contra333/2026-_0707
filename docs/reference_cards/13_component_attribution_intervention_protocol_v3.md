# Reference Card 13: Component-Attribution Intervention Protocol v3

## 0. Status, scope, and authority

This card defines protocol
`fixed_readout_component_attribution_intervention_v3`. It supersedes Card 12
for the active paper question and future scientific execution. Card 12 and its
failed Stage-2 radial execution remain immutable historical evidence; v3 does
not rewrite, relax, or resume that gate.

Status at creation:

```text
contract and manuscript outline: specified
historical component-attribution implementation: implemented; production run pending
shared-prefix fork runtime: minimally implemented; CPU tests pending
fresh WRN-28-10/CIFAR-10 intervention: NOT_RUN
all replication and modern-scale arms: NOT_RUN
```

Metric Contract v1.2 remains authoritative for its completed feature and score
artifacts. This card adds a new analysis question and a new fork operation; it
does not change v1.2 formulas, artifact keys, results, or reporting roles.

Adjacent authorities are Cards
[`01`](01_optimizers.md), [`02`](02_architectures.md),
[`04`](04_openood_v1_5_protocol.md), [`05`](05_training_protocol.md),
[`08`](08_raw_feature_artifact_contract.md), and
[`11`](11_metric_contract_v1_2.md).

## 1. Paper question and novelty boundary

Provisional title:

> **Where the Gap Lives: Tracing Training-Rule Sensitivity Through Fixed
> Mahalanobis OOD Readouts**

The paper asks:

> Starting from the same training prefix, does changing only decay coupling
> alter a fixed Mahalanobis readout, and through which additive score component
> do actual ID--OOD pair orderings change?

This is an **intervention paper with supporting theory**. It uses known score
decompositions as measurement instruments. It does not claim the following as
novel:

- similar ID accuracy can coexist with different OOD performance;
- the MD--Marginal--RMD relation;
- RMD's subtraction of a global Gaussian reference;
- L2 feature normalization or Mahalanobis++;
- the size--stretch factorization of a quadratic score;
- term-wise size--stretch analysis of RMD;
- radial, spectrum, allocation, or ID-only compatibility analyses already
  developed in *A Geometry-Based View of Mahalanobis OOD Detection*.

The contribution order is frozen as:

1. paired training-to-score-component attribution;
2. exact tie-aware accounting of changed ID--OOD pair orderings;
3. a shared-prefix coupled/decoupled/zero-decay intervention;
4. architecture, dataset/class-count, connectivity, and modern-scale checks.

Comparable ID utility is an interpretation condition, not the title, a
checkpoint-selection rule, or the primary novelty claim.

## 2. Score convention and exact identities

Every score is oriented so a larger value is more ID-like. For a penultimate
feature `z`, let

```text
d_c(z) = (z - mu_c)^T Sigma_W^{-1} (z - mu_c)
d_0(z) = (z - mu_0)^T Sigma_0^{-1} (z - mu_0)
s_MD(z)       = - min_c d_c(z)
s_Marginal(z) = - d_0(z)
s_RMD(z)      = max_c [d_0(z) - d_c(z)]
```

Because `d_0` is independent of class,

```text
s_MD(z) = s_RMD(z) + s_Marginal(z).
```

For ID sample `i`, OOD sample `o`, and detector/component `D`, define

```text
m_D(i,o) = s_D(i) - s_D(o).
```

Then

```text
m_MD(i,o) = m_RMD(i,o) + m_Marginal(i,o).
```

Tie-aware empirical AUROC is exactly the mean pair utility

```text
h(m) = 1 if m > 0; 1/2 if m = 0; 0 if m < 0.
```

An exact tie is never relabeled as an error. For branch `0` and branch `1`, the
artifact records all nine transitions among `{incorrect, tie, correct}`.

For exact two-component attribution of an AUROC change, v3 uses the symmetric
two-player Shapley identity. Let `Delta_R` and `Delta_M` be the branch changes
in RMD and Marginal pair margins. The RMD contribution is

```text
1/2 [h(m0 + Delta_R) - h(m0)
   + h(m0 + Delta_R + Delta_M) - h(m0 + Delta_M)],
```

and the Marginal contribution swaps `R` and `M`. Their pairwise and aggregate
sums reconstruct the exact tie-aware AUROC change. This is attribution of an
additive score change, not a claim that RMD or Marginal AUROCs themselves add.

## 3. Supporting theory and its boundary

### 3.1 Class-independent and class-dependent terms

Expanding the class-conditional quadratic form gives

```text
s_MD(z) = -z^T Sigma_W^{-1} z
          + max_c [2 mu_c^T Sigma_W^{-1} z
                   - mu_c^T Sigma_W^{-1} mu_c].
```

The first term is class-independent; the second is a class-dependent affine
envelope. This algebra explains which score terms are available for empirical
attribution. It does not prove that a training rule preferentially changes one
term.

### 3.2 Low-rank Woodbury statement

Under explicitly stated balanced-class conventions,
`Sigma_0 = Sigma_W + Sigma_B` and `rank(Sigma_B) <= K - 1`. If a positive
ridge makes both covariance matrices full-rank, write `Sigma_B = U U^T` and
apply Woodbury:

```text
(Sigma_W + ridge I + U U^T)^-1
= A^-1 - A^-1 U (I + U^T A^-1 U)^-1 U^T A^-1,
where A = Sigma_W + ridge I.
```

The precision correction has rank at most `rank(U) <= K - 1`. The paper does
not turn this into the stronger claim that a fixed number of raw feature
dimensions “exactly cancels” under pseudoinverse fitting. Pseudoinverse,
regularization, class imbalance, covariance convention, and numerical error
must be stated separately.

### 3.3 Size--stretch as a cited second-stage diagnostic

For each quadratic term and branch `b`, use the factorization from
*A Geometry-Based View of Mahalanobis OOD Detection*:

```text
q_b(x) = r_b(x) w_b(x),  r_b(x) = ||delta_b(x)||^2.
```

The branch change is split without order dependence:

```text
Delta q = ((w1 + w0) / 2) Delta r
        + ((r1 + r0) / 2) Delta w.
```

This identity is applied term-by-term and then to ID--OOD pair margins. It is
not presented as a new factorization. Spectrum and allocation are downstream
diagnostics only after an affected quadratic term and size/stretch channel
have been identified.

### 3.4 What is proved and what is measured

| Supporting theory | Empirical question |
| --- | --- |
| MD score expansion | whether decay coupling changes either term |
| score and pair-margin additive identities | size and direction of observed AUROC change |
| tie-aware pair accounting and Shapley reconstruction | which component carries changed pair utility |
| ridge/full-rank Woodbury rank bound | whether the observed covariance change follows that channel |
| size--stretch product/change identities | whether size or stretch changes under the branch |

## 4. Mechanism hierarchy and terminology

The required diagnostic order is:

```text
additive score component
-> affected quadratic term
-> size / stretch
-> spectrum / allocation diagnostic
-> ID--OOD pair transition
```

Radial geometry is one possible subchannel, not the whole mechanism contract.
RMD is called the **global-referenced class-relative component**. The term
“class-contrastive component” is not used because the global-reference
subtraction can retain more structure than a pure class contrast.

## 5. Historical discovery analysis

Inputs are the frozen 30 `last.pt`/`confirmatory_primary` bundles from Metric
Contract v1.2 and the already retrieved 19.6GB allowlisted reuse tree.

- all 30 bundles and all six existing OOD datasets are reported;
- MD, Marginal, and RMD are reconstructed for raw and L2-normalized fits;
- exact tie-aware pair counts are reported per score;
- only pairs frozen from pre-OOD v1.2 C-role metadata may receive descriptive
  cross-model pair attribution;
- optimizer, LR, and WD are confounded in this grid, so no optimizer-causal
  conclusion is permitted;
- nearest-accuracy matching is absent from the primary analysis;
- no checkpoint is reevaluated and no protected dataset is traversed;
- only files individually verified against the existing reuse allowlist are
  read; no new million-line checksum catalog is generated.

The v2 radial failure stays `FAILED`. v3 uses a new schema, question, output
directory, and analysis identity; it does not weaken the old tolerance or use
partial v2 evidence.

## 6. Shared-prefix confirmation

### 6.1 Main regime and branch families

Main regime: WRN-28-10/CIFAR-10.

```text
Adam zero-decay prefix
  -> Adam coupled
  -> AdamW decoupled
  -> zero-decay continuation

SGDM zero-decay prefix
  -> SGDM coupled
  -> SGDW decoupled
  -> zero-decay continuation
```

Adam is primary with five fresh prefix seeds. SGDM is the conventional-family
control with three fresh prefix seeds. Adam and SGDM prefix populations are
independent experimental blocks.

### 6.2 Fork invariant

`fork_from_prefix` is distinct from strict same-run resume. It requires an
epoch-boundary `last.pt` from an exact zero-decay prefix and preserves:

- model state;
- optimizer tensor state by exact parameter name and group order;
- scheduler state and boundary learning rate;
- Python, NumPy, CPU/CUDA Torch, and train-DataLoader generator RNG state;
- completed epoch, global step, history, and inherited best-validation state.

Only optimizer endpoint/coupling and decay dose may change inside a family.
Cross-family forks, a nonzero-decay prefix, parameter-order drift, scheduler
drift, seed/data/model changes, and source-checkpoint inconsistency fail before
training. A branch manifest records the source checksum and a digest over all
transferred sibling-invariant state. Ordinary resume remains strict.

### 6.3 Values still requiring a pre-OOD addendum

Before fresh protected OOD results are opened, a versioned addendum must freeze:

- switch epoch or prespecified switch-epoch set;
- Adam- and SGDM-family nominal decay dose(s);
- ID accuracy equivalence margin and NLL/ECE guardrails;
- AUROC/FPR95 practical margins;
- uncertainty interval, multiplicity, and main success rule.

Failure of ID equivalence does not delete a branch. It changes interpretation
to a Pareto/trade-off result.

## 7. Replication and scale order

The planned order is:

1. ResNet-18/CIFAR-10: architecture-only replication;
2. ResNet-18/CIFAR-100: dataset and class-count replication;
3. DenseNet-BC-100, growth rate 12/CIFAR-10: dense-connectivity appendix;
4. ConvNeXt-Tiny/ImageNet-200: modern-scale appendix.

DenseNet and ConvNeXt use three fresh Adam-family prefixes and only the focal
Mahalanobis family. They are required paper extensions but do not block the
main WRN causal test. Each unimplemented architecture/dataset receives its own
bounded implementation and actual-data validation task.

ConvNeXt-Tiny is trained from scratch. ImageNet-1K pretrained weights are
forbidden because they expose the model to the 800 ImageNet classes outside
ImageNet-200. Coupled and decoupled branches use the same nominal decay. If ID
equivalence fails, results are reported only as modern-scale Pareto/external-
validity evidence.

ImageNet-200 follows the OpenOOD v1.5 benchmark grouping:

- near OOD: SSB-hard, NINCO;
- far OOD: iNaturalist, Textures, OpenImage-O;
- covariate shifts: separate appendix only.

## 8. Detector roles

Focal family:

- Mahalanobis-Raw;
- Marginal-Mahalanobis-Raw;
- RMD-Raw;
- the three corresponding L2-normalized fits.

External controls: kNN-Raw/L2, CTM, Pure Residual, and Energy-T1. MSP and ViM
are appendix controls. ReAct is excluded. A formula-predicted detector
fragility ordering is excluded from the contribution list unless detector
loading is defined mathematically before results are inspected.

## 9. Artifacts and mandatory validation

Versioned artifacts record:

- prefix, branch, checkpoint, config, seed, dataset, and feature identity;
- per-sample MD, Marginal, and RMD raw/L2 score sources or arrays;
- score and pair-margin reconstruction residuals and scale-aware tolerances;
- exact pair outcome and branch-transition counts;
- component Shapley attribution of AUROC change;
- quadratic-term size/stretch arrays and symmetric branch contributions when
  that diagnostic is run;
- ID accuracy, NLL, ECE, AUROC, FPR95, and practical classification.

Mandatory tests are:

- MD score and pair-margin reconstruction;
- tie-aware AUROC parity;
- exact 3x3 pair-transition accounting;
- symmetric component and size/stretch reconstruction;
- Woodbury only under positive-ridge/full-rank assumptions;
- fork model/optimizer/scheduler/RNG/DataLoader identity;
- sibling first-minibatch identity;
- uninterrupted zero-decay continuation parity;
- ImageNet-200 membership and pretrained-leakage checks before that arm runs.

Tolerance is derived from data scale, float64 machine epsilon, and an
independently measured condition number when relevant. It is never selected by
looking at an observed failed residual.

## 10. Claim gate

The strongest allowed conclusion is:

> A controlled change in decay coupling alters fixed-readout ID--OOD
> orderings, and the effect is selectively concentrated in a
> formula-identified Mahalanobis score component.

That sentence may be used only if component concentration, practical OOD
change, ID interpretation, and required replication gates pass. Failed or
mixed results remain reported. The paper is not rescued with a new detector,
post-hoc checkpoint matching, seed deletion, or an outcome-chosen tolerance.
