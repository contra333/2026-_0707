# Where the Gap Lives: manuscript outline

This is the human-readable manuscript narrative. Exact run counts, estimands,
claim gates, and execution rules are governed only by
[`../reference_cards/13_active_paper_protocol.md`](../reference_cards/13_active_paper_protocol.md).
Changing this outline alone does not change the executable protocol.

## One-sentence paper

Starting from the same initialization, we change decay coupling from epoch 0
and trace how direction-changing updates produce non-affine representation
deformation, alter known Mahalanobis score components, and change or churn the
ordering of the same ID--OOD image pairs.

This is an intervention paper with supporting theory. It is not a new
Mahalanobis decomposition, a new detector, or an Adam-versus-AdamW leaderboard.

## Abstract logic

1. Fixed post-hoc OOD readouts depend on the learned representation, even when
   ID utility is similar; this is established motivation, not the novelty.
2. Existing geometry studies explain how to read an already learned feature
   space, but leave open which controlled training choices create the
   detector-relevant geometry over time.
3. We train coupled, decoupled, and zero-decay siblings from the same
   initialization and data stream, then follow their update and representation
   trajectories across checkpoints and network depth.
4. Affine-gauge invariance removes coordinate changes that a refitted raw MD
   cannot see; local update-direction and RMD low-rank-curvature propositions
   make training- and score-side predictions.
5. Exact MD--Marginal--RMD, size--stretch, and Gain/Loss/PairOrderChurn
   identities connect geometry to both net AUROC change and canceled
   same-pair reversals.
6. Channel-matched diagnostics test the explanation. The conclusion is
   conditional on practical paired effect, non-affine residual, component
   concentration, attenuation, ID-equivalence/Pareto classification, and
   replication.

Do not open with “same ID accuracy, different OOD.” That is the premise.

## 1. Introduction

The paper begins with the gap between two lines of work. One line shows that
training choices can change OOD behavior. The other explains how Mahalanobis
readouts use feature norm, covariance, size, stretch, spectrum, and sample
allocation. The missing link is formation:

> When one update-policy factor changes under otherwise paired training, which
> geometry changes first, which Mahalanobis term reads that change, and which
> actual ID--OOD decisions are reversed?

State contributions in this order:

1. paired from-scratch training-to-geometry-to-score trajectories;
2. theory-constrained attribution using affine gauge, local update direction,
   and RMD low-rank curvature;
3. exact component and pair-order-churn accounting;
4. controlled local factorial evidence plus architecture,
   dataset/class-count, connectivity, and scale replication.

Comparable ID accuracy/NLL/ECE qualifies interpretation. It is not used for
post-hoc checkpoint selection.

## 2. Related work and novelty boundary

Cover training-induced OOD variability, the original Mahalanobis detector,
Marginal Mahalanobis and RMD, Mahalanobis++, and *A Geometry-Based View of
Mahalanobis OOD Detection*.

Explicitly credit prior work for:

- `MD = RMD + Marginal`;
- global-reference subtraction in RMD;
- L2 feature normalization;
- size--stretch, spectrum, allocation, and radial analyses.

The difference is the unit of evidence:

| Axis | Prior geometry/cross-model work | This paper |
| --- | --- | --- |
| Source of variation | completed independently trained/pretrained models or post-hoc transforms | same initialization and data stream, update policy changed from epoch 0 |
| Main question | how to read a learned representation | how detector-relevant geometry is formed |
| Evidence | cross-model association or readout improvement | paired update, geometry, score, and ordering trajectories |
| Endpoint | aggregate detector performance | exact same-image pair transitions plus aggregate metrics |
| Intervention | often representation/readout side | controlled coupled/decoupled/zero-decay training |

## 3. Framework and supporting theory

All scores are ID-like. Define MD, Marginal, and RMD and show:

```text
s_MD(z) = s_RMD(z) + s_Marginal(z)
m_D(i,o) = s_D(i) - s_D(o)
m_MD(i,o) = m_RMD(i,o) + m_Marginal(i,o).
```

Explain that AUROC is the tie-aware fraction of ID--OOD pairs with positive
margin. This makes the same image pair, rather than an aggregate correlation,
the bridge between the score formula and detector behavior.

Organize the paper-specific supporting theory as one constraint chain.

**Proposition 1: affine gauge and residual deformation.** Under a common
invertible affine transformation and a full-rank, affine-equivariant refit,
raw MD score and pair order are unchanged. Therefore global scale, rotation,
and fixed channel rescaling are negative controls. Fit an ID-only branch
alignment `z_C=A z_D+b+e`; attribute the remaining quadratic difference to
same-image residual `e`, class-prototype residual, and precision residual.
Pseudoinverse/ridge/rank conditions are explicit.

**Proposition 2: local update direction.** In a locally positive-scale-
invariant parameter block, explicit decoupled decay is radial. Under a frozen
diagonal preconditioner, coupled L2 can acquire a tangential component unless
`P_t w` is parallel to `w`. Test the operational prediction with an exact
same-state, same-gradient WD-versus-zero counterfactual update. Do not extend
the premise to the whole residual WRN trunk.

**Proposition 3: RMD low-rank curvature.** With class-frequency-weighted
between-class covariance and full-rank inverses,

```text
Sigma_0 = Sigma_W + Sigma_B
rank(Sigma_B) <= K - 1
H = Sigma_W^-1 - Sigma_0^-1 >= 0
rank(H) <= K - 1
s_RMD(z) = -z^T H z + max_c(a_c^T z + b_c).
```

RMD is an affine envelope plus a shared low-rank quadratic correction. This
complements, rather than replaces, the prior term-wise RMD analysis. It yields
a class-count prediction: the bound grows from 9 on CIFAR-10 to 99 on
CIFAR-100, while curvature mass and sample allocation remain empirical.

**Proposition 4: pair-order balance.** For tie-aware pair correctness
`a_r in {0,1/2,1}`, define Gain and Loss as the positive directions of
`a_C-a_D`. Then

```text
DeltaAUROC = Gain - Loss
PairOrderChurn = Gain + Loss
abs(DeltaAUROC) <= PairOrderChurn.
```

Without ties, Gain and Loss are recovered exactly from DeltaAUROC and churn.
This formalizes aggregate cancellation and makes epoch-200 raw-MD
DeltaAUROC plus PairOrderChurn co-primary outcomes. FPR95 operating-point
disagreement is secondary.

Then cite and use the size--stretch factorization:

```text
q(x) = r(x) w(x)
Delta q = ((w_C+w_D)/2) Delta r + ((r_C+r_D)/2) Delta w.
```

In the covariance eigenbasis, use `q=sum_j a_j/lambda_j` to separate spectrum
from sample allocation. Use prespecified spectral bands as primary because
individual eigenvectors are unstable under near ties.

The four computational RMD/Marginal hybrids provide symmetric Shapley
accounting. They are not trained detectors and are not a unique causal
mediation decomposition.

The theory removes impossible explanations and creates measurements; it does
not establish the sign or size of the training effect, component
concentration, ID equivalence, attenuation, or replication.

## 4. Discovery: where the historical gap lives

Use the completed 30 frozen WRN `last.pt` bundles and six OOD datasets only as
descriptive discovery. Show raw and L2 MD/Marginal/RMD component ranges,
pair-outcome dispersion, and prespecified historical pair attribution.

Do not call the mixed optimizer/LR/WD population causal evidence. Do not use
nearest-accuracy matching as a primary analysis. Preserve the failed v2 radial
gate and the successful v3 component analysis exactly as recorded.

This section motivates the focal component question and nothing more.

## 5. Confirmation: paired from-scratch trajectories

### Training design

Main regime: WRN-28-10/CIFAR-10. Coupled, decoupled, and zero-decay siblings
begin at epoch 0 from the same initialization and use identical data order,
augmentation RNG, architecture, loss, batch size, parameter groups, schedule
shape, and checkpoint policy.

Adam-family 2 x 2:

| LR | WD `1e-4` | WD `1e-3` |
| --- | --- | --- |
| `3e-4` | Adam / AdamW | Adam / AdamW |
| `1e-3` | **primary anchor** Adam / AdamW | Adam / AdamW |

- primary anchor: zero/Adam/AdamW, five seeds, full trajectory analysis;
- remaining cells: three paired seeds, epoch-200 focal analysis;
- low-LR zero reference: three seeds;
- total Adam-family budget: 36 runs.

SGDM control: LR `0.1`, WD `5e-4`, zero/SGDM/SGDW, three seeds, nine runs.

Analyze each cell first. Never average raw runs across LR/WD. Use a cell-equal
summary only after showing sign and interaction patterns. Same nominal WD is a
controlled numerical input, not matched effective regularization or a tuned
optimizer comparison. Zero plus nominal WD `1e-4` and `1e-3` already give
three prespecified dose points at each LR. Coupled Adam has no uniquely
separable realized-decay vector because the L2 term changes its moments and
denominator; compare exact same-state, same-gradient WD-versus-zero
counterfactual operator differences instead.

### Time and depth

Use snapshots `0, 1, 10, 30, 60, 61, 120, 121, 160, 161, 200`. Evaluate cheap
ID-only geometry at all snapshots and full OOD at `10, 60, 120, 160, 200`.
Epoch-200 `last.pt` is primary; ID-validation `best_val.pt` is secondary.

Use the penultimate endpoint for the full time trajectory. At epoch 200,
compare stage1, stage2, stage3, and penultimate features. Only trace one
earlier stage over time if the final depth scan localizes the first large
divergence. Because stage widths and spatial resolutions differ, do not compare
raw norm, spectrum, effective-rank, or condition values across stages as one
scale. Compare standardized coupled--decoupled effects within each stage.

An optional three-seed primary-anchor extension to epochs 240 and 300 tests
whether the epoch-200 difference grows, plateaus, or shrinks. It is a
runtime-by-decay-exposure appendix, not a new primary endpoint.

### Concrete mechanism analysis

Follow the same deterministic probe images through four levels:

1. **Update dynamics:** parameter/gradient/update norm, relative update,
   update--weight cosine, radial/tangential update, Adam moments, BN/running
   statistics, residual/shortcut ratio, and exact one-step counterfactuals.
2. **Representation geometry:** ID-only affine alignment and held-out residual;
   global radial scale as a negative control; sample/class-conditioned radial
   heterogeneity; class-mean geometry; global/within covariance; spectral-band
   allocation; nearest versus full class-distance profile.
3. **Score geometry:** branch-specific ID-only Gaussian fits; raw/L2 MD,
   Marginal, and RMD; exact additive reconstruction; size/stretch and
   spectrum/allocation decomposition.
4. **OOD ordering:** the same ID--OOD pairs classified as incorrect/tie/correct
   at every checkpoint, with exact component attribution, Gain, Loss,
   PairOrderChurn, and net DeltaAUROC.

For every seed and cell report coupled--decoupled, coupled--zero, and
decoupled--zero. The update policies differ by construction at the first
nonzero-decay step, so “update onset precedes geometry onset” is not a claim.
Report standardized divergence curves, minimum detectable effects, early
slopes, and functional/cumulative summaries. A practical-threshold crossing is
called detectability time and is descriptive, not a causal onset. A mechanism
claim requires an
exact update manipulation check, repeatable non-affine representation
divergence, formula-level score/pair reconstruction, Gain/Loss attribution,
and selective channel attenuation. Treat seed, not image pair, as the
independent statistical unit; same-policy different-seed variation is a
natural-variability reference, not a null.

### Channel-matched confirmation

- sample/class-conditioned radial difference -> L2 normalization and refitting;
- non-affine deformation -> ID-only affine alignment plus residual/precision accounting;
- global/Marginal difference -> RMD;
- spectrum/stretch difference -> prespecified ID-only spectral-band ablation,
  clipping, or whitening diagnostic;
- class-distance-profile difference -> nearest-class versus full profile.

Do not add a detector after seeing results. Scalar correlations can support,
but cannot replace, exact accounting and selective attenuation.

### Inferential hierarchy

- Primary outcome: WRN anchor, epoch 200, penultimate, raw MD,
  DeltaAUROC and PairOrderChurn, prespecified near/far summaries.
- Primary mechanism: Marginal/RMD pair attribution, affine residual,
  global-scale negative control versus radial heterogeneity, counterfactual
  tangential update, and channel-matched attenuation.
- Secondary: L2/size--stretch, functional time curves, within-stage depth,
  Adam LR/WD interaction, SGDM control, best-validation, epoch 300.
- Exploratory/appendix: external detector panel, individual eigenvectors,
  extra geometry scalars, and optional fork.

Do not perform an unadjusted test for every checkpoint by depth by metric by
OOD dataset. Freeze family-wise uncertainty, practical margins, minimum
detectable effects, spectral bands, and multiplicity handling before protected
OOD execution.

## 6. Result map

| Observation | Allowed interpretation |
| --- | --- |
| Practical OOD gap/churn, non-affine residual, localized component, selective attenuation | strongest mechanism result |
| OOD gap with failed ID equivalence | training-rule Pareto result; remove comparable-ID wording |
| Same direction across four Adam cells | locally robust coupling effect |
| Larger effect at stronger WD | coupling-by-decay-strength response |
| Magnitude changes with LR | coupling effect is step-scale sensitive |
| Sign reversal across cells | recipe interaction, not a universal coupling effect |
| Geometry changes but OOD does not | detector-relevant versus irrelevant geometry distinction; central claim weakens |
| Pair identities change but AUROC cancels | decision-level instability hidden by the aggregate |
| Branches are almost entirely related by one affine map | raw-MD gap should be absent; lower the non-affine mechanism claim |
| Tangential counterfactual update and non-affine residual grow together | supports coupling-to-direction-to-representation pathway |
| Difference disappears after epoch 200 | finite-time/epoch-budget interaction |
| Adam only | adaptive preconditioning-by-coupling interaction |
| Adam and SGDM | more general coupling effect |
| No fresh effect | historical gap was not primarily caused by coupling under this design |

Null outcomes remain in the paper. They are not rescued by matching
checkpoints, changing epoch 200, or expanding the detector panel.

## 7. Replication and scale

1. ResNet-18/CIFAR-10 architecture-only replication.
2. ResNet-18/CIFAR-100 dataset/class-count replication and RMD
   curvature-rank/allocation prediction.
3. DenseNet-BC-100 k=12/CIFAR-10 focal appendix.
4. ConvNeXt-Tiny/ImageNet-200 from-scratch focal appendix.

DenseNet and ConvNeXt use three Adam-family fresh seeds and the focal
Mahalanobis family. ConvNeXt is external-validity evidence, not a clean
BN-versus-LN mechanism test. ImageNet-1K pretrained weights are forbidden.

## 8. Discussion and conclusion

Discuss local normalized-block effective-step dynamics without calling the
whole WRN trunk scale-invariant, the difference between comparable-ID and
Pareto evidence, the limits of affine/full-rank assumptions and CIFAR Gaussian
fitting, and the boundary between tested optimizer families and universal
claims.

Use the Card 13 strong sentence only if every corresponding gate passes.

## Figure plan

1. From-scratch paired design and the update -> geometry -> score -> ordering
   chain, including MD/RMD/Marginal pair-margin decomposition.
2. Historical component localization across 30 bundles and six OOD datasets.
3. Fresh trajectories: counterfactual tangential update, non-affine residual,
   and representation geometry over time and within-stage depth.
4. ID utility, DeltaAUROC versus PairOrderChurn, Gain/Loss transitions, and
   component/size/stretch attribution at the primary endpoint.
5. Adam 2 x 2, SGDM control, and WRN--ResNet--DenseNet--ConvNeXt replication
   matrix.

## Table plan

1. Prior work versus the paired formation study.
2. Exact run budget, checkpoints, depth taps, and execution status.
3. Main paired ID/OOD estimates and practical classifications.
4. Component, affine residual, divergence, churn, and attenuation evidence by OOD dataset.
5. Replication claim gate and numerical-validation appendix.
