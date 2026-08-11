# Where the Gap Lives: manuscript outline

## One-sentence paper

Starting from the same initialization, we change decay coupling from epoch 0
and trace how the resulting update trajectories create different
representation geometry, change known Mahalanobis score components, and
reverse the ordering of the same ID--OOD image pairs.

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
4. We use the exact MD--Marginal--RMD and size--stretch identities to connect
   geometry changes to the score and to every tie-aware ID--OOD pair
   transition.
5. Channel-matched diagnostics test the explanation: L2 for radial change,
   RMD for the global/Marginal channel, and fixed spectral controls for
   stretch/allocation.
6. The conclusion is conditional on practical OOD effect, component
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
2. exact pair-order and component accounting;
3. controlled local factorial evidence for decay coupling and its interaction
   with LR and nominal decay strength;
4. architecture, dataset/class-count, connectivity, and scale replication.

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

Expand the shared-covariance Mahalanobis score into its class-independent
quadratic term and class-dependent affine envelope. Under explicit ridge or
full-rank conditions, include the `rank(Sigma_B) <= K-1` Woodbury correction
as a supporting proposition only.

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

Theory establishes identities and rank bounds. It does not establish which
training rule moves which term, the AUROC effect size, or replication. Those
are empirical questions.

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
optimizer comparison.

### Time and depth

Use snapshots `0, 1, 10, 30, 60, 61, 120, 121, 160, 161, 200`. Evaluate cheap
ID-only geometry at all snapshots and full OOD at `10, 60, 120, 160, 200`.
Epoch-200 `last.pt` is primary; ID-validation `best_val.pt` is secondary.

Use the penultimate endpoint for the full time trajectory. At epoch 200,
compare stage1, stage2, stage3, and penultimate features. Only trace one
earlier stage over time if the final depth scan localizes the first large
divergence.

An optional three-seed primary-anchor extension to epochs 240 and 300 tests
whether the epoch-200 difference grows, plateaus, or shrinks. It is a
runtime-by-decay-exposure appendix, not a new primary endpoint.

### Concrete mechanism analysis

Follow the same deterministic probe images through four levels:

1. **Update dynamics:** parameter/gradient/update norm, relative update,
   update--weight cosine, radial/tangential update, and Adam moment summaries.
2. **Representation geometry:** feature norms; class-mean distances/angles and
   CDNV; global and within-class covariance spectrum, effective rank, and
   condition; spectral-band allocation; nearest versus full class-distance
   profile.
3. **Score geometry:** branch-specific ID-only Gaussian fits; raw/L2 MD,
   Marginal, and RMD; exact additive reconstruction; size/stretch and
   spectrum/allocation decomposition.
4. **OOD ordering:** the same ID--OOD pairs classified as incorrect/tie/correct
   at every checkpoint, with exact component attribution of pair flips.

For every seed and cell report coupled--decoupled, coupled--zero, and
decoupled--zero. A mechanism claim requires the update difference to precede
the geometry difference, the geometry to reconstruct a named score component,
and the named channel to account for and selectively attenuate pair changes.

Use two-consecutive-checkpoint onset after freezing seed-noise/practical
thresholds. Treat seed, not image pair, as the independent statistical unit.

### Channel-matched confirmation

- radial difference -> L2 normalization and refitting;
- global/Marginal difference -> RMD;
- spectrum/stretch difference -> prespecified ID-only spectral-band ablation,
  clipping, or whitening diagnostic;
- class-distance-profile difference -> nearest-class versus full profile.

Do not add a detector after seeing results. Scalar correlations can support,
but cannot replace, exact accounting and selective attenuation.

## 6. Result map

| Observation | Allowed interpretation |
| --- | --- |
| Practical OOD gap, localized component, correct temporal order, selective attenuation | strongest mechanism result |
| OOD gap with failed ID equivalence | training-rule Pareto result; remove comparable-ID wording |
| Same direction across four Adam cells | locally robust coupling effect |
| Larger effect at stronger WD | coupling-by-decay-strength response |
| Magnitude changes with LR | coupling effect is step-scale sensitive |
| Sign reversal across cells | recipe interaction, not a universal coupling effect |
| Geometry changes but OOD does not | detector-relevant versus irrelevant geometry distinction; central claim weakens |
| Pair identities change but AUROC cancels | decision-level instability hidden by the aggregate |
| Difference disappears after epoch 200 | finite-time/epoch-budget interaction |
| Adam only | adaptive preconditioning-by-coupling interaction |
| Adam and SGDM | more general coupling effect |
| No fresh effect | historical gap was not primarily caused by coupling under this design |

Null outcomes remain in the paper. They are not rescued by matching
checkpoints, changing epoch 200, or expanding the detector panel.

## 7. Replication and scale

1. ResNet-18/CIFAR-10 architecture-only replication.
2. ResNet-18/CIFAR-100 dataset/class-count replication.
3. DenseNet-BC-100 k=12/CIFAR-10 focal appendix.
4. ConvNeXt-Tiny/ImageNet-200 from-scratch focal appendix.

DenseNet and ConvNeXt use three Adam-family fresh seeds and the focal
Mahalanobis family. ConvNeXt is external-validity evidence, not a clean
BN-versus-LN mechanism test. ImageNet-1K pretrained weights are forbidden.

## 8. Discussion and conclusion

Discuss normalized-network effective-step dynamics as an alternative path,
the difference between comparable-ID and Pareto evidence, the limits of CIFAR
discovery and Gaussian fitting, and the boundary between tested optimizer
families and universal claims.

Use the Card 13 strong sentence only if every corresponding gate passes.

## Figure plan

1. From-scratch paired design and the update -> geometry -> score -> ordering
   chain, including MD/RMD/Marginal pair-margin decomposition.
2. Historical component localization across 30 bundles and six OOD datasets.
3. Fresh trajectories: update and representation geometry over time and depth.
4. ID utility, OOD effect, pair transitions, and component/size/stretch
   attribution at the primary endpoint.
5. Adam 2 x 2, SGDM control, and WRN--ResNet--DenseNet--ConvNeXt replication
   matrix.

## Table plan

1. Prior work versus the paired formation study.
2. Exact run budget, checkpoints, depth taps, and execution status.
3. Main paired ID/OOD estimates and practical classifications.
4. Component, geometry, onset, and attenuation evidence by OOD dataset.
5. Replication claim gate and numerical-validation appendix.
