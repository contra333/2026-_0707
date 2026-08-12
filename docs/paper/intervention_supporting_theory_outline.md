# When the Ranking Flips: manuscript outline

This is the human-readable manuscript narrative. Exact run counts, estimands,
claim gates, and execution rules are governed only by
[`../reference_cards/13_active_paper_protocol.md`](../reference_cards/13_active_paper_protocol.md).
Changing this outline alone does not change the executable protocol.

## One-sentence paper

Starting from the same initialization and data stream, changing decay coupling
can make a fixed OOD readout reverse the ordering of the same ID--OOD image
pairs beyond ordinary seed variation; we localize that pair-ranking
multiplicity to the update, representation, and Mahalanobis score channels
that carry it.

This paper treats pair-ranking multiplicity as the problem and identifies the
geometric formation and detector visibility of training-rule sensitivity. It
is an intervention paper with supporting theory,
not an Adam-versus-AdamW leaderboard. One secondary, theory-derived `RtMD`
method slot is preregistered, but it becomes a contribution only if its
derivation, novelty, ID-only tail, protected-OOD, and replication gates pass.

## Abstract logic

1. Fixed post-hoc OOD readouts can be unstable across seeds and training
   details even when ID utility is similar; that existence claim is prior
   motivation, not the novelty.
2. What is missing is controlled, same-image evidence of how much pair-ranking
   churn one training-rule factor creates beyond seed variation, and which
   score channel carries it.
3. We train coupled, decoupled, midpoint-mixed, and zero-decay siblings from
   the same initialization and data stream, then follow their update and
   representation trajectories across checkpoints and network depth. A fixed-total-decay
   `alpha in {0,0.5,1}` anchor tests interior/monotonic compatibility without
   claiming a resolved dose-response curve.
4. Exact Gain/Loss/PairOrderChurn makes aggregate cancellation visible and
   compares policy churn with same-policy seed churn.
5. Under explicit full-rank/common-ridge conditions, Raw MD and Marginal share
   the same `S-perp` residual term and RMD cancels it. Low-rank curvature is a
   corollary of RMD acting only on `S`; the theorem is the explanatory
   interface, not the headline problem.
6. Branch-internal `S-perp`/parallel-Marginal/RMD identities connect geometry
   to both net AUROC and canceled same-pair reversals. A raw/L2 x MD/RMD interaction and fixed
   residual-retention path distinguish nonlinear normalization from algebraic
   cancellation. Prior update, size--stretch, and spectrum--allocation tools
   explain how a localized channel formed.
7. Prespecified channel-matched diagnostics test the explanation. The
   conclusion is conditional on estimator applicability, practical paired
   effect, repeatable representation divergence, component concentration,
   attenuation, ID-equivalence/Pareto classification, and replication.
8. If the residual-tail gates pass, evaluate one frozen Residual-t
   Mahalanobis candidate that retains residual evidence while limiting
   heavy-tail domination. Its failure does not weaken items 1--7.

Do not open with “same ID accuracy, different OOD.” That is the premise.

## 1. Introduction

The paper begins with the reliability problem: two models with similar ID
utility can give different pair-level OOD rankings after a seemingly minor
training-rule change. Prior work establishes such instability in aggregate;
the missing link is controlled attribution and formation:

> When one update-policy factor changes under otherwise paired training, which
> geometry changes first, which Mahalanobis term reads that change, and which
> actual ID--OOD decisions are reversed?

State contributions in this order:

1. controlled pair-ranking multiplicity beyond same-policy seed variation;
2. paired from-scratch training-to-subspace-to-score trajectories;
3. discriminant--residual RMD cancellation as the explanatory interface,
   together with exact subspace component and pair-order-churn accounting;
4. a fixed-total-decay three-point alpha confirmation with a
   prespecified normalization-by-cancellation diagnostic and residual-channel
   attenuation path;
5. controlled local factorial evidence plus architecture,
   discriminant-capacity, connectivity, and scale replication.
6. conditionally, one preregistered residual-tail method derived from the same
   score interface, with no OOD-tuned formula or second rescue method.

Comparable ID accuracy/NLL/ECE qualifies interpretation. It is not used for
post-hoc checkpoint selection.

## 2. Related work and novelty boundary

Cover training-induced OOD variability, the original Mahalanobis detector,
Marginal Mahalanobis and RMD, Mahalanobis++, and *A Geometry-Based View of
Mahalanobis OOD Detection*. Treat WDiscOOD as the direct whitened
discriminative/residual score predecessor. Audit NECO, ViM, CORE, MaRS,
Neural-Collapse Mahalanobis scores, robust/t-Mahalanobis models,
classifier/principal/residual-subspace methods, and PCA projection filtering
as direct neighboring boundaries.

Use Szyc et al. and later training-induced OOD studies to establish that
seeds and training details already perturb aggregate OOD performance and
detector rankings. Use predictive-churn/multiplicity work for decision-level
language. The new target is narrower: controlled same-initialization
**ID--OOD pair-ranking** churn, its score-component localization, and its
formation trajectory. Do not blur this with thresholded classification churn.

Position Zhao et al. as upstream optimization/Neural-Collapse theory and as the
precedent for fixed-total-decay coupling interpolation. Zhao does not study
OOD detection or Mahalanobis readouts. Our question is which
optimizer-sensitive geometry is transmitted to the detector interface.
NC0--NC4 span classifier, within-class, class-mean, and alignment structure,
so do not simplify Zhao's prediction to “the `S` side changes.” Report the NC
profile as a supporting transfer analysis against `q_perp`, `P_S x`, score
components, and pair churn.

Explicitly credit prior work for:

- Fisher/LDA class-mean discriminant subspaces and the `K-1` rank bound;
- `MD = RMD + Marginal`;
- global-reference subtraction in RMD;
- L2 feature normalization;
- size--stretch, spectrum, allocation, and radial analyses.

Do not claim that the class-discriminative subspace itself is new. The theorem
candidate is the OOD-score statement that Raw MD and Marginal share the same
`S-perp` term and RMD cancels it. Avoid “first” language until the direct
RMD/LDA and subspace-detector literature audit is complete. Mahalanobis++
already refits MD and RMD after L2 normalization; the paper's possible
addition is a controlled, component-level explanation of when normalization
and cancellation interact, not the normalized detector itself.

WDiscOOD already combines nearest-class distance in a whitened discriminative
subspace with a weighted residual-centroid distance. Therefore neither the
`S/S-perp` split nor a linear residual weight is a method contribution here.
The conditional `RtMD` candidate must earn a separate novelty claim through
its residual-only heavy-tail likelihood, ID-only fitting, policy-stability
target, and controlled formation evidence. The full-text gate explicitly
includes Linderman et al.'s Bayesian-nonparametric/DPMM--RMDS connection and
predictive generalizations; a direct collision closes the slot. CORE is an
adjacent
classifier-aligned/residual combination; MaRS uses Mahalanobis scoring of
autoencoder reconstruction residuals and is related but not the same object.

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

**Theorem 1: discriminant--residual RMD cancellation.** Let
`Sigma_0=Sigma_W+Sigma_B` under the shared class-frequency-weighted `1/N`
sample convention. Under a numerically full-rank inverse, or one common
positive ridge applied to every conditional and marginal term, whiten with
`A=Sigma_W` or `A=Sigma_W+lambda I` and define

```text
x = A^(-1/2)(z-mu_0)
eta_c = A^(-1/2)(mu_c-mu_0)
S = span{eta_c}
x = x_parallel + x_perp.
```

Because every class-mean offset and the whitened between-class covariance lie
in `S`, the scores decompose exactly as

```text
s_perp = -||x_perp||^2
s_parallel_Marginal =
    -x_parallel^T (I+B|_S)^(-1) x_parallel
s_RMD = -min_c ||x_parallel-eta_c||^2 - s_parallel_Marginal
s_Marginal = s_perp + s_parallel_Marginal
s_MD = s_perp + s_parallel_Marginal + s_RMD
s_RMD(x) = s_RMD(x_parallel)

m_q(i,o) = s_q(i) - s_q(o)
m_MD = m_perp + m_parallel_Marginal + m_RMD.
```

Raw MD and Marginal therefore share the literal same `S-perp` term and RMD
cancels it. `dim(S)<=K-1`; the old low-rank-curvature result follows as a
corollary. This theorem says what the scores can see. It does not predict that
decay coupling changes `S-perp`, the sign of an OOD effect, or which detector
wins.

For one fixed branch, transform, and fit, summarize the sample-side input to
this Gaussian score family by the **detector interface**
`(q_perp, x_parallel)=(||P_S-perp x||^2,P_S x)`. The whitening map, `S`, class
centers, and `B|_S` remain branch-specific fitted state. Call this
score-sufficient relative to the fixed fit, not a general statistical
sufficient statistic or a complete description of the representation.

In the unregularized full-rank case, pooled ID-train within-class whitening
also gives exact mean residual energies `d-dim(S)` in `S-perp` and `dim(S)` in
`S`. This explains large mean residual energy in high dimension, not its
sample variance, correlation with raw feature norm, policy sensitivity, or OOD
utility. Under a common ridge these means become projected traces.

State the pinning boundary explicitly. Whitening pins the pooled ID-train
residual vector's first two moments and therefore `E[q_perp]=d-dim(S)`. It does
not pin `Var(q_perp)`, because that depends on fourth moments, nor does it pin
skewness, kurtosis, upper tails, held-out-ID drift, or OOD placement. Organize
the empirical search around these non-pinned categories without claiming that
they form an exhaustive list of exactly four degrees of freedom.

The theorem applies separately to each raw or L2-normalized fit. L2 maps each
sample nonlinearly and refits means, covariance, whitening, and `S`; it does
not literally scale the raw fit's `s_perp`. Therefore “L2 partially suppresses
the same channel that RMD fully removes” is a gated empirical interpretation,
not a corollary. Test it through the raw/L2 x MD/RMD interaction,
transform-specific component covariance, raw-norm/component correlations, and
pair-margin attenuation. Broad Mahalanobis++ results in which normalization
often also improves RMD forbid a universal claim that L2 is redundant or
harmful after RMD.

Inside a fixed branch and transform, define the score diagnostic

```text
s_MD^(rho) = rho s_perp + s_parallel_Marginal + s_RMD,
m_MD^(rho) = rho m_perp + m_parallel_Marginal + m_RMD,
rho in [0,1].
```

It connects MD (`rho=1`) to `S`-only MD (`rho=0`) and yields exact pair-flip
thresholds. It is a channel-dose diagnostic, not an optimized detector. A
putative feature map that scales `S-perp` by `gamma>0` and then refits MD is
affine-invariant and changes nothing; `gamma=0` is a singular projection.
Thus do not present the `rho` path as a new refitted normalization or select a
best `rho`.

**Conditional method track: Residual-t Mahalanobis.** Reserve one candidate of
the form

```text
q_S = min_c ||x_parallel-eta_c||^2
D_RtMD = q_S + g_(nu,k)(q_perp),  k=dim(S-perp),
```

where `g` is one frozen multivariate-t radial negative log likelihood. The
idea is not to discard `S-perp`, but to retain its OOD evidence without letting
a Gaussian quadratic tail dominate. The exact covariance-versus-scatter
parameterization, `nu`/scale fit, ID-only split, fallback, and MD limit must be
derived and frozen before the historical residual-tail preflight. The current
covariance-normalized working form proportional to
`(nu+k) log(1+q_perp/(nu-2))`, `nu>2`, is not executable until that lock.

Activate protected-OOD evaluation only after: the direct novelty audit passes;
historical held-out ID supports an estimable non-Gaussian residual tail; and
fresh paired ID-only trajectories show a preregistered policy effect on tail
shape beyond same-policy seed variation. Judge the method primarily by reduced
policy `PairOrderChurn` relative to Raw MD, with frozen AUROC/FPR95 and far-OOD
residual-signal guardrails, then require replication. On any failed gate, close
the method slot and keep the mechanism paper. Never tune `nu`, choose the
parameterization, or repair the formula with protected OOD.

An ID-only `nu-hat` may be explored as a predictor of fresh policy churn, not
detector performance. Heavy tails alone do not establish that prediction.

The frozen Metric Contract v1.2 uses Moore--Penrose-compatible precision
without explicit ridge. Do not silently change it. Apply exact cancellation to
the primary score only if the actual backend passes full-rank, condition,
inverse-parity, and reconstruction gates. Otherwise mark the theorem
inapplicable to that primary fit and name any common-ridge calculation as a
separate diagnostic.

Each branch refits its own whitening and `S`. Exact score attribution is
branch-internal. Interpret cross-branch formation only after ID-only gauge
alignment, with principal-angle and whitening-change reports plus a
zero-decay common-frame diagnostic. `S` is class-mean-discriminative for this
tied-covariance detector, not all information used by the classifier;
`S-perp` may still contain OOD or higher-order information.

**Supporting lemma 1: affine gauge and residual deformation.** Under a common
invertible affine transformation and a full-rank, affine-equivariant refit,
raw MD score and pair order are unchanged. Therefore global scale, rotation,
and fixed channel rescaling are negative controls. Fit an ID-only branch
alignment `z_C=A z_D+b+e`; attribute the remaining quadratic difference to
same-image residual `e`, class-prototype residual, and precision residual.
Fit on ID train, then report held-out ID and each OOD residual separately. The
held-out ID residual is the floor: the cleanest mechanism is an affine map that
still explains ID but breaks on OOD. If both sides are within the certified
affine bound while raw MD changes materially, the implementation or estimator
contract is wrong. Pseudoinverse/ridge/rank conditions are explicit.

**Cited update-direction audit.** In a locally positive-scale-
invariant parameter block, explicit decoupled decay is radial. Under a frozen
diagonal preconditioner, coupled L2 can acquire a tangential component unless
`P_t w` is parallel to `w`. Test the operational prediction with an exact
same-state, same-gradient WD-versus-zero counterfactual update. Do not extend
the premise to the whole residual WRN trunk. Run the audit both at the
zero-decay state and at states with accumulated decay history. WRN's `conv0`
and block `conv1` weights are rescaling-eligible; `conv2`, projection
shortcuts, and the classifier are scale-breaking. An optional anchor-only
parameter-location ablation asks where the effect is carried, but does not
pretend these are pure “effective-LR” and “functional” mechanisms.

**Corollary: RMD low-rank curvature.** With class-frequency-weighted
between-class covariance and full-rank inverses,

```text
Sigma_0 = Sigma_W + Sigma_B
rank(Sigma_B) <= K - 1
H = Sigma_W^-1 - Sigma_0^-1 >= 0
rank(H) <= K - 1
s_RMD(z) = -z^T H z + max_c(a_c^T z + b_c).
```

RMD is an affine envelope plus a shared quadratic correction supported on `S`.
This is no longer a separate central mechanism. Numerical rank, PSD behavior,
curvature mass, and allocation are supporting diagnostics. The guaranteed
residual dimension is the lower bound `d-(K-1)`, not an exact discarded
fraction; actual `dim(S)` is measured. The bound does not predict monotone OOD
performance.

**Supporting lemma 2: pair-order balance.** For tie-aware pair correctness
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

Give churn an interpretable scale by comparing same-seed coupled/decoupled
churn with same-policy, different-seed natural variability. Report both values
and their ratio, without treating overlapping seed pairs as independent.
Separately use ID-only and OOD-only score-replacement hybrids to reconstruct
the signed AUROC change. This shows which side moves the ordering, but is not a
unique causal partition of the nonlinear churn indicator.

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

Present these as three views of one pair-margin object: exact additive
components at pair level, the fixed `rho` residual-channel section with
analytic flip thresholds, and Shapley only for declared nonlinear aggregate
replacement games. Do not use Shapley where the exact additive identity
already answers the question.

The theory removes impossible explanations and creates measurements; it does
not establish the sign or size of the training effect, component
concentration, ID equivalence, attenuation, or replication.

## 4. Discovery: where the historical gap lives

Use the completed 30 frozen WRN `last.pt` bundles and six OOD datasets only as
descriptive discovery. Before fresh training, freeze the estimator/rank/frame
tolerances and run a compact discriminant--residual preflight: theorem
applicability, branch-internal `S-perp`/parallel-Marginal/RMD score and pair
reconstruction, RMD cancellation residual, actual `dim(S)`, branch principal
angles, zero-reference common frame, classifier alignment, and historical
component concentration. For ID train/test, include the complete three-component
variance--covariance matrix and raw-norm/component correlations. Do not label
`Var(s_perp)/Var(s_MD)` as a fragility percentage: covariance can move it
outside `[0,1]`, and one model's ID score variation is not retraining
variation. The completed C3 raw/L2 pattern motivates the fresh interaction
hypothesis but is not an ordered dose response because detector-wise maxima
occur on different OOD datasets. Curvature and spectrum/allocation remain
supporting diagnostics.

Only after the conditional method specification is locked, add an ID-only
residual-tail preflight: held-out-ID `q_perp` Q--Q deviation from `chi^2_k`,
empirical variance, standardized moments, upper quantiles, tail-fit stability,
and between-model variation. The fitted ID-train mean is pinned by whitening
and is not a tail or policy-sensitivity result. This historical mixed-recipe
analysis may close the `RtMD` slot, but cannot by itself activate a causal or
method claim.

Do not call the mixed optimizer/LR/WD population causal evidence. Do not use
nearest-accuracy matching as a primary analysis. Preserve the failed v2 radial
gate and the successful v3 component analysis exactly as recorded.

The bounded D1 survival reuse adds one discovery result without new inference:
all 96 reused score arrays passed their recorded hashes and sample ordering.
Across six historical cross-policy pairs, median Raw-MD PairOrderChurn was
`0.322` versus a same-policy seed reference of `0.220` on CIFAR-100 and
`0.359` versus `0.273` on MNIST. Raw-RMD churn was `0.123` versus `0.114`
and `0.111` versus `0.098`. This makes pair-ranking multiplicity a viable
fresh headline and suggests that RMD attenuates the policy-specific excess
toward, but not necessarily down to, seed variability. Because the historical
role pairs mix LR and WD and are not same-initialization siblings, this is not
evidence that coupling caused the churn or that RMD is immune.

This section chooses a falsifiable fresh-study multiplicity and channel
hypothesis and nothing more. It is not causal evidence and does not guarantee
five-seed power. If no
component concentrates or applicability fails, do not promote the theorem as
the central empirical explanation.

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
| `1e-3` | **primary anchor** zero / `alpha=0,0.5,1` | Adam / AdamW |

- primary anchor: zero / `alpha=0` AdamW / `alpha=0.5` mixed /
  `alpha=1` Adam, five paired seeds, full trajectory analysis;
- remaining cells: three paired seeds, epoch-200 focal analysis;
- low-LR zero reference: three seeds;
- total Adam-family budget: 41 runs; the alpha midpoint adds five runs.

SGDM control: LR `0.1`, WD `5e-4`, zero/SGDM/SGDW, three seeds, nine runs.

Analyze each cell first. Never average raw runs across LR/WD. Use a cell-equal
summary only after showing sign and interaction patterns. Same nominal WD is a
controlled numerical input, not matched effective regularization or a tuned
optimizer comparison. Zero plus nominal WD `1e-4` and `1e-3` remain separate
decay-presence/strength reference points at each LR. Coupled Adam has no
uniquely
separable realized-decay vector because the L2 term changes its moments and
denominator; compare exact same-state, same-gradient WD-versus-zero
counterfactual operator differences instead.

At the primary anchor, the existing `adam_coupled_decoupled` implementation
holds `total_weight_decay=1e-4` fixed and uses `coupled_ratio
alpha in {0,0.5,1}`. The endpoints are exact AdamW and Adam; the midpoint
sends half of the nominal decay through each path. Report the endpoint,
lower-half, and upper-half seed-paired contrasts for ID utility, NC0--NC4,
theorem-aligned components, Raw-MD/RMD churn, and net AUROC. Call the response
interior-compatible only if the midpoint lies between the endpoints in their
observed direction; otherwise report a non-monotone three-point response. This
tests interiority/monotonic compatibility, not curve shape, linearity, or
matched realized regularization. Zero decay is a separate reference, not an
alpha value. A shortened one-seed run is technical validation only.

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
   update--weight cosine, groupwise radial/tangential/angular update, Adam
   moments, BN/running statistics, residual/shortcut ratio, and exact one-step
   counterfactuals at both zero-decay and history-conditioned states. These are
   cited manipulation/pathway measurements, not a new theorem.
2. **Representation geometry:** make the fitted detector interface primary:
   branch-internal distributions of `q_perp` and `P_S x`, actual `dim(S)`,
   whitening, class centers, and `B|_S`; then ID-only gauge alignment,
   subspace principal angles, whitening change, a zero-decay common frame, and
   held-out ID as the affine fit floor. Use global radial scale as a negative
   control and keep norm/radial heterogeneity, class-mean/CDNV/NC,
   global/within covariance, spectral-band allocation, and nearest/full class
   profiles as supporting explanations rather than equal headline outcomes.
   Treat NC0--NC4 as a profile and compare its alpha/endpoint trajectory with
   the detector-interface and churn trajectory. Agreement supports an
   upstream-to-readout transfer; dissociation shows that optimizer-sensitive
   NC geometry need not be the channel a fixed OOD score reads. Neither result
   makes NC a scalar mediator or equates it with `S`.
3. **Score geometry:** branch-specific ID-only Gaussian fits; raw/L2 MD,
   Marginal, and RMD; explicit theorem-applicability status;
   `S-perp`/parallel-Marginal/RMD score and pair-margin reconstruction; RMD
   cancellation residual; raw/L2 x MD/RMD interaction; component
   variance--covariance and raw-norm correlations; fixed `rho` path and exact
   pair-flip thresholds; size/stretch and spectrum/allocation as supporting
   decompositions.
4. **OOD ordering:** the same ID--OOD pairs classified as incorrect/tie/correct
   at every checkpoint, with exact theorem-aligned component attribution, Gain, Loss,
   PairOrderChurn, net DeltaAUROC, policy/seed churn ratio, and ID/OOD-side
   replacement accounting.

If the `RtMD` slot passes its fresh ID-only activation gate, add its frozen
score and comparison panel as a separate secondary method analysis. It does
not enter the primary mechanism family or change the raw-MD endpoint.

For every seed and cell report coupled--decoupled, coupled--zero, and
decoupled--zero. The update policies differ by construction at the first
nonzero-decay step, so “update onset precedes geometry onset” is not a claim.
Report standardized divergence curves, minimum detectable effects, early
slopes, and functional/cumulative summaries. A practical-threshold crossing is
called detectability time and is descriptive, not a causal onset. A mechanism
claim requires an
exact update manipulation check, estimator applicability, repeatable
substantive representation divergence, formula-level score/pair
reconstruction, Gain/Loss attribution, and selective channel attenuation.
Treat seed, not image pair, as the
independent statistical unit; same-policy different-seed variation is a
natural-variability reference, not a null.

### Channel-matched confirmation

- `S-perp` residual -> RMD cancellation, `S`-only reconstruction, and the
  fixed `rho` residual-retention path;
- parallel-Marginal/RMD -> retained `S`-only gap and class-relative/estimator audit;
- sample/class-conditioned radial difference -> L2 normalization and refitting,
  interpreted through the raw/L2 x MD/RMD interaction rather than assumed to
  be raw-`S-perp` suppression;
- non-affine deformation -> ID-only affine alignment plus residual/precision accounting;
- spectrum/stretch difference -> prespecified ID-only spectral-band ablation,
  clipping, or whitening diagnostic;
- class-distance-profile difference -> nearest-class versus full profile.

Do not add a detector after seeing results. The single `RtMD` slot is allowed
only because its derivation, activation, and closure rules are registered
before the residual-tail and protected-OOD analyses. Scalar correlations can
support, but cannot replace, exact accounting and selective attenuation.

### Inferential hierarchy

- Primary outcome: WRN anchor, epoch 200, penultimate, raw MD,
  DeltaAUROC and PairOrderChurn, policy/seed churn ratio, prespecified near/far
  summaries.
- Prespecified alpha confirmation: fixed-total-decay endpoint/lower-half/
  upper-half contrasts and interior-compatible versus non-monotone
  classification for ID utility, NC0--NC4, components, Raw-MD/RMD churn, and
  net AUROC; no fitted curve or selected alpha.
- Primary mechanism: theorem applicability, branch-internal
  `S-perp`/parallel-Marginal/RMD pair attribution, RMD cancellation,
  raw/L2 x MD/RMD interaction, fixed `rho` attenuation,
  gauge-aligned subspace comparison, differential ID/OOD affine residual,
  ID/OOD score-side accounting, counterfactual update audit, and
  channel-matched attenuation.
- Secondary: norm/radial and class-mean/CDNV/NC profiles, L2/size--stretch,
  spectrum--allocation, functional time curves, within-stage depth,
  Adam LR/WD interaction, SGDM control, best-validation, epoch 300.
- Exploratory/appendix: external detector panel, individual eigenvectors,
  extra geometry scalars, and optional fork.
- Conditional secondary method family, only if activated: frozen `RtMD`
  policy-churn reduction, performance/far-OOD guardrails, WDiscOOD and other
  prespecified comparisons, and replication.

Do not perform an unadjusted test for every checkpoint by depth by metric by
OOD dataset. Freeze family-wise uncertainty, practical margins, minimum
detectable effects, spectral bands, and multiplicity handling before protected
OOD execution.

## 6. Result map

| Observation | Allowed interpretation |
| --- | --- |
| Applicability passes, practical gap/churn concentrates in `S-perp`, and RMD selectively attenuates it | strongest discriminant--residual mechanism result |
| L2 reduces transform-specific residual variation, helps MD much more than RMD, and the fixed `rho` path attenuates gap/churn | supports a shared empirical residual-sensitivity channel; L2 and RMD remain different operations |
| L2 changes parallel terms or helps RMD comparably | normalization acts beyond residual suppression; reject the simple partial-versus-complete account |
| Gap is dominated by parallel-Marginal or RMD | theorem remains; lower the empirical `S-perp` claim and report the observed `S` pathway |
| RMD retains the branch gap | residual cancellation is insufficient; investigate `S`-internal, class-relative, or estimator pathways |
| Separately whitened subspaces differ but align after gauge control | coordinate/frame effect; do not claim substantive subspace rotation |
| Pseudoinverse/rank gate fails | do not apply exact cancellation to the primary score; report the estimator boundary |
| OOD gap with failed ID equivalence | training-rule Pareto result; remove comparable-ID wording |
| Same direction across four Adam cells | locally robust coupling effect |
| Larger effect at stronger WD | coupling-by-decay-strength response |
| Magnitude changes with LR | coupling effect is step-scale sensitive |
| Sign reversal across cells | recipe interaction, not a universal coupling effect |
| Geometry changes but OOD does not | detector-relevant versus irrelevant geometry distinction; central claim weakens |
| Pair identities change but AUROC cancels | decision-level instability hidden by the aggregate |
| Policy churn is much larger than seed churn | coupling changes pair decisions beyond ordinary training variability |
| `alpha=0.5` lies between both endpoints across the focal outcomes | three-point evidence compatible with a graded coupling response; no curve-shape claim |
| Alpha midpoint is outside the endpoint interval or channels disagree | non-monotone or outcome-specific coupling response; do not summarize it as a scalar dose response |
| ID residual stays near floor while OOD residual grows | differential query-side deformation; cleanest affine mechanism |
| ID and OOD residuals both grow | global non-affine deformation; query-side-specific claim is unavailable |
| Branches are almost entirely related by one affine map | raw-MD gap should be absent; lower the non-affine mechanism claim |
| Tangential counterfactual update and non-affine residual grow together | supports coupling-to-direction-to-representation pathway |
| Angular update aligns with the OOD effect | report known rotational dynamics as a plausible upstream pathway, not unique mediation |
| Difference disappears after epoch 200 | finite-time/epoch-budget interaction |
| Adam only | adaptive preconditioning-by-coupling interaction |
| Adam and SGDM | more general coupling effect |
| No fresh effect | historical gap was not primarily caused by coupling under this design |
| Residual-tail or novelty gate fails | close `RtMD` and `nu-hat`; retain the mechanism paper unchanged |
| `RtMD` reduces policy churn, passes performance/far-OOD guardrails, and replicates | promote the conditional method contribution |
| `RtMD` helps the anchor only or loses far-OOD signal | report a local exploratory result or close the slot; do not make a general method claim |

Null outcomes remain in the paper. They are not rescued by matching
checkpoints, changing epoch 200, or expanding the detector panel.

## 7. Replication and scale

1. ResNet-18/CIFAR-10 architecture replication in a similar guaranteed
   residual-capacity regime; not a clean single-factor ablation.
2. ResNet-18/CIFAR-100 class-count/discriminant-capacity stress test.
3. DenseNet-BC-100 k=12/CIFAR-10 focal appendix.
4. ConvNeXt-Tiny/ImageNet-200 from-scratch focal appendix.

DenseNet and ConvNeXt use three Adam-family fresh seeds and the focal
Mahalanobis family. ConvNeXt is external-validity evidence, not a clean
BN-versus-LN mechanism test. ImageNet-1K pretrained weights are forbidden.
For the class-count ladder, report the guaranteed `dim(S-perp)` lower bound,
actual `dim(S)`, branch principal angles, measured curvature mass, and ID/OOD
allocation. The lower bounds are 631/640 for WRN/CIFAR-10, 503/512 for
ResNet/CIFAR-10, 413/512 for ResNet/CIFAR-100, and 569/768 for
ConvNeXt/ImageNet-200. They are not exact discarded proportions. Every regime
tests reconstruction; CIFAR-100 is a theory-motivated capacity stress test,
not the sole direct theorem test. Do not assume that larger `K` monotonically
changes detector performance.

## 8. Discussion and conclusion

Discuss local normalized-block effective-step dynamics without calling the
whole WRN trunk scale-invariant, the difference between comparable-ID and
Pareto evidence, the limits of affine/full-rank/common-ridge assumptions and
CIFAR Gaussian fitting, branch-dependent whitening frames, the fact that
`S-perp` is not meaningless noise, the distinction between nonlinear L2 refit
and algebraic RMD cancellation, and the boundary between tested decay policies
and universal optimizer claims. Decision stability is measured by policy
churn relative to same-policy seed churn; do not rename one model's ID score
variance as retraining reproducibility.

If active, discuss `RtMD` as a theorem-derived residual likelihood correction,
not a replacement for the formation contribution or a generic claim that
heavy-tailed scores always outperform MD. Contrast it directly with WDiscOOD's
linear discriminative/residual combination and RMD's complete cancellation.

Use the Card 13 strong sentence only if every corresponding gate passes.

## Figure plan

1. Pair-ranking multiplicity: exact Gain/Loss/Churn accounting, the bounded
   historical D1 survival result, and why small net AUROC need not imply stable
   same-image rankings.
2. From-scratch paired plus `alpha=0,0.5,1` design and the update ->
   `S/S-perp` geometry -> score -> ordering chain, including the
   discriminant--residual cancellation interface.
3. Fresh trajectories: zero-state/history-conditioned counterfactual update,
   differential ID/OOD affine residual,
   gauge-aligned subspace/frame diagnostics, and representation geometry over
   time and within-stage depth.
4. ID utility, DeltaAUROC versus PairOrderChurn, policy/seed churn ratio,
   ID/OOD-side and Gain/Loss transitions, and theorem-aligned component
   attribution at the primary endpoint, with the fixed `rho` attenuation path.
5. Alpha three-point response, Adam 2 x 2, SGDM control, and
   WRN--ResNet--DenseNet--ConvNeXt replication matrix.
6. Conditional only: residual-tail gate, frozen `RtMD` policy/seed churn and
   performance guardrails, and replication. Omit the figure if the slot closes.

## Table plan

1. Prior work versus the paired formation study.
2. Exact run budget, checkpoints, depth taps, and execution status.
3. Main paired ID/OOD estimates and practical classifications.
4. Applicability, subspace/frame, component covariance, normalization x
   cancellation, affine residual, churn, and attenuation evidence by OOD
   dataset.
5. Replication claim gate and numerical-validation appendix.
6. Conditional only: `RtMD` specification, ID-only fit/activation record,
   direct baselines, guardrails, and close-or-promote verdict.
