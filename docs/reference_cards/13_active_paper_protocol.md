# Reference Card 13: Active Paper Protocol

## 0. Authority and status

Protocol identifier:

```text
fixed_readout_theory_constrained_paired_trajectory_v6
```

The stable path of this file is the sole authority for the paper's executable
experiment design. The protocol version is recorded inside this file and in
Git history; future revisions replace this content without creating another
Card 13 filename. The current design version is `v6`, and its fresh
experiments are `NOT_RUN`.

This revision does not change completed evidence:

- Metric Contract v1.2 and its 30-model WRN-28-10/CIFAR-10 population remain
  valid descriptive discovery evidence;
- the Research Contract v2 radial Stage-2 result remains immutable `FAILED`;
- the v3 historical MD--Marginal--RMD analysis remains `PASS` and discovery
  only;
- the validated `fork_from_prefix` runtime remains available, but it is an
  optional follow-up tool rather than the main paper intervention;
- no fresh v6 trajectory, protected-OOD confirmation, or replication has run.

The known MD--Marginal--RMD relation, L2 feature normalization, RMD, and the
size--stretch factorization are measurement tools. They are not claimed as
new decompositions or detectors.

## 1. Paper question and claim boundary

The paper asks:

> Starting from the same initialization and data stream, does changing only
> decay coupling alter the trajectory of representation geometry, and can the
> resulting changes in a fixed Mahalanobis readout be traced to a specific
> score component and to exact ID--OOD pair-order transitions?

The primary contribution is a paired from-scratch trajectory study, not an
optimizer leaderboard and not a claim that similar ID accuracy alone causes
OOD differences. Comparable ID utility is an interpretation condition.

Contribution order:

1. paired training-rule-to-geometry-to-score trajectory attribution;
2. theory-constrained attribution using affine-gauge invariance, local
   update-direction geometry, and the low-rank curvature of RMD;
3. exact tie-aware accounting of pair-order churn, AUROC gain/loss
   cancellation, and Mahalanobis score components;
4. a controlled coupled/decoupled/zero-decay factorial with architecture,
   dataset/class-count, connectivity, and modern-scale replication.

The strongest allowed sentence is conditional:

> A controlled change in decay coupling produces a non-affine trajectory of
> Mahalanobis-relevant representation geometry, and the effect propagates
> through a formula-identified score component to alter and churn ID--OOD
> pair orderings.

This sentence is unavailable unless the practical paired effect,
non-affine-residual evidence, component concentration, channel-matched
attenuation, and fresh replication gates pass. If ID equivalence fails, the
result is retained and interpreted as a Pareto trade-off rather than discarded
or post-hoc checkpoint-matched.

## 2. What is controlled in a paired trajectory

For each seed and hyperparameter cell, coupled, decoupled, and zero-decay runs
start at epoch 0 from the same model initialization. They use the same:

- architecture and parameter-group policy;
- ID train/validation membership;
- minibatch order and augmentation RNG stream;
- loss, batch size, epoch budget, and precision policy;
- LR schedule shape and milestones;
- checkpoint and deterministic probe-image identities.

The intended contrast changes the update policy only. At nonzero nominal
decay, the primary pair is coupled versus decoupled. The zero-decay run is a
family and LR-specific reference for separating decay presence from decay
coupling.

`weight_decay=x` in Adam and AdamW is a shared numerical input, not a claim of
matched effective shrinkage or matched effective regularization. The primary
question is algorithmic: under controlled inputs, does coupling itself alter
the trajectory? A separately tuned optimizer comparison would answer a
different practical question and is not part of the causal main contrast.

Coupled Adam does not expose a unique additive `decay update`: the L2 term
enters first and second moments and the coordinate-wise denominator. Therefore
v6 does not place Adam and AdamW on a purported common realized-decay norm.
It uses an exact same-state, same-gradient one-step counterfactual operator
difference, defined in Section 6.1, and reports its radial and tangential
components. At each LR, the common zero reference plus nominal WD values
`1e-4` and `1e-3` already supply three prespecified dose points; no asymmetric
post-hoc WD point is added.

Training seed is the independent statistical unit. Repeated images and
ID--OOD pairs are paired observations, not independent replicates.

## 3. Main training design

### 3.1 WRN-28-10/CIFAR-10 Adam-family factorial

Use the local 2 x 2 design below. Each nonzero cell contains both Adam
(coupled L2) and AdamW (decoupled weight decay).

| LR | nominal WD `1e-4` | nominal WD `1e-3` |
| --- | --- | --- |
| `3e-4` | Adam / AdamW | Adam / AdamW |
| `1e-3` | **primary anchor:** Adam / AdamW | Adam / AdamW |

The schedule shape is shared: 200 epochs, Multistep milestones
`[60, 120, 160]`, and `gamma=0.2`. Only the initial LR differs by row.

Run allocation:

- primary anchor `(lr=1e-3, wd=1e-4)`: zero / Adam / AdamW, five seeds,
  all trajectory, depth, geometry, score, and pair analyses: 15 runs;
- `(1e-3, 1e-3)`: Adam / AdamW, three seeds, epoch-200 focal analysis:
  6 runs;
- `(3e-4, 1e-4)`: zero / Adam / AdamW, three seeds, epoch-200 focal
  analysis: 9 runs;
- `(3e-4, 1e-3)`: Adam / AdamW, three seeds, epoch-200 focal analysis:
  6 runs.

Total Adam-family budget: **36 runs**. One zero-decay baseline is shared by
the two WD cells at each LR because WD is inactive at zero.

Every cell is reported separately before any summary. For outcome `Y`, the
primary cell-level contrast is

```text
Delta(LR, WD) = Y_Adam(coupled) - Y_AdamW(decoupled).
```

The two WD levels test coupling-by-decay-strength interaction; the two LR
levels test coupling-by-step-scale interaction. Raw runs are never pooled
across cells. A cell-equal-weight summary is secondary and cannot hide sign
reversal or a primary-only result.

### 3.2 Conventional SGDM-family control

Use one conventional WRN/CIFAR cell:

```text
LR = 0.1
nominal WD = 5e-4
zero / SGDM coupled / SGDW decoupled
three seeds
```

Total SGDM-family budget: **9 runs**. This tests whether the coupling pattern
is specific to adaptive preconditioning or is also visible in a conventional
momentum family. It is a control, not a second HPO study.

### 3.3 ID utility and model-selection controls

At every reported endpoint, record accuracy, NLL, and ECE. Practical
equivalence margins and the inferential rule must be frozen before protected
OOD results are opened.

- If all ID guardrails pass, use `comparable-ID` language.
- If they do not pass, keep every run and show the ID/OOD Pareto relation.
- Do not select a matching checkpoint after seeing OOD performance.
- Epoch-200 `last.pt` is the primary endpoint.
- `best_val.pt` selected by ID validation only is a secondary
  model-selection control and is never mixed with the epoch-200 estimate.

Historical pair-control results already show that comparable ID behavior is
not automatic. It is therefore an interpretation gate, not a promised
outcome or a run-exclusion rule.

## 4. Time and depth design

### 4.1 Time axis

Candidate snapshots are:

```text
0, 1, 10, 30, 60, 61, 120, 121, 160, 161, 200
```

The pairs around milestones distinguish gradual formation from an immediate
schedule-boundary change. Cheap ID-only geometry is evaluated at all listed
snapshots. Full OOD score and pair-order evaluation is primary at:

```text
10, 60, 120, 160, 200
```

The primary anchor's first three paired seeds may be continued to epoch 300
as a terminal-phase appendix, with snapshots 240 and 300 and the final LR
held constant after epoch 160. This is a runtime-by-decay-exposure diagnostic,
not a replacement for epoch 200. `best<=200` and `best<=300` are reported as
separate selection windows.

The update policies differ by construction at the first nonzero-decay step.
Consequently, v6 does not use “update onset precedes geometry onset” as a
mechanism gate. For every prespecified quantity, report the seed-level paired
divergence curve, a standardized effect against same-policy natural training
variability, the minimum detectable effect, an early divergence slope, and a
functional or cumulative summary. A threshold-crossing time may be reported
as **detectability time**, not as a causal onset or as proof that one physical
process began before another. Same-policy different-seed variation is a
natural-variability reference, not a stochastic null.

### 4.2 Depth axis

The default depth scan is performed at epoch 200:

- `stage1`: output of WRN residual group 1, then global average pooling;
- `stage2`: output of residual group 2, then global average pooling;
- `stage3`: output of residual group 3 before final BN/ReLU, then pooling;
- `penultimate`: output after final BN/ReLU and global average pooling, before
  the classifier; 640 dimensions for WRN-28-10.

These are analysis taps, not additional learned heads. Time is studied densely
at the penultimate endpoint; depth is studied across the four taps at epoch
200. Only if the final depth scan identifies the first major divergence may
one immediately preceding stage be traced across the full time axis. This
keeps the design interpretable and computationally bounded.

WRN stage widths and spatial resolutions differ. Raw condition numbers,
effective ranks, spectra, or norm magnitudes are therefore not compared as if
they shared one scale across stages. The depth question uses standardized
coupled--decoupled effects **within each stage** to localize where policy
sensitivity grows. The penultimate endpoint remains the primary mechanism
space.

## 5. Exact score and pair-accounting framework

All scores are oriented so larger means more ID-like. With branch-specific
ID-train statistics fitted at the same checkpoint and depth:

```text
s_MD(z) = s_RMD(z) + s_Marginal(z)
m_D(i,o) = s_D(i) - s_D(o)
m_MD(i,o) = m_RMD(i,o) + m_Marginal(i,o)
```

RMD is called the **global-referenced class-relative component**. It is not
called a pure class-contrastive score.

AUROC is the tie-aware proportion of ID--OOD pairs with positive margin, with
half credit for ties. For the same deterministic ID and OOD probe images, each
pair is classified in each run as `incorrect`, `tie`, or `correct`, producing
an exact 3 x 3 transition table between coupled and decoupled runs.

For branch `r`, define tie-aware pair correctness
`a_r(i,o) in {0, 1/2, 1}`. Then `AUROC_r = E[a_r]`. Define

```text
Gain = E[(a_C - a_D)_+]
Loss = E[(a_D - a_C)_+]
PairOrderChurn = Gain + Loss
DeltaAUROC = Gain - Loss
```

Therefore `abs(DeltaAUROC) <= PairOrderChurn`; without ties,
`Gain=(PairOrderChurn+DeltaAUROC)/2` and
`Loss=(PairOrderChurn-DeltaAUROC)/2`. Epoch-200 raw-Mahalanobis
`DeltaAUROC` and `PairOrderChurn` are co-primary outcomes. This distinguishes
an aggregate change from large opposing decision reversals that cancel in
AUROC. ID-validation-threshold accept/reject disagreement and Cohen's kappa at
an approximately 95% ID-TPR operating point are deployment-oriented secondary
outcomes, not replacements for pair-order churn.

The two-component AUROC accounting uses four computational hybrids: both
components from the decoupled run, both from the coupled run, and the two
replacement orders. Symmetric Shapley contributions average the two orders.
The hybrids are calculation devices, not trained detectors. Exactness holds
inside this declared two-player replacement game; it is not claimed to be a
unique physical or causal mediation decomposition.

ID-side and OOD-side score motion receive a separate exact accounting. Let
`a_00` use decoupled ID and OOD scores, `a_11` use coupled ID and OOD scores,
and define the two hybrids:

```text
a_10 = a(s_C(i) - s_D(o))
a_01 = a(s_D(i) - s_C(o))
phi_ID  = 1/2 [(a_10-a_00) + (a_11-a_01)]
phi_OOD = 1/2 [(a_01-a_00) + (a_11-a_10)]
phi_ID + phi_OOD = a_11-a_00.
```

Their averages reconstruct `DeltaAUROC` exactly. The four hybrid transitions
also show whether replacing only the ID or only the OOD side crosses the
ordering boundary. Because threshold crossing is nonlinear, this is a declared
counterfactual/Shapley accounting, not a unique causal decomposition of churn.

To make churn interpretable, compare the same-seed policy contrast with
same-policy natural training variability:

```text
R_churn = median_s Churn(C_s,D_s)
          / median_{p in {C,D}, s<t} Churn(p_s,p_t).
```

Report numerator and denominator alongside the ratio for every OOD dataset;
do not call the denominator a null or treat its overlapping seed pairs as
independent replicates. If the denominator is below the frozen numerical or
practical floor, report the ratio as undefined rather than inflating it.

## 6. How training recipe is linked to geometry and ordering

The analysis follows the same fixed probe images through four levels. A scalar
correlation alone cannot close the mechanism claim.

### 6.1 Level A: update dynamics

At selected steps and for every depth block, record:

- parameter norm, gradient norm, and optimizer-update norm;
- update norm divided by parameter norm;
- update/weight cosine;
- radial update component parallel to the weight and tangential component
  orthogonal to it;
- Adam first- and second-moment summaries where applicable;
- BatchNorm scale/shift and running-variance summaries, pre/post-normalization
  activation scale, and residual-branch versus shortcut norm ratios.

At selected checkpoints, copy model state, optimizer state, and one fixed
minibatch loss gradient. Perform the audit at two kinds of state: the
zero-decay sibling, which supplies a decay-free instantaneous reference, and
each nonzero sibling, which reveals the state-conditioned marginal effect
after decay history has accumulated. At the zero state, calculate coupled,
decoupled, and zero-WD candidate operators. At each nonzero state, calculate
the actual operator and its zero-WD counterfactual. Apply none of them.
Decompose the differences into radial and tangential components. This is a
manipulation check, not a separate long-running branch experiment and not a
claim that coupled Adam has a uniquely separable physical decay vector.

For a parameter block whose function is locally invariant to positive radial
rescaling, decoupled weight decay is an explicitly radial step. Under a frozen
diagonal preconditioner, a coupled L2 contribution is proportional to
`P_t w` and can have a nonzero tangential component unless `P_t w` is parallel
to `w`. This is a local proposition and testable prediction. It is **not**
extended to the entire WRN trunk: epsilon terms, running statistics,
residual/projection additions, the final classifier, and training dynamics
break the blanket scale-invariance claim. A controlled small positive weight
rescaling audit measures how closely each block satisfies the local
approximation.

WRN-28-10 supplies a prespecified parameter-location stratification. `conv0`
and the 12 residual-block `conv1` weights are **rescaling-eligible** because
their outputs next enter BatchNorm. The 12 `conv2` weights, three projection
shortcuts, and final classifier are **scale-breaking** because their outputs
enter an unnormalized residual addition or logits. This is an architectural
classification, not an assertion of exact invariance: BatchNorm epsilon,
running statistics, residual context, and later optimization can break the
idealization. The small positive-rescaling audit quantifies the approximation.

If the primary anchor shows a practical policy effect, a prespecified
secondary parameter-location follow-up may train Adam and AdamW with decay
restricted to each group (two groups x two policies x three shared seeds; the
existing zero sibling is reused). It uses only the focal endpoint analysis.
This tests where the coupling contrast is carried; it must not be described as
a pure separation of “effective learning rate” and “functional regularization,”
because both mechanisms can coexist in either arm.

### 6.2 Level B: representation geometry

For recipe `r`, checkpoint `t`, depth `l`, and fixed image `x`, extract
`z_{r,t,l}(x)`. Fit and report these prespecified channels:

| Channel | Measurements |
| --- | --- |
| Affine gauge / residual | ID-only branch alignment and held-out same-image non-affine residual |
| Global radial scale | shared-scale negative control; mean/median norm trajectory |
| Radial heterogeneity | full and class-conditioned norm distributions, same-image multipliers, ID--OOD radial relation, radius--direction coupling |
| Class geometry | class-mean distances/angles and CDNV |
| Global geometry | global mean and covariance trace/spectrum/effective rank |
| Within-class geometry | pooled covariance spectrum and condition number |
| Allocation | sample displacement energy in covariance spectral bands |
| Class-distance profile | nearest-class distance versus the full class profile |

The basic paired contrasts within every seed and cell are coupled--decoupled,
coupled--zero, and decoupled--zero. This separates the presence of decay from
the way decay is coupled.

Under an invertible affine map `z_C=A z_D+b`, branch-refitted full-rank
Mahalanobis distances and pair orderings are identical when the covariance
estimator transforms equivariantly. Global scale, rotation, or fixed channel
rescaling is therefore an affine-gauge negative control, not an independent
raw-Mahalanobis carrier. Fit the branch alignment using ID train only and write
`z_C(x)=A z_D(x)+b+e(x)`; evaluate the residual `e(x)` separately on held-out
ID and every OOD dataset, together with class-mean and precision residuals.
Held-out ID test residual is the generalization floor for the ID-trained affine
fit. Report raw and feature-scale-normalized residuals plus each OOD dataset's
excess over that floor. Pseudoinverse, ridge, rank deficiency, and
finite-precision conditions must be recorded rather than silently extending
the full-rank identity. If both ID and OOD residuals stay within the certified
affine/numerical bound while a raw-MD gap exceeds its propagated score bound,
treat that as an implementation or estimator-contract failure, not a
scientific result.

### 6.3 Level C: Mahalanobis components and quadratic geometry

At each checkpoint and depth, fit model-specific Gaussian statistics using
only that run's ID-train features. Score the same probe images with:

- Mahalanobis-Raw, Marginal-Mahalanobis-Raw, and RMD-Raw;
- their separately refitted L2-normalized versions.

For each relevant quadratic term, use the cited size--stretch factorization:

```text
q(x) = r(x) w(x)
r(x) = ||delta(x)||^2
w(x) = delta(x)^T Sigma^{-1} delta(x) / ||delta(x)||^2
```

For coupled `C` and decoupled `D`, the change is exactly reconstructed as:

```text
Delta q
= ((w_C + w_D) / 2) Delta r
+ ((r_C + r_D) / 2) Delta w.
```

In an eigenbasis, `q(x)=sum_j a_j(x)/lambda_j`. The primary diagnostic uses
prespecified spectral bands because individual eigenvectors can rotate or
swap under near-degenerate eigenvalues. It asks whether branch differences
come from changed eigenvalues (spectrum), changed sample energy in those bands
(allocation), or both.

### 6.4 Level D: exact OOD ordering

For every fixed pair `(ID image i, OOD image o)`, reconstruct MD pair-margin
change from RMD and Marginal changes, then connect it to the transition table.
Report which component carries correct-to-incorrect and
incorrect-to-correct transitions, and report Gain, Loss, PairOrderChurn, and
DeltaAUROC rather than only aggregate AUROC correlation.

Also report the exact identity
`Delta m(i,o)=Delta s_ID(i)-Delta s_OOD(o)` and the four ID/OOD replacement
hybrids from Section 5. This determines whether ordering changes are exposed
mainly because ID scores move, OOD scores move, or both are required to cross
the boundary, while preserving the stated non-unique mediation boundary.

For metric `G`, define the paired trajectory contrast
`Delta G(t,l)=G_C(t,l)-G_D(t,l)`. Its curve, standardized effect, uncertainty,
minimum detectable effect, and functional summary are frozen before protected
OOD evaluation. A first practical-threshold crossing may summarize
detectability but does not define causal onset. Multiplicity rules are fixed in
a pre-protected-OOD addendum and are not estimated after observing the
protected OOD trajectory.

A coherent temporal chain would look like:

```text
update-path divergence
-> representation-channel divergence
-> additive/quadratic score divergence
-> changed ID--OOD pair ordering
```

The update-operator difference is a manipulation check. A mechanism statement
requires repeatable non-affine representation divergence, exact score/pair
reconstruction, pair gain/loss attribution, and matched attenuation. Temporal
ordering supports this chain but is neither sufficient evidence nor described
as complete causal mediation.

### 6.5 Channel-matched confirmation

After localization, weaken only the implicated channel:

| Localized channel | Confirmatory diagnostic |
| --- | --- |
| sample/class-conditioned radial heterogeneity | L2 normalization and refitting; global scale remains a negative control |
| non-affine branch deformation | ID-only affine alignment and residual/precision perturbation accounting |
| global/Marginal component | RMD subtraction |
| spectrum/stretch | ID-only spectral-band ablation, eigenvalue clipping, or whitening diagnostic |
| class-distance profile | nearest-class versus full-profile readout |

The confirmation must selectively attenuate the branch gap without being
chosen after OOD results. A new detector is not added to rescue a failed
mechanism claim.

## 7. Theory-constrained attribution and its limits

The supporting theory is one connected constraint chain, not a collection of
new Mahalanobis detectors.

### 7.1 Affine-gauge proposition and residual diagnostic

For full-rank branch-refitted covariance and a common invertible affine map
applied consistently to fit and query samples,

```text
z' = A z + b
mu'_c = A mu_c + b
Sigma' = A Sigma A^T
d'_c(z') = d_c(z)
```

so MD score and pair ordering are unchanged. The empirical consequence is an
ID-only affine-residual audit, not a novelty claim for this textbook identity.
A branch gap requires sample/class-dependent non-affine deformation or an
explicit estimator/rank/numerical departure from the proposition's
conditions. A quadratic perturbation accounting separates aligned same-image
residual, class-prototype residual, and precision residual; its terms and bound
must reconstruct or upper-bound the observed distance change under the stated
norm convention.

The sharper empirical prediction is **differential deformation**. If the
ID-trained affine map generalizes to held-out ID but fails on OOD, the fitted
Gaussian geometry and ID query scores should remain near their affine floor
while OOD queries expose the branch difference. If residuals are large on both
sides, the evidence supports global non-affine deformation but not this cleaner
query-side mechanism.

### 7.2 Local update-direction proposition

For a locally positive-scale-invariant parameter block, the explicit AdamW
decay term is radial. Under a frozen diagonal preconditioner, a coupled L2 term
can induce a tangential direction proportional to the tangential projection
of `P_t w`. The exact implementation test is the same-state, same-gradient
counterfactual operator difference from Section 6.1. The paper does not claim
that all WRN trunk weights are scale-invariant or that every decay effect is a
pure optimization-path effect.

The architecture-defined rescaling-eligible/scale-breaking split in Section
6.1 turns this proposition into a location-specific prediction. Its optional
training ablation localizes the carrier but does not label either arm a pure
mechanism.

### 7.3 RMD low-rank curvature proposition

Using class-frequency-weighted between-class covariance,

```text
Sigma_0 = Sigma_W + Sigma_B
rank(Sigma_B) <= K - 1
```

without requiring balanced classes. Under a full-rank inverse convention,
Woodbury gives

```text
H = Sigma_W^-1 - Sigma_0^-1 >= 0
rank(H) <= K - 1
s_RMD(z) = -z^T H z + max_c(a_c^T z + b_c)
```

Thus RMD is a class-dependent affine envelope plus a shared low-rank quadratic
correction. This complements the prior term-wise size--stretch view; it does
not create a single shared RMD covariance or solve that distinct problem. Test
the numerical rank, positive-semidefinite residual, curvature mass, centered
query allocation in `range(H)`, and the quadratic-versus-affine contribution.
CIFAR-10 (`K=10`), CIFAR-100 (`K=100`), and ImageNet-200 (`K=200`) form a
prespecified capacity ladder with bounds 9, 99, and 199. Interpret this
relative to feature dimension using `min(K-1,d)/d` and measured effective
curvature rank/mass; `199/768` is not “low-rank” in the same practical sense
as `9/640`. The theorem predicts the bound, not a monotone OOD effect. Effect
magnitude and stability remain empirical spectrum/allocation questions.

Before fresh GPU training, use the existing verified 30-bundle raw-feature
cache for a read-only historical preflight. Freeze the numerical-rank rule
before inspecting its output, then measure the covariance identity residual,
PSD/rank behavior of `H`, curvature mass, ID/OOD allocation, and exact
quadratic-versus-affine RMD reconstruction. This mixed-recipe population is
noncausal. The preflight gates whether low-rank allocation is promoted as a
central mechanism prediction; it does **not** cancel CIFAR-100 replication,
which remains independently valuable for dataset/class-count validity.

### 7.4 Pair-order balance proposition

The Gain/Loss/PairOrderChurn identities in Section 5 are exact and explain how
large opposing ordering changes can be hidden by a small net AUROC difference.
They constrain reporting: pair churn cannot be replaced by a thresholded
deployment metric or treated as millions of independent replicates.
The policy-to-natural-variability churn ratio supplies scale, while the
ID/OOD hybrid accounting localizes signed score motion. Neither creates a
unique causal partition of the nonlinear churn indicator.

The paper also derives the exact MD--Marginal--RMD score and pair-margin
identities, the symmetric size--stretch branch-change identity, and the
spectrum/allocation expansion while crediting their prior sources.

The theory does **not** establish the sign or size of the training effect,
component concentration, ID equivalence, attenuation, or replication. It does
not justify a universal detector-fragility ranking. Pseudoinverse, ridge,
covariance-centering, numerical-rank, and finite-precision conditions accompany
every applicable statement.

## 8. Inferential hierarchy and multiplicity

The independent statistical unit is the training seed. Images and ID--OOD
pairs are repeated paired observations within that unit.

Primary outcome family:

- WRN-28-10/CIFAR-10 primary anchor;
- epoch 200, penultimate endpoint, Mahalanobis-Raw;
- paired `DeltaAUROC` and `PairOrderChurn`;
- policy-to-natural-variability `R_churn` as a descriptive standardized effect;
- prespecified near/far summaries with per-seed effects.

Primary mechanism family:

- Marginal/RMD pair attribution;
- held-out-ID versus per-OOD differential affine residual, with global-scale
  negative control;
- ID/OOD score-side replacement accounting;
- sample/class-conditioned radial heterogeneity;
- counterfactual tangential update;
- channel-matched attenuation.

Secondary families are L2 and size/stretch variants, functional time and
within-stage depth localization, the Adam LR/WD interactions, SGDM control,
`best_val`, and the epoch-300 appendix. External detector panels, individual
eigenvectors, extra geometry scalars, and optional fork analyses are
exploratory or appendix. Do not perform separate unadjusted tests for every
metric by every checkpoint by every depth by every OOD dataset. Exact family
definitions, simultaneous/functional trajectory uncertainty, practical
margins, and multiplicity control are frozen in the pre-protected-OOD
addendum.

## 9. Detector roles

Focal Mahalanobis family:

- Mahalanobis-Raw;
- Marginal-Mahalanobis-Raw;
- RMD-Raw;
- separately refitted L2 versions of all three.

External controls:

- kNN-Raw/L2, CTM, Pure Residual, and Energy-T1;
- MSP and ViM in the appendix;
- ReAct excluded.

The focal family answers the component question. Controls test whether the
effect is specific to the Mahalanobis formula. They are not ranked by
popularity, and no post-hoc detector expansion is allowed.

## 10. Replication and scale order

1. ResNet-18/CIFAR-10: architecture-only replication.
2. ResNet-18/CIFAR-100: dataset/class-count replication and the prespecified
   RMD curvature-rank/allocation prediction.
3. DenseNet-BC-100, growth rate 12/CIFAR-10: dense-connectivity focal
   appendix.
4. ConvNeXt-Tiny/ImageNet-200: modern-scale focal appendix.

DenseNet and ConvNeXt use three Adam-family fresh seeds and the focal
Mahalanobis family only. Their architecture, dataset, and training support are
not currently implemented and require separate bounded tasks.

ConvNeXt-Tiny must be trained from scratch. ImageNet-1K pretrained weights are
forbidden because they expose the model to the 800 ImageNet classes outside
the ImageNet-200 ID subset. The planned OpenOOD v1.5 ImageNet-200 roles are:

- near OOD: SSB-hard, NINCO;
- far OOD: iNaturalist, Textures, OpenImage-O;
- covariate-shift datasets: separate appendix only.

ConvNeXt is external-validity evidence, not a clean BN-versus-LN mechanism
ablation. If ID equivalence fails at scale, report a Pareto result.

Across the three class-count regimes, report the theoretical rank bound,
feature dimension, normalized bound, measured numerical/effective rank,
curvature mass, and ID/OOD allocation. Do not infer a monotone detector effect
from `K` alone.

## 11. Historical discovery and optional shared-prefix follow-up

The completed 30-model v3 analysis uses 30 frozen `last.pt` bundles and six
OOD datasets. It is valid for locating descriptive MD/Marginal/RMD variation,
but optimizer, LR, and WD are mixed in that population. It cannot establish
that decay coupling caused the gap. Nearest-accuracy matching is excluded from
primary analysis and, if shown, is labeled selection-biased sensitivity.

The same immutable cache supports the read-only RMD curvature preflight in
Section 7.3 without new training or protected-data traversal. Its output must
use a fresh external artifact directory and a compact committed summary only;
do not recreate the retired Stage-2 gate, checksum catalog, or large generated
tables in Git.

The existing `fork_from_prefix` implementation may be used only as an optional
follow-up if the from-scratch divergence curves produce a quantitative
switch-time prediction. A future test must use prespecified switch times and
report the remaining fraction of the full from-scratch geometry/churn gap; it
must not choose the most favorable threshold-crossing checkpoint after seeing
OOD results. Forking is not required for the main contribution and does not
replace from-scratch evidence. Ordinary resume remains strict and distinct
from this optional fork operation.

## 12. Required artifacts and validation

Every v6 result must identify seed, initialization, config, branch policy,
checkpoint epoch, depth tap, probe-image membership, Gaussian-fit population,
and source code/config hashes. Required records include:

- ID accuracy/NLL/ECE and `last` versus `best_val` identity;
- per-sample MD, Marginal, and RMD raw/L2 scores;
- score and pair-margin reconstruction residuals;
- tie-aware pair-transition counts and component Shapley accounting;
- Gain, Loss, PairOrderChurn, DeltaAUROC, and operating-point disagreement;
- same-policy churn reference, `R_churn`, and ID/OOD replacement hybrids;
- quadratic size/stretch and symmetric branch contributions;
- spectrum/allocation spectral-band summaries;
- exact zero-state and history-conditioned counterfactual update differences
  and radial/tangential projections;
- affine alignment, held-out-ID and per-OOD residuals, prototype residual, and
  precision residual;
- update-dynamics and standardized geometry trajectories;
- numerical condition, ridge, failure, and leakage status.

Required checks include:

- exact identity/reconstruction tests with scale-aware tolerances;
- full-rank affine-gauge invariance plus deliberate rank/ridge boundary tests;
- brute-force tie-aware Gain/Loss/PairOrderChurn/AUROC balance tests;
- small-array ID/OOD replacement-Shapley reconstruction tests;
- tie-aware AUROC parity that does not count a broken numerical tie as a
  scientific rank error;
- ridge/full-rank-only Woodbury rank tests;
- small-array RMD low-rank quadratic-plus-affine reconstruction;
- same-state, same-gradient coupled/decoupled counterfactual update tests;
- same-initialization, RNG, DataLoader, and first-minibatch identity;
- zero-decay Adam/AdamW and SGDM/SGDW semantic parity where applicable;
- deterministic probe membership and no protected-result leakage;
- ImageNet-200 class-membership and pretrained-leakage checks before scale
  execution.

The current repository has part of this infrastructure, but v6 training,
multi-depth extraction, expanded trajectory snapshots, update logging, and
scale regimes are `NOT_RUN`/not yet implemented until separately authorized.

## 13. Pre-execution addendum and stopping rules

Before protected OOD execution, freeze in a versioned addendum:

- exact accuracy/NLL/ECE equivalence and guardrail margins;
- practical AUROC/FPR95 margins;
- seed-count/power or minimum-detectable-effect justification for the fixed
  five/three seed allocation;
- standardized divergence references, functional trajectory uncertainty,
  detectability summaries, multiplicity handling, and spectral-band boundaries;
- historical-preflight numerical-rank rule, propagated affine-score bound,
  churn denominator floor, and ID/OOD residual normalization;
- exact protected OOD evaluation schedule and go/no-go rule.

Null and adverse results remain reportable. Do not rescue them by adding a
detector, selecting a matching checkpoint, changing the primary epoch, or
pooling cells whose effects disagree.
