# Reference Card 13: Active Paper Protocol

## 0. Authority and status

Protocol identifier:

```text
fixed_readout_pair_ranking_multiplicity_paired_trajectory_v12
```

The stable path of this file is the sole authority for the paper's executable
experiment design. The protocol version is recorded inside this file and in
Git history; future revisions replace this content without creating another
Card 13 filename. The current design version is `v12`. When v12 was frozen,
its fresh experiments were `NOT_RUN`; this is freeze-time provenance, not the
live execution status. Current execution state lives only in
[`STATUS.md`](../STATUS.md). The frozen Task F execution is now complete; its
post-execution interpretation and claim boundaries are maintained separately in
[`task_f_result_analysis.md`](../paper/task_f_result_analysis.md). That result
document does not revise the estimands or decision rules frozen here.

V12 is an owner-approved governance and preservation-record revision. It
corrects one compact-record transcription error, documents the already-run
cached-parity diagnostic convention, and separates the primary mechanism
study from the optional RtMD method slot. It does not rerun or re-analyze the
preflight, change a numerical tolerance or estimator, or change any completed
scientific result or gate decision.

This revision does not change completed evidence:

- Metric Contract v1.2 and its 30-model WRN-28-10/CIFAR-10 population remain
  valid descriptive discovery evidence;
- the Research Contract v2 radial Stage-2 result remains immutable `FAILED`;
- the v3 historical MD--Marginal--RMD analysis remains `PASS` and discovery
  only;
- a bounded, checksum-verified D1 reuse analysis of historical per-sample
  scores is complete and gives a discovery-only `GO` for the pair-ranking
  multiplicity question; its cross-policy configurations mix LR/WD and are
  not same-initialization causal evidence;
- the validated `fork_from_prefix` runtime remains available, but it is an
  optional follow-up tool rather than the main paper intervention;
- the bounded historical discriminant--residual and residual-tail preflight is
  complete, with execution `PASS` and historical Gate 2 permanently
  `INCONCLUSIVE`; a follow-up ID-only archive diagnostic supports one
  operand-scale compliance clarification, but neither result activates RtMD;
- the single v2 compliance rerun is complete and makes the frozen estimator
  eligible for prospective fresh use without re-adjudicating Gate 2 or
  activating RtMD;
- at the v12 freeze point, no fresh v12 trajectory, protected-OOD
  confirmation, or replication had run; current execution state lives in
  [`STATUS.md`](../STATUS.md).

Fisher/LDA discriminant subspaces, the known MD--Marginal--RMD relation, L2
feature normalization, size--stretch, spectrum--allocation, and
radial/angular optimization dynamics are prior knowledge or measurement
tools. They are not claimed as new generic subspace facts or complete causal
mediation. Pair-ranking multiplicity under controlled training-rule variation
is the primary problem; the formation study and theorem explain it. Section
7.5 reserves one conditional, theory-derived Residual-t Mahalanobis (`RtMD`)
method slot. Its formula and ID-only fit are now frozen, but it is not
activated or validated and may be closed without weakening the primary paper.
Only the bounded ID-only fitter/preflight exists; the full detector and OOD
evaluation remain unimplemented and `NOT_RUN`.

## 1. Paper question and claim boundary

The paper asks:

> Starting from the same initialization and data stream, does changing only
> decay coupling alter exact ID--OOD pair orderings beyond same-policy seed
> variation; where in the update and representation trajectory does that
> multiplicity form; and do the within-class-whitened `S`/`S-perp` score
> components explain why Raw MD and RMD differ in sensitivity?

The primary contribution is a controlled pair-ranking-multiplicity study
implemented as paired from-scratch trajectories, not an optimizer leaderboard
and not a claim that similar ID accuracy alone causes OOD differences.
Comparable ID utility is an interpretation condition.

Contribution order:

1. controlled pair-ranking multiplicity: exact tie-aware
   Gain/Loss/PairOrderChurn for same-image pairs, scaled against same-policy
   seed variation;
2. paired from-scratch attribution from decay policy through update and
   representation trajectories to the score channels that carry that churn;
3. a discriminant--residual explanatory interface: under the stated estimator
   conditions, Raw MD and Marginal share the same `S-perp` term and RMD
   cancels it, with exact component-level pair accounting;
4. a fixed-total-decay `alpha in {0, 0.5, 1}` anchor confirmation, the
   prespecified normalization/cancellation diagnostics, and controlled
   factorial plus architecture/class-capacity replication;
5. only if Gate 1 and the fresh Gate 3 rule carrying the historical
   plausibility question pass, followed by the unchanged protected and
   replication gates, a secondary `RtMD` contribution that retains residual
   OOD information while replacing the Gaussian residual-tail model with one
   frozen ID-only heavy-tail model.

The positioning boundary is explicit. Prior OOD work already establishes
that seeds and training details can change detector performance and rankings;
this paper does not claim discovery of OOD instability. Classification churn
and predictive multiplicity provide the decision-level vocabulary, but the
primary estimand here is **ID--OOD pair-ranking churn**, not thresholded class
prediction disagreement. Zhao et al. establish that weight-decay coupling can
change Neural-Collapse formation and provide the fixed-total-decay coupling
interpolation precedent, but they do not study OOD detection or Mahalanobis
scores. The paper asks which optimizer-sensitive geometry is transmitted to a
fixed OOD readout. It treats NC0--NC4 as a supporting profile spanning
classifier, within-class, class-mean, and alignment structure; it never
identifies “NC” wholesale with the `S` side of this score decomposition.

The strongest allowed sentence is conditional:

> Changing only decay coupling can create pair-ranking multiplicity beyond
> ordinary seed variation while largely preserving class-discriminative
> utility. When that excess churn localizes to class-orthogonal residual
> geometry, Raw Mahalanobis reads the channel whereas RMD cancels it under the
> stated estimator conditions and attenuates the policy-specific excess.

If the normalization interaction gates also pass, the following extension is
allowed:

> Post-hoc L2 normalization and RMD stabilize this sensitivity by distinct
> operations on an empirically common residual channel: normalization reshapes
> the feature distribution and its refitted residual term, whereas RMD cancels
> the transform-specific residual term algebraically.

The first sentence is unavailable unless the practical paired effect, estimator
applicability, repeatable representation divergence, `S-perp` component
concentration, RMD attenuation, ID-equivalence/Pareto classification, and
fresh replication gates pass. The extension additionally requires the
transform-specific component, interaction, and fixed-`rho` attenuation gates.
The theorem alone does not establish that decay coupling changes `S-perp`, the
sign of an OOD effect, or causal mediation. If ID equivalence fails, the result
is retained and interpreted as a Pareto trade-off rather than discarded or
post-hoc checkpoint-matched.

L2 normalization and RMD are not declared to be the same operation. L2 is a
nonlinear feature transform followed by a new Gaussian fit; RMD is an
algebraic cancellation inside one fixed transformed feature space. The paper
may claim that they attenuate a common empirical sensitivity channel only if
the transform-specific component and interaction gates in Sections 6--8 pass.

`RtMD` is never required for either allowed sentence above. Section 13 freezes
its exact score, covariance/scatter convention, ID-only fit, failure rule,
comparison panel, and success/guardrail criteria before the historical
residual-tail preflight and before protected OOD is opened. It cannot be added
or altered to rescue a failed mechanism result.

V12 treats RtMD as an optional sidecar. The main fresh paired mechanism study,
Task F engineering, and their preserved claim boundaries do not depend on a
Gate 3 `PASS` or on the existence of an exact Gate 3 rule. RtMD failure or
closure is neither failure nor a stopping condition for the main study.

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
v12 does not place Adam and AdamW on a purported common realized-decay norm.
It uses an exact same-state, same-gradient one-step counterfactual operator
difference, defined in Section 6.1, and reports its radial and tangential
components.

At the primary nonzero-decay anchor, v12 also uses the already implemented
`adam_coupled_decoupled` operator with `total_weight_decay=1e-4` and
`coupled_ratio alpha in {0, 0.5, 1}`. Here `alpha=0` is the exact AdamW
endpoint, `alpha=1` is the exact Adam endpoint, and `alpha=0.5` sends half of
the nominal decay through each path. This fixes a nominal total input but does
not match realized regularization. Zero decay remains a separate reference,
not a fourth alpha value.

Training seed is the independent statistical unit. Repeated images and
ID--OOD pairs are paired observations, not independent replicates.

## 3. Main training design

### 3.0 V12 study and workstream priority

Fresh paired mechanism training is the primary study. It preserves the
same-initialization/data-stream control, fixed-total-decay alpha design,
trajectory snapshots, comparable-ID/Pareto interpretation boundary, claim
limits, and prohibition on protected-OOD access. Task F CPU engineering and
fixture validation may proceed in parallel with, and independently of, the
optional RtMD slot. Main paired training may likewise proceed after its own
pre-execution contracts and resource approval even if no exact RtMD Gate 3
rule exists.

The RtMD slot remains secondary and optional. Historical v2 establishes only
that the frozen estimator is numerically eligible for prospective fresh use;
it does not supply a Gate 2 `PASS`, a Gate 3 rule, method activation, or OOD
evidence.

### 3.1 WRN-28-10/CIFAR-10 Adam-family factorial

Use the local 2 x 2 design below. Each nonzero cell contains both Adam
(coupled L2) and AdamW (decoupled weight decay).

| LR | nominal WD `1e-4` | nominal WD `1e-3` |
| --- | --- | --- |
| `3e-4` | Adam / AdamW | Adam / AdamW |
| `1e-3` | **primary anchor:** zero / `alpha=0,0.5,1` | Adam / AdamW |

The schedule shape is shared: 200 epochs, Multistep milestones
`[60, 120, 160]`, and `gamma=0.2`. Only the initial LR differs by row.

Run allocation:

- primary anchor `(lr=1e-3, wd=1e-4)`: zero / `alpha=0` AdamW /
  `alpha=0.5` mixed / `alpha=1` Adam, five paired seeds, all trajectory,
  depth, geometry, score, and pair analyses: 20 runs;
- `(1e-3, 1e-3)`: Adam / AdamW, three seeds, epoch-200 focal analysis:
  6 runs;
- `(3e-4, 1e-4)`: zero / Adam / AdamW, three seeds, epoch-200 focal
  analysis: 9 runs;
- `(3e-4, 1e-3)`: Adam / AdamW, three seeds, epoch-200 focal analysis:
  6 runs.

Total Adam-family budget: **41 runs**. One zero-decay baseline is shared by
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

The alpha arm reports the three seed-paired point estimates and the contrasts
`Y_0.5-Y_0`, `Y_1-Y_0.5`, and `Y_1-Y_0` for ID utility, geometry components,
Raw-MD/RMD pair churn, and net AUROC. An outcome is called
**interior-compatible** only when the seed-mean `alpha=0.5` estimate lies
between the two seed-mean endpoint estimates in the endpoint direction;
otherwise it is reported as a non-monotone three-point response. No sign is
assumed in advance, and seed-level contrasts plus uncertainty remain visible.
With one
interior point the protocol may test midpoint/interiority and monotonic
compatibility, but it must not claim a resolved dose-response curve, linearity,
or a common effective-regularization scale. A one-seed shortened run may check
execution only and is never scientific evidence.

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
equivalence margins and the inferential rule must be frozen with independent
evidence before fresh research results are inspected, as specified in Section
13.4.

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
Consequently, v12 does not use “update onset precedes geometry onset” as a
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

For a score that passes the Section 7 applicability gate, fit the
within-class-whitened class-mean span `S` and write `x=x_parallel+x_perp`.
The primary theorem-aligned accounting is

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

The signs above are score signs: every component is ID-like and the three
pair-margin components add. Implementations must not mix these definitions
with unsigned positive distance terms. The `S-perp`, parallel-Marginal, and
RMD components are primary mechanism accounting; the older two-component
Marginal/RMD identity remains a coarser exact view and historical bridge.

For one fixed branch, transform, and fitted detector, the sample-side
**detector interface** for all three Gaussian scores is

```text
q_perp(x) = ||P_S-perp x||^2,
x_parallel = P_S x.
```

The branch-specific fitted objects--the whitening map, `S`, `eta_c`, and
`B|_S`--remain part of the detector state. Thus `(q_perp, x_parallel)` is
score-sufficient only relative to that fixed fit; it is not claimed to be a
general statistical sufficient statistic, a complete description of the
representation, or a common coordinate system across branches. Policy effects
reach this Gaussian score family through changes in the sample-side interface
or these fitted objects. Other geometry channels are retained to explain why
that interface moved or to test alternative pathways.

The prespecified residual-retention diagnostic is defined inside one fixed
branch and one fixed raw/L2 fit as

```text
s_MD^(rho) = rho s_perp + s_parallel_Marginal + s_RMD,
m_MD^(rho) = rho m_perp + m_parallel_Marginal + m_RMD,
rho in {0, 0.25, 0.5, 0.75, 1}.
```

`rho=1` is that fit's MD score and `rho=0` is its `S`-only MD score. The latter
is not RMD because it retains the parallel-Marginal term. For each ID--OOD pair
the margin is linear in `rho`, so every interior flip threshold is available
analytically. Report the five frozen grid values and every exact interior
pair-flip threshold; do not fit or select another `rho`. The aggregate
AUROC/churn path is reconstructed from those exact pair transitions. Use this
as a fixed channel-dose diagnostic, not as a new detector family. Do not choose
a favorable `rho` from ID or protected OOD data and do not make a best-`rho`
performance claim.

This score attenuation must not be mislabeled as a refitted feature
normalization. Scaling `S-perp` coordinates by any `gamma>0` is an invertible
linear map; if class means and covariance are transformed and refitted
consistently, affine invariance makes MD unchanged. `gamma=0` is a singular
projection rather than a continuous refitted-MD family. A `gamma^2 s_perp`
formula is therefore a frozen-metric score reweighting, which is exactly the
role served by `rho` above.

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

The declared two-component Marginal/RMD AUROC accounting uses four
computational hybrids: both
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

Keep the three accounting views distinct but connected: the component identity
is the exact additive account of a pair margin; the fixed `rho` path is its
one-parameter residual-channel section with analytic flip thresholds; Shapley
is used only for a declared nonlinear aggregate replacement game. Shapley is
neither needed nor used to replace the exact pair-level additive identity.

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
Call this decision stability under a controlled policy perturbation. Reserve
“retraining reproducibility” for the same-policy, different-seed denominator;
one model's ID score variance alone cannot establish it.

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
- angular update per parameter group and its distribution across layers;
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

Prior normalized-network optimization work motivates radial, tangential, and
angular measurements. For a parameter block whose function is locally
invariant to positive radial rescaling, decoupled weight decay is an explicitly
radial step. Under a frozen diagonal preconditioner, a coupled L2 contribution
is proportional to `P_t w` and can have a nonzero tangential component unless
`P_t w` is parallel to `w`. This is a cited manipulation/pathway audit, not the
paper's central theorem and not complete causal mediation. It is **not**
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

Before ResNet replication is launched, use the anchor update audit to decide
the prespecified location-ablation gate. If the anchor has a practical policy
effect and rescaling-eligible versus scale-breaking groups differ clearly, a
secondary follow-up may train Adam and AdamW with decay restricted to each
group (two groups x two policies x three shared seeds; the existing zero
sibling is reused). It uses only focal endpoint analysis. This tests where the
coupling contrast is carried; it must not be described as a pure separation of
“effective learning rate” and “functional regularization,” because both
mechanisms can coexist in either arm.

### 6.2 Level B: representation geometry

For recipe `r`, checkpoint `t`, depth `l`, and fixed image `x`, extract
`z_{r,t,l}(x)`. Fit and report these prespecified channels:

| Role | Channel | Measurements |
| --- | --- | --- |
| Primary score-facing interface | Discriminant/residual | distributions of `q_perp` and `P_S x` on ID train, held-out ID, and each OOD dataset; branch-internal score/pair components |
| Primary fitted geometry | Fit and branch frame | actual `dim(S)`, whitening, `eta_c`, `B|_S`, ID-only gauge-aligned principal angles, and zero-decay common-frame diagnostics |
| Primary negative control | Affine gauge / residual | ID-only branch alignment and held-out same-image non-affine residual |
| Supporting formation diagnostic | Global radial scale and heterogeneity | shared-scale negative control; norm trajectories and distributions; same-image multipliers; ID--OOD radial relation; radius--direction coupling |
| Supporting formation diagnostic | Class geometry | class-mean distances/angles, CDNV, and the prespecified NC profile |
| Supporting estimator diagnostic | Global/within-class geometry | global and pooled covariance trace/spectrum/effective rank/condition number |
| Supporting score diagnostic | Allocation and class-distance profile | spectral-band sample displacement plus nearest-class versus full class profile |

The basic paired contrasts within every seed and cell are coupled--decoupled,
coupled--zero, and decoupled--zero. This separates the presence of decay from
the way decay is coupled.

The NC profile is the explicit interface to Zhao et al.'s upstream formation
theory. Report whether alpha and endpoint contrasts in NC0--NC4 track, precede,
or dissociate from the detector-interface and churn contrasts. This is a
transfer analysis, not a claim that Zhao predicts OOD behavior or that NC is a
scalar mediator.

Each branch refits `mu_c` and `Sigma_W`, so its whitened subspace is generally
different. The primary detector attribution is exact **inside each branch**.
Cross-branch statements about feature energy or subspace rotation require an
ID-only affine/gauge alignment and a diagnostic common frame defined by the
zero-decay sibling. Report the principal-angle profile and whitening change;
do not compare two separately whitened coordinate systems as if they were one
fixed frame. Measure the actual `dim(S)`, which may be below `K-1`; equality is
not an applicability gate.

In the unregularized full-rank case, the fitted ID-train mean of `q_perp` is
pinned to `d-dim(S)`. Therefore a branch claim is not based on a larger
ID-train mean residual energy. Report the empirical variance, standardized
moments, upper quantiles, and held-out-ID Q--Q deviation from `chi^2_k`, with
`k=dim(S-perp)`, using a frozen cross-fit or calibration rule. Whitening does
not pin `Var(q_perp)` or its tail. These tail summaries are a primary
activation input only for the conditional Section 7.5 method slot; they do not
replace the score-component and pair-order outcomes.

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

Treat this as a prespecified `normalization x cancellation` 2 x 2:

| Feature fit | Reads transform-specific `S-perp` | Cancels transform-specific `S-perp` |
| --- | --- | --- |
| raw | MD-Raw | RMD-Raw |
| per-sample L2, then refit | MD++ | RMD++ |

The cancellation theorem applies separately inside each applicable raw or L2
fit. Per-sample L2 normalization changes the class means, covariance,
whitening, `S`, and potentially every parallel term; it does not literally
multiply the raw fit's `s_perp`. Therefore compare the four detector outcomes
and their transform-specific decompositions without identifying
`S_raw-perp` with `S_L2-perp`. Record raw feature norm correlations with each
raw score component, the component variance--covariance matrix on ID, and the
normalization effect on component pair margins, policy gap, and churn. A large
raw-to-L2 improvement is only consistent with residual-channel suppression
when these direct checks agree.

Before calling cancellation exact, record the actual precision backend,
covariance identity residual, numerical rank, retained condition number, and
score/component reconstruction residual. Metric Contract v1.2 continues to
use its frozen Moore--Penrose-compatible backend without explicit ridge. If
the fitted matrices are numerically full rank and the backend agrees with the
inverse within the frozen tolerance, the primary raw score may pass the
full-rank applicability gate. Otherwise, exact cancellation is unavailable
for that primary score. A same-positive-ridge calculation may be reported only
as a separately named diagnostic; it never silently replaces the v1.2 score.

For every applicable fit, save per-sample and pair-margin values for
`S-perp`, parallel-Marginal, and RMD, together with exact reconstruction and
RMD `S-perp` cancellation residuals. These are the primary score-geometry
channels. Also evaluate the fixed `rho` attenuation path in Section 5 and
store exact interior pair-flip thresholds; never select a best `rho`.

As a supporting diagnostic for each relevant quadratic term, use the cited
size--stretch factorization:

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
change from `S-perp`, parallel-Marginal, and RMD changes, then connect it to
the transition table. Also retain the coarser historical RMD/Marginal view.
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
-> theorem-aligned subspace/score divergence
-> changed ID--OOD pair ordering
```

The update-operator difference is a manipulation check. A mechanism statement
requires estimator applicability, repeatable substantive representation
divergence, exact score/pair reconstruction, pair gain/loss attribution, and
prespecified matched attenuation. Temporal ordering supports this chain but is
neither sufficient evidence nor described as complete causal mediation.

### 6.5 Channel-matched confirmation

After localization, weaken only the implicated channel:

| Localized channel | Confirmatory diagnostic |
| --- | --- |
| `S-perp` residual | RMD cancellation, `S`-only reconstruction, and the fixed `rho` path should selectively attenuate gap/churn |
| parallel-Marginal or RMD | retain the gap in `S`-only accounting and test class-relative/estimator pathways |
| sample/class-conditioned radial heterogeneity | L2 normalization and refitting; use the raw/L2 x MD/RMD interaction to test whether the changed channel is transform-specific `S-perp`; global scale remains a negative control |
| non-affine branch deformation | ID-only affine alignment and residual/precision perturbation accounting |
| spectrum/stretch | ID-only spectral-band ablation, eigenvalue clipping, or whitening diagnostic |
| class-distance profile | nearest-class versus full-profile readout |

The confirmation must selectively attenuate the branch gap without being
chosen after OOD results. A new detector is not added to rescue a failed
mechanism claim; the separately gated Section 7.5 slot cannot serve as this
confirmation or change the mechanism verdict.

## 7. Theory-constrained attribution and its limits

The supporting theory is one connected constraint chain, not a collection of
new Mahalanobis detectors.

### 7.1 Discriminant--residual RMD cancellation theorem

Use ID-train class frequencies `pi_c`, global mean `mu_0`, class means `mu_c`,
the pooled biased within-class covariance `Sigma_W`, and

```text
Sigma_B = sum_c pi_c (mu_c-mu_0)(mu_c-mu_0)^T
Sigma_0 = Sigma_W + Sigma_B.
```

The covariance identity is exact under the shared `1/N` sample convention; it
does not assume that every population class truly has the same covariance.
The tied covariance is the detector definition. For either a numerically
full-rank inverse or one common positive ridge `lambda` applied to every
relevant conditional and marginal term, let

```text
A = Sigma_W                         (full-rank case)
A = Sigma_W + lambda I              (common-ridge case)
x = A^(-1/2) (z-mu_0)
eta_c = A^(-1/2) (mu_c-mu_0)
S = span{eta_c};  x = x_parallel + x_perp.
```

Then `dim(S)<=K-1`, every `eta_c` lies in `S`, and the global covariance in
whitened coordinates is `I+B`, where `B` acts only on `S`. Consequently,

```text
d_c = ||x_perp||^2 + ||x_parallel-eta_c||^2
d_0 = ||x_perp||^2
      + x_parallel^T (I+B|_S)^(-1) x_parallel

s_perp = -||x_perp||^2
s_parallel_Marginal =
    -x_parallel^T (I+B|_S)^(-1) x_parallel
s_RMD = -min_c ||x_parallel-eta_c||^2 - s_parallel_Marginal
s_Marginal = s_perp + s_parallel_Marginal
s_MD = s_perp + s_parallel_Marginal + s_RMD
s_RMD(x) = s_RMD(x_parallel).
```

Thus Raw MD and Marginal share the literal same `S-perp` residual term and RMD
cancels it. For component `q`, define `m_q(i,o)=s_q(i)-s_q(o)`; then the same
additive identity holds for ID--OOD pair margins. This is the central theorem.
Fisher/LDA class-mean subspaces and the rank bound are prior facts; the proposed
paper contribution is the OOD-score cancellation formalization combined with
the controlled formation study. Do not use “first” language until the direct
RMD/LDA literature and the NECO, ViM, Neural-Collapse Mahalanobis,
classifier/principal/residual-subspace, and projection-filtering boundaries
are locked.

In the unregularized full-rank case, the pooled ID-train within-class residual
has identity covariance after whitening. If `r=dim(S)`, this gives the exact
training-set mean-energy identities

```text
mean_ID-train ||P_S-perp A^(-1/2)(z-mu_y)||^2 = d-r
mean_ID-train ||P_S      A^(-1/2)(z-mu_y)||^2 = r.
```

This explains why a high-dimensional residual channel can carry a large mean
share of within-class whitened distance. It does **not** imply that it carries
most sample variance, raw Euclidean norm variation, policy sensitivity, or OOD
signal. With a common ridge the corresponding quantities are projected traces
of `A^(-1/2) Sigma_W A^(-1/2)`, not simply dimensions.

**Pinning corollary.** In the unregularized full-rank fit, pooled ID-train
within-class residual vectors have zero mean and identity second moment. Hence
`mean(q_perp)=d-r` inside each branch, where `r=dim(S)`. If `r` differs across
branches, the pinned means differ only through that measured dimension. This
does **not** pin `Var(q_perp)`: it depends on fourth moments of the residual
vector, and skewness, kurtosis, upper-tail shape, and held-out behavior remain
free. The main non-pinned categories inspected by v12 are ID higher-order
interface shape, held-out-ID generalization drift, OOD placement in
`(q_perp,P_S x)`, and branch-specific fitted geometry/class heterogeneity.
These are an organizing list, not a theorem that exactly four statistical
degrees of freedom remain.

`S` means the class-mean-discriminative subspace for this pooled
tied-covariance detector; it is not all information used by a nonlinear
classifier. `S-perp` is not meaningless noise: it may contain OOD signal,
higher-order structure, or class-specific covariance information. The theorem
states what these three Gaussian scores can see, not which channel training
must change or which detector must perform better.

L2 normalization is outside the algebra of one fixed raw fit: it maps each
sample nonlinearly to the sphere and then refits all Gaussian statistics. The
theorem holds again inside the normalized fit if its own applicability gate
passes, but it does not prove that L2 partially removes the raw fit's
`S-perp`. The stronger empirical interpretation--L2 reduces a norm-driven
residual sensitivity whereas RMD cancels the transform-specific residual--is
available only if the Section 6.3 interaction, component covariance, norm
correlation, and pair-margin attenuation agree. Existing broad
Mahalanobis++ evidence that normalization can also improve RMD prevents a
universal claim that L2 is useless or harmful after RMD.

Metric Contract v1.2 uses `Sigma_W^dagger` and `Sigma_0^dagger` without an
explicit common ridge. A general Moore--Penrose pseudoinverse does not inherit
the theorem automatically. Preserve the frozen v1.2 primary score. Apply the
exact statement to it only when the actual matrices/backend pass the frozen
full-rank, conditioning, inverse-parity, and reconstruction gate. Otherwise
report the theorem as inapplicable to that primary fit and, if needed, report a
separately named common-ridge diagnostic. Never relabel the diagnostic as the
primary score.

The exact score attribution is branch-internal. Because each branch refits its
own whitening and class means, cross-branch geometry additionally reports
ID-only gauge-aligned principal angles, whitening change, and a zero-decay
common-frame diagnostic. These cross-branch diagnostics interpret formation;
they do not replace the exact branch-internal decomposition.

### 7.2 Affine-gauge proposition and residual diagnostic

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

### 7.3 Cited local update-direction audit

For a locally positive-scale-invariant parameter block, the explicit AdamW
decay term is radial. Under a frozen diagonal preconditioner, a coupled L2 term
can induce a tangential direction proportional to the tangential projection
of `P_t w`. Existing normalized-network optimization literature motivates the
radial/tangential/angular measurements. The exact implementation audit is the
same-state, same-gradient counterfactual operator difference from Section 6.1.
This is not a new central proposition, and the paper does not claim that all
WRN trunk weights are scale-invariant, that rotation uniquely mediates the OOD
effect, or that every decay effect is a pure optimization-path effect.

The architecture-defined rescaling-eligible/scale-breaking split in Section
6.1 turns this proposition into a location-specific prediction. Its optional
training ablation localizes the carrier but does not label either arm a pure
mechanism.

### 7.4 RMD low-rank curvature corollary

Using class-frequency-weighted between-class covariance,

```text
Sigma_0 = Sigma_W + Sigma_B
rank(Sigma_B) <= K - 1
```

without requiring balanced classes. Under the same full-rank/common-ridge
convention as Section 7.1, with `A=Sigma_W` or `A=Sigma_W+lambda I`, Woodbury
gives

```text
H = A^-1 - (A + Sigma_B)^-1 >= 0
rank(H) <= K - 1
s_RMD(z) = -z^T H z + max_c(a_c^T z + b_c)
```

Thus the earlier low-rank-curvature result is a corollary of RMD acting only on
`S`: its quadratic correction is supported on a subspace of dimension at most
`K-1`. This does not create a single shared RMD covariance. Numerical rank,
PSD residual, curvature mass, and allocation remain supporting diagnostics,
not a second central mechanism. The guaranteed residual dimension is a lower
bound `dim(S-perp)>=d-(K-1)`, not an exact discarded proportion. Actual
`dim(S)` is measured in every branch and checkpoint. The bound predicts
neither effect sign nor a monotone OOD trend with class count.

The completed v1 preflight used the existing verified 30-bundle raw-feature
cache read-only. Its rank/condition and reconstruction tolerances were frozen
before execution; it measured theorem applicability,
branch-internal `S-perp`/parallel-Marginal/RMD reconstruction, RMD cancellation
residual, actual `dim(S)`, branch principal angles, the zero-reference common
frame, classifier-row-space alignment, and historical Raw-MD gap
concentration. For ID train/test, report all three component variances and
their full covariance matrix so that they reconstruct total score variance;
also report raw-norm/component correlations. Do not call
`Var(s_perp)/Var(s_MD)` a percentage of fragility: covariance can put this
ratio outside `[0,1]`, and within-model ID variation is not retraining
variation. Use the completed C3 raw/L2 x MD/RMD pattern only to motivate the
fresh interaction prediction because its detector-wise maxima occur on
different OOD datasets and compare different optimizer families. Curvature
mass/allocation is secondary. This mixed-recipe
population is noncausal: it selects a useful measurement hypothesis but does
not confirm a decay-coupling effect or guarantee fresh-study power.

The one allowed compliance rerun, v2, is complete. It used the same ID-only
cache and changed only the RMD-cancellation denominator clarified in Section
13.1; v1 and its permanently `INCONCLUSIVE` Gate 2 record remain immutable.
The official production payload preserved all 60 fit keys and left theorem
applicability unchanged at `30/60`. All 30 applicable fits passed every
required numerical gate. Cached Metric Contract v1.2 parity was a diagnostic:
v1 had 24 PASS and 36 FAILED fits, while official v2 had 52 PASS and 8 FAILED
fits (`28` `FAILED→PASS`, `0` `PASS→FAILED`); all eight remaining failures were
theorem-inapplicable, and applicable cached-parity failures were zero. The
common-ridge diagnostic moved from 27 to 48 PASS fits, with 21
`FAILED→PASS` and no reverse transition. Required numerical PASS moved from
`9/60` to `30/60`, a gain of 21.

The compact preservation record's former value of nine inapplicable
cached-parity failures came from preliminary-output provenance rather than the
official production payload and is corrected to eight in v12. This is a
transcription/provenance correction, not another preflight execution,
remediation, Gate 2 re-adjudication, or scientific-result change. V2 refreshed
measurement coverage and exact accounting but did not activate RtMD.

### 7.5 Conditional Residual-t Mahalanobis method slot

V12 reserves exactly one optional post-hoc method slot. Its frozen statistical
object keeps the class-discriminative `S` distance and replaces only the
Gaussian radial likelihood in `S-perp` by one covariance-normalized
multivariate-t radial likelihood:

```text
q_S(x) = min_c ||x_parallel-eta_c||^2
q_perp(x) = ||x_perp||^2
k = dim(S-perp)

g_(nu,k)(q) = (nu+k) log(1 + q/(nu-2)),  2 < nu < infinity
g_(infinity,k)(q) = q

D_RtMD(x) = q_S(x) + g_(nu,k)(q_perp(x))
s_RtMD(x) = -D_RtMD(x).
```

The residual t distribution has covariance `I` and scale matrix
`((nu-2)/nu) I`. Thus `g` is twice its negative log likelihood after removing
the sample-independent constant, and `nu -> infinity` recovers the Raw-MD
quadratic exactly. The full residual negative log likelihood used to fit `nu`
is

```text
nll(q;k,nu)
 = -lgamma((nu+k)/2) + lgamma(nu/2)
   + (k/2) log((nu-2) pi)
   + ((nu+k)/2) log(1 + q/(nu-2)).
```

No free residual scale is permitted. For every branch and raw/L2 transform,
fit one `nu` from ID train only. Split stable sample identities into a
deterministic class-stratified two-fold partition. Fit the geometry on the
opposite fold for each observation, pool the out-of-fold `q_perp` values, fit
one branch/transform-specific `nu`, and then refit the final query geometry on
all ID train. The finite domain is `nu in [2.05,1000]`, optimized
deterministically in `theta=log(nu-2)` with a bounded scalar optimizer, plus an
exact `nu=infinity` Gaussian candidate. Retain finite t only if
`2(LL_t-LL_Gaussian)>log(N)`. Otherwise, or on a non-finite objective,
optimization failure, or lower-bound solution, fall back to `nu=infinity` and
record RtMD activation failure for that fit. ID test, protected OOD, AUROC,
and FPR95 are forbidden for fitting, selection, or fallback.

The intended interpretation is: MD trusts the quadratic residual fully, RMD
cancels it, and `RtMD` tests whether a tail correction can retain residual
evidence while limiting domination by that channel. This is a conditional
method candidate, not a current result, central theorem, or fifth contribution.

The completed direct-collision subgate of Gate 1 is a **narrow PASS**. WDiscOOD
already occupies the within-class-whitened discriminative/residual split and a
linear residual combination. Linderman et al. occupy the RMDS/DPMM connection
and generic Student-t predictive extensions. D-KNN uses PCA
principal/residual spaces with dual-space KNN calibration; CORE uses a
classifier-row-space residual; MaRS uses an autoencoder reconstruction
residual. None of these audited methods combines the exact frozen block score
above with training-rule pair-ranking churn as its target, but that absence is
not a “first” claim. The only allowed novelty scope is to test whether
tail-correcting the class-orthogonal residual block identified by RMD
cancellation reduces training-rule pair-ranking churn while preserving
far-OOD residual evidence. Do not claim the first subspace detector, the first
Student-t/robust Mahalanobis detector, the first residual-score combination,
or that a heavy tail implies method success.

The gates are sequential only for the optional RtMD slot, with the historical
Gate 2 governance outcome preserved rather than retroactively thresholded.
They are not launch gates for Task F or the main paired mechanism study:

1. **Derivation and novelty gate:** the frozen likelihood is coherent and the
   WDiscOOD, D-KNN, CORE, MaRS, direct RMD/LDA, robust/t-Mahalanobis, and
   Bayesian-nonparametric/DPMM--RMDS audit, including Linderman et al., gives
   the narrow direct-collision PASS above. This does not activate the method.
2. **Historical ID-only plausibility record:** the completed cache run tested
   held-out-ID `q_perp` deviation from `chi^2_k`, tail estimability, and
   between-model variation, but Card 13 had no frozen aggregate decision rule
   and only half of the raw/L2 fits had numerically applicable two-fold
   geometry. Its governance status is permanently `INCONCLUSIVE`; its finite
   subset is conditional discovery and is never re-adjudicated as PASS or
   FAILED. The unresolved plausibility question transfers to Gate 3.
3. **Fresh ID-only activation gate:** v12 does not yet freeze an exact Gate 3
   statistic, threshold, same-policy reference, or finite-time aggregate rule.
   Confirmatory use of the paired anchor requires a versioned addendum frozen
   before its fresh ID-only residual results are inspected, including the exact
   primary transform, tail-shape statistic, same-policy seed reference,
   pairing, multiplicity, inapplicability/fallback handling, and deterministic
   activation rule. If the rule is designed after inspecting the anchor's
   ID-only results, that anchor's RtMD evidence is `discovery-informed` or
   `discovery-only`; it cannot be reused as a confirmatory PASS, and method
   activation requires a later preregistered replication Gate. Any prospective
   rule must jointly establish numerical applicability, ID-only tail
   estimability, and a repeatable policy effect on held-out residual tail shape
   beyond same-policy seed variation. Mean ID-train residual energy cannot
   satisfy this gate because it is pinned by whitening. Weak residual evidence,
   insufficient estimator applicability, or a failed prospective Gate 3 closes
   the RtMD slot without a replacement estimator while the main study
   continues.
4. **Frozen protected-OOD evaluation:** only after Gate 1 and a prospectively
   registered fresh Gate 3, including any required replication Gate carrying
   the transferred plausibility content, pass, evaluate the
   unchanged score once against Raw MD, RMD, their L2 fits, WDiscOOD, and
   the already prespecified appendix controls. The primary method estimand is
   reduction of coupled--decoupled `PairOrderChurn` relative to Raw MD and the
   same-policy churn reference. Prespecified AUROC/FPR95 non-inferiority and
   preservation of far-OOD residual signal relative to RMD are guardrails.
5. **Replication gate:** a method contribution requires the frozen effect to
   replicate outside the WRN/CIFAR-10 anchor. Otherwise report it as local or
   close the slot.

The full RtMD detector and every protected-OOD path remain prohibited from
implementation and execution. Protected OOD may be connected only after a
separate frozen addendum specifies the necessary gate, unchanged evaluation
schedule, and resource report, followed by explicit owner approval. No
protected-OOD path or dataset identity is part of the pre-Task-F interface.

`nu-hat` may be recorded as an exploratory ID-only candidate predictor of
fresh policy churn, not detector performance. Heavy tails alone do not imply
that `nu-hat` predicts retraining instability. The predictor is promoted only
with a frozen target and fresh paired confirmation.

If any gate fails, `RtMD`, `nu-hat`, and their method claims are removed from
the contribution list while the theorem and paired mechanism study continue
unchanged. No second new-score slot, best-`nu` OOD search, or post-result
formula repair is allowed.

### 7.6 Pair-order balance proposition

The Gain/Loss/PairOrderChurn identities in Section 5 are exact and explain how
large opposing ordering changes can be hidden by a small net AUROC difference.
They constrain reporting: pair churn cannot be replaced by a thresholded
deployment metric or treated as millions of independent replicates.
The policy-to-natural-variability churn ratio supplies scale, while the
ID/OOD hybrid accounting localizes signed score motion. Neither creates a
unique causal partition of the nonlinear churn indicator.

The paper combines the discriminant--residual score/pair identity with the
known MD--Marginal--RMD relation. It uses the cited symmetric size--stretch
identity and spectrum/allocation expansion as supporting measurements.

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

Prespecified alpha confirmation family:

- the same five anchor seeds at `alpha=0,0.5,1` with fixed
  `total_weight_decay=1e-4`;
- endpoint, lower-half, and upper-half paired contrasts for ID utility,
  NC0--NC4, theorem-aligned components, Raw-MD/RMD churn, and net AUROC;
- interior-compatibility versus non-monotone three-point classification;
- no fitted curve, post-hoc alpha selection, or claim of matched realized
  regularization.

Primary mechanism family:

- theorem applicability and exact branch-internal
  `S-perp`/parallel-Marginal/RMD score and pair attribution;
- RMD cancellation residual, the raw/L2 x MD/RMD interaction, and the
  prespecified `rho` residual-retention path including `S`-only attenuation;
- gauge-aligned branch principal angles, zero-decay common-frame diagnostics,
  and held-out-ID versus per-OOD affine residual;
- ID/OOD score-side replacement accounting;
- counterfactual radial/tangential/angular update audit;
- channel-matched attenuation.

Secondary families are the prespecified norm/radial, class-mean/CDNV/NC,
size--stretch, spectrum--allocation, functional-time, and within-stage-depth
diagnostics; the Adam LR/WD interactions; SGDM control; `best_val`; and the
epoch-300 appendix. External detector panels, individual eigenvectors, extra
geometry scalars, and optional fork analyses are exploratory or appendix. Do
not perform separate unadjusted tests for every metric by every checkpoint by
every depth by every OOD dataset. Exact family definitions,
simultaneous/functional trajectory uncertainty, practical margins, and
multiplicity control are frozen in the pre-protected-OOD addendum.

If and only if Section 7.5 reaches its protected-OOD gate, `RtMD` forms a
separate secondary method family. Its churn-reduction estimand, performance
guardrails, and replication rule are not pooled with the primary mechanism
family. Failure of this family does not alter the primary paper result.

## 9. Detector roles

Focal Mahalanobis family:

- Mahalanobis-Raw;
- Marginal-Mahalanobis-Raw;
- RMD-Raw;
- separately refitted L2 versions of all three.

External controls:

- kNN-Raw/L2, CTM, Pure Residual, and Energy-T1;
- WDiscOOD as the direct whitened discriminative/residual score control;
- MSP, ViM, and CORE in the appendix;
- ReAct excluded.

The focal family answers the component question. Controls test whether the
effect is specific to the Mahalanobis formula. They are not ranked by
popularity. No post-hoc detector expansion beyond the explicit Section 7.5
slot is allowed. If that slot activates, `RtMD` is compared with Raw MD, RMD,
their L2 fits, WDiscOOD, and the prespecified appendix controls without OOD
tuning. The fixed `rho` path is an exact computational intervention on an
already declared component, not an additional benchmark detector; no `rho` is
selected or entered into a performance leaderboard.

## 10. Replication and scale order

1. ResNet-18/CIFAR-10: architecture replication in a similar guaranteed
   residual-capacity regime; it is not a clean single-factor ablation.
2. ResNet-18/CIFAR-100: class-count/discriminant-capacity stress test in which
   the guaranteed residual lower bound is materially smaller.
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

The theorem implies lower bounds, not exact discarded proportions:

| regime | feature `d` | `dim(S)` upper bound | `dim(S-perp)` lower bound |
| --- | ---: | ---: | ---: |
| WRN-28-10/CIFAR-10 | 640 | 9 | 631 (at least 98.6%) |
| ResNet-18/CIFAR-10 | 512 | 9 | 503 (at least 98.2%) |
| ResNet-18/CIFAR-100 | 512 | 99 | 413 (at least 80.7%) |
| ConvNeXt-Tiny/ImageNet-200 | 768 | 199 | 569 (at least 74.1%) |

Every regime tests theorem reconstruction directly. CIFAR-100 is not the sole
or logically unique theorem test; it is a theory-motivated stress test of the
empirical mechanism when possible discriminant capacity is larger. Report the
actual branch/checkpoint `dim(S)`, principal angles, numerical rank, curvature
mass, and allocation. Do not infer a monotone detector effect from `K` alone.

## 11. Historical discovery and optional shared-prefix follow-up

The completed 30-model v3 analysis uses 30 frozen `last.pt` bundles and six
OOD datasets. It is valid for locating descriptive MD/Marginal/RMD variation,
but optimizer, LR, and WD are mixed in that population. It cannot establish
that decay coupling caused the gap. Nearest-accuracy matching is excluded from
primary analysis and, if shown, is labeled selection-biased sensitivity.

A bounded D1 survival reuse has also been completed without checkpoint
inference. It verified 96 score-array hashes and exact sample ordering for four
frozen role configurations, then compared six historical cross-policy pairs
with 12 same-policy seed pairs. For Raw MD, median PairOrderChurn was `0.322`
versus `0.220` on CIFAR-100 and `0.359` versus `0.273` on MNIST
(cross-policy versus same-policy seed reference). For Raw RMD it was `0.123`
versus `0.114` and `0.111` versus `0.098`. This is discovery-only evidence
that ranking multiplicity is worth a controlled test and that RMD may
attenuate policy-specific excess churn; it does not establish coupling
causality, RMD immunity, all-dataset generality, or independence of the
historical LR/WD recipe differences.

The same immutable cache supports the read-only discriminant--residual
preflight in Section 7.4 without new training or protected-data traversal. It
first checks estimator applicability and exact `S-perp`/parallel-Marginal/RMD
reconstruction, then historical component concentration, branch-frame
alignment, classifier alignment, and supporting curvature/allocation. Its
output must use a fresh external artifact directory and a compact committed
summary only; do not recreate the retired Stage-2 gate, checksum catalog, or
large generated tables in Git. This preflight is discovery, not causal
evidence or a five-seed power guarantee.

After the Section 7.5 method specification is frozen, the same cache may also
support an ID-only residual-tail plausibility preflight. It must use no OOD
score to select a tail model, `nu`, scale, gate, or formula. Its results are
discovery and do not activate `RtMD` without the fresh ID-only paired gate.

The existing `fork_from_prefix` implementation may be used only as an optional
follow-up if the from-scratch divergence curves produce a quantitative
switch-time prediction. A future test must use prespecified switch times and
report the remaining fraction of the full from-scratch geometry/churn gap; it
must not choose the most favorable threshold-crossing checkpoint after seeing
OOD results. Forking is not required for the main contribution and does not
replace from-scratch evidence. Ordinary resume remains strict and distinct
from this optional fork operation.

## 12. Required artifacts and validation

Every v12 result must identify seed, initialization, config, branch policy,
checkpoint epoch, depth tap, probe-image membership, Gaussian-fit population,
and source code/config hashes. Alpha-arm records must also identify
`total_weight_decay` and `coupled_ratio`.

### 12.1 Minimum pre-Task-F artifact identity interface

Task F must preserve the following common identity fields for every emitted
feature/checkpoint artifact:

```text
run_id
training_seed
branch_policy
total_weight_decay
coupled_ratio
checkpoint_epoch
checkpoint_sha256
depth_tap
dataset_split
ordered_sample_id_sha256
feature_shape
feature_dtype
oge_git_sha
specification_sha256
execution_only
```

The manifest must explicitly encode the zero / `alpha=0` / `alpha=0.5` /
`alpha=1` sibling relationship and preserve verifiable same-initialization and
data-stream identity across siblings. Protected-OOD paths and dataset
identities are excluded from this pre-Task-F interface. The concrete
serialization schema, validation rules, and specification-hash generation were
implemented in bounded Task F Issues after v12 was frozen. The current
pre-execution specification identity is
`0ac3101e6d6aaed1a5a0d4891792d4700540bac5247cca8f7d6e67d664ffe9ba`;
the fresh-training addendum must carry this identity forward before launch.

### 12.2 Full-study records and checks

Required records include:

- ID accuracy/NLL/ECE and `last` versus `best_val` identity;
- per-sample MD, Marginal, and RMD raw/L2 scores;
- actual precision backend; `Sigma_0=Sigma_W+Sigma_B` residual; numerical rank,
  condition, ridge, inverse-parity, and applicability status;
- actual `dim(S)`, branch-internal basis/projector metadata, `S-perp` feature
  energy, and per-sample `S-perp`/parallel-Marginal/RMD score components;
- ID-train and held-out-ID `q_perp` variance, standardized moments, upper-tail
  quantiles, and frozen Q--Q/tail-fit diagnostics;
- transform-specific raw/L2 component variance--covariance matrices,
  raw-norm/component correlations, and the MD/RMD normalization interaction;
- component pair margins, additive reconstruction residuals, and RMD
  cancellation residuals;
- fixed-`rho` score/margin paths and exact interior pair-flip thresholds, with
  no selected best `rho`;
- gauge-aligned branch principal angles, whitening change, classifier-row-space
  alignment, and zero-decay common-frame diagnostics;
- tie-aware pair-transition counts and component Shapley accounting;
- Gain, Loss, PairOrderChurn, DeltaAUROC, and operating-point disagreement;
- same-policy churn reference, `R_churn`, and ID/OOD replacement hybrids;
- alpha endpoint/lower-half/upper-half paired contrasts and the frozen
  interior-compatibility classification;
- quadratic size/stretch and symmetric branch contributions;
- spectrum/allocation spectral-band summaries;
- exact zero-state and history-conditioned counterfactual update differences
  and radial/tangential/angular group profiles;
- affine alignment, held-out-ID and per-OOD residuals, prototype residual, and
  precision residual;
- update-dynamics and standardized geometry trajectories;
- numerical condition, ridge, failure, and leakage status.
- if the Section 7.5 slot activates, the frozen `RtMD` specification hash,
  ID-only split and fit record, `nu`/scale/fallback status, method churn and
  guardrails, and every prespecified comparison result.

Required checks include:

- small-array discriminant--residual score and pair-margin reconstruction under
  full-rank and common-ridge conventions, including RMD `S-perp` invariance;
- full-rank ID-train mean-energy identities and the common-ridge projected-trace
  boundary;
- a matched-covariance counterexample showing that whitening pins
  `mean(q_perp)` but not `Var(q_perp)` or the residual tail;
- exact fixed-`rho` score/margin reconstruction and pair-flip-threshold tests;
- an invertible `S-perp`-scaling/refit affine-invariance fixture that rejects
  the incorrect claim that `gamma>0` creates a continuous refitted-MD family;
- a deliberate rank-deficient pseudoinverse boundary fixture that prevents an
  unsupported exact-cancellation claim;
- orthogonal-projector, actual-rank, principal-angle, and common-frame tests;
- exact score/pair identity tests with scale-aware tolerances;
- full-rank affine-gauge invariance plus deliberate rank/ridge boundary tests;
- brute-force tie-aware Gain/Loss/PairOrderChurn/AUROC balance tests;
- small-array ID/OOD replacement-Shapley reconstruction tests;
- tie-aware AUROC parity that does not count a broken numerical tie as a
  scientific rank error;
- ridge/full-rank-only Woodbury rank and low-rank-corollary tests;
- same-state, same-gradient coupled/decoupled counterfactual update tests;
- same-initialization, RNG, DataLoader, and first-minibatch identity;
- zero-decay Adam/AdamW and SGDM/SGDW semantic parity where applicable;
- deterministic probe membership and no protected-result leakage;
- if activated, covariance/scatter parameterization, `nu -> infinity` MD
  limit, deterministic ID-only fitting, fallback, and no-OOD-selection tests
  for `RtMD`;
- ImageNet-200 class-membership and pretrained-leakage checks before scale
  execution.

When this v12 pre-execution text was frozen, the repository implemented the
historical v1 preflight, its bounded ID-only archive diagnostic, and the
completed v2 compliance rerun. V2 left estimator applicability unchanged at
`30/60`, and all 30 applicable fits passed every required numerical gate.
Historical Gate 2 remains permanently `INCONCLUSIVE`. At that freeze point,
Task F CPU engineering implemented multi-depth taps, versioned ID-only export,
the 50-run matrix, paired provenance, expanded snapshot identities, resume
checks, and update-telemetry plumbing with synthetic fixtures; fresh v12
training and production extraction, fresh Gate 3, GPU work, the full RtMD
detector, protected OOD, and scale regimes remained `NOT_RUN`. Current
execution state lives only in [`STATUS.md`](../STATUS.md).

## 13. Pre-execution addendum and stopping rules

The v10 Task B addendum was frozen before the historical v1
discriminant--residual and residual-tail preflight. V11 adds only the
operand-scale compliance clarification supported by the bounded Issue-75
diagnostic; it does not alter or re-adjudicate v1. V12 preserves that
historical clarification and adds only the compact-record provenance
correction, cached-parity convention documentation, main/RtMD governance
separation, and minimum pre-Task-F artifact interface described here. Every
covariance, whitening, eigendecomposition/SVD, projector, and reference-score
calculation uses `float64`; covariance matrices are symmetrized as
`(M+M.T)/2`.

### 13.1 Applicability and algebra tolerances

Let `eps64=finfo(float64).eps`, `d` be feature dimension, and
`tau_spec=lambda_max*d*eps64`. Eigenvalues below `-tau_spec` fail the fit;
values in `[-tau_spec,0)` are clipped to zero and counted. The primary score is
full-rank applicable only when the numerical rank is `d` and
`kappa_2<=1e8`. Preserve a failed or inapplicable Metric Contract v1.2 score,
but do not attach the exact-cancellation claim to it.

For every applicable fit define

```text
tau_alg(kappa) = max(1e-10, 10*kappa*eps64)
score_scale = max(1, |s_direct|, sum_j |s_j|)
margin_scale = max(1, |m_direct|, sum_j |m_j|).
rmd_cancellation_scale
  = max(1,
        |direct_global| + |direct_class|,
        |q_parallel_global| + |q_parallel_class|).
```

At the allowed condition ceiling, `tau_alg` is approximately `2.22e-7`.
Backend/inverse parity, inverse backward residual, score reconstruction, RMD
cancellation, pair-margin reconstruction, and full-rank ID energy pinning must
each be no larger than `tau_alg(kappa)`. The RMD-cancellation residual alone
uses `rmd_cancellation_scale`, which follows the two large-operand differences
whose equality is being checked. This v11 clarification does not change
`tau_alg`, `score_scale`, `margin_scale`, the estimator, or any score value.
It was justified by algebra and a condition-grid high-precision fixture rather
than historical pass counts. Matrix parity uses Frobenius residual
divided by `max(1,||A||_F,||B||_F)`; inverse backward error uses
`||Sigma P-I||_2/max(1,||Sigma||_2||P||_2)`. Other score and margin checks use
their respective scales above, which remain defined for near ties. ID energy checks use
`max(1,|observed|,|target|)`. Covariance identity
`Sigma_0=Sigma_W+Sigma_B` uses relative Frobenius tolerance `1e-10`.
Projector symmetry and idempotence each use relative Frobenius tolerance
`1e-10`. A fit enters a primary theorem figure only if every required check
passes; failures are never hidden by aggregation.

The frozen `rmd_cancellation_scale` above applies only to the required RMD
cancellation identity. Cached Metric Contract v1.2 parity is a separate
diagnostic comparison. The already-executed v2 implementation used

```text
cached_parity_rmd_scale
  = max(
      1,
      |current_direct_global| + |current_direct_class|,
      |cached_global_distance| + |cached_mahalanobis_score|
    ).
```

This comparison convention is necessary because the cached pipeline does not
preserve the current fit's internal quadratic operands. The record also
preserves the legacy output-scaled parity residual. Cached parity does not
enter `required_numerical_pass`, `primary_channel`, Gate 2, Gate 3, or RtMD
activation. Documenting the convention in v12 is post-execution provenance
clarification, not a new remediation and not a change to any v2 result.

The only common-ridge diagnostic is named
`discriminant_residual_common_ridge_lmax_1e-6` and uses

```text
lambda = 1e-6 * lambda_max(Sigma_W)
A = Sigma_W + lambda I
global = Sigma_0 + lambda I = A + Sigma_B.
```

The same positive `lambda` is applied to conditional and marginal terms. If
`lambda_max(Sigma_W)<=0`, the diagnostic fails. It never replaces or inherits
the name of the primary v1.2 score.

### 13.2 Subspace, alignment, and common-frame rules

Construct `E=[sqrt(pi_c) eta_c]` and define

```text
tau_S = sigma_max(E) * max(d,K) * eps64
dim(S) = count(sigma_i(E) > tau_S).
```

If `sigma_max(E)=0`, set `dim(S)=0`. If any singular value lies in
`[tau_S/10,10*tau_S]`, record a rank-sensitive flag and make no cross-branch
rank-difference claim; branch-internal identities may continue if their
projectors and reconstructions pass.

Compare branches only after ID-only gauge alignment. Equal-rank comparisons
report the full principal-angle vector. Unequal-rank comparisons report the
`min(r_C,r_D)` angles, rank gap, and chordal projector distance. Principal
angles are descriptive and have no pass threshold. For classifier alignment,
remove the common-logit direction from the classifier rows, compare the row
span of `W_centered A^(1/2)` with `S`, and store the angle profile and
`||P_S Q_W||_F^2/rank(Q_W)`; a zero classifier rank is `N/A`.

The zero common frame is fit from the same seed/checkpoint/depth/transform
zero-decay sibling: estimate its `mu_0`, `A`, and `S`, then project the
coupled, decoupled, and zero features into that frame without refitting. It is
a formation diagnostic only; exact score attribution remains branch-internal.

### 13.3 Component, interaction, and primary-channel rules

For every split/dataset/branch/transform, store the biased `1/N` covariance of
`[s_perp,s_parallel_Marginal,s_RMD]`. Require `1^T Cov 1` to reconstruct the
direct `Var(s_MD)` within relative `1e-10`. Raw feature norm versus each raw
component uses Spearman correlation as primary and Pearson as secondary;
zero-variance inputs are `N/A`.

The primary normalization-by-cancellation interaction for the same seed and
OOD dataset is

```text
I_churn = (Churn_MD_raw - Churn_RMD_raw)
          - (Churn_MD_L2 - Churn_RMD_L2).
```

Apply the same difference-in-differences to signed `DeltaAUROC` as secondary.
This does not identify raw and L2 `S-perp` as the same space. The frozen `rho`
grid is `{0,0.25,0.5,0.75,1}` together with every exact interior pair-flip
threshold; no best `rho` is selected.

`S-perp` is the fresh anchor's only prespecified primary empirical-channel
candidate. This choice is explicitly **discovery-informed** by the already
observed historical two-component MD/Marginal/RMD result; the finer
three-component split between `S-perp` and parallel-Marginal was not observed
when this rule was frozen. Historical preflight may test numerical
applicability and measurability but may not switch the candidate. If all
required numerical gates pass, primary=`S-perp`; otherwise primary=`none` and
all three components remain descriptive. Parallel-Marginal or RMD is never
promoted post hoc, even if it moves more in historical or fresh results.

### 13.4 Frozen conditional-method boundary

Section 7.5 freezes the RtMD covariance convention, score, ID-only two-fold
fit, finite domain, BIC-style fallback, narrow Gate-1 novelty verdict,
sequential activation gates, comparison panel, churn estimand, performance and
far-OOD guardrails, replication rule, and close-the-slot rule. Historical Gate
2 is permanently `INCONCLUSIVE`; Gates 3--5 are `NOT_RUN`. Gate 4 requires
Gate 1 plus a prospectively registered fresh Gate 3 containing the transferred
plausibility question, not a retroactive Gate 2 PASS. No exact Gate 3 rule is
frozen in v12. RtMD remains outside the abstract and
active contribution list unless all later gates pass. The paper does not
introduce a general new
“stability-targeted scoring” contribution at this stage; if RtMD later
replicates, it may become a secondary method contribution only.

Task F CPU-side engineering and fixture tests require no GPU approval. They
must create the concrete serialization/schema contract and specification-hash
procedure in their own bounded implementation Issue before a research launch.
An RtMD Gate 3 rule is not a prerequisite for Task F, its fixtures, or main
paired training.

Numeric accuracy/NLL/ECE ID-equivalence margins must be frozen with independent
evidence before fresh research results are inspected. They are the main
study's comparable-ID/Pareto interpretation gate and are not combined with
RtMD Gate 3. No research verdict or numeric equivalence margin applies to an
execution-only pilot. Before full fresh research training, a separate
versioned pre-execution addendum must freeze these margins, the Task F
specification identity, and the alpha interior-compatibility implementation;
it need not contain a Gate 3 rule unless the owner chooses to preregister a
confirmatory RtMD anchor test.

The one-seed shortened `alpha=0.5` pilot is only for execution, resume, and
export verification and is excluded from research evidence. Before that pilot,
the owner must receive the execution SHA and estimated GPU-hours, wall time,
and storage, then give explicit approval. Full fresh GPU training requires a
separate current resource check and explicit owner approval.

Before protected OOD execution, a separate frozen addendum must carry forward
the pre-execution identities and add the remaining protected-evaluation items:

- the already-frozen accuracy/NLL/ECE equivalence and guardrail margins;
- alpha-arm seed pairing, endpoint/lower-half/upper-half uncertainty, and the
  exact interior-compatibility decision rule without adding alpha points;
- practical AUROC/FPR95 margins;
- seed-count/power or minimum-detectable-effect justification for the fixed
  five/three seed allocation;
- standardized divergence references, functional trajectory uncertainty,
  detectability summaries, multiplicity handling, and spectral-band boundaries;
- propagated affine-score bound, churn denominator floor, and ID/OOD residual
  normalization;
- the fresh ID-only `RtMD` activation verdict; if active, its unchanged
  specification hash, separate multiplicity family, and replication rule;
- exact protected OOD evaluation schedule and go/no-go rule.

Protected OOD remains a separate one-shot approval: report its frozen schedule
and expected resources to the owner and obtain explicit approval immediately
before access. Until then, neither the full RtMD detector nor protected-OOD
evaluation may be implemented, connected, or executed.

Null and adverse results remain reportable. Do not rescue them by adding a
detector, selecting a matching checkpoint, changing the primary epoch, or
pooling cells whose effects disagree.

### 13.5 Task F fresh-training pre-execution addendum v2

This subsection is the frozen pre-execution contract for the Task F
execution-only pilot and later main fresh training. It is frozen before any
fresh result is inspected. The pilot remains a separate execution, resume, and
export check; it is not research evidence and is excluded from every research
aggregation.

V2 retains every v1 training, seed, telemetry, ID-equivalence, and alpha
decision. The completed v1 pilot used the frozen CPU export check below. Its
execution-only throughput exposed a production export bottleneck but supplied
no research result and changed no scientific rule. Bounded Issue #89 and PR
#90 added explicit `cuda:<local-index>` feature export while keeping CPU as a
supported default. Export device selection is a runtime control, not a
training semantic: every artifact remains float32, ordered by the same sample
identity, and bound to the same checkpoint, sibling, and specification
contract. Exact CUDA device/runtime identity and the resulting feature bytes
are recorded in runtime and output identity. A GPU artifact must pass its own
shape, finiteness, ordering, provenance, checksum, and CPU-reference numerical
parity validation before production use.

The only authorized pilot proposal is:

```yaml
execution_only_pilot:
  seed: 9000
  max_epochs: 2
  first_leg: run_to_epoch_1_boundary
  resume_leg: resume_the_identical_max_epochs_2_config
  research_evidence: false
  audit_steps: [1, 352, 353, 704]
  export_check:
    checkpoint: epoch_2_last.pt
    depth_tap: penultimate
    dataset_split: id_train
    device: cpu
```

Both legs use the same configuration with `max_epochs=2` from the start.
Running with `max_epochs=1` and changing it to `2` for resume is prohibited.
The first leg may stop only after the durable epoch-1 checkpoint and provenance
boundary using a documented safe procedure for this exact process; an
arbitrary kill is not an allowed substitute. The synthetic sibling quartet in
pilot metadata declares artifact identity only and must not be interpreted as
four executed pilot runs.

The frozen research seed allocation is:

```yaml
research_seeds:
  anchor: [0, 1, 2, 3, 4]
  adam_factorial: [0, 1, 2]
  sgdm: [0, 1, 2]
  reuse_seed_numbers_across_families: true
  historical_seed_number_overlap: allowed_but_not_replication
```

Reusing a seed number across families intentionally controls initialization
and data-stream inputs. The number alone is not evidence that the realized
controls match: every applicable sibling comparison must verify both
`initialization_sha256` and `data_stream_sha256`. Overlap with a historical
seed number is never described as replication.

For the frozen 45,000-member training split and batch size 128,

```text
S = ceil(45000 / 128) = 352.
```

All 50 research runs use the same audit schedule:

```yaml
research_audit_steps:
  [1, 352, 3520, 10560, 21120, 21121,
   42240, 42241, 56320, 56321, 70400]
```

It is determined by

```text
{1}
∪ {S*e : e ∈ [1, 10, 30, 200]}
∪ {S*m, S*m+1 : m ∈ [60, 120, 160]}.
```

Resume must not add, remove, or change an audit step.

The following main-study ID interpretation guardrail is frozen before fresh
results are seen. It does not apply to the execution-only pilot.

```yaml
id_equivalence:
  primary_endpoint: epoch_200_last.pt
  secondary_control: best_val.pt_selected_by_id_validation_only
  accuracy_margin: 0.01
  nll_margin: 0.08
  ece_margin: 0.02
  joint_rule: all_three_required_for_comparable_id
  uncertainty: paired_90_percent_ci_reported
  formal_tost_claim: false
  failure_action:
    keep_all_runs: true
    exclude_or_retrain_runs: false
    interpretation: report_failed_guardrail_and_id_ood_pareto
```

Failure of any one guardrail forbids `comparable-ID` language. It does not
discard a run, stop training, retrain a run, or authorize post-hoc checkpoint
selection. The `0.08` NLL margin is a prospective practical margin grounded in
existing ID-only, same-policy seed dispersion. It is not derived from fresh
results or protected OOD.

The versioned fresh-ID aggregation implementation fixes the decision statistic
as the absolute value of the seed-paired mean difference for each of accuracy,
NLL, and ECE. Each metric passes only when that statistic is at or below its
frozen practical margin, and all three must pass jointly. The two-sided paired
90% t-interval is reported descriptively as uncertainty but is not part of the
PASS rule and is not a TOST claim. Until the protected `id_test` endpoint is
separately authorized and evaluated, the implementation must report
`PENDING_PROTECTED_ID_TEST`; `id_validation` cannot substitute for it.

Alpha confirmation uses exactly the three existing points and this
classification:

```yaml
alpha_interior_compatibility:
  rule: >
    seed-mean Y(0.5) is interior-compatible when it lies in the closed interval
    [min(Y(0),Y(1)), max(Y(0),Y(1))]
  endpoint_tie_rule: >
    when |Y(1)-Y(0)| <= 1e-12 * max(1,|Y(0)|,|Y(1)|), classify the outcome as
    undefined_degenerate_endpoints
  role: reported_alpha_confirmation_classification_not_a_gate
```

This is a reported confirmation classification, not a training-launch gate or
primary-result gate. No alpha point may be added.

The Task F serialization specification SHA-256, re-derived from the current
generator and validator before this addendum was frozen, is:

```text
0ac3101e6d6aaed1a5a0d4891792d4700540bac5247cca8f7d6e67d664ffe9ba
```

This specification identity is distinct from runtime and output identity.
Runtime/output records separately bind the exact execution Git SHA, host and
device, environment, data root, run and output paths, checkpoints, exports,
and their artifact hashes. Neither kind of identity substitutes for the other.

Any change to the specification identity, pilot seed or pilot configuration,
research seed lists, or telemetry audit schedule invalidates the existing
approval and requires a new pre-execution approval. A concrete pilot approval
also binds its reported execution SHA and runtime/output proposal.

Pilot approval and full-training approval are separate. Approval of the one
`alpha=0.5`, seed-9000, two-epoch execution-only run never authorizes any of
the 50 research runs. Task F pilot and main ID training do not require
protected-OOD access, an RtMD Gate 3 rule, or RtMD activation. Protected OOD
remains prohibited until its own later contract and approval.

### 13.6 Task F pre-protected-OOD addendum v1

This subsection is frozen before inspecting the fresh Task F ID-only geometry,
paired aggregates, residual-tail values, protected `id_test`, or OOD results.
It carries forward the source training SHA
`9eb3c1fa56d880ea5220badac7bc71ba75786d22`, the Task F specification
SHA-256
`0ac3101e6d6aaed1a5a0d4891792d4700540bac5247cca8f7d6e67d664ffe9ba`,
the seed allocation, and the ID-equivalence margins in Section 13.5 without
change. A checksummed local-compute relay may aggregate the three host summaries
before the large source upload finishes. That relay is transport and ID-only
compute evidence, not the final remote research terminal. It does not authorize
protected access.

The protected population is the frozen CIFAR-10 `id_test` set of 10,000
members plus the existing OpenOOD-compatible near sets `cifar100` and `tin`
and far sets `mnist`, `svhn`, `texture`, and `places365`, in that order. The
9,000-member compatibility-only `id_test_openood` and the 1,000-member
`ood_validation_tin` remain excluded. No protected sample is used for fitting,
temperature selection, checkpoint selection, detector selection, fallback, or
early stopping.

The one-shot main protected export schedule is:

```yaml
main_protected_schedule:
  splits: [id_test, cifar100, tin, mnist, svhn, texture, places365]
  all_50_runs_penultimate_last_epochs: [10, 60, 120, 160, 200]
  primary_anchor_20_runs_epoch_200_extra_depths: [stage1, stage2, stage3]
  all_50_runs_penultimate_best_val: true
  checkpoint_depth_contexts_per_split: 360
  logical_feature_records: 2520
  primary_endpoint: epoch_200_last_penultimate
  best_val_role: secondary_id_validation_selected_control
```

The epoch-200 penultimate Raw-MD coupled--decoupled contrast in the five-seed
Adam anchor remains primary. Near and far summaries are arithmetic means of
the per-dataset seed-level metrics after each dataset is reported separately.
The four primary confirmatory hypotheses are `DeltaAUROC` and
`PairOrderChurn`, each for the near and far macro summaries. Two-sided paired
seed tests use familywise alpha `0.10` with Holm adjustment across these four
hypotheses. Paired mean, sample SD, and paired 90% intervals are always shown.
Dataset-specific, other-cell, SGDM, depth, L2, RMD, component, affine, and
`best_val` results are prespecified secondary or descriptive outputs and are
not expanded into a grid of unadjusted claims.

The five-epoch anchor trajectory uses one simultaneous 90% max-absolute-t
band over `[10,60,120,160,200]`, obtained from all exact paired-seed sign
flips. Its functional summary is the normalized trapezoidal area over epoch
10--200, and its early slope is the signed change from epoch 10 to 60 divided
by 50. A first practical-threshold crossing is called detectability time only.
For a two-sided `alpha=0.10`, power `0.80` paired-t design, the approximate
minimum detectable standardized paired effects are `1.36` for five seeds and
`2.30` for three seeds. Smaller uncertainty intervals may still be reported,
but absence of significance at this seed count is not evidence of no effect.

The prospective practical interpretation margins are `0.01` absolute AUROC
and `0.03` absolute FPR95. They are effect-size labels and RtMD
non-inferiority guardrails, not run-exclusion, checkpoint-selection, or
training gates. Higher AUROC and lower FPR95 are better. The primary mechanism
result remains reportable when a practical margin is not reached.

`R_churn` is undefined when its same-policy denominator is below
`max(1e-4, 10/(N_ID*N_OOD))`; the numerator and denominator are still reported.
The spectral allocation diagnostic sorts branch-internal covariance
eigenvalues in descending order and uses cumulative trace bands
`[0%,50%]`, `(50%,90%]`, and `(90%,100%]`. An equal-eigenvalue tie is never
split across a boundary; an empty band is `NOT_APPLICABLE`. Individual
eigenvectors are not promoted after inspection.

The ID-train affine map is evaluated with the same-image residual
`e=z_C-(A z_D+b)`. The normalized residual divides `||e||_2` by
`max(||z_C-mu_C||_2, ||A(z_D-mu_D)||_2, 1e-12)`. Each OOD dataset reports the
raw and normalized distribution plus its median excess over protected
`id_test`. The propagated quadratic-score check uses the stored fitted
precision and the exact first-order-plus-quadratic perturbation bound; a score
gap beyond that certified bound is an estimator/implementation failure, not a
scientific effect.

The fresh ID-only RtMD Gate 3 is exactly one test on the primary anchor's
coupled and decoupled roles, five paired seeds, epoch-200 `last.pt`,
penultimate raw features, and held-out `id_validation`. For each branch, the
tail statistic is
`log(Q_0.99(q_perp,id_validation)/chi2_ppf(0.99,k))`. The deterministic
class-stratified two-fold ID-train fit, finite-`nu` domain, Gaussian candidate,
and BIC-style selection in Section 7.5 are unchanged. Gate 3 passes only when:

```text
RtMD Gate 3 specification SHA-256:
30e7f212c6e91b84885a7d06568820caa15c48fdcbe924af28818d07c428d270
```

1. all ten role-by-seed fits are present and numerically applicable;
2. at least four of five fits in each role select a finite t tail;
3. the paired 90% t interval for coupled minus decoupled excludes zero;
4. at least four of five seed effects have the mean-effect sign; and
5. the absolute paired mean is at least the maximum of `0.10`, `1e-6`, and
   the pooled within-role median absolute between-seed tail-statistic
   difference.

This single prospectively declared Gate 3 contrast has no multiplicity search.
Missingness, numerical inapplicability, fallback frequency, weak effect,
inconsistent sign, or failure of any item closes the optional RtMD slot for
this anchor. It does not change, delay, or invalidate the main protected OOD
study, and no replacement estimator or detector may be introduced.

If and only if Gate 3 passes, the unchanged Section 7.5 RtMD score is added to
the one-shot protected run for the ten coupled/decoupled anchor models at the
epoch-200 last penultimate endpoint. Raw is primary and L2 is secondary; there
is no RtMD checkpoint, depth, `nu`, dataset, or transform search. RtMD's
separate secondary family requires all of the following at the seed level:

- mean `PairOrderChurn(RtMD)-PairOrderChurn(RawMD) <= -0.02` and its paired
  90% interval has upper bound below zero;
- AUROC is no worse than Raw MD by more than `0.01` and FPR95 is no worse by
  more than `0.03` for both near and far macro summaries; and
- relative to RMD on the far macro summary, AUROC is no worse by more than
  `0.01` and FPR95 is no worse by more than `0.03`.

These RtMD outcomes are evaluated jointly as guardrails, with dataset-level
values visible. Failure closes the method-contribution claim; it cannot rescue
or reverse the main mechanism conclusion. Passing on WRN-28-10/CIFAR-10 permits
only an anchor-local secondary finding. A method contribution still requires
the separately approved replication in Section 10.

Protected execution may be proposed only after the exact 50-run ID-only
inventory, 1,320 bridge records, 660 geometry units, alignment coverage,
paired aggregate, and Gate 3 verdict are terminal and checksummed. Completion
of the large HF source upload is not a scientific prerequisite when these
local source and compute identities are independently verified, but the final
remote terminal remains pending until that upload verifies. Immediately before
protected execution, the owner must receive the execution SHA, exact split and
context counts, target GPUs, time/storage estimate, Gate 3 verdict, and RtMD
included/excluded status and must give a new explicit one-shot approval.

## 14. Post-result classifier-insensitive geometry fast kill gate v1

This section is a prospective post-result analysis addendum layered on the
completed v12 experiment. It does not alter the v12 training, protected-OOD
schedule, stored estimands, or completed results. Its sole purpose is to decide
whether the paper should test a new candidate mechanism or retain the current
Raw-MD pair-instability framing. Status before execution is `NOT_RUN`.

### 14.1 Candidate Main and fallback

The candidate question is:

> Does decay coupling produce held-out, sample-dependent representation
> deformation that is concentrated per dimension in directions to which the
> trained classifier is insensitive?

If the gate passes, the candidate paper identity becomes training-rule control
of classifier-insensitive geometry and its downstream feature-OOD readout.
Passing does not establish an OOD mechanism and does not authorize protected
evaluation. If the gate fails or is mixed, the paper retains the completed
controlled Raw-MD pair-ranking-instability result as Plan B. No threshold,
subspace, normalization, or alternative carrier may be selected after seeing
this gate.

### 14.2 Frozen inputs and scope

The analysis reads only existing checksum-verified Task F artifacts:

- raw penultimate `id_validation` features for the coupled and decoupled roles;
- the coupled classifier weight and the two roles' stored validation logits;
- the stored raw affine matrix and bias from the existing
  `coupled_minus_decoupled` alignment artifact; and
- identity and checksum manifests.

The population is the 14 Adam C--D endpoint pairs at epoch-200 `last`,
penultimate: five anchor seeds and three seeds in each of the other three
LR--WD cells. The saved alignment direction maps decoupled features into the
coupled frame. The analysis does not load a checkpoint, reprocess `id_train`,
refit an affine map or detector, access protected ID/OOD data, or inspect another
epoch/depth. Large feature arrays remain on their source hosts.

### 14.3 Frozen statistic

Let `T_D_to_C(z)=z A+b` be the already-fitted ID-train affine map and define on
held-out `id_validation`

```text
R = T_D_to_C(Z_D) - Z_C.
```

Remove the common-logit direction from the coupled classifier rows,
`Wc = W_C - row_mean(W_C)`, and obtain its exact numerical row-space basis `Q`
using the existing float64 SVD tolerance
`sigma_max * max(Wc.shape) * eps64`. Let `r=rank(Q)` and `d` be the feature
dimension. The dimension-normalized residual energies and primary ratio are

```text
e_parallel = ||R Q||_F^2 / (n r)
e_perp     = (||R||_F^2 - ||R Q||_F^2) / (n (d-r))
rho        = e_perp / e_parallel.
```

`rho > 1` is the prespecified direction supporting classifier-insensitive
concentration. `rho <= 1` does not support the required carrier. Prediction
disagreement, predictive-distribution Jensen--Shannon divergence, absolute
top-two logit-margin difference, and the existing Accuracy/NLL/ECE statuses are
supporting outputs without new equivalence thresholds.

### 14.4 Numerical and compute policy

Every context must pass source checksum and C--D sibling identity, have
`1 <= r < d`, finite positive energies, and reconstruct total residual energy
within relative `1e-5`. GPU float32 is the production path with TF32 disabled.
Anchor seed 0 is compared with CPU float64 and must have relative `rho`
difference at most `1e-4`. A context with `abs(log(rho)) < 0.05`, or a failed
float32 reconstruction check, is rerun once on CPU float64. A persistent failure
is `NOT_APPLICABLE`; it is not repaired by changing precision or the statistic.

Each source host may use every currently idle GPU with at least 2 GiB free and
no active compute process, one worker per GPU. Existing processes are never
stopped or preempted. With no eligible GPU, use the existing host CPU-worker
limits `curie/lise/precision_medicine = 4/2/4`. Only small canonical JSON host
summaries are collected centrally.

### 14.5 Decision rule

The gate is `GO` only when all conditions hold:

1. all 14 contexts pass identity, checksum, and numerical validation;
2. at least four of five anchor seeds have `rho > 1`;
3. at least two of three seeds in the all-ID-guardrail-PASS
   `LR=3e-4, WD=1e-4` cell have `rho > 1`;
4. at least one of the two remaining high-WD cells has at least two of three
   seeds with `rho > 1`; and
5. every required CPU confirmation preserves the `rho > 1` decision side.

All other complete outcomes are `FAIL`; missing or technically inapplicable
coverage is `BLOCKED`, not scientific evidence. For paper routing, a mixed
pattern is treated as `FAIL`. A `GO` permits only a separately frozen
bidirectional affine confirmation and anchor formation analysis. Any projected
OOD readout still requires a new pre-protected addendum and explicit owner
approval.

## 15. Post-result Plan B freeze and ResNet-18 replication v1

This section is a post-result addendum. It does not alter Sections 1--14, the
completed Task F training, its protected evaluation, or any frozen estimand.
Section 14 has now completed: all 14 contexts passed source and numerical
validation, but zero of 14 had `rho > 1`. Its candidate route is therefore
`FAIL` and is closed without reverse-alignment, subspace, normalization,
trajectory, or projected-OOD rescue.

### 15.1 Frozen paper identity

The working title is:

> **Training-Rule-Induced Pair-Ranking Multiplicity in Mahalanobis OOD
> Detection**

The paper question is:

> Holding architecture, data, objective, initialization, and minibatch order
> fixed, how much can changing only the decay-coupling rule reorganize the
> same ID--OOD pair ordering of a protocol-fixed, branch-refitted Mahalanobis
> detector; in which score component, epoch, and depth does that sensitivity
> appear?

`Protocol-fixed, branch-refitted` is required terminology: each branch uses
the same ID-only fitting procedure but refits its own means and covariance.
`Fixed detector parameters`, `same ID performance`, `Marginal causal
mediator`, and a universal claim that coupling is harmful are prohibited.

The completed WRN-28-10/CIFAR-10 paper body is fixed to five results:

1. controlled Raw-MD C--D pair-ranking non-invariance;
2. Gain/Loss cancellation and Churn hidden by net AUROC;
3. predominantly Marginal score localization, with RMD and L2 as prior-work
   probes rather than new methods;
4. early-detectable, later-amplified and deep-layer formation; and
5. local LR/WD dependence, SGDM reversal, ID Pareto status, and failed
   mechanism gates as explicit boundaries.

### 15.2 Prospective ResNet-18/CIFAR-10 matrix

Architecture replication precedes any broader WRN LR/WD grid. The research
matrix is exactly:

```yaml
study: resnet18_cifar10_replication_v1
model:
  name: resnet18
  variant: cifar
  feature_dim: 512
dataset: cifar10
epochs: 200
endpoint: last.pt_epoch_200_penultimate
learning_rates: [1.0e-3, 3.0e-4]
weight_decay: 1.0e-4
roles: [adam_coupled, adamw_decoupled]
seeds: [0, 1, 2, 3, 4]
research_runs: 20
```

For each seed, all four LR-by-role arms share one initialization and one
minibatch stream. Each same-LR C--D pair has a sibling identity, and the four
arms also store a `cross_lr_pairing_block_id`. This identifies the ResNet
fixed-WD LR-by-coupling comparison; it does not retroactively identify a WRN
LR effect. Zero, midpoint, SGDM, depth taps, epoch trajectories, and new
mechanism metrics are excluded.

Before research execution, one seed-9000, two-epoch, four-arm
execution-only pilot must validate:

- 20-run planner coverage and four-arm identity construction;
- logits `[B,10]`, penultimate features `[B,512]`, and classifier weight
  `[10,512]`;
- seconds per epoch, peak GPU memory, checkpoint size, and projected storage;
  and
- deterministic restart and artifact/provenance identity.

The pilot is not research evidence. Neither the pilot nor this addendum
authorizes main GPU training. Main training requires a separate owner approval
after the measured execution packet. Protected evaluation requires another
approval after all 20 checkpoints and ID-only guardrails are complete.

### 15.3 Endpoint evaluation and prospective gate

The protected population and Near/Far dataset roles remain exactly those in
Section 13.6. The new study runs one endpoint-only, penultimate,
branch-refitted Raw MD/RMD/Marginal/L2 evaluation. Statistical units are the
five paired seeds; dataset observations and ID--OOD pairs are not independent
replicates. Report seed means, sample SD, paired 90% intervals, and every
dataset before Near/Far macro averaging.

The two LR settings are named by their WRN roles:

- `large-context`: `LR=1e-3, WD=1e-4`;
- `small-context`: `LR=3e-4, WD=1e-4`.

The replication status is `FULL` only if all conditions hold:

1. all 20 run identities, checksums, and numerical checks pass;
2. in the large-context, Near and Far Raw-MD mean C--D are negative and at
   least four of five seeds have that sign in each region;
3. large-context Raw-MD Churn is at least `0.10` in both regions;
4. the absolute small-context Raw-MD C--D mean is smaller than the
   large-context mean in both regions;
5. in the large-context, `abs(Delta RMD) < abs(Delta Raw MD)`, Marginal
   accounting has the Raw-MD effect sign, and its absolute contribution is
   greater than 50% of `abs(Delta Raw MD)` in both regions; and
6. the small-context Accuracy, NLL, and ECE equivalence guardrails all pass.

If items 2--3 pass but any of items 4--6 fails, the status is `PARTIAL`. If
item 2 or 3 fails, the status is `FAIL`. A technical coverage failure is
`BLOCKED`, not scientific evidence.

- `FULL` permits an architecture-replicated Plan B claim.
- `PARTIAL` is reported as an architecture boundary and stops ICLR main-paper
  promotion; it does not trigger a rescue grid or new mechanism.
- `FAIL` closes the current ICLR direction without a replacement hypothesis.

No WRN common-stream LR factorial or CIFAR-100 extension runs before this
gate. After `FULL`, the default is to stop new training and finish the paper;
those extensions remain post-submission work unless separately frozen and
approved.
