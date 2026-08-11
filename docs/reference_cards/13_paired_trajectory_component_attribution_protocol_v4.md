# Reference Card 13: Paired-Trajectory Component-Attribution Protocol v4

## 0. Authority, status, and supersession

Protocol identifier:

```text
fixed_readout_paired_trajectory_component_attribution_v4
```

This card is the active paper and experiment authority after the 2026-08-11
direction decision. It supersedes Card 13 v3's shared-prefix main experiment
without changing completed evidence:

- Metric Contract v1.2 and its 30-model WRN-28-10/CIFAR-10 population remain
  valid descriptive discovery evidence;
- the Research Contract v2 radial Stage-2 result remains immutable `FAILED`;
- the v3 historical MD--Marginal--RMD analysis remains `PASS` and discovery
  only;
- the validated `fork_from_prefix` runtime remains available, but it is an
  optional follow-up tool rather than the main paper intervention;
- no fresh v4 trajectory, protected-OOD confirmation, or replication has run.

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
2. exact tie-aware accounting of changed ID--OOD pair orderings;
3. a controlled coupled/decoupled/zero-decay comparison within optimizer
   families, with LR and nominal-decay sensitivity;
4. architecture, dataset/class-count, connectivity, and modern-scale
   replication.

The strongest allowed sentence is conditional:

> A controlled change in decay coupling alters fixed-readout ID--OOD
> orderings, and the effect is selectively concentrated in a
> formula-identified Mahalanobis score component.

This sentence is unavailable unless the practical OOD effect, component
concentration, channel-matched attenuation, and fresh replication gates pass.
If ID equivalence fails, the result is retained and interpreted as a Pareto
trade-off rather than discarded or post-hoc checkpoint-matched.

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

The two-component AUROC accounting uses four computational hybrids: both
components from the decoupled run, both from the coupled run, and the two
replacement orders. Symmetric Shapley contributions average the two orders.
The hybrids are calculation devices, not trained detectors. Exactness holds
inside this declared two-player replacement game; it is not claimed to be a
unique physical or causal mediation decomposition.

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
- Adam first- and second-moment summaries where applicable.

At selected checkpoints, an optional one-step audit may copy one model state
and one minibatch gradient, calculate the coupled and decoupled candidate
updates, and apply neither. This is a manipulation check showing the immediate
algorithmic difference, not a separate long-running branch experiment.

### 6.2 Level B: representation geometry

For recipe `r`, checkpoint `t`, depth `l`, and fixed image `x`, extract
`z_{r,t,l}(x)`. Fit and report these prespecified channels:

| Channel | Measurements |
| --- | --- |
| Radial | feature-norm distribution and same-image norm change |
| Class geometry | class-mean distances/angles and CDNV |
| Global geometry | global mean and covariance trace/spectrum/effective rank |
| Within-class geometry | pooled covariance spectrum and condition number |
| Allocation | sample displacement energy in covariance spectral bands |
| Class-distance profile | nearest-class distance versus the full class profile |

The basic paired contrasts within every seed and cell are coupled--decoupled,
coupled--zero, and decoupled--zero. This separates the presence of decay from
the way decay is coupled.

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
incorrect-to-correct transitions, not only aggregate AUROC correlation.

For metric `G`, define the paired trajectory contrast
`Delta G(t,l)=G_C(t,l)-G_D(t,l)`. Its onset is the first of two consecutive
prespecified checkpoints at which the paired effect exceeds a frozen
seed-noise/practical threshold. Thresholds and multiplicity rules must be
fixed in a pre-protected-OOD addendum; they are not estimated after observing
the protected OOD trajectory.

A coherent temporal chain would look like:

```text
update-path divergence
-> representation-channel divergence
-> additive/quadratic score divergence
-> changed ID--OOD pair ordering
```

Temporal order plus exact reconstruction and matched attenuation is strong
mechanism evidence, but it is not described as complete causal mediation.

### 6.5 Channel-matched confirmation

After localization, weaken only the implicated channel:

| Localized channel | Confirmatory diagnostic |
| --- | --- |
| radial feature norm | L2 normalization and refitting |
| global/Marginal component | RMD subtraction |
| spectrum/stretch | ID-only spectral-band ablation, eigenvalue clipping, or whitening diagnostic |
| class-distance profile | nearest-class versus full-profile readout |

The confirmation must selectively attenuate the branch gap without being
chosen after OOD results. A new detector is not added to rescue a failed
mechanism claim.

## 7. Supporting theory and its limits

The paper may prove or derive:

- the class-independent quadratic term plus class-dependent affine envelope
  of the shared-covariance Mahalanobis score;
- the exact MD--Marginal--RMD score and pair-margin identities;
- the exact symmetric size--stretch branch-change identity;
- under a stated ridge/full-rank convention,
  `Sigma_0 = Sigma_W + Sigma_B`, `rank(Sigma_B) <= K-1`, and the associated
  Woodbury low-rank precision correction.

It does not prove:

- that training changes Marginal more than RMD;
- the sign or magnitude of the OOD AUROC effect;
- that a single radial or spectral channel always dominates;
- a formula-predicted universal detector-fragility ranking;
- that Woodbury supplies a new covariance estimator or a separate main
  novelty.

Pseudoinverse, ridge, covariance-centering, and numerical-tolerance conditions
must accompany every applicable statement. Do not claim exact cancellation
in a named number of feature dimensions without those conditions.

Because normalized networks can turn weight-norm changes into effective-step
changes, the path from decay policy to representation may be direct or may be
mediated by effective learning rate. This is an alternative mechanism to
measure and discuss, not a pre-established conclusion.

## 8. Detector roles

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

## 9. Replication and scale order

1. ResNet-18/CIFAR-10: architecture-only replication.
2. ResNet-18/CIFAR-100: dataset and class-count replication.
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

## 10. Historical discovery and optional shared-prefix follow-up

The completed 30-model v3 analysis uses 30 frozen `last.pt` bundles and six
OOD datasets. It is valid for locating descriptive MD/Marginal/RMD variation,
but optimizer, LR, and WD are mixed in that population. It cannot establish
that decay coupling caused the gap. Nearest-accuracy matching is excluded from
primary analysis and, if shown, is labeled selection-biased sensitivity.

The existing `fork_from_prefix` implementation may be used only as an
optional follow-up if the from-scratch trajectory identifies a narrow onset
window that a local intervention could test efficiently. It is not required
for the main contribution and does not replace from-scratch evidence.
Ordinary resume remains strict and distinct from this optional fork operation.

## 11. Required artifacts and validation

Every v4 result must identify seed, initialization, config, branch policy,
checkpoint epoch, depth tap, probe-image membership, Gaussian-fit population,
and source code/config hashes. Required records include:

- ID accuracy/NLL/ECE and `last` versus `best_val` identity;
- per-sample MD, Marginal, and RMD raw/L2 scores;
- score and pair-margin reconstruction residuals;
- tie-aware pair-transition counts and component Shapley accounting;
- quadratic size/stretch and symmetric branch contributions;
- spectrum/allocation spectral-band summaries;
- update-dynamics and geometry trajectories;
- numerical condition, ridge, failure, and leakage status.

Required checks include:

- exact identity/reconstruction tests with scale-aware tolerances;
- tie-aware AUROC parity that does not count a broken numerical tie as a
  scientific rank error;
- ridge/full-rank-only Woodbury rank tests;
- same-initialization, RNG, DataLoader, and first-minibatch identity;
- zero-decay Adam/AdamW and SGDM/SGDW semantic parity where applicable;
- deterministic probe membership and no protected-result leakage;
- ImageNet-200 class-membership and pretrained-leakage checks before scale
  execution.

The current repository has part of this infrastructure, but v4 training,
multi-depth extraction, expanded trajectory snapshots, update logging, and
scale regimes are `NOT_RUN`/not yet implemented until separately authorized.

## 12. Pre-execution addendum and stopping rules

Before protected OOD execution, freeze in a versioned addendum:

- exact accuracy/NLL/ECE equivalence and guardrail margins;
- practical AUROC/FPR95 margins;
- prefix-count/power or minimum-detectable-effect justification for the fixed
  five/three seed allocation;
- onset noise bands, multiplicity handling, and spectral-band boundaries;
- exact protected OOD evaluation schedule and go/no-go rule.

Null and adverse results remain reportable. Do not rescue them by adding a
detector, selecting a matching checkpoint, changing the primary epoch, or
pooling cells whose effects disagree.
