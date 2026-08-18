# Optimizer-side formation of Mahalanobis readout sensitivity

This is the human-readable post-result manuscript outline. Exact estimands and
execution rules remain governed by
[`../reference_cards/13_active_paper_protocol.md`](../reference_cards/13_active_paper_protocol.md).
Numerical evidence and claim boundaries live in
[`task_f_result_analysis.md`](task_f_result_analysis.md).

## One-sentence paper

Under identical initialization and data streams, controlled optimizer-side
decay choices can produce different representation geometry and reverse the
ordering of the same ID--OOD pairs under a fixed Raw Mahalanobis readout; in
the completed Task F case study, the adverse score movement is predominantly
Marginal, is detectable early and amplified later/deeper, and is not a general
loss of class-relative OOD information.

## Scope: broad question, bounded answer

The research-program question is:

> Which optimizer-side training choices create representations that are
> sensitive or incompatible with a fixed Raw Mahalanobis readout, and how does
> the relevant geometry form during training?

The current paper does not yet answer that question over optimizer recipes in
general. Its completed controlled answer is:

```text
same initialization and data stream
-> zero / decoupled / coupled sibling trajectories
-> exact pair Gain/Loss/Churn
-> Raw-MD versus RMD/L2 readout sensitivity
-> Marginal score localization
-> time/depth and covariance-geometry concordance
```

`Mahalanobis-hostile` is not used as an intrinsic model label. If used at all,
it means poor absolute AUROC/FPR95 and adverse paired ordering on the frozen
benchmark under a specified fit/readout.

## Abstract logic

1. Prior work establishes that Mahalanobis behavior depends on representation
   geometry and that RMD and L2 normalization can improve the readout.
2. The less controlled question is where detector-sensitive geometry comes
   from during training under fixed architecture, data, and objective.
3. Task F changes decay policy among same-init, same-stream sibling arms and
   observes exact same-image pair transitions, not only aggregate AUROC.
4. Raw-MD C-D effects repeat with the same sign in four Adam LR/WD contexts but
   differ greatly in magnitude. Same-LR WD contrasts show a genuine local
   `WD x coupling` interaction; cross-LR differences remain descriptive.
5. RMD retains high absolute performance more consistently, while L2-MD gives
   strong absolute recovery but heterogeneous signed-gap attenuation.
6. Exact `MD = RMD + Marginal` replacement accounting localizes adverse movement
   predominantly to Marginal. Primary OOD-side dominance is strong but not
   universal across contexts.
7. The effect is early detectable, later amplified, and concentrated in deep
   representation stages. Covariance/norm geometry is broadly concordant with
   effect magnitude but is not a monotone or unique mediator.
8. Primary component-theorem applicability fails, so neither `S_perp` nor a
   coupled spectral band receives causal or unique attribution.

## 1. Introduction

Start from the training-side origin gap, not from a detector leaderboard.

```text
classifier training choice
-> learned representation
-> fixed post-hoc Gaussian readout
-> exact pair ordering
```

Prior representation-side work explains why a completed representation can be
friendly or unfriendly to Mahalanobis scoring. Broad training-induced OOD work
shows that training strategies alter detector rankings. The missing controlled
unit is the formation of readout-sensitive geometry under one optimizer-side
intervention with architecture/data/objective held fixed.

Primary question for the completed paper:

> Can one local optimizer-side rule change alter the geometry read by Raw MD
> and reorganize the same ID--OOD pairs, and which stored score channel carries
> that change?

Secondary question:

> Does the controlled effect vary across local LR/WD contexts, and what does
> that imply for a future systematic optimizer-recipe study?

Do not open with “same ID accuracy, different OOD.” Primary NLL fails the
comparable-ID guardrail in the improvement direction. Present it as a
multidimensional ID/OOD Pareto result and use the low-LR/low-WD all-PASS cell
as a controlled comparable-ID boundary.

## 2. Related work and novelty boundary

Explicitly credit prior work for:

- the original Mahalanobis detector and representation-dependent failure;
- `MD = RMD + Marginal` and RMD as a near-OOD correction;
- L2 feature normalization and Mahalanobis++;
- Fisher/LDA discriminant subspaces and the `K-1` bound;
- size--stretch, eigen-direction, spectrum--allocation, LID, and spectral
  ablation analyses;
- optimizer/weight-decay effects on representation and Neural Collapse;
- broad training-strategy effects on OOD detector rankings.

Do not claim novelty for RMD, L2-MD, Marginal, spectrum-allocation, a generic
pair churn concept, or the fact that optimizer choice changes geometry.

The paper-specific combination is:

1. same-init/common-stream zero/decoupled/coupled intervention;
2. exact ID--OOD Gain/Loss/Churn under that intervention;
3. controlled local WD contrasts and `WD x coupling` DiD;
4. paired score replacement and ID/OOD-side accounting;
5. time/depth formation and update telemetry in the same experimental system;
6. explicit numerical/applicability boundaries that prevent invalid
   `S_perp` or spectral attribution.

Position *A Geometry-Based View* as the detector-side explanatory lens. This
paper asks how optimizer-side choices move a representation into or out of a
known readout-sensitive geometry regime.

## 3. Experimental intervention

Zero, AdamW, midpoint, and Adam are parallel sibling arms from a common
initialization and data stream. Never draw `Zero -> AdamW -> Adam` as a
sequential causal path.

- `C-D`: coupling policy at fixed nominal WD;
- `D-Z`: adding decoupled WD;
- `C-Z`: adding coupled decay dynamics;
- same-LR `WD1e-3 - WD1e-4`: controlled branch-specific WD contrast;
- `[(C-D)_highWD - (C-D)_lowWD]`: controlled local `WD x coupling` DiD.

High-LR and low-LR sibling groups do not share the same data-stream identity.
Cross-LR differences are context descriptions, not LR causal effects. The 2 x
2 set is a local factorial, not a phase map or dose-response curve.

## 4. Primary phenomenon: exact pair-ranking multiplicity

Lead with absolute Raw-MD performance and exact pair accounting.

Primary C-D:

| Region | AdamW | Adam | DeltaAUROC | Gain | Loss | Churn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Near | 0.5922 | 0.4178 | -0.1743 | 0.0871 | 0.2615 | 0.3486 |
| Far | 0.6709 | 0.3874 | -0.2835 | 0.0514 | 0.3349 | 0.3863 |

The four Adam cells all have negative C-D Raw-MD signs. The low-LR/low-WD
cell is a small-effect all-ID-guardrail-PASS control. SGDM has the opposite
sign and refutes a universal “coupling is bad” claim.

Zero--AdamW has near-zero net AUROC but `0.14-0.19` churn. Use it to show that
aggregate cancellation is not behavioral equivalence. Flip burden is broadly
distributed across samples; it is not carried by a tiny fixed set of pairs.

## 5. Local training-context interaction

Show the four cells without averaging over LR/WD. Then show the controlled
same-LR WD contrast.

- At low LR, higher WD changes coupled Raw MD by `-0.0962/-0.0779` Near/Far,
  while decoupled changes by `-0.0050/+0.0018`.
- The low-LR `WD x coupling` DiD is `-0.0912/-0.0797`.
- At high LR, DiD is `+0.0004/+0.0618`; the same direction does not repeat.

Conclusion: coupling sensitivity depends on the local optimization context.
Do not infer a monotone WD law or a causal LR main effect.

## 6. Readout localization

Use RMD and L2 as prior-work probes, not new methods.

Primary absolute performance:

- Raw RMD C/D: `0.8806/0.8908` Near and `0.9006/0.9111` Far;
- L2-MD C/D: `0.8466/0.8716` Near and `0.9435/0.9334` Far.

RMD attenuation repeats more consistently across the four Adam cells. L2
strongly recovers absolute performance, but local signed-gap attenuation is
heterogeneous and cannot be stated as universal.

For ID-like scores:

```text
s_MD(z) = s_RMD(z) + s_Marginal(z)
m_MD(i,o) = m_RMD(i,o) + m_Marginal(i,o)
```

Use symmetric two-player replacement accounting for RMD/Marginal and ID/OOD
sides. It is score accounting, not causal mediation.

- Primary dataset-level Marginal share: `69%-94%` using
  `abs(mean phi_Marginal) / abs(mean Delta)`.
- Marginal remains the adverse component in all four Adam cells.
- Share above one means RMD offsets some Marginal deterioration.
- OOD-side negative motion dominates primary and high-WD contexts, but not the
  low-LR/low-WD Near result.

The valid conclusion is predominantly Marginal score localization. The stronger
claim of universal OOD-side mediation is rejected.

## 7. Formation over time and depth

Use `early detectable, later amplified`, not `late formation`.

Primary Near Raw-MD Delta:

```text
epoch 10   -0.0739
epoch 60   -0.1072
epoch 120  -0.1771
epoch 160  -0.1781
epoch 200  -0.1743
```

Depth Near Delta:

```text
stage1       +0.0032
stage2       -0.0313
stage3       -0.1540
penultimate  -0.1743
```

Connect these trajectories to Marginal share, ID/OOD-side motion, fixed
geometry metrics, and telemetry on the same axes. Telemetry is a passive
witness at each branch's actual state, not a same-state causal update
decomposition.

## 8. Geometry: supporting concordance, not rediscovery

Use a fixed panel:

- feature norm mean/CV and class-wise norm;
- within/global numerical and effective rank;
- retained condition number;
- trace and top-trace concentration;
- affine held-out-ID residual;
- principal-angle profile/chordal distance;
- applicability status.

The small-effect low-LR/low-WD cell has the smallest geometry differences. The
three larger-effect contexts show larger concentration/condition differences.
However, the two high-LR WD levels are not monotone in Raw-MD gap despite a
large geometry change. Call this `concordant but not monotone`.

Exact spectral allocation is a prior-work-informed diagnostic. Decoupled
primary fits reconstruct Marginal, but coupled raw fits are `NOT_APPLICABLE`.
Therefore cross-branch spectral-band attribution is `NOT AVAILABLE` and belongs
in the limitation/supporting appendix, not the headline mechanism.

## 9. Conditional supporting theory

Under a numerically full-rank inverse, or one common positive ridge applied to
all conditional and marginal terms, within-class whitening and the class-mean
span `S` yield:

```text
s_Marginal = s_perp + s_parallel_Marginal
s_MD       = s_perp + s_parallel_Marginal + s_RMD
s_RMD(x)   = s_RMD(P_S x)
```

This identity explains why RMD can cancel a class-orthogonal term. It does not
predict that coupling changes `S_perp`, which direction OOD scores move, or
which detector wins.

Primary raw/L2 component fits are 30/30 `NOT_APPLICABLE`. Keep the theorem as
a conditional interface and limitation. Do not report primary `S_perp`
mediation.

An optimizer-side local heuristic may motivate directional geometry change:
coupled Adam places decay inside moment updates, whereas AdamW applies a
decoupled shrinkage. A frozen-moment expression involving `(P_t-I)theta` is not
the exact Adam trajectory because moment history and denominators also change.
Likewise `Delta z approximately J_x Delta theta` motivates covariance drift but
is not a deep-network causal theorem. Keep this material in motivation or an
appendix unless a new same-state analysis is preregistered.

## 10. Contributions

State contributions in this order:

1. **Controlled training-side origin case:** decay-policy siblings show large,
   reproducible Raw-MD same-pair multiplicity.
2. **Cancellation-aware evidence unit:** Gain/Loss/Churn reveals behavior hidden
   by near-equal aggregate AUROC.
3. **Local interaction:** same-LR WD contrasts separate a genuine local
   `WD x coupling` interaction from invalid cross-LR causal language.
4. **Score-channel localization:** adverse movement is predominantly Marginal;
   RMD and L2 reveal distinct cancellation/normalization behavior.
5. **Formation evidence:** the sensitivity is early detectable, amplified over
   training, and concentrated in deep representation stages.
6. **Boundary discipline:** SGDM reversal, ID Pareto status, non-monotone
   geometry concordance, and applicability failure delimit the claim.

Do not list architecture replication, broad optimizer mapping, RtMD, or a new
spectral mechanism as completed contributions.

## 11. Figures

1. Parallel Zero/AdamW/Adam sibling design and causal contrast definitions.
2. Four-cell absolute Raw-MD bars plus C-D Gain/Loss/Churn.
3. Zero--AdamW cancellation versus AdamW--Adam directional loss.
4. Raw MD/RMD/L2 and Marginal replacement accounting by context/dataset.
5. Time and depth formation with Marginal/OOD-side overlay.
6. Fixed geometry panel with a clearly descriptive four-cell concordance view.
7. Appendix: flip burden, alignment, SGDM boundary, spectral applicability gate.

## 12. Reviewer-facing boundaries

| Objection | Response |
| --- | --- |
| AdamW absolute Raw MD is already low | Zero is equally low in the primary context; coupling adds a controlled directional deterioration. Cross-LR baseline differences remain descriptive. |
| Primary ID is not equivalent | Correct; NLL guardrail fails in the improvement direction. Use Pareto language and the all-PASS low-LR/low-WD cell. |
| Spectrum/RMD/L2 are prior work | Correct; they are diagnostic lenses, not method novelty. |
| Marginal attribution is causal | It is not. It is exact score replacement accounting. |
| `S_perp` theorem fails | Primary applicability failure is explicit; no `S_perp` attribution is made. |
| WD should have a monotone effect | It does not in current results; report local interaction heterogeneity. |
| This does not identify optimizer recipes generally | Correct. Task F is a controlled case study; the broad RQ requires a paired LR factorial and replication. |
| One architecture | Keep generality local until ResNet-18/CIFAR-10 replication passes. |

## 13. Next experiment gates

### If the current narrow paper is retained

Run ResNet-18/CIFAR-10 focal replication first using a large-effect and a
small-effect representative context. Preserve common init/stream sibling
pairing. Replicate Raw-MD pair multiplicity, RMD/L2 probes, Marginal accounting,
and the minimal endpoint geometry panel before extending time/depth.

### If the broad optimizer-origin RQ becomes the headline

First run a paired LR bridge/factorial in which every LR arm within a seed shares
one initialization and one data stream. This is required to estimate an LR
main effect and LR interactions. Include zero/decoupled/coupled at minimum;
include both WD levels only if the current local interaction remains central.

Additional adaptive optimizers, ConvNeXt/ViT, and a broad phase map are later
external-validity studies, not immediate prerequisites.

## 14. Discussion language

Safe conclusion:

> The results establish a controlled optimizer-side route to Raw-Mahalanobis
> pair-ranking multiplicity and localize its adverse score movement primarily
> to the Marginal term. They also show when the sensitivity is amplified and
> which covariance diagnostics move with it, while leaving a unique spectral or
> `S_perp` mediation and a general optimizer-recipe map open.

Unsafe conclusion:

> Adam coupling, larger weight decay, or higher learning rate generally creates
> the spectral tail that causes Mahalanobis failure.

## 15. Conditional classifier-insensitive geometry route

The current body remains the completed Plan B until the Card 13 Section 14
ID-only gate is run. The candidate main paper asks whether decay coupling forms
sample-dependent, non-affine representation differences predominantly in
directions that the trained classifier does not use, and whether feature-space
OOD readouts subsequently consume that hidden geometry.

```text
Candidate Main, only after GO:
training rule
-> classifier-insensitive non-affine representation change
-> feature-space uncertainty-readout sensitivity

Plan B after FAIL, MIXED, or unusable evidence:
controlled decay-coupling intervention
-> Raw-MD exact pair-ranking multiplicity
-> Marginal score localization and time/depth formation
```

The fast gate is ID-only and cannot establish the downstream OOD arrow. Neural
Collapse is not a gate or a proxy for unknown OOD ordering. The completed Task F
results, including the primary NLL failure and the all-PASS low-LR/low-WD
boundary, retain their current interpretation in either route.
