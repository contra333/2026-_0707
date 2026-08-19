# Training-Rule-Induced Pair-Ranking Multiplicity in Mahalanobis OOD Detection

This is the human-readable post-result manuscript outline. Exact estimands and
execution rules remain governed by
[`../reference_cards/13_active_paper_protocol.md`](../reference_cards/13_active_paper_protocol.md).
Numerical evidence and claim boundaries live in
[`task_f_result_analysis.md`](task_f_result_analysis.md).

Status as of 2026-08-19: the prospective ResNet-18/CIFAR-10 gate completed with
technical `PASS` and scientific **`PARTIAL`**. The frozen gate therefore did
not unlock architecture-general wording and stopped the current ICLR
main-paper promotion. This outline preserves the WRN-scoped paper logic and
records the architecture boundary; it is not a claim that the complete pattern
replicated.

## Frozen paper identity

The paper asks:

> Holding architecture, data, objective, initialization, and minibatch order
> fixed, how much can changing only the decay-coupling rule reorganize the same
> ID--OOD pair ordering of a protocol-fixed, branch-refitted Mahalanobis
> detector; in which score component, epoch, and depth does that sensitivity
> appear?

The one-sentence answer supported by completed Task F is:

> Matched decay-coupling siblings can substantially reorganize the ID--OOD
> pair ordering of branch-refitted Raw Mahalanobis; the adverse movement is
> predominantly localized to the Marginal score component, is early detectable
> and later amplified in deep representations, and varies across local LR/WD
> contexts.

`Protocol-fixed, branch-refitted` is important. Every branch follows the same
ID-only fitting procedure, but refits its own class means and covariance. The
paper does not compare one frozen set of detector parameters across branches.

The paper is a high-control empirical study of readout non-invariance, not an
Adam-versus-AdamW benchmark, a Neural Collapse paper, a new OOD detector, or a
claim that one geometry metric uniquely mediates the effect.

## Abstract logic

1. Aggregate AUROC is a net probability over ID--OOD pairs; equal AUROC need
   not mean equal pair behavior.
2. Prior work explains Mahalanobis through completed representation geometry,
   but does not isolate this matched training-rule intervention and trace its
   exact pair reordering through training.
3. Task F changes decay coupling among same-initialization, same-stream sibling
   trajectories and measures Gain, Loss, and Churn on identical pairs.
4. Raw-MD C--D is negative in all four tested Adam cells, but its magnitude is
   strongly context-dependent. Same-LR WD contrasts are controlled; current
   cross-LR WRN differences are descriptive.
5. RMD remains much more stable, and exact two-player replacement accounting
   localizes most adverse Raw-MD movement to Marginal. This is score
   localization, not causal mediation.
6. The effect appears early, grows through epoch 120, and is concentrated in
   stage3 and penultimate features.
7. SGDM reversal, ID guardrail failures, non-monotone geometry concordance,
   `S_perp` inapplicability, and the failed classifier-insensitive gate delimit
   the claim.
8. A prospectively frozen ResNet-18/CIFAR-10 study reproduced Raw-MD direction,
   Churn, large-versus-small context, Marginal accounting, and ID guardrails,
   but failed the large-context Near RMD-attenuation condition; the verdict is
   `PARTIAL`.

## 1. Introduction

Open with the distinction between aggregate and behavioral equivalence.

For two matched policies, define pair transitions as Gain, Loss, and Churn.
With the frozen tie convention,

```text
DeltaAUROC = Gain - Loss
Churn      = Gain + Loss.
```

Therefore `DeltaAUROC approximately 0` can coexist with large Gain and Loss.
The Zero--AdamW result supplies the motivating example: net AUROC is nearly
unchanged while `0.14--0.19` of pair orderings change.

Then introduce the training-side intervention:

```text
matched initialization and minibatch stream
-> decoupled versus coupled decay
-> protocol-fixed, branch-refitted Mahalanobis
-> exact same-pair behavior
-> score localization and formation
```

Do not open with `same ID performance`. Primary Accuracy and ECE pass, but NLL
fails in the improvement direction. Present that cell as an ID/OOD Pareto
result and use the low-LR/low-WD all-PASS cell as the comparable-ID boundary.

The five contributions are:

1. controlled decay-coupling evidence for Raw-MD pair-ranking non-invariance;
2. cancellation-aware Gain/Loss/Churn evidence hidden by aggregate AUROC;
3. predominantly Marginal score localization with RMD/L2 as diagnostic probes;
4. matched epoch/depth evidence for formation during training; and
5. explicit context and applicability boundaries, including the prospective
   architecture replication's `PARTIAL` outcome.

## 2. Related work and novelty boundary

### Mahalanobis geometry and normalization

Credit prior work for representation-dependent Mahalanobis behavior, RMD and
Marginal, L2 normalization, size--stretch factorization, spectrum--allocation,
intrinsic dimension, and post-hoc geometric interventions. These tools are
explanatory lenses in this paper, not new methods.

### OOD experimental reliability

Credit prior work showing that seeds, splits, epochs, and training details can
change detector performance and ranking. The present unit is narrower and
more controlled: one decay-coupling factor, matched initialization and stream,
identical evaluated pairs, and formation in the same trajectories.

### Optimizer-induced representation geometry

Credit optimizer/weight-decay and Neural Collapse work for the proposition that
training rules can shape representation geometry. Do not transfer their NC
theory into an OOD-ordering theorem. Our downstream object is a branch-refitted
post-hoc Gaussian readout.

### Multiplicity

Use `pair-ranking multiplicity` operationally. Do not claim novelty for the
generic idea of churn or conflate this object with predictive multiplicity.
The paper-specific contribution is the controlled intervention plus exact
pair accounting, score localization, and matched formation evidence.

## 3. Problem setup and estimands

### Parallel sibling design

Zero, AdamW, midpoint, and Adam are parallel siblings; they are never drawn as
a sequential path.

- `C-D`: coupled minus decoupled at fixed nominal WD;
- `D-Z`: adding decoupled WD;
- `C-Z`: adding coupled decay dynamics;
- same-LR `WD1e-3 - WD1e-4`: controlled branch-specific WD contrast; and
- `[(C-D)_highWD - (C-D)_lowWD]`: local `WD x coupling` DiD.

High- and low-LR WRN groups do not share one data-stream identity. Their
absolute difference is descriptive, not an LR main effect.

### Branch-refitted readout

For every branch, fit class means and covariance only from that branch's ID
training features. Apply the same fitting and scoring protocol to all branches.
State the OOD-positive score orientation, tie convention, checkpoint, layer,
and dataset roles before every aggregate.

### Pair and score accounting

Report Gain, Loss, Churn, and net AUROC on the same ID--OOD pair population.
For ID-like scores,

```text
s_MD(z) = s_RMD(z) + s_Marginal(z)
m_MD(i,o) = m_RMD(i,o) + m_Marginal(i,o).
```

Use symmetric two-player replacement accounting for RMD/Marginal and ID/OOD
sides. The Shapley terms exactly reconstruct the observed score change but are
not causal mediators.

## 4. Controlled experimental design

Specify WRN-28-10/CIFAR-10, four Adam LR/WD cells, role and seed counts,
epoch-200 `last.pt` penultimate primary endpoint, six protected OOD datasets,
Near/Far macro construction, and the Accuracy/NLL/ECE guardrails.

The statistical unit is the seed. Average datasets within seed before Near/Far
aggregation; report seed means, sample SD, and paired 90% intervals. Samples,
datasets, and ID--OOD pairs are not independent replicates.

Keep the protected evaluation and artifact provenance visible but concise.
Move implementation inventory and checksum catalogs to the appendix.

## 5. Results

### 5.1 Controlled Raw-MD non-invariance

Lead with the four C--D cells. Raw-MD mean C--D is negative in every tested
Adam cell:

- Near: `-0.0218` to `-0.1780`;
- Far: `-0.0787` to `-0.2835`.

The primary `LR=1e-3, WD=1e-4` result is `-0.1743/-0.2835` Near/Far with Churn
`0.3486/0.3863`. Do not average over LR/WD or infer a universal optimizer law.

### 5.2 Aggregate AUROC hides pair multiplicity

Show Zero--AdamW next. Its net AUROC difference is near zero, yet Churn is
`0.14--0.19`. Plot Gain and Loss separately so their cancellation is visible.
Pair-burden analysis shows a distributed phenomenon rather than one tiny fixed
set of OOD samples; retain seed as the inferential unit.

### 5.3 Sensitivity is score-structured

RMD remains high and its C--D gap is smaller in the tested cells. L2-MD strongly
recovers absolute AUROC, but does not uniformly attenuate the signed gap.

Marginal is the main adverse replacement component in all four Adam cells.
Primary dataset-level absolute Marginal share is `69%--94%`. A share above one
means RMD offsets part of the Marginal deterioration; it does not mean more
than 100% causal mediation. OOD-side dominance is strong in primary and
high-WD contexts but is not universal.

### 5.4 Sensitivity forms during training

The primary Near C--D sequence is
`-0.0739, -0.1072, -0.1771, -0.1781, -0.1743` at epochs
`10, 60, 120, 160, 200`. Say `early detectable, later amplified`, not `late
formation`.

The depth sequence is small at stage1/2 and large at stage3/penultimate.
Telemetry is a passive witness at each branch's actual state, not a same-state
optimizer counterfactual.

### 5.5 Local context and negative boundaries

At low LR, the local `WD x coupling` DiD is `-0.0912/-0.0797` Near/Far. At
high LR it is `+0.0004/+0.0618`. Stronger WD therefore does not universally
amplify coupling damage. SGDM has the opposite C--D sign.

The fixed geometry panel is broadly concordant with effect size but not
monotone. Coupled spectral attribution is unavailable because the relevant fit
is inapplicable. The `S_perp` and classifier-insensitive gates are reported as
failed negative diagnostics, not replaced with a new mechanism.

## 6. ResNet-18/CIFAR-10 replication boundary

Use the exact 20-run Card 13 Section 15 matrix:

```text
2 LR x 1 WD x 2 coupling roles x 5 common seeds = 20 runs.
```

This endpoint-only study asked whether the large-versus-small context pattern,
Raw-MD sign and Churn, RMD attenuation, Marginal localization, and
small-context ID guardrails replicate under ResNet-18. It introduced no new
mechanism metric.

Technical validation passed 20/20 runs. In the large context, Raw-MD C--D was
`-0.0356/-0.1758` Near/Far with Churn `0.3244/0.3617`. The small context had
smaller Raw-MD gaps `-0.0130/-0.0973`; Marginal accounting and small-cell ID
guardrails passed. The frozen large-context Near RMD condition failed because
`abs(Delta RMD)=0.1426` exceeded `abs(Delta Raw MD)=0.0356`; two of five seeds
had large adverse Near RMD deltas while three were near zero. The resulting
scientific verdict is `PARTIAL`.

Present this result as an architecture/seed boundary, not as complete
replication. It does not unlock architecture-general language, a rescue grid,
or a replacement mechanism. Under the frozen contract it also stops the
current ICLR main-paper promotion.

## 7. Discussion and limitations

Discuss the result as training-rule-induced non-invariance of a commonly used
post-hoc readout. The practical implication is that matching architecture,
dataset, objective, accuracy, or aggregate AUROC does not certify identical
feature-OOD decisions.

Bound the conclusions:

- Primary NLL is not equivalent; one tested cell passes all ID guardrails.
- Current WRN cross-LR comparison is not causal.
- Marginal is a score localization, not a causal mediator.
- RMD and L2 are prior methods used as probes.
- Geometry concordance does not identify a unique carrier.
- SGDM refutes a universal coupling law.
- The prospective ResNet gate was `PARTIAL`; WRN-scoped findings do not become
  architecture-general claims.

## 8. Conclusion

Conclude with the controlled chain only:

```text
matched decay-coupling intervention
-> exact same-pair Gain/Loss/Churn
-> predominantly Marginal score localization
-> matched time/depth formation
```

Do not conclude that Adam coupling is generally harmful or that the study maps
optimizer recipes broadly.

## Main figure and table contract

| Item | Question answered | Placement |
| --- | --- | --- |
| Figure 1 | What is held fixed, what changes, and which contrasts are valid? | Main |
| Figure 2A | How large is C--D Raw-MD non-invariance in each cell? | Main composite |
| Figure 2B | How do Gain and Loss hide behind net AUROC? | Main composite |
| Figure 3 | Which readout and score component carries the movement? | Main |
| Figure 4 | When and at what depth does it form? | Main |
| Figure 5 | Which WRN predictions replicate or break in ResNet-18? | Appendix/boundary; no architecture-general claim |
| Geometry and negative gates | Which explanations were diagnostic or rejected? | Appendix |

Use individual seed dots plus paired means and 90% intervals. Every delta plot
has a zero line. Show Gain and Loss in opposite directions. Plot signed
`phi_RMD`, `phi_Marginal`, and reconstructed total rather than a share alone.
Use actual epoch coordinates; show depth as a discrete axis without a fitted
trend. Do not use a four-point geometry regression in the main paper.

Main tables are:

1. design, roles, seeds, sibling identity, and ID guardrails;
2. AdamW/Adam AUROC, C--D interval, Gain, Loss, Churn, and D--Z cancellation;
3. Raw MD/RMD/L2, signed score accounting, and within-LR WD/DiD results.

Dataset-level values, FPR95, SGDM, geometry/alignment, telemetry, and negative
gates belong in appendix tables.

## Human reading ledger for Geometry V3

Read *A Geometry-Based View / Dissecting Mahalanobis* by recording, for every
major figure:

1. the claim the figure actually establishes;
2. the comparison or intervention supporting it; and
3. what our paper must not repeat as novelty.

Prioritize detector definitions and refitting semantics; cross-model versus
matched-pair experimental units; geometry correlation and the ID-only proxy;
radial post-hoc interventions; size--stretch and spectrum--allocation; and the
rotation/smoothing/shaping appendices. Maintain this ledger:

| Prior work establishes | Diagnostic reused here | New matched claim here | Prohibited wording |
| --- | --- | --- | --- |
|  |  |  |  |

The novelty boundary is the complete combination, not any single geometry
quantity:

```text
matched decay-coupling intervention
-> exact same-pair Gain/Loss/Churn
-> RMD/Marginal score-response localization
-> matched epoch/depth formation.
```

## Claim-language guardrails

| Avoid | Use |
| --- | --- |
| fixed detector/readout parameters | protocol-fixed, branch-refitted readout |
| same ID performance | ID/OOD Pareto result; all-PASS boundary cell |
| Marginal mediates the effect | adverse movement is localized to Marginal |
| coupling is bad | tested Adam C--D direction; SGDM and context boundaries |
| LR causes the WRN gap | cross-LR descriptive difference |
| geometry explains the gap | geometry is concordant but not monotone |
| classifier-insensitive carrier | prespecified carrier gate failed |
| optimizer-side origin generally established | controlled decay-coupling case study |
