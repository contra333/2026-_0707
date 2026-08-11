# Where the Gap Lives: manuscript outline

## One-sentence paper

Known Mahalanobis decompositions are used as measurement instruments to trace
a controlled decay-coupling change from a shared training prefix, through an
additive score component and quadratic geometry channel, to changed ID--OOD
pair orderings.

## Abstract logic

1. Premise: fixed post-hoc readouts can vary across learned representations;
   similar ID performance does not resolve that underspecification.
2. Gap: cross-model comparisons do not tell us which score component carries
   the change or whether the training rule caused it.
3. Method: branch coupled, decoupled, and zero-decay continuations from the
   same prefix; decompose MD into global-referenced RMD and Marginal terms;
   account for every tie-aware ID--OOD pair transition.
4. Mechanism: localize the changed additive component, then diagnose its
   quadratic size/stretch and spectrum/allocation channel.
5. Evidence: historical discovery, fresh WRN intervention, architecture and
   dataset replications, dense-connectivity and modern-scale appendices.
6. Conditional conclusion: use the strong claim only if the preregistered
   component, ID-equivalence/trade-off, attenuation, and replication checks
   pass.

Do not open with “same ID accuracy, different OOD.” That is context, not the
claimed contribution.

## 1. Introduction

- Fixed readout means identical formula, feature endpoint, fitting split,
  preprocessing, and hyperparameters; model-specific ID statistics are refit.
- Prior work establishes representation-dependent Mahalanobis behavior,
  feature normalization effects, RMD, and rich geometric decompositions.
- The unresolved question is intervention-level: after identical training
  history, which exact score term changes enough to reverse ID--OOD orderings?
- State contributions in this order:
  1. paired training-to-score attribution;
  2. exact pair-order accounting;
  3. shared-prefix decay-coupling intervention;
  4. replication across architecture, dataset/class count, connectivity, and
     scale.

## 2. Related work and novelty boundary

### 2.1 Training-induced OOD variability

Summarize seed, split, optimizer, pretraining, and fine-tuning sensitivity as
motivation. Do not claim the existence of variability as new.

### 2.2 Mahalanobis family

- original class-conditional Mahalanobis readout;
- Marginal Mahalanobis;
- RMD as subtraction of the marginal global reference;
- Mahalanobis++ as L2 normalization and refitting.

### 2.3 Geometry-based analyses

Credit *A Geometry-Based View of Mahalanobis OOD Detection* for exact
size--stretch factorization, ID-instability decomposition, term-wise RMD,
spectrum/allocation mechanisms, radial transforms, and ID-only compatibility
analysis. State that this paper applies those diagnostics after a controlled
training-side fork and attributes actual cross-branch pair transitions.

### 2.4 Difference table

| Axis | Prior geometry/cross-model studies | This paper |
| --- | --- | --- |
| Source of variation | independently trained or pretrained models | siblings from one training prefix |
| Primary target | detector performance/geometry association | branch-induced pair-order change |
| Score diagnosis | known quadratic/radial/spectral analyses | exact MD--RMD--Marginal transition attribution |
| Intervention | usually model/pretraining/post-hoc transform | decay coupling inside optimizer family |
| Claim | cross-model compatibility or detector improvement | controlled training-to-score mechanism |

## 3. Framework and supporting theory

### 3.1 Scores and pair margins

Define the ID-like score convention, MD, Marginal, RMD, their exact additive
identity, pair margin, and tie-aware AUROC.

### 3.2 Branch effects and exact accounting

Define branch score change, the 3x3 `{incorrect,tie,correct}` transition table,
and symmetric two-component Shapley accounting. Emphasize that AUROCs of the
three detectors do not add; pair-margin changes do.

### 3.3 Formula expansion

Show the class-independent quadratic term plus class-dependent affine envelope.
Use it to name empirical channels, not to assert which channel training changes.

### 3.4 Ridge/full-rank low-rank statement

State assumptions, Woodbury expression, and `rank <= K-1`. Put pseudoinverse
and covariance-convention limitations beside the proposition.

### 3.5 Cited size--stretch diagnostic

Introduce the cited `q=rw` factorization and the symmetric branch-change
identity. Apply it separately to class and global-reference quadratic terms.

### 3.6 Theory boundary table

Use Card 13 §3.4 verbatim in substance: algebraic identities are proved;
training selectivity, AUROC magnitude, and replication are measured.

## 4. Historical discovery: where the existing gap lives

- population: 30 frozen `last.pt` bundles, 10 configurations x 3 seeds;
- datasets: six existing OOD sets, reported separately;
- scores: MD/Marginal/RMD raw and L2;
- primary output: component-specific pair-outcome dispersion and the
  prespecified historical pair attribution;
- role: discovery only;
- forbidden claim: optimizer, LR, or WD caused the historical difference;
- sensitivity only: nearest-accuracy matching, if shown at all, is appendix
  and explicitly selection-biased.

## 5. Shared-prefix confirmation

### 5.1 Design

WRN-28-10/CIFAR-10. Five Adam-family prefixes and three SGDM-family prefixes.
Each prefix produces coupled, decoupled, and zero continuations.

### 5.2 Identity checks

Report source SHA, transferred-state digest, parameter names/order, scheduler,
RNG, DataLoader generator, and first minibatch. Show uninterrupted zero branch
parity before interpreting any OOD result.

### 5.3 ID utility

Report accuracy, NLL, and ECE with the frozen equivalence/guardrail decision.
If equivalence fails, retain the result and switch the language to Pareto
trade-off.

### 5.4 OOD effect and component localization

For each OOD dataset, show branch AUROC/FPR95, practical classification, pair
transition matrix, and exact RMD/Marginal attribution. Report raw and L2 as a
radial control, not as a novelty claim.

### 5.5 Size/stretch follow-through

Only after a component is localized, identify its affected quadratic term and
apply size/stretch, then spectrum/allocation diagnostics. A scalar correlation
alone cannot close this section.

## 6. Replication and scale

1. ResNet-18/CIFAR-10 architecture-only replication.
2. ResNet-18/CIFAR-100 dataset/class-count replication.
3. DenseNet-BC-100 k=12/CIFAR-10 focal appendix.
4. ConvNeXt-Tiny/ImageNet-200 from-scratch focal appendix.

The main claim can establish that the channel is not WRN-specific after step
1, but the final paper plan retains steps 2--4 for dataset and modern-scale
external validity. ConvNeXt pretrained leakage is a hard failure.

## 7. Discussion

- distinguish controlled effect within tested families from universal
  optimizer claims;
- discuss ID-equivalence versus Pareto interpretation;
- explain why component localization precedes size/stretch and spectral detail;
- state limits of CIFAR discovery, from-scratch scale recipes, tied-Gaussian
  fitting, and the selected OOD benchmark.

## 8. Conclusion

Use the Card 13 strong sentence only if all corresponding gates pass. Otherwise
write a result-specific weaker conclusion without changing the detector panel
or matching checkpoints after the fact.

## Figure plan

1. Shared-prefix design plus MD/RMD/Marginal and pair-margin decomposition.
2. Historical 30-bundle component localization across six OOD datasets.
3. Fresh branches: ID utility versus OOD effect with equivalence/Pareto labels.
4. Exact pair transitions, component Shapley attribution, and size/stretch.
5. WRN--ResNet--DenseNet--ConvNeXt replication matrix.

## Table plan

1. Prior work versus this intervention/attribution design.
2. Regime, branch family, prefix count, detector scope, and execution status.
3. Main paired ID and OOD estimates with practical classifications.
4. Replication summary with claim gate per regime.
5. Appendix numerical validation: reconstruction, condition number, ties,
   transferred-state identity, and leakage checks.

## Results language gate

Before results, freeze four sentence families: component concentration passed,
ID equivalence passed, selective attenuation passed, and replication passed.
When any gate fails, its strong sentence is not used. Failed branches and null
effects remain in tables and are not replaced.
