# Reference Card 12: Fixed-Readout Training-Rule Intervention Protocol v2

## 0. Status, scope, and authority

This card freezes protocol
`fixed_readout_training_rule_intervention_v2` for the next paper phase. It is
the authority for the paper claim hierarchy, experiment populations, optimizer
roles, detector tiers, shared-prefix intervention, mechanism gate, and
replication boundary.

Status on 2026-08-10:

```text
contract: frozen at Stage 1
Stage 2 protected-artifact reuse: authorized by the owner fast-path task;
  selection manifest and checksum verification still required
shared-prefix fork runtime: not implemented or validated
fresh paired confirmation: not executed
ResNet-18/CIFAR-100 replication: not executed; dataset/OOD contract pending
causal or cross-regime conclusion: not established
```

Metric Contract v1.2 and its completed WRN-28-10/CIFAR-10 results remain an
immutable **discovery foundation**. This card does not change any v1.2 formula,
artifact key, reporting tier, checkpoint, role, result, or validation record.
In particular, v1.2's optimizer grid is descriptive rather than the causal
intervention, and its seed 0 participated in role selection.

Adjacent authorities remain:

- [`01_optimizers.md`](01_optimizers.md) for optimizer and parameter-group
  semantics;
- [`02_architectures.md`](02_architectures.md) for feature endpoints;
- [`04_openood_v1_5_protocol.md`](04_openood_v1_5_protocol.md) for the current
  CIFAR-10 data/OOD membership and ID-like score convention;
- [`05_training_protocol.md`](05_training_protocol.md) for ordinary training,
  checkpoint, and same-run resume semantics;
- [`08_raw_feature_artifact_contract.md`](08_raw_feature_artifact_contract.md)
  for raw checkpoint-feature artifacts;
- [`11_metric_contract_v1_2.md`](11_metric_contract_v1_2.md) for immutable
  v1.2 formulas and completed historical outputs.

Where this card adds a v2 detector or fork operation, the addition is
version-scoped and must use a separate v2 registry/schema. It must not mutate
the v1.2 registry to make old bundles appear to contain new measurements.

## 1. Fixed research question and contribution boundary

The paper asks:

> When models have comparable ID utility, can a fixed feature-based OOD
> readout be practically non-invariant to the training rule, and which
> detector-relevant representation channel changes ID--OOD score overlap?

The intended evidence chain is:

```text
shared training prefix
  -> coupled / decoupled / zero-decay continuation
  -> detector-formula-relevant geometry component
  -> per-sample ID/OOD score overlap
  -> fixed-readout AUROC and FPR@95
  -> independent dataset--architecture replication
```

The fixed contribution hierarchy is:

1. **Phenomenon:** the paired readout change exceeds seed noise and a
   prespecified practical margin, or is classified as inconclusive/invariant.
2. **Mechanism:** a score-relevant geometry channel changes the overlap through
   terms in the detector formula. A scalar geometry--AUROC correlation alone is
   insufficient.
3. **Intervention:** branches forked from the same prefix isolate the decay
   coupling rule within optimizer family.
4. **Replication:** the focal result is checked in ResNet-18/CIFAR-100.

The paper is **not** an optimizer leaderboard, a universal-best-detector claim,
a new-detector SOTA paper, an ID-only detector-compatibility predictor paper,
or a claim that L2 normalization itself is novel. Prior work already documents
training-induced OOD variation and norm/geometry effects; the intended addition
is the controlled chain from training-rule intervention to formula-linked score
overlap and replication. See source keys
`szyc_ood_reliability_paper`, `training_induced_ood_wacv2026_paper`,
`mahalanobis_pp_paper`, and `geometry_based_mahalanobis_ood_paper` in
[`docs/sources.lock.yaml`](../sources.lock.yaml).

An ID-only audit predictor may be reported only as a separately labeled stretch
analysis. It is not required for the v2 paper and cannot substitute for the
paired intervention or score-level mechanism evidence.

## 2. Meaning of fixed readout and comparable ID utility

**Fixed readout** means the detector formula, feature endpoint, preprocessing
rule, fitting split, hyperparameters, score direction, and metric implementation
are identical across compared branches. Model-specific ID statistics are
refitted independently on each branch's deterministic `id_train` artifact;
reusing means, covariance, reference banks, subspaces, or other fitted state
across models is forbidden.

“Comparable ID utility” is an evidence condition, not a training target and not
permission to post-hoc select matching checkpoints. Every branch reports ID
accuracy, NLL, ECE, and their paired differences. Before fresh protected OOD
evaluation, the protocol addendum must freeze an accuracy equivalence margin
and NLL/ECE guardrails.

- If the paired ID interval lies inside the frozen equivalence region, the OOD
  contrast may use the comparable-ID wording.
- If it lies outside, the OOD result remains reportable but must be framed as an
  ID/OOD Pareto trade-off rather than readout non-invariance at comparable ID
  utility.
- If it overlaps the boundary, the comparable-ID claim is inconclusive.

Exact ID accuracy equality is neither expected nor required. No branch is
discarded, replaced, or rerun because its ID result is inconvenient.

## 3. Four-stage execution order

### Stage 1 -- contract and truth synchronization

Freeze this card, reconcile current repository status, record local historical
materials without moving them, validate the documentation, and push `main`.

### Stage 2 -- existing-artifact mechanism gate

Use only checksum-verified, already produced Metric Contract v1.2 artifacts.
This owner-authorized reuse does not authorize checkpoint reevaluation, dataset
traversal, feature re-extraction, mutation of remote artifacts, or a new upload.

Before download or analysis, commit a selection manifest that fixes:

- the exact discovery checkpoint identities (primary scope: all authorized
  `last.pt` identities; `best_val.pt` is post-freeze sensitivity only);
- the six OOD datasets and exact file allowlist;
- remote URI, size, and expected checksum for every reused file;
- a non-Git destination and no-overwrite policy;
- the parity checks against the completed v1.2 aggregate.

Stage 2 must reproduce existing AUROC/FPR95 from raw score arrays before
forming new results. It then evaluates score overlap, reconstructs detector
scores from saved components, tests invariant controls and non-invariant
witnesses, and selects at most one focal geometry channel/targeted transform.
Stage 2 is protected **discovery** evidence, not confirmation or causality.

The gate passes only if all of the following hold:

1. artifact identities and checksums pass with no missing selected file;
2. v1.2 score/metric parity passes at the declared tolerance;
3. the candidate channel changes a term in the detector formula and predicts
   the direction of ID--OOD misordering changes;
4. a targeted transform selectively attenuates that gap while a prespecified
   invariant null control remains stable;
5. the pattern is not supported only by a single hand-picked OOD dataset or
   configuration.

If the gate fails, do not launch the paired production study. Narrow the claim,
change the mechanism hypothesis in a new versioned contract, or stop.

### Stage 3 -- fresh paired main confirmation

Implement and validate the shared-prefix fork, then run fresh prefix seeds in
the WRN-28-10/CIFAR-10 main regime. Freeze all items in Section 11 before any
fresh protected OOD result is opened.

### Stage 4 -- second-regime replication

After the main analysis is frozen, reproduce the focal phenomenon/mechanism in
ResNet-18/CIFAR-100. Its dataset membership, preprocessing, OOD membership, and
success criterion require a separate versioned addendum before execution.

The order is mandatory: Stage 1 push, then Stage 2 gate, then the Stage 3
protocol addendum and implementation, then Stage 4. A later stage cannot be
used to revise an earlier result-dependent choice silently.

## 4. Regimes and optimizer roles

### 4.1 Main: WRN-28-10/CIFAR-10

The primary family is Adam/AdamW. From one Adam-family zero-weight-decay prefix
for each fresh prefix seed, fork three continuations:

| Branch | Update rule | Role |
| --- | --- | --- |
| `adam_coupled` | Adam with coupled weight decay `lambda` | primary endpoint |
| `adamw_decoupled` | AdamW with decoupled weight decay, same nominal `lambda` | primary endpoint |
| `adam_zero` | Adam with weight decay zero | shared-base control |

The conventional-family control starts from an independent SGDM zero-decay
prefix and forks:

| Branch | Update rule | Role |
| --- | --- | --- |
| `sgdm_coupled` | SGD+Momentum with coupled weight decay `lambda` | control endpoint |
| `sgdw_decoupled` | SGDW with decoupled weight decay, same nominal `lambda` | control endpoint |
| `sgdm_zero` | SGD+Momentum with weight decay zero | shared-base control |

Adam and SGDM prefixes are independent experimental blocks. Cross-family
forks such as Adam-prefix to SGD are forbidden. Equal numerical weight decay
is a controlled nominal input; it does not imply equal effective
regularization between coupled and decoupled updates.

The Adam/AdamW family carries the primary paper claim. SGDM/SGDW checks whether
the coupling-sensitive pattern is restricted to the adaptive family; it is not
an additional optimizer-ranking contest.

### 4.2 Replication: ResNet-18/CIFAR-100

Replication uses the paired Adam/AdamW family and an independently trained
conventional SGDM reference. The SGDM reference is a sanity anchor for the
conventional CIFAR training regime and is not a shared-prefix causal contrast.
No new full SGD HPO grid is required. Its exact recipe is frozen in the Stage 4
addendum using ID-only information.

## 5. Shared-prefix fork contract

`fork_from_prefix` is a new scientific operation. It is **not** ordinary
resume, and Card 05's same-run unchanged-config resume checks must not be
weakened.

The source is a full epoch-boundary `last.pt` with weight decay zero.
`best_val.pt` and model-only snapshots are forbidden. The first branch epoch is
`switch_completed_epoch + 1`. Before any stochastic operation, every sibling
receives identical:

- model tensors and parameter-name ordering;
- optimizer tensor state (`step`, `exp_avg`, `exp_avg_sq` for Adam family;
  `momentum_buffer` for SGDM family);
- scheduler state and global step;
- Python, NumPy, CPU/CUDA RNG and train-loader generator state;
- immutable prefix history and dataset membership.

Do not load the prefix optimizer parameter-group dictionary wholesale into a
different branch optimizer: that would overwrite the branch's intended decay
configuration. Import tensor state by exact parameter name/order, then
re-apply and assert LR, betas/epsilon or momentum/Nesterov, parameter groups,
weight-decay policy, nominal dose, and coupling rule.

Every fork records at least:

```text
contract_version, regime_id, pair_id, optimizer_family, prefix_seed
switch_completed_epoch, prefix_run_id, prefix_checkpoint_sha256
prefix_config_hash, branch_name, branch_nominal_weight_decay
branch_config_hash, fork_policy_version, branch_start_state_digest
state_transfer = {model, optimizer_tensors, scheduler, rng, train_generator}
```

`pair_id` binds regime, family, prefix seed, switch epoch, and prefix SHA.
`branch_id` additionally binds branch name, dose, and resolved branch config.
A checksum-sealed fork manifest lists all siblings and their terminal status.
The source prefix is read-only and terminal branch artifacts cannot be
overwritten.

Sibling branches must use one locked runtime and host/GPU policy. Any device
difference is recorded and cannot be silently treated as the decay-only
contrast.

## 6. Detector panel v2

All scores are ID-like. The existing v1.2 keys keep their historical meaning;
new definitions use `mechanism_contract_v2` records and a separate v2 registry.

### 6.1 Main confirmatory panel (seven)

| Paper name | Artifact key | Fixed definition |
| --- | --- | --- |
| Mahalanobis-Raw | `detector/mahalanobis_raw` | Card 11 GAUSS-1 |
| Mahalanobis++ | `detector/mahalanobis_pp` | Card 11 GAUSS-4 |
| kNN-Raw K=50 | `detector/knn_raw_k50` | negative raw-feature squared distance to the exact 50th ID-train neighbor |
| kNN-L2 K=50 | `detector/knn_l2_k50` | Card 11 SUB-1, including its `+1e-10` denominator |
| CTM / Prototype-Cosine | `detector/ctm_prototype_cosine` | Card 11 SUB-3 |
| Pure Residual | `detector/vim_residual_author_dim` | negative ViM residual at the frozen author dimension |
| Energy-T1 | `ood_score/energy_t1` | Card 11 LOGIT-3 |

For raw kNN, fit/query features are not normalized:

```text
s_knn_raw(z) = -d^2_(50)(z, {z_i : i in id_train}).
```

Search is exact float64 squared Euclidean with stable `(distance, sample_id)`
ties. There is no self-exclusion for ID-test or OOD queries. Store the 50th
distance/ID and, in mechanism bundles, the complete top-50 IDs and distances.

Pure Residual reuses ViM's classifier origin and assume-centered ID subspace:

```text
u = -W^+ b
Sigma_u = (Z_train-u)^T (Z_train-u) / N
DIM = floor(p/2)
r(z) = ||(z-u) NS||_2
s_residual(z) = -r(z).
```

The negative sign is required because project scores are ID-like. No Energy or
`alpha` term enters Pure Residual. `DIM=floor(p/2)` is the frozen default for
both WRN (`320/640`) and ResNet-18 (`256/512`); Stage 2 must verify it rather
than choose a best OOD dimension.

### 6.2 Secondary, appendix, and excluded

- Secondary matched/robustness: Relative Mahalanobis-Raw,
  Relative Mahalanobis++, and MSP.
- Appendix decomposition: ViM author rule, with Energy, residual, and
  `alpha * residual` components retained.
- Excluded: ReAct. It is not needed to test the representation-geometry claim
  and adds an activation-clipping hyperparameter/intervention axis.

The v2 panel does not retroactively change v1.2, where raw kNN was forbidden,
ViM/RMD/MSP had different reporting tiers, and 19 detector identities were
already emitted.

## 7. Score overlap and mechanism evidence

For an ID-like score `s`, per-dataset AUROC satisfies

```text
1 - AUROC
= P[s(X_ID) < s(X_OOD)]
  + 0.5 P[s(X_ID) = s(X_OOD)].
```

Every mechanism result must therefore retain raw ID/OOD score arrays and show
how a channel changes misordered/tied pairs. Required outputs include:

- ID and OOD score/component empirical distributions and quantiles;
- exact AUROC reconstruction from pairwise ranking, implemented without
  materializing an infeasible full pair matrix;
- counts/mass of incorrect-to-correct, correct-to-incorrect, and tie
  transitions under the targeted transform;
- score reconstruction from saved formula components;
- per-dataset results before any near/far/overall macro summary.

A correlation between a scalar geometry metric and AUROC may accompany this
analysis but cannot establish the mechanism. “Restoration” is not assumed.
The confirmatory term is **selective gap attenuation**: the targeted transform
reduces the paired training-rule gap for the detector predicted to depend on
that channel, beyond changes seen in its invariant/null control.

The transform manifest records its ID-only or parameter-free fit source,
operation order, detector-refit rule, handling of logits/classifier
parameters, expected score-value relation, expected rank/AUROC relation, and
checksum.

## 8. Invariance and non-invariance controls

Score-value invariance and rank/AUROC invariance are distinct and must be
reported separately.

- Raw Mahalanobis/RMD refitted after a common translation, orthogonal
  transformation, or nonzero global scaling is an invariant control in exact
  arithmetic. General invertible-affine invariance is claimed only for
  full-rank, well-conditioned covariance. The v1.2 Moore-Penrose path may
  violate it in rank-deficient settings and requires an explicit witness.
- Mahalanobis++ is invariant to positive per-sample scaling when no norm-status
  threshold is crossed and is invariant to a common orthogonal transform; it
  is not generally translation- or anisotropic-transform invariant.
- Raw kNN is value-invariant to common translation/orthogonal transforms. A
  common global scale multiplies squared-distance scores by a positive
  constant, so rank/AUROC is invariant; per-sample scaling and anisotropic
  transforms need not be.
- Card 11 kNN-L2 uses `z/(||z||+1e-10)` and is therefore only approximately
  stable, not mathematically exact, under positive sample scaling. Its v1.2
  formula must not be silently changed.
- CTM is invariant to positive scaling of each query, but independently
  scaling training samples can rotate the raw-before-average class prototype.
- Pure Residual is orthogonally invariant only under a coordinated transform of
  representation and classifier coordinates. A common global scale changes
  score values but preserves rankings. Energy-T1, computed from frozen logits,
  is the feature-only post-hoc transform null control.

Every invariant test has a non-invariant witness. Numerical tolerances,
rank/condition diagnostics, exact/near-zero status, and any exception to the
expected relation are stored rather than hidden.

## 9. Estimands, uncertainty, and reporting

For prefix block `i`, detector `D`, and OOD dataset `k`, the primary paired
contrast is

```text
delta_i,D,k = AUROC_i,D,k(decoupled) - AUROC_i,D,k(coupled).
```

The zero-decay branch supplies two additional within-prefix contrasts but does
not convert the same nominal dose into an estimate of equal regularization.
The focal targeted-transform attenuation is

```text
A_D,k,T = E_i[delta_i,D,k(raw)] - E_i[delta_i,D,k(T)].
```

This is not called a causal mediation proportion. Evaluation-sample
uncertainty and between-prefix uncertainty are estimated and displayed
separately. OOD datasets are environments, not independent seed replicates.
Every dataset is reported separately; near/far/overall arithmetic macro means
are summaries only, and raw samples are never pooled across datasets.

`last.pt` is the primary terminal endpoint. Any `best_val.pt` analysis is a
separate post-freeze deployment sensitivity. Fresh prefixes are the unit of
confirmation. v1.2 seeds and configurations may inform Stage 2 channel
selection and prospective power only; they do not count as fresh confirmation.

Before fresh OOD evaluation, each outcome receives a practical margin
`epsilon_D,k` and an interval rule:

- interval wholly inside `[-epsilon, +epsilon]`: practically invariant;
- interval wholly above `+epsilon` or below `-epsilon`: practically
  non-invariant with direction;
- otherwise: inconclusive.

The same three-way logic applies to the ID equivalence margin. Multiple-testing
and claim-family handling are frozen in the Stage 3 addendum; raw per-detector,
per-dataset effects are retained regardless of significance.

## 10. Required implementation oracles before Stage 3

The fork implementation must pass at least:

1. zero-decay Adam/AdamW state-and-update parity and SGDM/SGDW parity, including
   populated moment/momentum state;
2. exact model, optimizer-tensor, scheduler, RNG, loader-generator, and
   parameter-name transfer with branch decay surviving import;
3. manual first-post-switch update oracles for coupled, decoupled, and zero
   branches, including no-decay parameter groups;
4. identical branch-start digests and first minibatch/sample/augmentation
   identity;
5. zero branch parity with an uninterrupted zero-decay continuation;
6. rejection of cross-family conversion, model-only/best checkpoints,
   incompatible parameter order, and overwrite attempts;
7. new raw-kNN and Pure Residual score/component, tie, batching, orientation,
   and reconstruction tests;
8. invariance plus non-invariance witness fixtures from Section 8.

Toy/CPU fixtures are implementation evidence only. Production readiness still
requires bounded actual-data validation on the execution SHA; neither is a
research result.

## 11. Decision register

### Frozen now

- main and replication dataset--architecture regimes;
- Adam/AdamW primary and SGDM/SGDW conventional-family control roles;
- independent family prefixes and three-way continuation design;
- paired Adam replication plus conventional SGDM reference;
- seven main, three secondary, ViM appendix, and ReAct exclusion;
- fixed-readout refitting rule, per-dataset reporting, raw-score mechanism
  evidence, fresh-seed requirement, and four-stage order;
- v1.2 non-retroactivity and discovery-only role.

### Freeze after Stage 2 and before any fresh protected OOD result

- focal geometry channel and one targeted transform;
- exact invariant/null controls used for the focal claim;
- switch epoch or prespecified switch-epoch set;
- family-specific nominal decay dose or dose set;
- ID accuracy equivalence margin and NLL/ECE guardrails;
- detector/dataset practical OOD margins;
- prospective prefix count/power rule and multiplicity plan;
- main success/no-go criterion.

These are stage-gated decisions, not permission to choose after seeing fresh
confirmation OOD outcomes.

### Freeze before Stage 4

- CIFAR-100 train/validation/test membership and normalization;
- replication OOD datasets and near/far roles;
- exact conventional SGDM reference recipe;
- replication seed count and success criterion.

No ResNet-18/CIFAR-100 protected run is authorized until that addendum and its
data/runtime validation are committed.

## 12. Literature and interpretation guardrails

- `Flavors of Margin` is a precedent for studying optimizer-dependent
  trajectories and switch-style intervention; it is not evidence for this
  project's OOD or coupled/decoupled result.
- Szyc et al. and `One Model, Many Behaviors` establish that similar or higher
  closed-set performance does not guarantee stable post-hoc OOD behavior; they
  do not supply this project's controlled mechanism.
- Mahalanobis++ already establishes L2 normalization as a strong Mahalanobis
  repair, and the geometry-based Mahalanobis work studies geometry/performance
  prediction and radial transformations. This project must therefore claim a
  training-rule-linked, score-decomposed, selectively attenuated paired effect,
  not L2 normalization or ID-only geometry auditing as its novelty.
- Zhao et al. motivates optimizer/decay-coupling effects on Neural Collapse,
  but an NC metric alone is not a detector mechanism. It must connect to raw
  score components and overlap.

Negative or inconclusive results are valid outcomes under this contract. No
post-hoc detector, transform, optimizer dose, checkpoint, seed replacement, or
dataset deletion may be introduced to rescue the claim.
