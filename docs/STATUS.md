# Project status

Last updated: 2026-08-17

## Current phase

The active paper protocol is
[`13_active_paper_protocol.md`](reference_cards/13_active_paper_protocol.md).
The active design is v12. The bounded historical D1 survival check, Task B
theory/estimator/literature lock, ID-only archive diagnostic, and single
authorized CPU preflight v2 are complete. V2 left theorem applicability at
30/60 and made all 30 applicable fits pass every required numerical gate, up
from 9 in v1. Historical Gate 2 remains permanently `INCONCLUSIVE`; no Gate 2
or scientific conclusion changed.

The v2 compact preservation record now matches the official production
payload: cached-parity diagnostic failures are eight theorem-inapplicable fits,
not the preliminary run's nine. This is a transcription/provenance correction,
not a preflight rerun or method activation. The deterministic repeat reproduced
all payload hashes on the same runtime with the fixed four-thread numerical
environment.

RtMD is an optional sidecar, not a prerequisite for Task F or the main fresh
paired mechanism study. No exact Gate 3 rule is frozen. Task F model/export and
paired-training engineering are complete, including full-matrix sibling
provenance. The one authorized execution-only pilot completed its safe epoch-1
stop, identical-config resume through epoch 2, global-step/audit continuity
checks, and CPU ID-train export; it remains excluded from research evidence.
The CPU export took 20 minutes 13 seconds and was a confirmed production
bottleneck. PR #90 therefore added explicit GPU feature export with accelerator
runtime/output identity and progress reporting; focused tests and the full CPU
suite pass. The resulting specification identity is
`0ac3101e6d6aaed1a5a0d4891792d4700540bac5247cca8f7d6e67d664ffe9ba`, now
frozen in the Task F fresh-training pre-execution addendum v2. A production
A6000 re-export of the existing pilot checkpoint completed twice in 19.89 and
19.67 seconds at batch size 512, versus 20 minutes 13 seconds on the bounded
four-thread CPU run. The repeated GPU artifact was byte-identical; its CPU
reference comparison had maximum absolute feature difference `6.72e-4`,
maximum per-sample relative L2 difference `3.37e-4`, and minimum cosine
`0.99999995`. This removes the production export-throughput blocker without
changing training semantics. The 50-run and pilot launch plans were regenerated
and validated against the new specification. The owner-approved 50-run ID-only
full fresh training completed at execution SHA
`9eb3c1fa56d880ea5220badac7bc71ba75786d22` on the fixed sibling-colocated
shards `curie=20`, `lise=11`, and `precision_medicine=19`. All 50 runs reached
epoch 200 and `global_step=70400`; the pre-existing GPU feature-export queue
also completed all 310 planned jobs. These exports are evaluation inputs, not
research results, and protected-OOD evaluation has not started.

The versioned fresh ID evaluation bridge validates the exact 50-run inventory,
checkpoint/feature/membership/classifier identities, and the complete
1,320-record `id_train` plus `id_validation` logical coverage. The existing 310
jobs are a validated subset; the 1,010 supplemental exports have not yet
started. The fresh ID-only geometry and paired-aggregation implementation is
fixture-tested for raw/L2 discriminant-component bundles, NC/norm/spectrum
records, ID-train affine and gauge alignment with held-out `id_validation`
residuals, cell-and-seed-matched paired 90% t-intervals, non-dropping terminal
status, and deterministic table templates. Comparable-ID remains
`PENDING_PROTECTED_ID_TEST`; no protected inference was run.

PR #96 added the reusable staged-evaluation supervisor and Task F adapter at
evaluation execution SHA
`be666c04ec9a1e0075a4b82d788161d850f1663e`. PR #98 corrected the source
finalizer's active-`RUNNING` handling, PR #100 corrected indexing of the flat
hash-addressed feature-artifact layout, and PR #102 made empty Hugging Face
prefix listings compatible with the installed CLI. The Task F specification
remains unchanged. All three source hosts now pass local checksum validation for
their exact run and export shards and are uploading about 153 GiB of training
bundles plus 34 GiB of existing features with no-delete semantics. Source
`REMOTE_VERIFIED` markers are pending upload completion.

Clean detached evaluation worktrees remain fixed at the evaluation SHA.
Detached gate watchers on all three hosts wait for the three source terminal
markers and then start the fixed 1,010 supplemental GPU exports, 1,320 bridge
bindings, 660 ID-only geometry units, and 657 sibling alignments. A detached
collector gate on `precision_medicine` waits for all three evaluation terminal
markers before central paired aggregation. This chain is independent of the
local workstation, but it has not crossed the source gate yet. No supplemental
export, protected `id_test`, OOD, or RtMD execution has started.

The planned chain is:

```text
same initialization and data stream
→ coupled / decoupled / zero-decay trajectories plus the alpha=0.5 anchor arm
→ update and non-affine representation divergence
→ discriminant subspace S / residual subspace S-perp
→ S-perp / parallel-Marginal / RMD score attribution
→ raw/L2 x MD/RMD interaction and fixed residual-retention path
→ exact pair gain, loss, churn, and net AUROC
→ fixed-total-decay alpha = 0 / 0.5 / 1 interiority check at the anchor
→ channel-matched attenuation and replication
→ conditional only: frozen residual-tail RtMD method gate and evaluation
```

## Validated foundation

- Optimizer factory, shared parameter groups, CIFAR training/checkpoint/resume, and
  `fork_from_prefix` are implemented and tested.
- `wrn28_10`, CIFAR `resnet18`, and the common logits/penultimate-feature API are
  implemented. WRN stage1/stage2/stage3/penultimate taps and the separate
  CPU/GPU Task F ID-feature artifact contract are implemented and tested.
- The Task F run-plan generator conserves 41 Adam-family and nine SGDM-family
  runs, preserves same-initialization/data-stream witnesses, exact snapshot
  identities, strict resume, execution-only exclusion, and non-mutating update
  telemetry. Every planned run now emits the versioned checkpoint provenance
  required by the feature exporter. These are fixture-validated engineering
  records, not completed research runs.
- CIFAR-10 membership and OpenOOD-compatible roles are frozen and validated.
- The WRN-28-10/CIFAR-10 v1.2 population completed: 36 seed-0 grid cells plus the
  frozen-role follow-up population.
- Metric Contract v1.2 protected evaluation completed for 60 checkpoint bundles.
- The historical 30-model MD--Marginal--RMD component analysis completed with `PASS`.
  It is descriptive discovery, not decay-coupling confirmation.
- The Card-13 historical ID-only CPU preflight verified 450 allowlisted files from
  30 cached bundles and completed 60 raw/L2 fits without checkpoint inference.
  Required numerical gates passed for 9 fits (`S-perp`; the other 51 retained
  primary=`none`). Deterministic two-fold residual-tail fits completed for 15 of
  30 bundles per transform and selected finite tails in all 30 completed fits;
  Gate 2 remains `INCONCLUSIVE`, not a method activation or OOD claim. The compact
  record is [`results/discriminant_residual_preflight_v1.json`](../results/discriminant_residual_preflight_v1.json).
- The bounded follow-up diagnostic re-derived 30 applicability failures
  (`22` rank, `8` condition), 21 applicable required-gate failures confined to
  RMD cancellation, and 36 cached-parity diagnostic failures. Its pre-frozen
  NC1 direction was supported overall on the finite-condition subset but was
  heterogeneous by optimizer family, so it remains descriptive. A
  high-precision condition-grid fixture supports an operand-aware cancellation
  scale without changing `tau_alg`; the compact record is
  [`results/discriminant_residual_diagnostic_v1.json`](../results/discriminant_residual_diagnostic_v1.json).
- The one allowed Card-13 v11 compliance rerun completed 60 raw/L2 fits on the
  same verified ID-only cache. Applicability remained 30/60; all 30 applicable
  fits passed the required gates and cached Metric Contract v1.2 parity.
  Across all fits, cached parity was a diagnostic with 52 PASS and eight
  theorem-inapplicable FAILED fits. V1 remained immutable, all 60 fit keys were
  conserved, and tail model/fold status did not change. Gate 2 remains
  permanently `INCONCLUSIVE`. The compact record is
  [`results/discriminant_residual_preflight_v2.json`](../results/discriminant_residual_preflight_v2.json).
- A checksum-verified, no-inference D1 reuse analysis recovered exact per-sample
  scores for four frozen historical role configurations. Across six confounded
  cross-policy pairs, median Raw-MD PairOrderChurn was `0.322` on CIFAR-100 and
  `0.359` on MNIST, versus same-policy seed references `0.220` and `0.273`.
  Raw-RMD medians were `0.123` and `0.111`, close to seed references `0.114`
  and `0.098`. This is a `GO` for the fresh churn-first question, not a causal
  coupling result or an RMD-immunity claim. The historical independently
  trained configurations remain confounded and noncausal; preserving them did
  not create fresh scientific evidence. Their byte-exact reproducibility archive
  is stored at
  `hf://buckets/contra333/ICLR_RUN/artifacts/d1_historical_survival/v1/25f9b588f52bf539cc86c0221424f68d4ba846fb9adefe26a2e1f9b3c1d6380a/d1_historical_survival_v1.tar`
  with SHA-256
  `25f9b588f52bf539cc86c0221424f68d4ba846fb9adefe26a2e1f9b3c1d6380a`.
- The historical radial Stage-2 mechanism gate ended `FAILED` on its numerical witness.
  It is closed and is not an active launch dependency.

Historical commands, hashes, and detailed evidence are indexed in
[`history/README.md`](history/README.md). Generated C1--C4 tables and figures are stored
outside Git at the hash-addressed HF archive recorded there.

## Frozen active design

- Central theorem: under the stated full-rank/common-ridge applicability
  conditions, Raw MD and Marginal share the same `S-perp` residual term and RMD
  cancels it. Metric Contract v1.2 pseudoinverse scores remain unchanged and
  require an explicit applicability gate.
- Primary detector attribution is exact within each branch. Cross-branch
  representation interpretation additionally requires ID-only gauge alignment,
  subspace principal angles, and a zero-decay common-frame diagnostic.
- L2 normalization and RMD are separate operations. The theorem is applied
  inside each transform-specific fit; a common residual-sensitivity
  interpretation requires the frozen raw/L2 x MD/RMD interaction and component
  checks. The fixed residual-retention path is diagnostic and never selects a
  best detector parameter.
- For a fixed branch/transform/fit, `(q_perp, P_S x)` is the sample-side
  Gaussian-score interface; branch-specific whitening, `S`, class centers, and
  `B|_S` remain fitted state. ID-train whitening pins `mean(q_perp)` but not
  `Var(q_perp)` or its tail.
- One optional `RtMD` method slot is registered. Its covariance-normalized
  residual-t score, deterministic ID-only two-fold fit, finite-domain/fallback
  rule, comparison panel, and guardrails are frozen. The direct-collision
  subgate is a narrow `PASS` after WDiscOOD, Linderman/DPMM--RMDS, D-KNN, CORE,
  and MaRS boundary checks. The historical ID-only Gate 2 preflight is
  `INCONCLUSIVE`; empirical Gates 3--5 are `NOT_RUN`. It closes on any failed
  gate without changing or stopping the primary paper, Task F, or main fresh
  mechanism training.
- Core preflight rules are frozen: condition-aware algebra tolerances,
  numerical `dim(S)`, rank-sensitive reporting, gauge-aligned angles,
  classifier alignment, the zero-decay common frame, component covariance,
  the raw/L2 x MD/RMD interaction, and the fixed five-point `rho` path.
- `S-perp` is the only fresh primary-channel candidate. This is explicitly
  discovery-informed by the historical two-component result; historical
  preflight can return primary=`none` but cannot promote another component.
- Main: WRN-28-10/CIFAR-10 from-scratch paired trajectories.
- Adam family: the existing 2 x 2 LR/nominal-WD design plus a fixed-total-decay
  anchor interpolation at `coupled_ratio alpha in {0, 0.5, 1}`. The midpoint
  adds five runs, for 41 total; the primary anchor uses five paired seeds and
  the other cells three. The three points test midpoint/interiority and
  monotonic compatibility, not a dose-response curve shape.
- SGDM family: one conventional zero/SGDM/SGDW cell, three seeds, nine runs.
- Epoch-200 `last.pt` is primary; ID-selected `best_val.pt` is a separate control.
- Raw MD `DeltaAUROC` and `PairOrderChurn` are co-primary at the primary endpoint.
- Replication order: ResNet-18/CIFAR-10, ResNet-18/CIFAR-100,
  DenseNet-BC/CIFAR-10, ConvNeXt-Tiny/ImageNet-200.

Exact estimands, checkpoints, geometry channels, theory conditions, claim gates, and
replication rules live only in the active protocol.

## Open before protected OOD

- ID accuracy/NLL/ECE equivalence and Pareto guardrails
- practical OOD margins and seed/power justification
- standardized trajectory uncertainty, multiplicity, and spectral-band boundaries
- exact protected-OOD schedule and launch rule
- CIFAR-100 and ImageNet-200 data contracts; DenseNet/ConvNeXt implementations

## Explicitly not run

- no fixed residual-retention diagnostic, fresh Gate 3, full `RtMD` detector,
  or `RtMD` OOD evaluation; Gate 2 is `INCONCLUSIVE` and Gates 3--5 remain
  `NOT_RUN`
- no completed fresh active-protocol research result or protected OOD; the
  completed training and 310 source exports are inputs to the still-pending
  ID-only evaluation, and the execution-only pilot remains excluded from
  research evidence
- no new architecture/dataset replication
- no comparable-ID, causal decay-coupling, or cross-regime conclusion
