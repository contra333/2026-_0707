# Project status

Last updated: 2026-08-12

## Current phase

The active paper protocol is
[`13_active_paper_protocol.md`](reference_cards/13_active_paper_protocol.md).
The active design is v11. A bounded historical D1 survival check is complete
and supports the pair-ranking-multiplicity framing as discovery evidence. The
Task B theory/estimator/literature lock is complete: the core numerical and
subspace rules, discovery-informed `S-perp` hypothesis, and exact conditional
`RtMD` specification are frozen in Card 13. The bounded existing-cache
discriminant--residual and ID-only residual-tail CPU preflight is complete. Its
execution passed, but Gate 2 is `INCONCLUSIVE`: Card 13 has no frozen aggregate
decision rule and only 15 of 30 fits per transform had numerically applicable
two-fold geometry. Fresh paired training has not started.

An ID-only archive diagnostic has now re-derived the 60-fit failure layers,
checksum-verified all 30 NC1 inputs, and found synthetic forward-error support
for treating the RMD-cancellation denominator as a Card-13 wording ambiguity.
This authorizes one prospective contract clarification and CPU preflight v2;
it does not re-adjudicate Gate 2 or activate RtMD.
Card 13 v11 now records that clarification, makes historical Gate 2 permanently
`INCONCLUSIVE`, transfers its unresolved plausibility question to a fresh
pre-registered Gate 3, and requires explicit owner approval before either a
fresh GPU launch or protected-OOD access.

The single authorized CPU preflight v2 is complete. The operand-aware
cancellation scale left theorem applicability unchanged at 30 of 60 fits and
made all 30 applicable fits pass every required numerical gate, up from 9 in
v1. The frozen two-fold tail and pair payloads are byte-identical to v1, and a
second fixed-thread execution reproduced every v2 payload hash. Historical
Gate 2 remains permanently `INCONCLUSIVE`; this result only makes the frozen
estimator eligible for a prospective fresh Gate-3 addendum.

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
  implemented. WRN multi-depth taps are not.
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
  fits passed the required gates, cached Metric Contract v1.2 parity, and RMD
  cancellation. V1 remained immutable, all 60 fit keys were conserved, and
  tail model/fold status did not change. Gate 2 remains permanently
  `INCONCLUSIVE`. The compact record is
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
  gate without changing the primary paper.
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

- no fixed residual-retention diagnostic, full `RtMD` detector, or `RtMD` OOD
  evaluation; Gate 2 is `INCONCLUSIVE` and Gates 3--5 remain `NOT_RUN`
- no fresh active-protocol training, alpha midpoint run, GPU work, checkpoint
  inference, or protected OOD
- no new architecture/dataset replication
- no comparable-ID, causal decay-coupling, or cross-regime conclusion
