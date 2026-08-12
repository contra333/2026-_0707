# Project status

Last updated: 2026-08-12

## Current phase

The active paper protocol is
[`13_active_paper_protocol.md`](reference_cards/13_active_paper_protocol.md).
The next task is the v9 theory/estimator/literature lock, including the exact
conditional `RtMD` derivation and close-the-slot rule, followed by a bounded
implementation of the existing-cache discriminant--residual and ID-only
residual-tail preflight. Fresh paired training has not started.

The planned chain is:

```text
same initialization and data stream
→ coupled / decoupled / zero-decay trajectories
→ update and non-affine representation divergence
→ discriminant subspace S / residual subspace S-perp
→ S-perp / parallel-Marginal / RMD score attribution
→ raw/L2 x MD/RMD interaction and fixed residual-retention path
→ exact pair gain, loss, churn, and net AUROC
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
- One optional `RtMD` method slot is registered. It is secondary, must freeze
  its covariance/scatter convention, ID-only fit, activation gates, comparison
  panel, and guardrails before the tail preflight/protected OOD, and closes on
  any failed gate without changing the primary mechanism paper.
- Main: WRN-28-10/CIFAR-10 from-scratch paired trajectories.
- Adam family: 2 x 2 LR/nominal-WD design, 36 total runs; the primary anchor uses five
  seeds and the other cells three.
- SGDM family: one conventional zero/SGDM/SGDW cell, three seeds, nine runs.
- Epoch-200 `last.pt` is primary; ID-selected `best_val.pt` is a separate control.
- Raw MD `DeltaAUROC` and `PairOrderChurn` are co-primary at the primary endpoint.
- Replication order: ResNet-18/CIFAR-10, ResNet-18/CIFAR-100,
  DenseNet-BC/CIFAR-10, ConvNeXt-Tiny/ImageNet-200.

Exact estimands, checkpoints, geometry channels, theory conditions, claim gates, and
replication rules live only in the active protocol.

## Open before protected OOD

- full-rank/common-ridge applicability, numerical-rank/condition/reconstruction
  tolerances, and the separately named common-ridge diagnostic
- branch-alignment and common-frame summaries; RMD/LDA, WDiscOOD, CORE, MaRS,
  robust/t-Mahalanobis, NECO, ViM, Neural-Collapse Mahalanobis, and
  projection-filtering novelty audit
- component variance--covariance/norm summaries and the normalization x
  cancellation interaction estimand
- exact `RtMD` likelihood parameterization, ID-only split/estimator/fallback,
  historical and fresh activation thresholds, churn estimand, performance and
  far-OOD guardrails, and replication/closure rule
- ID accuracy/NLL/ECE equivalence and Pareto guardrails
- practical OOD margins and seed/power justification
- standardized trajectory uncertainty, multiplicity, and spectral-band boundaries
- exact protected-OOD schedule and launch rule
- CIFAR-100 and ImageNet-200 data contracts; DenseNet/ConvNeXt implementations

## Explicitly not run

- no `S`/`S-perp` analysis interface, historical discriminant--residual or
  residual-tail preflight, fixed residual-retention diagnostic, or `RtMD`
  implementation/evaluation
- no fresh active-protocol training, GPU work, checkpoint inference, or protected OOD
- no new architecture/dataset replication
- no comparable-ID, causal decay-coupling, or cross-regime conclusion
