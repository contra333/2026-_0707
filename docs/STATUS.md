# Project Status

Last updated: 2026-07-15

## Current phase

The first optimizer, model, OpenOOD v1.5-aligned CIFAR-10 data/MSP, and
reproducible classifier-training foundations are merged and validated. The
first completed WRN-28-10 SGD seed-0 200-epoch baseline has also been run and
independently validated at pinned training commit
`d3fb1db222e755fe721c78efd0eb52915dcef7fd`. See the
[Issue #14 server validation report](validation/issue14_wrn200_sgd_seed0_server_validation.md).
Issue #22 and merged PR #23 added the deterministic optimizer-HPO orchestration
foundation and completed its bounded department-server smoke. Production
discovery HPO has not started.

## Validated or implemented

- Optimizer reference semantics are documented.
- Shared weight-decay parameter-group policy is implemented.
- Optimizer and parameter-group tests exist.
- Common model API for logits and penultimate features is documented.
- `toy_cifar_cnn` is implemented as an API smoke-test fixture only.
- `resnet18` is implemented as a research backbone.
- `wrn28_10` is implemented as a research backbone.
- The OpenOOD v1.5-aligned CIFAR-10 dataset and evaluation contract is documented and implemented.
- Official released OpenOOD imglists and all eight required archives were validated on the department server.
- Actual-data manifest validation confirmed fixed split counts, image-path existence, sample-ID uniqueness, and label ranges.
- A bounded CUDA WRN-28-10 MSP vertical slice completed with random-model output marked as infrastructure-only.
- Pull Request #11 merged the Issue #10 WRN-28-10 training protocol,
  deterministic loader state, scheduler boundaries, checkpoints, resume, and
  run artifacts. Complete tests and bounded actual-data CUDA SGD, resume,
  Adam, and AdamW validation passed on the department server.
- Issue #14 produced and independently revalidated the first complete
  WRN-28-10 SGD seed-0 200-epoch classifier baseline, including all fixed
  snapshots and full ID validation/test recomputation.
- Issue #22 and PR #23 implemented versioned study/trial/attempt records,
  canonical configuration hashing, deterministic ranking and freeze records,
  code-enforced deferred ID-test selection mode, independent single-GPU trial
  scheduling, and attempt-preserving failure/retry accounting.
- The protocol-v1.1 discovery bundle is frozen at 16 rows per optimizer and 64
  rows total. Its manifest hash is
  `5b2f915d2337924cbb67077216bbcc4a49835f7927cea96c45004f0b76576b54`.
- The Issue #22 department-server validation passed the complete 167-test suite,
  actual OpenOOD membership verification, and a bounded two-GPU/two-trial,
  one-epoch smoke with checkpoint, checksum, GPU-identity, and deferred-ID-test
  validation. The smoke consumed no production discovery slot.

## Documented but not executed or implemented

- `docs/reference_cards/07_optimizer_comparison_hpo_protocol.md` fixes the
  four-optimizer tuned comparison, accuracy matching, pairwise coupling
  controls, search budget, seeds, checkpoints, provenance, and rerun rules.
  Its orchestration foundation is implemented, but discovery, confirmation,
  final training, and the resulting scientific comparisons are not executed.
- `docs/reference_cards/06_feature_ood_detectors.md` fixes the planned DDU
  name, class-wise full unbiased covariance, official adaptive-jitter ladder,
  `logsumexp` ID-like score, and explicit PCA/Diag/L2/Shrinkage post-hoc variant
  boundaries.
- SN on/off is documented as a training ablation and does not rename DDU.
- These decisions are documentation only. No DDU detector code, test, config,
  checkpoint evaluation, or OOD result has been implemented or validated.

## Still missing

- Production 64-run discovery HPO and top-3 freeze
- Confirmation, tuned-winner freeze, accuracy matching, pair controls, and
  final five-seed classifier execution
- Penultimate feature extraction pipeline
- Geometry and Neural Collapse metrics
- Feature-based OOD detector implementations, including DDU and its planned
  PCA/Diag/L2/Shrinkage variants

## Active next phase

Issue #25 is the bounded A3a task for executing the frozen 64-run discovery
study only. Before production launch it requires a fresh clean-SHA,
dataset-membership, GPU UUID/availability, concurrency, storage/inode,
artifact-root, retention, and backup preflight. Every assigned discovery slot
uses training seed `0` for 200 epochs. Only after all 64 slots are terminal may
the deterministic per-optimizer top-3 freeze be created.

Confirmation seeds, final seeds, accuracy matching, pair-control execution,
ID-test release, OOD evaluation, feature extraction, geometry/Neural Collapse,
and detector work remain outside Issue #25 and require later bounded tasks. The
Issue #10 CUDA runs, Issue #14 baseline, and Issue #22 orchestration smoke are
infrastructure or single-run evidence, not optimizer-comparison results.

Issue #24 is a documentation-only Track B task for raw-feature artifact and
representation-metric definitions. It may proceed independently and does not
authorize or block Track A discovery execution.

The DDU reference-card decision does not replace that next-phase Issue and does
not authorize detector implementation without a separately bounded task.

## Known workflow maintenance

- `docs/reference_cards/03_architecture_implementation_checklist.md` describes a historical first implementation task whose listed models are now implemented. Treat its durable API and validation rules as useful context, but do not treat its one-time scope as the current active task.
- New one-time implementation tasks should be created as GitHub Issues rather than new permanent checklists under `docs/reference_cards/`.

## Blockers and unknowns

- Only one optimizer and one seed have a completed long-run baseline. No
  optimizer-comparison or multi-seed conclusion is currently supported.
- Production GPU availability, storage/inode capacity, artifact retention and
  backup behavior, and optimizer-specific 200-epoch wall time remain unverified
  until the fresh Issue #25 server preflight.
- Confirmation and every later comparison phase require separate bounded
  Issues after discovery is complete and reviewed.
- The canonical DDU shrinkage estimator and PCA component-selection rule remain
  literature-backed decisions for a later implementation Issue.

## Update rule

Update this file only when the project phase, validated foundation, major blocker, or next-phase priority changes. Do not use it as a daily log or duplicate Pull Request descriptions.
