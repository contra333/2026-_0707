# Project Status

Last updated: 2026-07-14

## Current phase

The first optimizer, model, OpenOOD v1.5-aligned CIFAR-10 data/MSP, and
reproducible classifier-training foundations are merged and validated. The
first completed WRN-28-10 SGD seed-0 200-epoch baseline has also been run and
independently validated at pinned training commit
`d3fb1db222e755fe721c78efd0eb52915dcef7fd`. See the
[Issue #14 server validation report](validation/issue14_wrn200_sgd_seed0_server_validation.md).

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

## Documented but not implemented

- `docs/reference_cards/07_optimizer_comparison_hpo_protocol.md` fixes the
  four-optimizer tuned comparison, accuracy matching, pairwise coupling
  controls, search budget, seeds, checkpoints, provenance, and rerun rules.
- `docs/reference_cards/06_feature_ood_detectors.md` fixes the planned DDU
  name, class-wise full unbiased covariance, official adaptive-jitter ladder,
  `logsumexp` ID-like score, and explicit PCA/Diag/L2/Shrinkage post-hoc variant
  boundaries.
- SN on/off is documented as a training ablation and does not rename DDU.
- `docs/reference_cards/08_raw_feature_artifact_contract.md` fixes the future
  deterministic checkpoint-feature cache, provenance, checksum, and
  protected-split authorization contract.
- `docs/reference_cards/09_core_representation_metrics.md` freezes the future
  confirmatory geometry, logit-control, and low-complexity distance/angle
  panel. Covariance effective rank remains explicitly `UNSPECIFIED` rather
  than receiving an unaudited formula.
- These decisions are documentation only. No DDU detector code, test, config,
  checkpoint evaluation, or OOD result has been implemented or validated.

## Still missing

- Hyperparameter optimization
- Penultimate feature extraction pipeline
- Geometry and Neural Collapse metrics
- Feature-based OOD detector implementations, including DDU and its planned
  PCA/Diag/L2/Shrinkage variants
- Multi-seed experiment orchestration

## Active next phase

Implement the documented optimizer-comparison orchestration in a separate
bounded Issue, including deterministic frozen trial tables, deferred ID-test
evaluation, independent-GPU scheduling, provenance, and failure accounting.
The Issue #10 CUDA runs remain infrastructure validation; the Issue #14 run is
the single-seed SGD baseline. Neither is optimizer-comparison, geometry,
Neural Collapse, or OOD-detector evidence.

The DDU reference-card decision does not replace that next-phase Issue and does
not authorize detector implementation without a separately bounded task.

## Known workflow maintenance

- `docs/reference_cards/03_architecture_implementation_checklist.md` describes a historical first implementation task whose listed models are now implemented. Treat its durable API and validation rules as useful context, but do not treat its one-time scope as the current active task.
- New one-time implementation tasks should be created as GitHub Issues rather than new permanent checklists under `docs/reference_cards/`.

## Blockers and unknowns

- Only one optimizer and one seed have a completed long-run baseline. No
  optimizer-comparison or multi-seed conclusion is currently supported.
- Optimizer-comparison orchestration and execution require separately bounded
  Issues before further long-running experiments.
- The canonical DDU shrinkage estimator and PCA component-selection rule remain
  literature-backed decisions for a later implementation Issue.

## Update rule

Update this file only when the project phase, validated foundation, major blocker, or next-phase priority changes. Do not use it as a daily log or duplicate Pull Request descriptions.
