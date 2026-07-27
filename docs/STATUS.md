# Project Status

Last updated: 2026-07-27

## Current phase

The first optimizer, model, OpenOOD v1.5-aligned CIFAR-10 data/MSP, and
reproducible classifier-training foundations are merged and validated. The
first completed WRN-28-10 SGD seed-0 200-epoch baseline has also been run and
independently validated at pinned training commit
`d3fb1db222e755fe721c78efd0eb52915dcef7fd`. See the
[Issue #14 server validation report](validation/issue14_wrn200_sgd_seed0_server_validation.md).
Issue #22 and merged PR #23 added the deterministic optimizer-HPO orchestration
foundation and completed its bounded department-server smoke. No
protocol-v1.2 36-cell production grid has started. Issue #35 replaces the
current-tree v1.1 random-search path with the v1.2 deterministic grid,
`C1`-`C4` role freeze, seed/pair-control reuse plan, and protected-evidence
gate. The active classifier data configuration uses
`oge_cifar10_holdout_v1`; its department-server path/loader smoke and the
common-server numerical preflight are still required before production.
Issue #37 now provides fail-closed capture/comparison, throughput-decision,
parity, and host-shard tooling, but desktop WSL cannot resolve the department
hosts. The common dependency lock, `precision_medicine` hostname, server
measurements, parity, and smokes therefore remain explicitly unresolved.

## Validated or implemented

- Optimizer reference semantics are documented.
- Shared weight-decay parameter-group policy is implemented.
- Optimizer and parameter-group tests exist.
- Common model API for logits and penultimate features is documented.
- `toy_cifar_cnn` is implemented as an API smoke-test fixture only.
- `resnet18` is implemented as a research backbone.
- `wrn28_10` is implemented as a research backbone.
- The OpenOOD v1.5-aligned CIFAR-10 dataset and evaluation contract is documented and implemented.
- Issue #33 locally implements the deterministic project CIFAR-10 ID
  membership: official train is split into class-balanced 45k/5k using the
  frozen SHA-256 rule, official test is reconstructed as the verified
  OpenOOD 1k/9k union, and all three imglists plus a 60,000-row provenance
  manifest are committed. The active training config uses
  `oge_cifar10_holdout_v1`, WRN `msr_fan_in`, and train-only reflection
  padding. Local deterministic/hash tests are not a substitute for the pending
  department-server image-path and loader smoke.
- Official released OpenOOD imglists and all eight required archives were validated on the department server.
- Actual-data manifest validation confirmed fixed split counts, image-path existence, sample-ID uniqueness, and label ranges.
- A bounded CUDA WRN-28-10 MSP vertical slice completed with random-model output marked as infrastructure-only.
- Pull Request #11 merged the Issue #10 WRN-28-10 training protocol,
  deterministic loader state, scheduler boundaries, checkpoints, resume, and
  run artifacts. Complete tests and bounded actual-data CUDA SGD, resume,
  Adam, and AdamW validation passed on the department server.
- Issue #14 produced and independently revalidated the first complete
  WRN-28-10 SGD seed-0 200-epoch classifier baseline, including all fixed
  snapshots and full ID validation/test recomputation. That run predates the
  2026-07-27 initialization correction and used `fan_out`; its checkpoints and
  report are retained unchanged as a historical single-seed baseline under the
  prior protocol and are excluded from protocol-v1.2 aggregation.
- WRN-28-10 convolution initialization was audited against the pinned official
  repository on 2026-07-27 and corrected from `fan_out` to `fan_in`, matching
  both `models/utils.lua` `MSRinit` and `pytorch/utils.py`. The policy is now
  carried by the `init_policy` model-config field, materialized into the
  resolved config, and included in the canonical scientific config hash.
  Reference card 02 records the full architecture-side deviation audit.
- Issue #22 and PR #23 implemented versioned study/trial/attempt records,
  canonical configuration hashing, deterministic ranking and freeze records,
  code-enforced deferred ID-test selection mode, independent single-GPU trial
  scheduling, and attempt-preserving failure/retry accounting.
- The Issue #22 department-server validation passed the complete 167-test suite,
  actual OpenOOD membership verification, and a bounded two-GPU/two-trial,
  one-epoch smoke with checkpoint, checksum, GPU-identity, and deferred-ID-test
  validation. The smoke consumed no production study slot.
- Issue #35 implements protocol v1.2 as three deterministic 12-cell tables,
  with no sampler. The frozen manifest binds row/table/grid hashes to the
  45k/5k/10k membership and split-manifest hashes. Selection requires all 36
  seed-0 cells to be terminal, implements C1-C4 including widening and
  absent/unresolved states, freezes the result identity, deduplicates role and
  pair-control follow-ups, and gates protected evidence to frozen role
  config/seed identities.
- Historical decision log: an earlier 64-run random-search study exists only
  on the department server, used the old 50k/1k/9k split, and is not imported
  or used for v1.2 selection, baseline, table, ranking, analysis, or numeric
  justification. Its executable config, tables, sampler, default path, and
  golden hashes are absent from the current tree.
- Issue #37 locally implements versioned server-preflight records, exact
  environment/numerical comparison, legacy and PyTorch 2.9+ TF32 API capture,
  the common 15% DataLoader worker gate, deterministic 5/2/5-capable balanced
  host shards, synthetic A5000/A6000 parity comparison, and study-runner gates
  for frozen dependency/preflight/throughput/shard/GPU identities. The
  dependency lock is intentionally pending and no server result is claimed;
  see `docs/validation/issue37_server_preflight_blocked.md`.

## Documented but not executed

- `docs/reference_cards/07_optimizer_comparison_hpo_protocol.md` fixes the
  deterministic grid, C1-C4 selection, pairwise coupling controls, budget,
  seeds, checkpoints, provenance, and rerun rules. The v1.2 definition and
  selection implementation exist, but no v1.2 grid, role-replication,
  pair-control, or downstream run has been executed.
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
- `docs/reference_cards/02_architectures.md` now fixes the research lineup
  (2026-07-23): WRN-28-10/CIFAR-10 main (full protocol), ResNet-18 on
  CIFAR-10 and CIFAR-100 plus VGG-16-BN/CIFAR-10 (reduced protocol), and a
  pilot-gated from-scratch `vit_small`/CIFAR-10 arm with recorded fallbacks.
  `vgg16` and `vit_small` remain unimplemented.
- `docs/reference_cards/10_optimizer_grid_literature_anchors.md` maps every
  v1.2 grid value and lineup row to pinned sources or labeled project
  judgment, and records prior-work positioning found in the 2026-07-23
  literature pass.
- These decisions are documentation only. No DDU detector code, test, config,
  checkpoint evaluation, or OOD result has been implemented or validated.

## Still missing

- Department-server verification of the committed
  `oge_cifar10_holdout_v1` lists against the assembled image root and a bounded
  loader smoke
- Execution of the v1.2 grid, the `C1`-`C4` role freeze, three-seed role
  replication, and the pair controls
- Common environment installation and three-host preflight completion,
  DataLoader measurements, parity, current-UUID shard generation, and
  per-host one-epoch smoke
- Penultimate feature extraction pipeline
- Geometry and Neural Collapse metrics
- Feature-based OOD detector implementations, including DDU and its planned
  PCA/Diag/L2/Shrinkage variants

## Active next phase

Complete separate bounded Issues for the three-server common environment and
measured TF32 policy, DataLoader throughput decision, A5000/A6000 parity,
immutable mixed-GPU host shards, and production preflight. Only after those
gates pass may the execution Issue start the 200-epoch SGD canary and the
remaining frozen grid.
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
- Optimizer-comparison server hardening and execution require separately
  bounded Issues before long-running experiments.
- Production GPU availability and identity, storage and inode capacity,
  artifact retention and backup behavior, and optimizer-specific 200-epoch wall
  time remain unverified until a fresh server preflight.
- The canonical DDU shrinkage estimator and PCA component-selection rule remain
  literature-backed decisions for a later implementation Issue.

## Update rule

Update this file only when the project phase, validated foundation, major blocker, or next-phase priority changes. Do not use it as a daily log or duplicate Pull Request descriptions.
