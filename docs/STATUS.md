# Project Status

Last updated: 2026-08-04

## Current phase

The first optimizer, model, OpenOOD v1.5-aligned CIFAR-10 data/MSP, and
reproducible classifier-training foundations are merged and validated. The
first completed WRN-28-10 SGD seed-0 200-epoch baseline has also been run and
independently validated at pinned training commit
`d3fb1db222e755fe721c78efd0eb52915dcef7fd`. See the
[Issue #14 server validation report](validation/issue14_wrn200_sgd_seed0_server_validation.md).
Issue #22 and merged PR #23 added the deterministic optimizer-HPO orchestration
foundation and completed its bounded department-server smoke. Issue #35 and merged PR #36
replaced the current-tree v1.1 random-search path with the v1.2 deterministic grid,
`C1`-`C4` role freeze, seed/pair-control reuse plan, and protected-evidence
gate. The active classifier data configuration uses
`oge_cifar10_holdout_v1`; its practical runtime, numerical-policy,
actual-data loader, and one-epoch smoke checks passed on `curie`,
`precision_medicine`, and `lise` at clean execution SHA
`e9bfde43bb40f3ea2a6a11da9da86178049ecc40`. Issue #37 and merged PR #38
completed the three-host practical acceptance ledger. Benchmark-grade Conda/cache parity,
DataLoader throughput experiments, cross-GPU parity gates, and immutable UUID
shards are not production prerequisites. On 2026-07-28, an owner-authorized
`fast path` task bounded the seed-0 production execution directly on `main`.
The committed exact-once assignment uses `curie` 14 cells, `lise` 8 cells, and
`precision_medicine` 14 cells. A follow-up owner `fast path` fixes
`grid-sgd-06` as a concurrently running operational sentinel rather than a
blocking gate; all 36 cells remain pinned to production Git SHA
`0d30054b38f0dc7a513c3eacc5c4e5435670fc4d`. The 36 cells subsequently
completed, their uploaded metadata and checksums passed the Issue #47 integrity
gate, and C1-C4 were frozen from epoch-200 `last.pt` ID-validation metrics.
Issue #49 and PR #50 completed the exact-once 13/10/7 follow-up at production
Git SHA `3556841340e6f6b92782af045ed4a468e6e271bd`. All 30 new rows are
epoch-200 terminal, independently validated, and uploaded with zero deletes.
Together with 12 reused seed-0 rows, they form 42 unique `(config_hash, seed)`
identities across 14 configurations. The checksum-sealed aggregate retains
per-seed validation values, mean/sample SD, paired-control differences, and
the seed-0 selection-bias disclosure.

## Validated or implemented

- Optimizer reference semantics are documented.
- Shared weight-decay parameter-group policy is implemented.
- Optimizer and parameter-group tests exist.
- Common model API for logits and penultimate features is documented.
- `toy_cifar_cnn` is implemented as an API smoke-test fixture only.
- `resnet18` is implemented as a research backbone.
- `wrn28_10` is implemented as a research backbone.
- The OpenOOD v1.5-aligned CIFAR-10 dataset and evaluation contract is documented and implemented.
- Issue #33 and merged PR #34 implement the deterministic project CIFAR-10 ID
  membership: official train is split into class-balanced 45k/5k using the
  frozen SHA-256 rule, official test is reconstructed as the verified
  OpenOOD 1k/9k union, and all three imglists plus a 60,000-row provenance
  manifest are committed. The active training config uses
  `oge_cifar10_holdout_v1`, WRN `msr_fan_in`, and train-only reflection
  padding. The committed membership, image paths, and bounded loaders passed on
  all three approved hosts at the clean Issue #37 execution SHA.
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
- Issue #35 and merged PR #36 implement protocol v1.2 as three deterministic
  12-cell tables, with no sampler. The frozen manifest binds row/table/grid hashes to the
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
- Issue #37 and merged PR #38 record the practical runtime contract and add the
  effective FP32/TF32/cuDNN policy, Python executable, hostname, driver, and
  GPU metadata to each training run's `environment.json`. The existing study
  runner remains responsible for bounded one-epoch smokes; no new
  preflight/throughput/shard gate is inserted into production execution. See
  `docs/validation/issue37_practical_runtime_status.md`.
- Issue #37's `precision_medicine` runtime candidate passed `pip check`, the
  complete 175-test suite, bounded CUDA/cuDNN probes, and the required numerical
  policy checks. Its committed-holdout, bounded-loader, and actual-data
  one-epoch smoke then passed at clean Git
  `e9bfde43bb40f3ea2a6a11da9da86178049ecc40`, with ID-test deferred and no
  protected-evidence artifact.
- Issue #37's `lise` validation passed at clean Git
  `e9bfde43bb40f3ea2a6a11da9da86178049ecc40`: the hash-locked Python 3.11.9,
  PyTorch `2.5.1+cu121`, TorchVision `0.20.1+cu121`, and CUDA 12.1 runtime
  passed `pip check`, the complete 175-test suite, bounded A5000 CUDA/cuDNN
  probes, committed 45k/5k/10k holdout path and loader checks, and one
  existing-runner actual-data epoch on an idle GPU. The smoke deferred ID-test
  and created no protected-evidence artifact.
- Issue #37 current-scope `curie` validation passed at the same clean Git SHA
  with the same 35-entry runtime/test lock surface and effective numerical
  policy: `pip check`, the complete 175-test suite, bounded A5000 CUDA/cuDNN
  probes, committed holdout and bounded loaders, and one actual-data epoch all
  passed on idle GPU 0. The smoke completed 352 optimizer steps, strict-reloaded
  both checkpoints, deferred ID-test, and created no evaluation artifact.
- Issue #47 verified 12/12 stage-level metadata sidecars and 36/36 trial-record
  sidecars from `contra333/ICLR_RUN`, exact-once host assignment, schema,
  grid/dataset/config identities, production Git SHA, terminal epoch-200
  metrics, deferred ID-test state, and protected-field absence. The official
  C1-C4 freeze hash is
  `fdf67c1184abc489542ca64cad2410ff38aa816acb1e9e5289d60461600373fa`;
  see
  `docs/validation/seed0_20260728_grid_role_freeze.md`.
- Issue #49 verified 30/30 epoch-200 follow-up records as `curie=13`,
  `lise=10`, and `precision_medicine=7`, including checkpoint and metadata
  checksums, deferred protected fields, three `REMOTE_COMPLETE.json` markers,
  and zero-delete uploads. The committed 42-row aggregate contains 12 reused
  seed-0 rows and 30 new rows without duplicate identities; see
  `docs/validation/issue49_followup_execution.md`.

## Documented but not executed

- `docs/reference_cards/07_optimizer_comparison_hpo_protocol.md` fixes the
  deterministic grid, C1-C4 selection, pairwise coupling controls, budget,
  seeds, checkpoints, provenance, and rerun rules. The seed-0 v1.2 grid,
  C1-C4 freeze, role replication, and pair controls are complete; downstream
  protected evaluation has not been executed.
- `docs/reference_cards/06_feature_ood_detectors.md` fixes the SN-off
  `GDA-ClassDensity` name, class-wise full unbiased covariance, official
  adaptive-jitter ladder, empirical-prior `logsumexp` score, and the boundary
  that reserves `DDU` for a future spectral-normalization training ablation.
- `docs/reference_cards/08_raw_feature_artifact_contract.md` fixes the future
  deterministic checkpoint-feature cache, provenance, checksum, and
  protected-split authorization contract.
- `docs/reference_cards/09_core_representation_metrics.md` freezes the future
  confirmatory geometry, logit-control, and low-complexity distance/angle
  panel, including Moore-Penrose NC1 and separate covariance entropy-rank,
  trace-to-top-rank, and participation-ratio definitions.
- `docs/reference_cards/02_architectures.md` now fixes the research lineup
  (2026-07-23): WRN-28-10/CIFAR-10 main (full protocol), ResNet-18 on
  CIFAR-10 and CIFAR-100 plus VGG-16-BN/CIFAR-10 (reduced protocol), and a
  pilot-gated from-scratch `vit_small`/CIFAR-10 arm with recorded fallbacks.
  `vgg16` and `vit_small` remain unimplemented.
- `docs/reference_cards/10_optimizer_grid_literature_anchors.md` maps every
  v1.2 grid value and lineup row to pinned sources or labeled project
  judgment, and records prior-work positioning found in the 2026-07-23
  literature pass.
- `docs/reference_cards/11_metric_contract_v1_2.md` is the authoritative
  WRN-28-10/CIFAR-10 metric dictionary. It fixes the paper formulas, reporting
  tiers, checkpoint/split roles, artifact keys, degeneracy rules, source pins,
  and validation oracles before implementation.
- These decisions are documentation only. No GDA-ClassDensity, DDU, expanded
  metric code, checkpoint evaluation, or OOD result has been implemented or
  validated by the metric-contract task.

## Still missing

- Penultimate feature extraction pipeline
- Geometry and Neural Collapse metrics
- Feature-based OOD detector implementations, including GDA-ClassDensity and
  any future DDU/SN ablation

## Active next phase

Issue #49's role replication and pair controls are complete. The next phase
requires a separately bounded task for protected feature extraction and
checkpoint evaluation using only the frozen role config/seed identities.
ID-test, OOD, geometry/Neural Collapse, and detector evaluation were not
executed by Issue #49. Issue #37 and PR #38 remain readiness evidence rather
than production evidence.
The Issue #10 CUDA runs remain infrastructure validation; the Issue #14 run is
the single-seed SGD baseline. Neither is optimizer-comparison, geometry,
Neural Collapse, or OOD-detector evidence.

The metric-contract and GDA/DDU naming decisions do not replace that next-phase
Issue and do not authorize detector implementation without a separately
bounded task.

## Known workflow maintenance

- `docs/reference_cards/03_architecture_implementation_checklist.md` describes a historical first implementation task whose listed models are now implemented. Treat its durable API and validation rules as useful context, but do not treat its one-time scope as the current active task.
- New one-time implementation tasks should be created as GitHub Issues rather than new permanent checklists under `docs/reference_cards/`.

## Blockers and unknowns

- Protected feature extraction and checkpoint evaluation remain unimplemented
  and require their own bounded Issue and validation evidence.
- DDU post-hoc shrinkage and PCA choices remain outside metric-contract v1.2
  and require a separately bounded future ablation if the project retains them.

## Update rule

Update this file only when the project phase, validated foundation, major blocker, or next-phase priority changes. Do not use it as a daily log or duplicate Pull Request descriptions.
