# Project Context

## Research objective

This repository supports experiments on the causal chain:

```text
optimizer / training rule
→ penultimate representation geometry
→ detector score behavior
→ ID/OOD reliability
```

The active v2 question is whether a fixed feature-based OOD readout is
practically non-invariant to a paired training-rule change at comparable ID
utility, which detector-formula-relevant geometry channel changes ID--OOD score
overlap, and whether that chain reproduces in a second dataset--architecture
regime. The target evidence is paired intervention and score-level diagnosis,
not an optimizer leaderboard or a scalar geometry--AUROC correlation alone.

## Repository as the source of truth

This repository is authoritative for:

- code and tests;
- implementation and experiment semantics;
- active task specifications;
- experiment configurations and metadata;
- validation records and pull-request history.

ChatGPT conversations, Work outputs, Codex sessions, and local notes are supporting interfaces. A decision that changes code behavior, experimental meaning, or validation requirements must be written back into this repository before it is treated as authoritative.

## Current scope

Implemented foundation:

- optimizer semantics and shared parameter-group policy;
- optimizer and parameter-group tests;
- model API for logits and penultimate features;
- `toy_cifar_cnn`, `resnet18`, and `wrn28_10` model endpoints;
- deterministic `oge_cifar10_holdout_v1` 45k/5k/10k CIFAR-10 ID membership,
  OpenOOD v1.5-compatible OOD roles, imglist loaders, manifests, and
  preprocessing;
- bounded MSP OOD inference and metric infrastructure;
- reproducible classifier training, scheduling, checkpoint/reload, epoch-boundary resume, and run artifacts;
- bounded actual-data CUDA validation for the OpenOOD data/evaluation and WRN-28-10 training paths;
- deterministic study/trial/attempt orchestration for the WRN-28-10 optimizer
  HPO protocol v1.2;
- code-enforced deferred ID-test selection mode, deterministic ranking/freeze
  logic, independent single-GPU trial scheduling, and attempt-preserving
  failure/retry accounting;
- deterministic 36-cell grid generation and hashing, C1-C4 role freezing,
  deduplicated role/pair-control reuse planning, and protected-evidence
  authorization by frozen config/seed identity;
- bounded actual-data, two-GPU orchestration smoke validation with checkpoint,
  checksum, GPU-identity, and artifact-provenance verification;
- a practical three-host gate with one hash-locked runtime/test distribution
  set, an agreeing measured FP32/TF32/cuDNN policy, committed 45k/5k/10k
  data-path and bounded-loader checks, and one actual-data smoke epoch per
  approved host;
- completed execution and metadata-integrity validation of all 36 seed-0
  protocol-v1.2 grid cells across the three approved hosts, plus the immutable
  C1-C4 role freeze and deduplicated follow-up plan;
- completed 30-row seed 1/2 role-replication and pair-control execution,
  three independently verified no-delete uploads, and the checksum-sealed
  12-reused-plus-30-new aggregate with per-seed and sample-SD summaries;
- Metric Contract v1.2 evaluation modules for deterministic checkpoint feature
  extraction, ID/calibration metrics, OOD scores, representation geometry,
  explicit numerical status, canonical artifact records, and bounded cache
  execution, with tiny-fixture/parity/failure/invariance tests and a clean
  non-protected Curie checkpoint smoke recorded in
  `docs/validation/issue53_metric_runtime_curie.md`.
- checksum-addressed planning for the protected Metric Contract v1.2 run:
  42 source config/seed identities, 30 authorized role identities, 12 explicit
  pair-control-only exclusions, 60 separate checkpoint jobs, and deterministic
  source-local 20/20/20 assignment;
- completed protected Metric Contract v1.2 execution at scientific evaluator
  SHA `c38b09694be88aa74de0741b39e9d3ba0d6ff61a`: 60/60 checkpoint bundles and
  three host operational shards are `REMOTE_VERIFIED`, and the checksum-valid
  central aggregate contains separate `last` primary and `best_val` control
  rows for all 30 authorized training identities and seeds `{0,1,2}`. See
  `docs/validation/issue57_metric_evaluation_execution.md`.
- completed checksum-bounded local C1-C4 analysis of that aggregate, with
  `last.pt` primary and `best_val.pt` control kept separate, all central
  seed-0 and three-seed scalars exported, six OOD datasets partitioned into
  independent tables, 19-detector appendix coverage, 11-detector paper panels,
  seed-matched descriptive deltas, and dataset-specific exploratory
  geometry-OOD associations. See
  `docs/analysis/metric_contract_v1_2_c1_c4/`.

Documented and implemented result boundaries:

- `GDA-ClassDensity` as the SN-off class-wise full-covariance Gaussian density
  readout using the pinned adaptive jitter and class-prior-weighted `logsumexp`;
- the complete DDU name reserved for a future spectral-normalization training
  ablation, with optional post-hoc variants outside metric-contract v1.2;
- the WRN-28-10/CIFAR-10 Metric Contract v1.2 formulas, paper names, artifact
  keys, numerical failure semantics, validation oracles, and runtime entrypoints.

Active but not yet executed research contract:

- [`reference_cards/12_fixed_readout_intervention_protocol_v2.md`](reference_cards/12_fixed_readout_intervention_protocol_v2.md)
  freezes WRN-28-10/CIFAR-10 as the main regime and
  ResNet-18/CIFAR-100 as replication;
- Adam/AdamW is the primary shared-prefix family, SGDM/SGDW is the
  conventional-family control, and the replication adds a conventional SGDM
  reference;
- the v2 detector panel and score-overlap mechanism requirements are fixed,
  while switch epoch, decay dose, practical margins, power, and the focal
  channel/transform are explicitly Stage-2-gated decisions;
- the completed v1.2 artifacts are discovery inputs only and are never
  retroactively labeled causal confirmation.

Not part of the validated foundation unless added by a later bounded task:

- a completed shared-prefix fork implementation or paired causal optimizer-rule
  conclusion;
- the CIFAR-100 dataset/OOD contract and ResNet-18 replication result;
- a dataset-pooled geometry-OOD association;
- final manuscript prose and final main/appendix table selection;
- spectral-normalization training and any future DDU ablation;
- protected research-result evaluation for any architecture, dataset, or
  checkpoint population outside the completed WRN-28-10/CIFAR-10 inventory.

## Document roles

- `AGENTS.md`: mandatory entry point and agent operating rules.
- `docs/PROJECT_CONTEXT.md`: stable research objective and repository role.
- `docs/WORKFLOW.md`: end-to-end human/AI work process.
- `docs/STATUS.md`: current validated state, active phase, and blockers.
- `docs/reference_cards/`: durable implementation and experiment semantics.
- `docs/reference_cards/06_feature_ood_detectors.md`: durable feature-based
  detector naming, fitting, score, numerical-stability, and variant semantics.
- `docs/reference_cards/07_optimizer_comparison_hpo_protocol.md`: durable
  optimizer HPO, comparison, seed, checkpoint, and provenance semantics.
  Protocol v1.2 is authoritative; the earlier random-search execution is only
  a one-sentence excluded-study decision record.
- `docs/reference_cards/08_raw_feature_artifact_contract.md`: durable raw
  checkpoint-feature artifact, checksum, and protected-split contract.
- `docs/reference_cards/09_core_representation_metrics.md`: frozen
  confirmatory representation-metric and low-complexity detector panel.
- `docs/reference_cards/10_optimizer_grid_literature_anchors.md`: literature
  anchor mapping for the v1.2 grids and the research lineup.
- `docs/reference_cards/11_metric_contract_v1_2.md`: authoritative
  WRN-28-10/CIFAR-10 metric definition dictionary for paper notation,
  reporting tiers, artifact keys, degeneracy states, and implementation tests.
- `docs/reference_cards/12_fixed_readout_intervention_protocol_v2.md`: active
  claim hierarchy, mechanism gate, shared-prefix intervention, v2 detector
  roles, statistics boundary, and replication protocol.
- `docs/history/local_research_draft_manifest.md`: metadata-only inventory of
  preserved local historical/superseded drafts and handoff archives; those
  untracked files are never repository authorities.
- `docs/validation/issue37_practical_runtime_status.md`: completed three-host
  practical runtime, data, and one-epoch smoke evidence.
- `docs/validation/issue53_metric_runtime_curie.md`: Metric Contract v1.2
  implementation checks and clean bounded non-protected Curie checkpoint smoke.
- `docs/validation/issue55_metric_evaluation_plan.md`: checksum-sealed
  checkpoint inventory, protected-evaluation exclusions, and deterministic
  three-host launch plan; it is planning evidence, not a research result.
- `docs/validation/issue57_metric_evaluation_execution.md`: completed
  20/20/20 protected execution, checkpoint-centric Hugging Face publication,
  operational-shard readback, deterministic seed aggregation, and local
  checksum-verified analysis handoff.
- `docs/analysis/metric_contract_v1_2_c1_c4/`: reproducible Issue #59 C1-C4
  technical report, Methods-ready English text, numerical CSV/Markdown/LaTeX
  tables, seed-0 audit, and SVG/PDF figures generated only from the verified
  central aggregate.
- `docs/validation/issue59_metric_contract_v1_2_analysis.md`: Issue #59 input
  hashes, analysis population, validation commands, deterministic rerun, and
  claim-boundary evidence.
- `docs/validation/seed0_20260728_grid_role_freeze.md`: completed seed-0 grid
  integrity, validation metrics, and immutable C1-C4 selection evidence.
- GitHub Issues: one-time task scope and acceptance criteria.
- Pull Requests: actual changes, validation evidence, and unresolved limitations.

## Rule for external AI workspaces

When this repository is attached to a ChatGPT Project, Work session, desktop Codex project, or server Codex CLI session, the AI must read the repository files directly when possible. Do not maintain an independent edited copy of these context documents.

When direct repository access is unavailable, provide a snapshot of at least:

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/WORKFLOW.md`
4. `docs/STATUS.md`
5. reference cards relevant to the current task
6. the active GitHub Issue

A copied snapshot is temporary context. The repository remains authoritative, and decisions made outside it must be reconciled back into the repository.
