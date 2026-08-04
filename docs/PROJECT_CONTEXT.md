# Project Context

## Research objective

This repository supports experiments on the causal chain:

```text
optimizer / training rule
→ penultimate representation geometry
→ detector score behavior
→ ID/OOD reliability
```

The primary research question is whether changes in optimization alter learned representation geometry and whether those geometric changes help explain differences in out-of-distribution performance.

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
  12-reused-plus-30-new aggregate with per-seed and sample-SD summaries.

Documented or prepared, but not yet implemented or executed:

- `GDA-ClassDensity` as the SN-off class-wise full-covariance Gaussian density
  readout using the pinned adaptive jitter and class-prior-weighted `logsumexp`;
- the complete DDU name reserved for a future spectral-normalization training
  ablation, with optional post-hoc variants outside metric-contract v1.2;
- the WRN-28-10/CIFAR-10 metric-contract v1.2 formulas, paper names, artifact
  keys, numerical failure semantics, and validation oracles.

Not yet part of the validated foundation unless added by a later merged pull request:

- feature extraction pipelines;
- geometry metrics;
- feature-based OOD detector implementations such as Mahalanobis, kNN, generic
  GMM baselines, GDA-ClassDensity, future DDU, CTM, ViM, and NECO;
- research-result OOD evaluation beyond the bounded MSP infrastructure smoke.

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
- `docs/validation/issue37_practical_runtime_status.md`: completed three-host
  practical runtime, data, and one-epoch smoke evidence.
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
