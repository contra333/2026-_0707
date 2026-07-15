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
- OpenOOD v1.5-aligned CIFAR-10 ID/OOD loaders, fixed split manifests, and preprocessing;
- bounded MSP OOD inference and metric infrastructure;
- reproducible classifier training, scheduling, checkpoint/reload, epoch-boundary resume, and run artifacts;
- bounded actual-data CUDA validation for the OpenOOD data/evaluation and WRN-28-10 training paths.

Documented for later implementation, but not yet implemented or validated:

- the WRN-28-10 `SGD`/`SGDW`/`Adam`/`AdamW` tuned, accuracy-matched,
  coupled-vs-decoupled, and shared-number comparison protocol;
- DDU as a class-wise full-covariance GDA density detector using the pinned
  official adaptive jitter and `logsumexp` score convention;
- explicit post-hoc DDU variants for PCA, diagonal covariance, L2-normalized
  features, and statistical covariance shrinkage;
- SN on/off as a training ablation that does not rename the detector.

Not yet part of the validated foundation unless added by a later merged pull request:

- completed optimizer-comparison and multi-seed research results;
- experiment orchestration;
- HPO and multi-seed execution;
- feature extraction pipelines;
- geometry metrics;
- feature-based OOD detector implementations such as Mahalanobis, kNN, generic
  GMM baselines, DDU and its variants, CTM, ViM, and NeCo;
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
  four-optimizer HPO, comparison, seed, checkpoint, and provenance semantics.
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
