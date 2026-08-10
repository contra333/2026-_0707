# Project Status

Last updated: 2026-08-10

## Current phase

Research Contract v2 is the active paper protocol. The repository has finished
the WRN-28-10/CIFAR-10 Metric Contract v1.2 foundation and descriptive analysis;
the next executable phase is the **Stage 2 existing-artifact mechanism gate**.
Fresh shared-prefix confirmation and ResNet-18/CIFAR-100 replication have not
started.

The active claim is deliberately narrower than “optimizer X is better”:

```text
paired training-rule change
-> detector-relevant representation channel
-> ID/OOD score-overlap change under a fixed readout
-> fresh main confirmation
-> second-regime replication
```

[`reference_cards/12_fixed_readout_intervention_protocol_v2.md`](reference_cards/12_fixed_readout_intervention_protocol_v2.md)
is authoritative for that sequence. It does not retroactively change v1.2.

## Completed and validated foundation

### Training and optimizer infrastructure

- Optimizer semantics, shared `weights_only_no_bias_norm` parameter groups,
  SGD/SGDW, Adam/AdamW, and coupled/decoupled endpoints are implemented and
  tested.
- `toy_cifar_cnn`, CIFAR `resnet18`, and `wrn28_10` expose the common logits /
  penultimate-feature API. The toy model remains infrastructure-only.
- The classifer runner implements resolved configuration, deterministic loader
  state, scheduling, atomic `last.pt`/`best_val.pt`/snapshot artifacts, strict
  loading, and same-run epoch-boundary resume.
- The three approved department hosts passed the practical runtime, data, and
  bounded actual-CUDA gate recorded in
  [`validation/issue37_practical_runtime_status.md`](validation/issue37_practical_runtime_status.md).

### Protocol v1.2 optimizer population

- All 36 seed-0 WRN-28-10/CIFAR-10 grid cells completed and passed metadata
  integrity checks.
- C1--C4 were frozen using only epoch-200 `last.pt` ID-validation evidence.
- All 30 new role-replication/pair-control rows completed; combined with 12
  reused seed-0 rows, the aggregate contains 42 unique `(config_hash, seed)`
  identities across 14 configurations.
- Per-seed results, sample SD, attempt history, checksums, and the seed-0
  selection-bias disclosure are preserved. See
  [`validation/seed0_20260728_grid_role_freeze.md`](validation/seed0_20260728_grid_role_freeze.md)
  and [`validation/issue49_followup_execution.md`](validation/issue49_followup_execution.md).

### Metric Contract v1.2

- Deterministic raw-feature extraction, ID/calibration metrics, logit and
  feature OOD detectors, representation geometry, explicit numerical states,
  checkpoint-centric records, and checksum validation are implemented.
- The protected evaluation completed 60/60 checkpoint bundles at scientific
  evaluator SHA `c38b09694be88aa74de0741b39e9d3ba0d6ff61a` with three host
  shards `REMOTE_VERIFIED` as 20/20/20.
- The central aggregate contains 95,160 successful per-checkpoint scalar
  records, 31,720 successful seed aggregates, 40 detector-rank-concordance
  records, and zero non-success seed aggregates. See
  [`validation/issue53_metric_runtime_curie.md`](validation/issue53_metric_runtime_curie.md),
  [`validation/issue55_metric_evaluation_plan.md`](validation/issue55_metric_evaluation_plan.md),
  and [`validation/issue57_metric_evaluation_execution.md`](validation/issue57_metric_evaluation_execution.md).
- The checksum-bounded C1--C4 package separates `last.pt` primary from
  `best_val.pt` control, preserves all six OOD datasets, exports every central
  scalar, and labels geometry--OOD associations exploratory. See
  [`analysis/metric_contract_v1_2_c1_c4/`](analysis/metric_contract_v1_2_c1_c4/)
  and [`validation/issue59_metric_contract_v1_2_analysis.md`](validation/issue59_metric_contract_v1_2_analysis.md).

## Evidence boundary of completed v1.2 work

The v1.2 grid, frozen roles, pair controls, protected metrics, and local
analysis are valid descriptive/discovery evidence. They do **not** establish:

- a shared-prefix causal effect of decay coupling;
- comparable-ID equivalence under a prespecified margin;
- a detector-formula mechanism from scalar association alone;
- a universal detector ranking or dataset-pooled generalization;
- ResNet-18/CIFAR-100 replication.

`GDA-ClassDensity` is implemented and evaluated for SN-off checkpoints. `DDU`
remains reserved for an unexecuted spectral-normalization training ablation.

## Research Contract v2 decisions frozen

- Main regime: WRN-28-10/CIFAR-10.
- Replication regime: ResNet-18/CIFAR-100.
- Main optimizer design: Adam/AdamW primary paired family plus SGDM/SGDW
  conventional-family control, each from its own zero-decay prefix and each
  with a zero-decay continuation.
- Replication optimizer design: Adam/AdamW paired family plus an independent
  conventional SGDM reference.
- Main detector panel: Mahalanobis-Raw, Mahalanobis++, kNN-Raw K=50, kNN-L2
  K=50, CTM, Pure Residual, and Energy-T1.
- Secondary: Relative Mahalanobis-Raw/++, MSP. Appendix: ViM and components.
  ReAct is excluded.
- The contribution is fixed-readout practical non-invariance plus
  formula-linked score-overlap mechanism and replication, not optimizer
  ranking, detector SOTA, L2-normalization novelty, or an ID-only audit rule.

The old local drafts and ZIP handoffs remain untouched and untracked. Their
hash-addressed historical/superseded status is recorded in
[`history/local_research_draft_manifest.md`](history/local_research_draft_manifest.md).

## Active next phase: Stage 2 mechanism gate

Stage 2 may reuse the already produced Issue #57 protected artifacts under the
owner's 2026-08-10 fast-path authorization. It may not reevaluate checkpoints,
traverse protected datasets, overwrite/mutate remote artifacts, or upload a new
result population.

Before retrieval, Stage 2 must commit a checksum-addressed selection manifest
for the exact `last.pt` discovery population, datasets, file allowlist, remote
identities, and non-Git destination. It must then:

1. verify every selected file and identity;
2. reproduce existing AUROC/FPR95 from raw scores;
3. add tested raw-kNN, Pure Residual, score-overlap, component-decomposition,
   targeted-transform, and invariant-control analyses;
4. keep all six OOD datasets separate and use all selected discovery
   configurations rather than a favorable subset;
5. freeze one focal channel/transform and the Stage 3 addendum before any
   fresh paired OOD result is opened.

Stage 2 is discovery only. A failed gate stops or reframes Stage 3.

## Stage-gated decisions still open

These are intentional pre-confirmation decisions, not unresolved permission to
tune on fresh OOD outcomes:

- focal geometry channel, targeted transform, and exact invariant/null control;
- switch epoch or prespecified switch-epoch set;
- family-specific nominal decay dose or dose set;
- ID accuracy equivalence margin and NLL/ECE guardrails;
- per-outcome practical OOD margins;
- fresh prefix count/power rule and multiplicity handling;
- main go/no-go criterion.

Before Stage 4, the CIFAR-100 membership/preprocessing/OOD contract, exact SGDM
reference recipe, replication seed count, and replication success rule must be
frozen separately.

## Implementation and execution blockers

- The shared-prefix operation does not exist. It must be implemented as a new
  `fork_from_prefix` path; ordinary resume must remain strict and unchanged.
- Raw kNN and the v2 Pure Residual key/registry do not yet exist.
- The Stage 2 reuse selection manifest and local raw-array cache do not yet
  exist; substantial non-Git working space is required.
- ResNet-18 is implemented, but the CIFAR-100 dataset/OOD protocol and its
  actual-data validation are pending.

The numerical items above are stage-gated design choices rather than blockers
to Stage 2 discovery.

## Explicitly not run by Stage 1

- no GPU, server training, checkpoint inference, or protected dataset traversal;
- no raw artifact download or external upload;
- no shared-prefix branch, fresh seed, or CIFAR-100 experiment;
- no new scientific result.

## Update rule

Update this file only when the active phase, validated foundation, major gate,
or next priority changes. Detailed command logs and immutable evidence belong
in a bounded validation record rather than this status summary.
