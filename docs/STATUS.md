# Project Status

Last updated: 2026-08-10

## Current phase

Research Contract v2 is the active paper protocol. The repository has finished
the WRN-28-10/CIFAR-10 Metric Contract v1.2 foundation and descriptive analysis.
The **Stage 2 existing-artifact mechanism gate** was executed at tooling SHA
`b433c13ad6ec736fcfbca4c6c7ff0d876ddf0a03`. Its 30-bundle, 210-variant
raw-kNN cache completed and passed the committed full-cache status rehash. The
full evidence build then failed closed on the prespecified exact targeted
witness for `radial_l2_mahalanobis:cifar100`; no 720/360 evidence package was
published. The reducer's checksum-bound preflight package has status `FAILED`,
with no selected candidate and `scientific_launch_allowed=false`. Stage 3 is
stopped pending an explicitly versioned contract/tooling revision. Fresh
shared-prefix confirmation and ResNet-18/CIFAR-100 replication have not
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

## Stage 2 mechanism gate result: `FAILED`

Stage 2 may reuse the already produced Issue #57 protected artifacts under the
owner's 2026-08-10 fast-path authorization. It may not reevaluate checkpoints,
traverse protected datasets, overwrite/mutate remote artifacts, or upload a new
result population.

The pre-retrieval freeze selects the exact 30 `last.pt` /
`confirmatory_primary` bundles (10 configurations x seeds 0--2), all six OOD
datasets, and 184 files per bundle. The resulting exact allowlist contains
5,520 files and 19,620,378,841 bytes. Its remote checksum catalog, reuse
manifest, candidate registry, and gate policy are stored under
[`configs/evaluation/fixed_readout_stage2/`](../configs/evaluation/fixed_readout_stage2/).
The destination is fixed outside every Git worktree, existing matching files
may only be checksum-verified and skipped, and mismatched or unexpected files
fail closed. The exact construction and validation evidence is in
[`validation/fast_path_20260810_stage2_preretrieval_freeze.md`](validation/fast_path_20260810_stage2_preretrieval_freeze.md).

After that freeze, all 5,520 allowlisted files were retrieved to
`/home/contra333/2026여름방학실험코드/fixed_readout_stage2_reuse/fixed_readout_stage2_reuse_manifest_v1`
and independently size/SHA-256 verified. The immutable retrieval receipt has
status `CHECKSUM_VERIFIED_EXACT_ALLOWLIST`, 5,520 files,
19,620,378,841 bytes, and verified catalog SHA-256
`1691467f5a054ac29fb2e6a18068e965d63ebc17f856b40eb074e7bb0e86f410`.
At that retrieval boundary, hashing had copied/read file bytes but no selected
`.npy` payload had yet been parsed with NumPy or used to compute a scientific
result. Tooling and retrieval validation are recorded in
[`validation/fast_path_20260810_stage2_gate_tooling.md`](validation/fast_path_20260810_stage2_gate_tooling.md).

The production cache completed as 30/30 bundles and 210/210 variants. The
committed `status` command rehashed the complete 2,730-file cache tree and
returned `PASS`; its completion catalog binds 30 completion records to 30
checksum-verified materialization receipts.

The evidence extractor then loaded the selected arrays in the frozen Lise
runtime and stopped on the first bundle. Row L2 normalization removed the
positive sample-wise scale to float64 closeness, but roughly `1e-16`
normalization differences were amplified by the covariance precision path and
broke exact score ties. Scores remained within the frozen mixed
absolute/relative tolerance, while 6/19,000 weak-rank signatures changed and
AUROC drifted by `5.56e-9`. The policy requires zero rank disagreement and
AUROC drift at most `1e-12`, so the required witness failed and atomic evidence
publication did not occur.

The reducer was invoked on the expected absent evidence paths and emitted its
intentional preflight `FAILED` package: four empty JSONLs, one gate decision,
and five checksum bindings. This is not a candidate-level `NO_GO` or a
scientific disproof of radial invariance; no scientific estimand was reduced.
It is a hard numerical-oracle failure that prohibits Stage 3 under the frozen
v2 policy. Exact identities, commands, diagnostics, hashes, and the verified
local metadata archive are recorded in
[`validation/fast_path_20260810_stage2_gate_execution.md`](validation/fast_path_20260810_stage2_gate_execution.md).

The next permissible repository-changing research task is a versioned
contract/tooling revision. It must not relax the observed result after the
fact, reuse partial evidence for selection, or overwrite this execution. A new
committed SHA, analysis ID, and fresh cache/evidence/gate roots are required
before any rerun. Stage 3 remains stopped.

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

- The current v2 numerical witness is terminally `FAILED`. A retry under the
  same frozen policy is not authorized; a versioned remediation must first
  define a numerically exact witness implementation and align worker/reducer
  tolerance semantics without using partial outcomes for candidate selection.
- The production raw-kNN cache is complete and reusable only under its frozen
  SHA/analysis identity. The successful 720-row candidate and 360-row
  diagnostic evidence populations do not exist because fail-closed extraction
  stopped before atomic publication.
- The shared-prefix operation does not exist. It must eventually be
  implemented as a new `fork_from_prefix` path while ordinary resume remains
  strict, but implementation and fresh training are blocked until a revised
  Stage 2 gate passes.
- ResNet-18 is implemented, but the CIFAR-100 dataset/OOD protocol and its
  actual-data validation are pending.

The later experiment-design items above remain stage-gated. The numerical
witness remediation is now a blocker to any revised Stage 2 execution.

## Explicitly not run after the Stage 2 execution

- no GPU work, server training, checkpoint inference, feature re-extraction,
  or new protected dataset traversal;
- no successful 720/360 evidence publication, candidate ranking, focal
  selection, or preconfirmation freeze;
- no shared-prefix branch, fresh seed, or CIFAR-100 experiment;
- no Hugging Face upload, source mutation, overwrite, or deletion;
- no comparable-ID, causal decay-coupling, or replication conclusion.

## Update rule

Update this file only when the active phase, validated foundation, major gate,
or next priority changes. Detailed command logs and immutable evidence belong
in a bounded validation record rather than this status summary.
