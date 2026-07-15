# Reference Card 07: WRN-28-10 Optimizer Comparison and HPO Protocol

## Purpose, authority, and evidence boundary

This card fixes protocol version `wrn28_10_optimizer_hpo_v1` for comparing
`SGD`, `SGDW`, `Adam`, and `AdamW` with WRN-28-10 on the repository's OpenOOD
v1.5-aligned CIFAR-10 membership. The primary result is the tuned optimizer
comparison. Accuracy matching, coupled-vs-decoupled pair controls, and the
shared-number appendix are separate controls and must never be pooled with the
primary result.

This is a protocol decision, not experimental evidence. No HPO, additional
training, feature extraction, geometry/Neural Collapse analysis, detector
fitting, or OOD evaluation was run to write this card. In particular, the
search ranges, trial count, and seed count below are versioned project choices
made before execution; they are not known-optimal ranges or empirically
validated compute requirements.

The repository audit for Issue #18 used `main` at:

```text
c95f33f33cd4c0b96afc187577eefc0427e9d473
```

The relevant merged evidence is bounded as follows:

- Issue #14 and PR #15 provide one completed WRN-28-10, `SGD`, seed-0,
  200-epoch baseline and its independent validation record. They do not provide
  an optimizer comparison, HPO result, or multi-seed scientific conclusion.
- Issue #16 and PR #17 document DDU semantics. No DDU or other feature-based
  detector implementation or detector result exists at this boundary.
- The merged optimizer implementation (PR #1) provides the optimizer factory,
  shared parameter groups, and coupled/decoupled endpoints. Merged PRs #7, #9,
  #11, and #13 provide the model, data/MSP, training, checkpoint, resume, and
  artifact foundations on which this protocol depends. Their bounded validation
  does not constitute this study.

The adjacent durable authorities remain:

- [`01_optimizers.md`](01_optimizers.md) for optimizer update semantics and
  shared parameter grouping;
- [`02_architectures.md`](02_architectures.md) for WRN-28-10 and its
  640-dimensional penultimate feature;
- [`04_openood_v1_5_protocol.md`](04_openood_v1_5_protocol.md) for fixed ID/OOD
  membership and evaluation semantics;
- [`05_training_protocol.md`](05_training_protocol.md) for run-level training,
  checkpoint, resume, and artifact semantics;
- [`06_feature_ood_detectors.md`](06_feature_ood_detectors.md) for future
  feature-detector semantics, not implemented capability.

The Adam and decoupled-weight-decay papers pinned in `docs/sources.lock.yaml`
support the distinction between adaptive updates and coupled/decoupled decay.
The pinned random-search paper supports random search as a general HPO design;
it does not validate this card's numeric ranges, budgets, or expected winners.

## Repository facts at the audited boundary

These are observations of the repository, not new decisions introduced here.

- The optimizer factory implements `SGD` with coupled L2 weight decay, `SGDW`
  with decoupled weight decay, `Adam` with coupled L2 weight decay, and `AdamW`
  with decoupled weight decay. It also implements
  `SGDCoupledDecoupled` and `AdamCoupledDecoupled` endpoints.
- Every optimizer uses the shared `weights_only_no_bias_norm` parameter-group
  builder: convolution and linear weights are decayed while bias and
  normalization parameters are excluded.
- The training engine is an optimizer-independent single-run executor with
  end-of-epoch `MultiStepLR`, atomic `last.pt`, validation-selected
  `best_val.pt`, fixed snapshots, strict reload, epoch-boundary resume, and
  stable run artifacts.
- Within one run, `best_val.pt` is selected by highest ID-validation accuracy,
  then lowest ID-validation NLL, then earliest epoch.
- The ordinary completed-run path automatically evaluates both `last.pt` and
  `best_val.pt` on ID test and writes test artifacts. That behavior is suitable
  for an ordinary frozen final run but is a leakage risk during discovery,
  confirmation, accuracy matching, or pair selection.
- There is no study runner, multi-seed orchestration, independent-GPU queue,
  deferred-ID-test mode, feature extraction pipeline, geometry/Neural Collapse
  implementation, or feature-based detector implementation.

## Fixed comparison taxonomy and order

The execution order is fixed so that later controls cannot change the primary
study.

1. Freeze this protocol, the complete discovery tables, seed sets, Git SHA,
   dataset manifest hashes, and output-storage location.
2. Run the four-optimizer tuned comparison and freeze one tuned configuration
   per optimizer.
3. Freeze the accuracy-matching target and candidates from confirmation-only
   records.
4. Freeze pairwise coupled-vs-decoupled candidates and the shared-number table.
5. Run the five-seed final classifier comparison for every selected
   configuration, reusing an identical selected configuration rather than
   training it twice.
6. Only after every configuration, seed, and primary checkpoint role is frozen,
   permit ID-test evaluation and downstream feature, geometry, Neural Collapse,
   detector, or OOD work in separately authorized Issues.
7. Consider coupled/decoupled interpolation only after the four-optimizer core
   and endpoint controls are stable, in a separate controlled study.

### Primary: tuned optimizer comparison

Each optimizer receives the same assigned discovery-trial count and its own
predeclared, optimizer-appropriate search space. Selection uses ID train and ID
validation only. This is the primary comparison.

Optimizer-specific ranges are permitted because a single numeric range can
grant very different useful resolution to different update rules. Equal trial
count provides equal selection opportunity; it does not assert equal effective
regularization or equal elapsed compute.

### Separate control: accuracy-matched comparison

Accuracy matching asks whether downstream differences remain among classifiers
with similar terminal ID-validation accuracy. It is not a replacement for the
tuned comparison and does not revise its winners.

### Separate control: coupled-vs-decoupled pairs

Two controlled endpoint comparisons are defined:

- `SGD` versus `SGDW`;
- `Adam` versus `AdamW`.

They use shared numeric hyperparameters within each pair to isolate the stated
coupling distinction as far as the implemented optimizer semantics allow. They
are not substituted for optimizer-appropriate tuning.

### Diagnostic appendix: shared numbers

The complete shared-number endpoint table is reported as an appendix
diagnostic. Equal numeric `lr` and `weight_decay` do not imply equal effective
regularization, equal optimization difficulty, or a fairness-preserving tuned
comparison.

### Later study: coupling interpolation

`SGDCoupledDecoupled` and `AdamCoupledDecoupled` are excluded from protocol v1
discovery, confirmation, and final core comparison. A later Issue may define a
coupling-ratio study with fixed numeric `total_weight_decay`, but it must not
silently add that variable to the four-optimizer study.

## Selection boundary and deterministic ranking

### Data permitted for selection

Every discovery and confirmation model is fitted on `id_train`. Configuration
selection and checkpoint selection may use only `id_validation` accuracy and
NLL. The following are forbidden as objectives, tie-breaks, pruning signals,
search-space revision signals, or reasons to add/repeat trials:

- ID-test accuracy, NLL, or any ID-test artifact;
- OOD validation or OOD test results;
- geometry or Neural Collapse metrics, including NC0, NC1, NC2, and NC3;
- feature-detector scores or AUROC, AUPR, and FPR95;
- any downstream qualitative inspection.

OpenOOD's OOD validation remains `compatibility_only` and is not a selection
split. The Issue #14 final ID-test accuracy is not an accuracy-matching target.

### Run-level checkpoint selection

The existing run-level rule is unchanged:

1. highest ID-validation accuracy;
2. lowest ID-validation NLL;
3. earliest epoch.

This rule chooses `best_val.pt` within one training run. It does not rank
configurations across a study.

### Discovery ranking

A complete discovery trial is ranked within its optimizer by:

1. highest `best_val.pt` ID-validation accuracy;
2. lowest corresponding ID-validation NLL;
3. earliest best-validation epoch;
4. ascending `trial_id`;
5. ascending canonical config hash.

The top `K = 3` discovery configurations per optimizer are frozen before any
confirmation run is inspected. If fewer than three discovery configurations
complete, freeze all completed configurations, report the shortfall, and do not
add replacement trials. If none complete, that optimizer has no v1 winner.

### Confirmation ranking

For each frozen configuration, aggregate the three confirmation runs. Rank
configurations within an optimizer by:

1. highest arithmetic mean of best-validation accuracy;
2. lowest arithmetic mean of the corresponding validation NLL;
3. earliest arithmetic mean best-validation epoch;
4. ascending discovery `trial_id`;
5. ascending canonical config hash.

Only configurations with all three valid confirmation outcomes are eligible;
an incomplete configuration is not replaced. The first ranked configuration is
the tuned winner and is frozen before final seeds. If none is eligible, report
the optimizer as having no v1 tuned winner and do not improvise a final choice.
Final-seed results describe the frozen choice and never trigger reselection.

Canonical config hashes are lowercase SHA-256 over the UTF-8 bytes of the
resolved configuration serialized as canonical JSON with recursively sorted keys
and compact separators `(',', ':')`. The hash covers only the scientific
configuration: the optimizer family and its numeric fields, model, loss, data
contract, and scheduler. Phase- and run-varying fields are excluded so that one
configuration keeps a single identity across discovery, confirmation, matching,
pair control, and final phases. `training.seed`, `checkpoint.snapshot_epochs`,
runtime paths, timestamps, and device assignment are provenance, not config-hash
inputs.

## Protocol v1 search budget and generation

### Primary budget and resource accounting

- Assign exactly 16 discovery trials to each optimizer: 64 total assigned
  discovery slots.
- Each table contains two explicit anchors followed by 14 fixed random draws.
- Equal assigned trial count is the primary fairness budget.
- Record actual elapsed GPU-hours per optimizer and overall as a resource
  diagnostic. GPU-hours do not authorize extra trials and are not normalized
  into the selection objective.
- Configuration- or optimizer-induced failures consume an assigned slot.
  Infrastructure retries preserve the same slot and do not become a new trial.

### Sampler and frozen tables

Use NumPy `Generator(PCG64(1800))` to generate log-uniform draws. Before the
first training run, materialize and hash all four ordered 16-row tables. No
adaptive sampler, Bayesian optimizer, pruning, or result-dependent proposal is
allowed in v1.

For the 14 random rows, use common uniform quantiles within the `SGD`/`SGDW`
pair and separately within the `Adam`/`AdamW` pair. Map those quantiles through
each optimizer's declared log range. This makes proposal randomness comparable
within a coupling pair without pretending that all four optimizers need the
same numeric domain.

For `x ~ LogUniform(a, b)`, generate `u ~ Uniform[0, 1)` and set:

```text
x = exp(log(a) + u * (log(b) - log(a)))
```

Every table row must include a stable `trial_id`, full resolved config, and
canonical config hash. Trial execution order or GPU assignment may change for
availability, but table membership may not.

### Fixed and searched parameters

All trials use WRN-28-10, `dropout_rate: 0.0`, unsmoothed cross entropy, FP32,
the fixed CIFAR-10 data contract, and `weights_only_no_bias_norm`.

| Optimizer | Learning rate | Weight decay | Fixed optimizer fields |
| --- | --- | --- | --- |
| `SGD` | log-uniform `[1e-2, 3e-1]` | zero anchor plus log-uniform `[1e-5, 1e-3]` | `momentum=0.9`, `nesterov=true`, coupled |
| `SGDW` | log-uniform `[1e-2, 3e-1]` | zero anchor plus log-uniform `[1e-5, 1e-3]` | `momentum=0.9`, `nesterov=true`, decoupled |
| `Adam` | log-uniform `[1e-4, 3e-3]` | zero anchor plus log-uniform `[1e-6, 1e-3]` | `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, coupled |
| `AdamW` | log-uniform `[1e-4, 3e-3]` | zero anchor plus log-uniform `[1e-5, 1e-2]` | `beta1=0.9`, `beta2=0.999`, `eps=1e-8`, decoupled |

The two explicit anchors are:

| Optimizer | Anchor 0: current/default-positive | Anchor 1: no decay |
| --- | --- | --- |
| `SGD` | `lr=0.1`, `weight_decay=5e-4` | `lr=0.1`, `weight_decay=0` |
| `SGDW` | `lr=0.1`, `weight_decay=5e-4` | `lr=0.1`, `weight_decay=0` |
| `Adam` | `lr=1e-3`, `weight_decay=1e-4` | `lr=1e-3`, `weight_decay=0` |
| `AdamW` | `lr=1e-3`, `weight_decay=1e-4` | `lr=1e-3`, `weight_decay=0` |

For each of the remaining 14 rows, both learning rate and strictly positive
weight decay are drawn from the table ranges. Zero is represented only by the
explicit no-decay anchor; a log distribution never samples zero.

These ranges are protocol-v1 proposals informed by the current SGD baseline,
standard optimizer parameterization, broad log-scale coverage, and the need to
bound numerical risk. They have not been demonstrated to contain every
optimizer's competitive region. Coupled and decoupled numeric decay are not
assumed to have equal effects.

### Versioning and boundary hits

If the selected configuration lies on a numeric search boundary, report the
boundary hit. Do not append trials, widen only one optimizer's table, or revise
v1 after seeing results. A follow-up `v1.1` study may declare new ranges, a new
search-space hash, and equal assigned budgets for all four optimizers before it
runs. The v1 result remains preserved and separately identifiable.

## Scheduler and training horizon

Every discovery, confirmation, matching, pairwise, and final run uses:

```yaml
training:
  max_epochs: 200
scheduler:
  name: multistep
  milestones: [60, 120, 160]
  gamma: 0.2
  step_timing: end_of_epoch
```

The scheduler is a common control and is not tuned. There is no early stopping,
pruning, lower-budget screening, successive halving, or other multi-fidelity
selection. Every valid trial receives the same 200-epoch opportunity so that
convergence speed and terminal-phase geometry are not selectively truncated.

## Seed policy

The roles and exact seeds are disjoint:

| Role | Seed(s) | Use |
| --- | --- | --- |
| sampler | `1800` | generate and freeze discovery tables only |
| discovery training | `1801` | every one of the 64 discovery configurations |
| confirmation | `1802`, `1803`, `1804` | every frozen top-3 configuration; also the pair-control and shared-number anchor seeds |
| final comparison | `1805`, `1806`, `1807`, `1808`, `1809` | every selected tuned, matched, or pairwise configuration |
| downstream | same `1805`-`1809` classifiers | geometry/OOD work uses frozen final checkpoints; it does not retrain replacement seeds |

This is single-seed discovery followed by top-3, three-seed confirmation. The
Issue #14 seed-0 run is a historical anchor only and is excluded from discovery
ranking, confirmation, accuracy matching, and final aggregation.

All seed lists, `K`, and table hashes must be frozen before results are viewed.
An unfavorable but valid seed outcome is retained. Seeds are never dropped,
replaced, or rerun to improve a mean. A final classifier seed may be reused
across later authorized geometry/OOD analyses only through the same frozen
checkpoint and provenance.

## Checkpoint and artifact roles

- `last.pt` is the primary scientific classifier state: the 200-epoch
  trajectory endpoint. It is the primary state for accuracy matching and later
  downstream representation analysis.
- `best_val.pt` is the validation-selected performance control. It supplies the
  HPO objective and may be evaluated as a separately labeled downstream
  control.
- Fixed snapshots are for training-trajectory analysis only. They never replace
  `last.pt` or `best_val.pt` after downstream results are seen.

Discovery, confirmation, matching, and pair-selection runs set the snapshot
epoch list to empty to avoid unnecessary storage while retaining atomic
`last.pt`, `best_val.pt`, history, and provenance. Frozen final runs retain the
existing fixed snapshot set:

```text
0, 60, 61, 120, 121, 160, 161, 200
```

Results for `last.pt` and `best_val.pt` must remain separately named. They may
not be silently pooled, and the primary checkpoint role may not change after
feature, geometry, Neural Collapse, detector, ID-test, or OOD inspection.

## Accuracy-matching control

Freeze matching only after confirmation finishes and before final-seed runs.

1. For each tuned optimizer winner, compute mean `last.pt` epoch-200
   ID-validation accuracy across confirmation seeds `1802`-`1804`.
2. Define the target as the minimum of the available tuned-winner means. This is
   a functional rule, not a post-hoc hand-selected number. If one or more
   optimizers have no tuned winner, the target remains the minimum of the winners
   that exist; if no optimizer has a tuned winner, accuracy matching is
   unresolved in v1.
3. Use an absolute tolerance of `0.002` accuracy (0.2 percentage points).
4. Each optimizer's candidate pool is exactly its already confirmed top-3
   configurations. Thus the equal matching search budget is three candidates
   per optimizer and no new matching HPO trial is allowed.
5. Among candidates whose mean terminal accuracy is within the band, choose by:
   smallest absolute target distance, lower mean terminal validation NLL,
   lower terminal-accuracy standard deviation, then ascending config hash.
6. If no candidate is in the band, record that optimizer as `unmatched`. Do not
   change the target, widen the tolerance, match at an earlier epoch, consult ID
   test, or add replacement candidates.
7. If the matched candidate equals the tuned winner, reuse its five final runs.
   Otherwise run the matched candidate on the same final seeds.

Matching uses `last.pt` at exactly epoch 200. Epoch matching and
`best_val.pt` matching are prohibited. The target, band, candidate hashes, and
selection record must be stored before any final run or downstream analysis.

## Coupled-vs-decoupled pair control and shared-number appendix

Use the following four predeclared numeric anchors per pair:

| Pair | Learning rates | Weight decays | Cartesian product |
| --- | --- | --- | --- |
| `SGD` / `SGDW` | `0.03`, `0.1` | `1e-4`, `5e-4` | 4 configs per endpoint |
| `Adam` / `AdamW` | `3e-4`, `1e-3` | `1e-5`, `1e-4` | 4 configs per endpoint |

Run every endpoint/configuration on confirmation seeds `1802`-`1804`: 24 runs
per pair and 48 runs total. This budget is fixed independently of tuned HPO.
An anchor is eligible for pair selection only when both endpoints have all
three valid outcomes. Failed or incomplete anchors are reported and are not
replaced; if no anchor remains, report the pair as unresolved in v1.

Select one shared-number configuration per pair symmetrically by:

1. highest minimum of the two endpoints' mean best-validation accuracy;
2. highest mean of the two endpoint means;
3. lowest mean validation NLL across both endpoints;
4. earliest mean best-validation epoch across both endpoints;
5. ascending shared config hash.

Freeze the selected pair configuration, then run both endpoints on final seeds
`1805`-`1809`. Report the entire four-anchor table as the shared-number appendix
even when only one anchor advances. Same numeric decay is a controlled input,
not proof of equal effective regularization.

## ID-test defer contract

The follow-up orchestration implementation must provide a code-enforced mode in
which discovery, confirmation, accuracy-matching, and pair-selection runs do
not call ID-test evaluation and do not create ID-test metrics. Merely hiding
already generated files from a report is insufficient because it permits human
or automated leakage.

ID-test evaluation becomes eligible only after an immutable freeze record
contains all selected config hashes, final seeds, checkpoint roles, matching
target/candidates, and pair-control candidates. Ordinary final-run evaluation
may then evaluate the frozen `last.pt` and separately labeled `best_val.pt`
without feeding results back into selection. The implementation must preserve
the current ordinary-run contract outside deferred study mode.

## GPU allocation and compute recording

Assign one independent, single-device trial to one physical GPU. Do not use
multiple GPUs or DDP for one trial. Run at most one study trial concurrently on
each physical GPU.

For every attempt record the physical GPU UUID, GPU model/class,
`CUDA_VISIBLE_DEVICES`, scheduler-visible/local device mapping, assigned
`trial_id`, concurrent study-trial count, and start/end timestamps. The number,
identity, homogeneity, and availability of server GPUs are unverified until the
execution Issue performs a fresh preflight.

## Study, trial, and attempt provenance

### Study record

Store at least:

- `study_id`, protocol version, search-space version and hash;
- sampler name and seed, optimizer families, assigned trial count;
- ordered discovery-table hashes and discovery/confirmation/final seed lists;
- objective, every tie-break, checkpoint policy, and accuracy-matching policy;
- explicit no-ID-test-selection, no-OOD-selection, and
  no-geometry/detector-selection declarations;
- Git SHA and clean/dirty state, dataset manifest and membership hashes;
- created/finished timestamps, status, and total/per-optimizer GPU-hours.

### Trial record

Store at least:

- `trial_id`, `study_id`, optimizer family, full resolved config, canonical
  config hash, training seed, and study phase;
- Git SHA and dirty state, dataset hashes, environment/package inventory;
- GPU name/UUID and visible-device mapping, start/end time and elapsed time;
- status, failure class/reason, history path, best-validation record;
- checkpoint paths and hashes, selection rank or exclusion reason;
- ordered attempt IDs.

### Attempt record

Store at least:

- `attempt_id`, parent `trial_id`, attempt number, and resume/retry origin;
- failure/retry reason and links to preserved prior attempts;
- verification that config hash, seed, Git SHA, dataset hashes, and study phase
  match the parent trial;
- checkpoint used for resume and its integrity result;
- environment, GPU, timing, terminal status, logs, and output paths.

## Failure, resume, retry, and rerun policy

Classify every non-success before deciding whether retry is permitted.

| Failure class | Budget and retry rule |
| --- | --- |
| OOM caused by assigned config/optimizer | consumes the slot; no replacement |
| non-finite loss or model state | consumes the slot; preserve evidence; no replacement |
| invalid resolved config | consumes the slot; no result-driven repair in v1 |
| OOM from unrelated GPU contention | infrastructure failure; retry same trial |
| external interruption or preemption | retry/resume same trial |
| data or mount failure | infrastructure failure if external; retry same trial after repair |
| dataset manifest mismatch | stop affected study; retry only after restoring the frozen membership |
| checkpoint corruption | preserve corrupt artifact; retry same trial from last earlier valid boundary or start |
| GPU reset | infrastructure failure; retry same trial |
| driver or other infrastructure failure | retry same trial |

An infrastructure retry keeps the same `trial_id`, config hash, training seed,
Git SHA, dataset hashes, and assigned slot. It creates a new `attempt_id`. If a
valid atomic `last.pt` exists, epoch-boundary resume is preferred; otherwise the
same trial restarts from epoch 0. All attempts, partial logs, checkpoint
integrity decisions, and reasons are retained.

Low validation accuracy, an unfavorable scientific result, or a result that
changes a ranking never permits retry. No optimizer receives a selective
replacement trial. A code/protocol defect that invalidates comparability stops
the affected study and requires a new version; it is not patched mid-study.

## Git and external artifact storage

Git stores code, protocol/reference cards, frozen small configuration tables,
schemas, hashes, manifests or manifest references, and compact validation or
study summaries suitable for review. Every study is tied to an immutable Git
SHA, and a dirty execution checkout is recorded and normally rejected.

Large checkpoints, snapshots, histories, per-attempt logs, predictions,
features, geometry arrays, and detector scores remain outside Git in durable
artifact storage. Git records stable relative artifact identifiers, checksums,
storage root/version, retention class, and provenance sufficient to locate and
verify them. No execution may begin until storage capacity, atomic publication,
retention, and backup expectations are confirmed.

## Requirements for the orchestration implementation Issue

The next bounded code Issue must, at minimum:

1. implement versioned study/trial/attempt schemas and canonical config hashing;
2. materialize, validate, freeze, and hash all trial tables before execution;
3. enforce exact assigned budgets, phase/seed separation, deterministic ranking,
   top-3 freeze, matching freeze, and pair-control freeze;
4. add code-enforced deferred ID-test evaluation while preserving ordinary
   final-run behavior;
5. schedule at most one independent single-device trial per physical GPU and
   record device identity, concurrency, timing, and GPU-hours;
6. reuse the existing single-run engine, optimizer factory, shared parameter
   groups, scheduler, checkpoint, atomic save, and epoch-boundary resume
   contracts rather than duplicating them;
7. implement the failure taxonomy, attempt-preserving retry rules, checkpoint
   integrity checks, and prohibition on result-driven reruns;
8. keep large artifacts outside Git and write auditable checksums/references;
9. provide focused tests for table determinism, hashes, ranking tie-breaks,
   budget accounting, seed separation, deferred ID test, matching, failure
   classification, and retry identity;
10. perform only a bounded orchestration smoke after code review; actual HPO,
    GPU inventory validation, and long training require separately authorized
    execution scope.

The implementation Issue must not select or install an HPO framework, add a
dependency, or revise protocol v1 implicitly. Any such choice must be explicit
in that Issue and preserve the frozen semantics here.

## Unverified assumptions and decision log

### Unverified before execution

- The declared search ranges may not contain every optimizer's competitive
  region; boundary handling is therefore versioned rather than adaptive.
- Sixteen discovery trials, top-3 confirmation, and five final seeds are a
  feasibility/fairness decision, not a demonstrated statistical-power result.
- Future server GPU count, UUIDs, homogeneity, availability, driver state, and
  per-optimizer wall time require fresh preflight.
- External artifact capacity, atomic publication, retention, and backup policy
  are not yet verified.
- The deferred-ID-test mode must be shown not to break the existing ordinary
  final-run evaluation contract.
- Snapshot omission for non-final phases is supported by the present artifact
  schema but still requires orchestration regression coverage.

### Decisions frozen by Issue #18

- Tuned comparison is primary; matching, pairwise, and shared-number results
  remain separately labeled.
- Equal assigned trial count is primary; GPU-hours are reported, not equalized.
- Fixed-seed random search uses two anchors plus 14 draws per optimizer.
- All valid trials train 200 epochs with one common fixed scheduler; early
  stopping and multi-fidelity are prohibited.
- Discovery uses one seed, top-3 confirmation uses three disjoint seeds, and
  frozen final comparison uses five further disjoint seeds.
- `last.pt` is the scientific endpoint, `best_val.pt` is the
  validation-selected control, and fixed snapshots are trajectory artifacts.
- Selection is ID-only and automatic ID-test evaluation must be deferred in
  study phases.
- Failures and retries follow assigned-slot and attempt-preservation rules; no
  result-driven replacement is allowed.
- Interpolation is a later study, not part of the initial four-optimizer HPO.
