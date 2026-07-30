# Reference Card 07: WRN-28-10 Optimizer Comparison Protocol

## Status and authority

This card fixes the active protocol
`wrn28_10_optimizer_hpo_v1.2` for WRN-28-10 on CIFAR-10. It is normative
for the 36-cell grid, validation-only selection, `C1`-`C4` roles, seed
reuse, pair controls, checkpoints, and protected-evidence boundary.

The executable protocol and frozen tables live under
`configs/studies/wrn28_10_optimizer_hpo_v1_2/`. The grid manifest records
the exact row, table, grid, dataset-membership, dataset-manifest, and
whole-manifest hashes.

The `seed0_20260728` v1.2 production grid completed all 36 assigned cells and
its metadata-integrity validation and C1-C4 role freeze are recorded in
`docs/validation/seed0_20260728_grid_role_freeze.md`. Local tests and server
smoke runs remain infrastructure evidence rather than research results.

## Superseded-study decision log

An earlier random-search study was executed outside this repository on the
department server using the old 50k/1k/9k ID split. Its 64 runs are not
imported and must not be used as v1.2 quantitative evidence, HPO selection,
baseline, table, ranking, or analysis. Its external physical deletion is
outside this repository task. The current Git tree contains no v1.1 study
config, 64-row table, sampler, generator, default execution path, or golden
hash.

The old design remains visible only through GitHub Issue/Pull Request and Git
history. It motivated the deterministic-grid redesign but supplies no current
numeric result or grid justification.

Issue #14 is a `historical single-seed baseline, excluded from v1.2
aggregation`. Its validation report remains an accurate historical record;
its accuracy is not used to choose the v1.2 grid or report v1.2 results.

## Fixed training and data contract

Every primary grid cell uses:

- dataset protocol `oge_cifar10_holdout_v1`;
- `id_train`: official CIFAR-10 train membership 45,000;
- `id_validation`: disjoint official-train holdout membership 5,000;
- `id_test`: untouched official CIFAR-10 test membership 10,000;
- WRN-28-10, `dropout_rate=0.0`, `init_policy=msr_fan_in`;
- batch size 128, 200 epochs, unsmoothed cross entropy;
- multistep schedule `[60, 120, 160]`, `gamma=0.2`, stepped at epoch end;
- train-only flip and reflection-padded crop augmentation;
- `weights_only_no_bias_norm`;
- FP32 parameter/activation/storage. Issue #37 measured an agreeing candidate
  numerical policy on all three approved hosts, but the execution Issue must
  reverify and explicitly pin the effective values on its final clean SHA;
- no AMP or BF16;
- `cudnn.benchmark=false`, `training.deterministic=false`;
- `last.pt` as the scientific endpoint and `best_val.pt` as a
  validation-selected control.

The generated grid rows contain the resolved logical dataset membership and
manifest hashes but omit host-specific data roots and device paths.

## Primary grid

The primary study has exactly 36 seed-0 slots: 12 per optimizer.
Enumeration order is ascending learning rate, then ascending weight decay.
No random sampler exists.

| Optimizer | Learning rate | Weight decay | Fixed fields |
| --- | --- | --- | --- |
| SGD | `0.03, 0.1, 0.3` | `0, 1e-4, 5e-4, 1e-3` | momentum `0.9`, Nesterov, coupled |
| Adam | `3e-4, 1e-3, 3e-3` | `0, 1e-4, 5e-4, 1e-3` | betas `(0.9, 0.999)`, eps `1e-8`, coupled |
| AdamW | `3e-4, 1e-3, 3e-3` | `1e-4, 1e-3, 1e-2, 1e-1` | betas `(0.9, 0.999)`, eps `1e-8`, decoupled |

Each assigned cell is immutable. A divergence, non-finite model, or
configuration-induced OOM consumes the scientific slot, is preserved as a
failure in the landscape, and is excluded from role candidacy. A grid-edge
selection is reported as a boundary hit; it does not authorize extending the
grid or rerunning it mid-study.

Card 10 maps these values to pinned literature and explicit project judgment.
The old random-search results are not a source.

## Checkpoint and ranking rule

Role selection begins only after all 36 assigned cells are terminal. It uses
only seed-0 `last.pt` epoch-200 ID-validation metrics:

1. highest ID-validation accuracy;
2. lowest ID-validation NLL at the same checkpoint;
3. earliest `best_val.pt` epoch;
4. ascending learning-rate rank;
5. ascending weight-decay rank;
6. ascending canonical config hash.

Only completed cells with accuracy at least `0.90` are valid role candidates.
`0.002` and `0.005` are predeclared operational accuracy bands, not
statistical-significance thresholds.

ID-test, OOD, geometry/Neural Collapse, and detector fields are forbidden in
the selection input. Their presence makes role freezing fail.

## C1-C4 role definitions

- **C1, tuned best per optimizer:** highest-ranked valid cell.
- **C2, near-optimal alternative per optimizer:** among valid non-C1 cells
  within `0.002` of C1 accuracy, choose the largest
  `|log10(lr/lr_C1)| + d_wd`. For two positive decays,
  `d_wd=|log10(wd/wd_C1)|`; when exactly one decay is zero, `d_wd=2.0`;
  when both are zero, `d_wd=0`. Distance ties use the standard ranking
  chain. If no cell qualifies, widen once to `0.005`; otherwise record C2
  absent.
- **C3, primary matched triple:** enumerate valid `(SGD, Adam, AdamW)`
  triples whose max-minus-min accuracy spread is at most `0.002`. Choose
  highest mean accuracy, then lowest mean NLL, then ascending concatenated
  config hashes. If no triple qualifies, widen once to `0.005`; otherwise
  record C3 unresolved.
- **C4, secondary matched triple:** among triples qualifying under C3's
  effective band and whose mean accuracy is at most `C3 mean - 0.005`,
  choose the same highest-mean/lowest-NLL/hash ranking. There is no widening.
  If absent or C3 is unresolved, omit C4.

Roles may coincide. An identical resolved configuration is trained once per
seed and carries all earned labels.

The role-freeze record includes the frozen grid manifest hash, a hash of all
36 terminal validation records, all selected/absent/unresolved decisions, and
the complete config-hash-to-role mapping. Mutation invalidates its freeze
hash.

## Seed replication and reuse

The full grid uses training seed 0. Every unique role configuration adds
seeds 1 and 2. The three-seed evidence unit therefore reuses seed 0 rather
than training it again.

Because selection itself uses seed 0, role-bundle statistics have known
selection bias toward that seed. Reports must disclose this and retain all
per-seed values. Seeds are never dropped, replaced, or rerun to improve a
mean.

The full 36-cell seed-0 grid is descriptive landscape evidence only. It is
not treated as 36 repeated measurements and is not used to revise the frozen
roles.

## Pair controls

Pair controls are diagnostic and separately labeled.

### Adam versus AdamW

Use both optimizers at both shared cells:

- `(lr=1e-3, wd=1e-4)`;
- `(lr=3e-3, wd=1e-3)`.

Each endpoint uses seeds 0, 1, and 2. Existing grid and role runs are reused;
only missing config/seed pairs are scheduled.

### SGD versus SGDW

- Cell A: `(lr=0.1, wd=5e-4)`.
- Cell B: the highest-ranked valid SGD cell different from A.

SGD reuses its grid/role runs and adds missing seeds. SGDW, which has no
primary grid, trains A and B at seeds 0, 1, and 2. If no valid non-A SGD cell
exists, this pair control is unresolved rather than invented.

Equal numeric learning rate and weight decay are controlled inputs; they do
not imply equal effective regularization across coupled and decoupled
updates.

The frozen follow-up plan deduplicates by `(scientific_config_hash,
training_seed)`, lists seed-0 reuse separately from newly scheduled work, and
has its own plan hash.

## Protected-evidence release

Before the immutable role freeze:

- ID-test evaluation is forbidden;
- OOD evaluation is forbidden;
- geometry and Neural Collapse analysis are forbidden;
- detector fitting/scoring results are forbidden as selection signals;
- OOD validation is compatibility-only and cannot tune the classifier,
  roles, or detector per optimizer.

After role freeze, protected evaluation is authorized only for role config
hashes at seeds 0, 1, and 2. Pair-control-only configurations are not silently
promoted to primary role evidence.

## Failure, retry, and provenance rules

- Scientific failures consume their assigned slot.
- Infrastructure failures may resume only from a validated same-trial
  epoch-boundary checkpoint with identical Git SHA, resolved config,
  membership identity, and trial identity.
- Every retry is a new preserved attempt; earlier attempts are never
  overwritten.
- Low accuracy, unfavorable ranking, boundary hits, or a changed scientific
  interpretation never authorize a retry.
- A host lost permanently does not authorize cross-host checkpoint resume.
  The trial starts from scratch on the replacement host as a new attempt.
- Git must be clean for execution.
- Long-running artifacts live outside Git. Git stores only code, small
  configs, frozen plans/hashes, and validation reports.

## Execution boundary

This card and Issue #35 authorize study-definition code only. Long training
requires a separate execution Issue. Issue #37 and merged PR #38 completed the
practical runtime, data, numerical-policy, and one-epoch smoke readiness gate
at clean execution SHA `e9bfde43bb40f3ea2a6a11da9da86178049ecc40`; those
smokes consumed no production slot. The execution Issue must verify that this
evidence remains applicable to its final production SHA and rerun any
invalidated gate. It must verify:

- clean Git SHA;
- byte-verified 36-row grid and dataset hashes;
- Python 3.11 and one hash-locked PyTorch/TorchVision/CUDA runtime distribution
  set on every approved host, while recording exact Python patch and driver
  versions as metadata;
- identical measured and explicitly pinned numerical flags on all hosts;
- continued actual-data loader readiness and external artifact/backup
  locations;
- applicability of the completed one-epoch smoke on every server, with any
  invalidated smoke rerun before production;
- a recorded execution assignment in which every optimizer appears on every
  host, without treating GPU UUIDs or LR/WD-rank balance as immutable gates;
- successful 200-epoch SGD canary with train/ID-validation and artifact
  integrity only.

The canary is included as its normal grid cell only after it completes
unchanged. No ID-test/OOD/geometry/detector result may be opened during that
check.

The owner-authorized `fast path` task on 2026-07-28 is the bounded exception
that authorizes this seed-0 production execution without a separate Issue. Its
exact-once multi-host assignment is committed at
`configs/studies/wrn28_10_optimizer_hpo_v1_2/seed0_execution.yaml`.
The follow-up owner `fast path` on 2026-07-28 fixes the bounded execution's
launch policy as `concurrent_sentinel`: `grid-sgd-06` starts first and remains
the named operational sentinel, but the frozen `remaining` cells may run
concurrently after the sentinel has started healthily. All cells remain pinned
to production Git SHA `0d30054b38f0dc7a513c3eacc5c4e5435670fc4d`.
This changes launch order only; it does not change scientific configs,
terminal-slot accounting, role selection, or retry rules. The complete
assignment uses the approved 4:2:4 GPU capacities:

- `curie`: 14 total cells at concurrency 4;
- `lise`: 8 total cells at concurrency 2;
- `precision_medicine`: 14 total cells at concurrency 4, including the canary.

The assignment covers all 36 cells exactly once and includes every optimizer
on every host. Each host uploads its complete study-artifact directory,
including trial records and checkpoints, under
`hf://buckets/contra333/ICLR_RUN/servers/<host_id>/seed0_20260728/<stage>`.
All 36 uploaded records were verified terminal and combined under Issue #47.
The resulting immutable role-freeze hash is
`fdf67c1184abc489542ca64cad2410ff38aa816acb1e9e5289d60461600373fa`.
The execution exception did not authorize role replication, pair controls,
protected-evidence execution, or another production study; those remain
separately bounded tasks.

Issue #49 is the separate bounded task for the frozen role replication and
pair controls. Its committed
`configs/studies/wrn28_10_optimizer_hpo_v1_2/followup_execution.yaml` binds
the role-freeze and follow-up-plan hashes, assigns the 30 new rows exactly
once as `curie=13`, `lise=10`, and `precision_medicine=7`, rotates every
configuration across all three hosts over seeds 0/1/2, and preserves the
declared pair co-locations. Production uses clean Git SHA
`3556841340e6f6b92782af045ed4a468e6e271bd` and detached host supervisors.
This running execution does not become multi-seed evidence until every row is
terminal, independently validated, uploaded without delete, and combined with
the 12 reused seed-0 records.
