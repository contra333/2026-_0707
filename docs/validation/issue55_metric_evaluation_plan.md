# Issue 55 Metric Contract v1.2 Protected Evaluation Plan

## Scope and verdict

Issue #55 freezes the checkpoint population and source-local three-host plan
for the WRN-28-10/CIFAR-10 Metric Contract v1.2 evaluation. It does not launch
feature extraction, read protected ID-test or OOD samples, calculate research
metrics, or upload evaluation results.

Planning verdict on 2026-08-04: **PASS** for deterministic inventory
construction, schema validation, focused tests, and read-only source
availability. The plan imports the checksum-sealed 42-row Issue #49 aggregate,
authorizes only its 30 frozen role identities, creates 60 separate checkpoint
jobs, and retains the other 12 pair-control-only identities as explicit
exclusions.

Protected execution remains **NOT_RUN** and requires a separate active Issue
after this plan is reviewed and merged.

## Frozen identities

| Item | Frozen value |
| --- | --- |
| Runtime merge Git SHA | `e22070ed436c772888e08ca1a4cb5a21760812a9` |
| Metric definition | `metric_contract_v1.2` |
| Source aggregate file SHA-256 | `1e0dee035ca77afd63c00b6ae9e74b8ed82246f69d95e1a8483f91624baf596b` |
| Source aggregate internal hash | `a403d23b747ba6b927211dd3a9acc7526fabcbba1daf9b56ad735219191fc856` |
| Inventory hash | `35a1a66f78dfd6255c4d804586ee6863969be78d21acec5cd0ccf00476ddaa5c` |
| Inventory file SHA-256 | `39c89b30aecbea0984cafd1ac75cf8f8676808ee12b4728fb4fb21e845f6e7a1` |
| Execution-manifest hash | `04645b9af91732cfd79a417b41236325b470ba239d84e9be05246f8a6c7cf105` |
| Execution file SHA-256 | `8d242183d9907a2b6383a999057f922723469ab413bdf1b451699169f1f18b97` |
| Dataset-policy hash | `f80984d28ef9a7550633fb3aea514f783f3db1fa655a6d3f49850b10857e6afc` |

The source aggregate covers 14 configurations with seeds `{0,1,2}` and no
duplicate `(config_hash, seed)`. Ten configurations have at least one frozen
`role:<optimizer>:C1..C4` label, giving 30 authorized training identities.
Four pair-control-only configurations give 12 excluded identities and 24
excluded checkpoint references.

Every authorized identity produces exactly two jobs:

- `last.pt`: `confirmatory_primary`;
- `best_val.pt`: `deployment_control`.

The 60 checkpoint SHA-256 values, job IDs, and checkpoint-centric output
namespaces are unique. Both roles retain their own fit state, score arrays,
artifacts, and result rows.

## Dataset and host plan

The executable split policy is exactly:

- ID: `id_train=45,000`, `id_validation=5,000`, and
  `id_test_primary=10,000`;
- near OOD: `cifar100`, `tin`;
- far OOD: `mnist`, `svhn`, `texture`, `places365`;
- excluded: `id_test_openood=9,000` compatibility provenance and
  `ood_validation_tin`.

Assignment follows the existing checkpoint location and does not copy models
between servers:

| Host | Checkpoint jobs | Planned concurrency | Role balance |
| --- | ---: | ---: | --- |
| `curie` | 20 | 4 | 10 `last` + 10 `best_val` |
| `lise` | 20 | 2 | 10 `last` + 10 `best_val` |
| `precision_medicine` | 20 | 4 | 10 `last` + 10 `best_val` |

`curie` is the future coordinator. The plan records
`execution_started=false`, `protected_accessed=false`, server sync and upload
as `not_started`, and requires a separate execution Issue before any of these
states may change.

## Commands and observed results

Deterministic generation and checked-in parity:

```bash
PYTHONPATH=src \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  scripts/build_metric_evaluation_plan.py
PYTHONPATH=src \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  scripts/build_metric_evaluation_plan.py --check
```

Result: **PASS**. Repeated materialization is byte-identical and both JSON/YAML
files match their SHA-256 sidecars.

Focused planning tests:

```bash
PYTHONPATH=src \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  -m pytest -q -s tests/test_metric_evaluation_planning_v1_2.py
```

Result: **PASS**, `17 passed in 2.91s`. The tests cover source and authorization
counts, seed coverage, checkpoint roles, aggregate identity, split exclusions,
20/20/20 source-local assignment, 4/2/4 concurrency, deterministic bytes,
immutable bindings, and rejection of hash, role, dataset, assignment, output,
launch-state, and delete-policy drift.

The local complete suite ran under Python 3.12.3, NumPy 2.5.1, and PyTorch
2.13.0+cu130:

```bash
PYTHONPATH=src \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  -m pytest -q -s
```

Result: **FAILED**, `252 passed, 3 failed`. All three failures are in the
pre-existing AURC path because NumPy 2.5 removed `np.trapz`; no planning test
failed. The production Curie environment validated for Issue #53 uses the
project runtime and must run the complete suite on the clean planning commit
before merge. Until that clean run is recorded, the complete-suite gate is
**PENDING**.

The following read-only Hugging Face query shape was applied to all 60 source
checkpoint URIs in parallel:

```bash
hf buckets list <exact-checkpoint-uri> --json --no-truncate
```

Result: **PASS**, 60/60 exact `last.pt` or `best_val.pt` paths returned a file
record. This establishes current source-path availability only. It does not
replace the checkpoint SHA-256 evidence sealed in the Issue #49 aggregate and
does not download a checkpoint.

The first parallel wrapper attempt was **FAILED** before issuing useful checks
because its nested shell quoting was malformed. The corrected wrapper above
completed 60/60. `git diff --check` is **PASS**.

## Deliberately not run

- Protected feature extraction and all `id_test_primary`/OOD metric
  calculations: **NOT_RUN** by Issue scope.
- 9k compatibility ID and OOD validation splits: **EXCLUDED** by contract.
- Server repository synchronization and launch: **NOT_RUN** until the planning
  commit is merged and a separate execution Issue is active.
- Hugging Face evaluation upload, remote completion markers, aggregation,
  figures, and paper analysis: **NOT_RUN**.
- Pair-control-only protected evaluation: **NOT_AUTHORIZED**.

## Interpretation boundary

This plan proves which completed models would be evaluated, on which server,
under which checkpoint and data identities. It is not evidence that evaluation
has started or that any metric value, optimizer difference, or OOD conclusion
exists. Those claims require terminal per-host records, checksum-verified
no-delete uploads, central remote validation, and checkpoint/seed aggregation
from the later protected execution task.
