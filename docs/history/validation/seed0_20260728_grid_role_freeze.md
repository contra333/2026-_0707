# Seed-0 v1.2 Grid Validation and C1-C4 Role Freeze

## Scope

This report records the metadata-only validation and official role freeze for
the 36-cell `wrn28_10_optimizer_hpo_v1.2` seed-0 grid executed under
`seed0_20260728`. The production runs used Git SHA
`0d30054b38f0dc7a513c3eacc5c4e5435670fc4d`.

This work did not run training, seeds 1 or 2, pair controls, ID-test, OOD,
geometry/Neural Collapse, or detector evaluation. The 36 cells are a
descriptive configuration landscape, not 36 independent repeated
measurements.

## Source and server accounting

The authenticated WSL Hugging Face CLI read
`hf://buckets/contra333/ICLR_RUN/servers`. The bucket contained 780 files
(70,096,527,246 bytes) at validation time. Only 96 metadata files were
downloaded: 12 stage-level JSON files, 36 `trial.json` records, and their 48
SHA-256 sidecars. No checkpoint was downloaded.

| Host | Stage | Assigned | Terminal | Completed | Optimizer allocation |
| --- | --- | ---: | ---: | ---: | --- |
| `curie` | `remaining` | 14 | 14 | 14 | SGD 6, Adam 4, AdamW 4 |
| `lise` | `remaining` | 8 | 8 | 8 | SGD 2, Adam 4, AdamW 2 |
| `precision_medicine` | `canary` | 1 | 1 | 1 | SGD 1 |
| `precision_medicine` | `remaining` | 13 | 13 | 13 | SGD 3, Adam 4, AdamW 6 |
| **Total** |  | **36** | **36** | **36** | SGD 12, Adam 12, AdamW 12 |

The four uploaded stage assignments exactly match
`configs/studies/wrn28_10_optimizer_hpo_v1_2/seed0_execution.yaml`. Trial IDs
are unique, with no omission or overlap.

## Integrity gate

All required checks passed:

- 12/12 `ordered_plan.json`, `study_manifest.json`, and
  `study_summary.json` files match their uploaded SHA-256 sidecars;
- 36/36 `trial.json` files match their uploaded SHA-256 sidecars;
- every study and trial record passes the repository schema validator;
- protocol, frozen grid manifest, optimizer/slot/config identity, dataset
  membership hashes, training seed 0, and production Git SHA agree;
- every trial is `completed` at epoch 200 with finite last-checkpoint
  ID-validation accuracy and NLL;
- each `study_summary.json` agrees with its individual trial records;
- every attempt records ID-test as deferred, with no ID-test metric or artifact;
- no ID-test, OOD, geometry/Neural Collapse, or detector field is present in
  the role-selection records.

The selection input is the epoch-200 `last.pt` validation result. The
`best_val.pt` metric is not the selection objective; only its epoch participates
in the declared tie-break chain.

## Seed-0 validation results

### SGD

| Trial | Host/stage | LR | WD | Last acc | Last NLL | Best epoch | Best acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `grid-sgd-00` | `curie/remaining` | 0.03 | 0 | 0.9472 | 0.294223 | 174 | 0.9496 |
| `grid-sgd-01` | `curie/remaining` | 0.03 | 0.0001 | 0.9478 | 0.257496 | 81 | 0.9492 |
| `grid-sgd-02` | `curie/remaining` | 0.03 | 0.0005 | 0.9580 | 0.188175 | 172 | 0.9592 |
| `grid-sgd-03` | `curie/remaining` | 0.03 | 0.001 | 0.9580 | 0.182228 | 165 | 0.9588 |
| `grid-sgd-04` | `lise/remaining` | 0.1 | 0 | 0.9458 | 0.317342 | 102 | 0.9484 |
| `grid-sgd-05` | `lise/remaining` | 0.1 | 0.0001 | 0.9534 | 0.219303 | 173 | 0.9548 |
| `grid-sgd-06` | `precision_medicine/canary` | 0.1 | 0.0005 | 0.9582 | 0.186770 | 176 | 0.9600 |
| `grid-sgd-07` | `precision_medicine/remaining` | 0.1 | 0.001 | 0.9534 | 0.201869 | 191 | 0.9568 |
| `grid-sgd-08` | `precision_medicine/remaining` | 0.3 | 0 | 0.9384 | 0.384875 | 113 | 0.9412 |
| `grid-sgd-09` | `precision_medicine/remaining` | 0.3 | 0.0001 | 0.9582 | 0.192001 | 189 | 0.9592 |
| `grid-sgd-10` | `curie/remaining` | 0.3 | 0.0005 | 0.9552 | 0.200821 | 199 | 0.9562 |
| `grid-sgd-11` | `curie/remaining` | 0.3 | 0.001 | 0.9566 | 0.171985 | 183 | 0.9580 |

### Adam

| Trial | Host/stage | LR | WD | Last acc | Last NLL | Best epoch | Best acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `grid-adam-00` | `lise/remaining` | 0.0003 | 0 | 0.9450 | 0.376302 | 187 | 0.9458 |
| `grid-adam-01` | `lise/remaining` | 0.0003 | 0.0001 | 0.9392 | 0.329983 | 125 | 0.9418 |
| `grid-adam-02` | `precision_medicine/remaining` | 0.0003 | 0.0005 | 0.9404 | 0.319728 | 174 | 0.9416 |
| `grid-adam-03` | `precision_medicine/remaining` | 0.0003 | 0.001 | 0.9410 | 0.263800 | 166 | 0.9414 |
| `grid-adam-04` | `precision_medicine/remaining` | 0.001 | 0 | 0.9430 | 0.441255 | 141 | 0.9436 |
| `grid-adam-05` | `precision_medicine/remaining` | 0.001 | 0.0001 | 0.9384 | 0.336446 | 169 | 0.9388 |
| `grid-adam-06` | `curie/remaining` | 0.001 | 0.0005 | 0.9306 | 0.343512 | 185 | 0.9310 |
| `grid-adam-07` | `curie/remaining` | 0.001 | 0.001 | 0.9126 | 0.408328 | 183 | 0.9162 |
| `grid-adam-08` | `curie/remaining` | 0.003 | 0 | 0.9412 | 0.479841 | 134 | 0.9442 |
| `grid-adam-09` | `curie/remaining` | 0.003 | 0.0001 | 0.9042 | 0.504426 | 136 | 0.9118 |
| `grid-adam-10` | `lise/remaining` | 0.003 | 0.0005 | 0.8752 | 0.521718 | 121 | 0.8814 |
| `grid-adam-11` | `lise/remaining` | 0.003 | 0.001 | 0.8676 | 0.429869 | 123 | 0.8736 |

### AdamW

| Trial | Host/stage | LR | WD | Last acc | Last NLL | Best epoch | Best acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `grid-adamw-00` | `precision_medicine/remaining` | 0.0003 | 0.0001 | 0.9416 | 0.390630 | 184 | 0.9438 |
| `grid-adamw-01` | `precision_medicine/remaining` | 0.0003 | 0.001 | 0.9460 | 0.352624 | 198 | 0.9474 |
| `grid-adamw-02` | `curie/remaining` | 0.0003 | 0.01 | 0.9444 | 0.377452 | 198 | 0.9446 |
| `grid-adamw-03` | `curie/remaining` | 0.0003 | 0.1 | 0.9444 | 0.343079 | 185 | 0.9460 |
| `grid-adamw-04` | `curie/remaining` | 0.001 | 0.0001 | 0.9466 | 0.423168 | 198 | 0.9480 |
| `grid-adamw-05` | `curie/remaining` | 0.001 | 0.001 | 0.9408 | 0.452999 | 189 | 0.9444 |
| `grid-adamw-06` | `lise/remaining` | 0.001 | 0.01 | 0.9424 | 0.443229 | 139 | 0.9456 |
| `grid-adamw-07` | `lise/remaining` | 0.001 | 0.1 | 0.9462 | 0.355354 | 188 | 0.9480 |
| `grid-adamw-08` | `precision_medicine/remaining` | 0.003 | 0.0001 | 0.9368 | 0.513360 | 149 | 0.9384 |
| `grid-adamw-09` | `precision_medicine/remaining` | 0.003 | 0.001 | 0.9378 | 0.480337 | 180 | 0.9400 |
| `grid-adamw-10` | `precision_medicine/remaining` | 0.003 | 0.01 | 0.9460 | 0.415791 | 198 | 0.9460 |
| `grid-adamw-11` | `precision_medicine/remaining` | 0.003 | 0.1 | 0.9564 | 0.296998 | 168 | 0.9566 |

Adam slots 10 and 11 are terminal but fall below the protocol's 0.90
role-candidate accuracy floor. All other completed cells are valid candidates.

## Frozen roles

| Role | Selected cell or status | Band |
| --- | --- | ---: |
| SGD C1 | `grid-sgd-06` (`lr=0.1`, `wd=5e-4`, acc 0.9582) | — |
| SGD C2 | `grid-sgd-09` (`lr=0.3`, `wd=1e-4`, acc 0.9582) | 0.002 |
| Adam C1 | `grid-adam-00` (`lr=3e-4`, `wd=0`, acc 0.9450) | — |
| Adam C2 | `grid-adam-03` (`lr=3e-4`, `wd=1e-3`, acc 0.9410) | 0.005 |
| AdamW C1 | `grid-adamw-11` (`lr=3e-3`, `wd=0.1`, acc 0.9564) | — |
| AdamW C2 | absent after the one permitted widening | 0.005 |
| C3 | SGD-04 / Adam-00 / AdamW-04; mean 0.945800, spread 0.001600 | 0.002 |
| C4 | SGD-08 / Adam-01 / AdamW-09; mean 0.938467, spread 0.001400 | 0.002 |

Adam C1 and Adam C3 reuse the same resolved configuration. AdamW C1 selects
the largest weight decay in the frozen AdamW table, so it is explicitly
reported as a weight-decay boundary hit. The grid is not extended or rerun.

The immutable role-freeze hash is
`fdf67c1184abc489542ca64cad2410ff38aa816acb1e9e5289d60461600373fa`.
The deduplicated follow-up-plan hash is
`3a3b00dbcf0ee3dc20c0959013665bf4243a2644e21f4552a831e0fdaed69264`.
The follow-up plan records 12 seed-0 reuse entries and 30 not-yet-authorized
scheduled entries. Generating this plan does not execute or authorize those
runs.

## Repository artifacts

The authoritative small artifacts are under
`results/wrn28_10_optimizer_hpo_v1_2/seed0_20260728/`.

| Artifact | SHA-256 |
| --- | --- |
| `grid_results.json` | `8110185c299e27511a30156c2194b07291e1e3f1b7062f2e7d7c4cad25da6292` |
| `validation_metrics.csv` | `6d73d2f9ef9a896ff9a7bd25f154752ae288ce3577fabeafcf7c442f89c1a78d` |
| `role_freeze.json` | `548c7c23fc6ec194b9df881df4098463a739e95aeb6eb68da767b1b42a07103d` |
| `followup_plan.json` | `333eed9586f3fc001dac3742e01626e2fc55f78d97a94071dbf21cee4b9fc130` |

Two independent executions of `scripts/freeze_optimizer_roles.py` produced
byte-identical role-freeze and follow-up-plan files.

The eight aggregate files above, including their four sidecars, were previewed
with a no-delete Hugging Face sync and uploaded to
`hf://buckets/contra333/ICLR_RUN/aggregate/seed0_20260728/`. A fresh download
from that path was byte-compared with every local file, and all four payload
SHA-256 values matched the table. The post-upload bucket listing contained 788
files (70,096,949,407 bytes).

## Validation commands and outcomes

- `PASS`: `/home/contra333/.local/bin/hf auth whoami` returned `contra333`.
- `PASS`: `/home/contra333/.local/bin/hf buckets info contra333/ICLR_RUN`.
- `PASS`: metadata-only `hf buckets sync` dry run reported 96 downloads and
  zero deletes; the actual download contained 96 files.
- `PASS`: repository schema and identity checks over the 12 stage JSON files
  and 36 trial records.
- `PASS`: two invocations of `scripts/freeze_optimizer_roles.py` followed by
  `cmp` for both generated outputs.
- `PASS`: `TMPDIR=/tmp PYTHONPATH=src .venv/bin/python -m pytest
  tests/test_study_selection.py tests/test_study_orchestration.py -q -s`
  returned `32 passed`.
- `PASS`: `TMPDIR=/tmp PYTHONPATH=src .venv/bin/python -m pytest -q -s`
  returned `178 passed, 1 warning`. The warning is the expected local
  CUDA-driver/runtime mismatch; no CUDA test was claimed.
- `PASS`: `git diff --check`, result-sidecar validation, JSON/CSV alignment,
  freeze validation, and relative Markdown-link checks.
- `PASS`: no-delete aggregate upload dry run reported eight uploads and zero
  deletes; fresh remote download and byte comparison passed.
- `NOT_RUN`: the first prescribed pytest invocation without `-s` stopped
  before collecting tests because pytest's WSL capture temporary file
  disappeared (`FileNotFoundError`; `no tests ran`). The two successful
  `TMPDIR=/tmp ... -s` runs above replace that invocation.

No new department-server, CUDA, training, ID-test, OOD, geometry, or detector
validation was requested or run.

## Authorization boundary

The immutable freeze makes only the frozen role config hashes and seeds 0, 1,
and 2 eligible under the protocol's protected-evidence gate. Issue #47 itself
does not authorize protected evaluation, seed replication, or pair-control
execution. Those actions require a separately bounded task.
