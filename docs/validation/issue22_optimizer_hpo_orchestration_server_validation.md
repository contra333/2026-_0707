# Issue 22 optimizer-HPO orchestration server validation

## Evidence boundary

This report records a bounded department-server infrastructure smoke for
`wrn28_10_optimizer_hpo_v1.1`. It is not an HPO result, optimizer comparison,
confirmation result, final-training result, ID-test release, OOD result,
feature extraction, geometry/Neural Collapse result, or detector result.

Exactly two smoke-only discovery attempts were run for one epoch each. No
production discovery slot was consumed. No actual 64-run discovery HPO,
confirmation, final five-seed training, accuracy-matched execution, pair-control
execution, ID-test metric, OOD evaluation, feature extraction, geometry/NC, or
detector work was performed.

## Git and review state

- Branch: `feat/issue-22-optimizer-hpo-orchestration-v1-1`
- Starting PR head: `46d0b195a30ef4ede4d5079112a06cf022029572`
- Smoke execution SHA: `674acf23011570cd28fcc4e8144f6c5df937f786`
- Runtime-fix commit: `674acf23011570cd28fcc4e8144f6c5df937f786`
- Report commit: the later commit containing this file; it is intentionally not
  the smoke execution SHA.
- Pull Request: <https://github.com/contra333/2026-_0707/pull/23>
- PR state at validation start: open Draft, mergeable, base `main`.

The original checkout contained the user-owned untracked `docs/prompts/` path.
It was not modified, staged, or deleted. Review and smoke were performed in the
clean worktree:

```text
/home/ghjin/0707_exp/2026-_0707-issue22-server
```

## Pre-smoke implementation finding and correction

Review of PR #23 found one acceptance blocker before any GPU smoke was run.
A child exit status of zero was sufficient to mark the attempt and trial
completed, but orchestration did not read the child `summary.json`, `last.pt`,
or `best_val.pt`. Consequently, trial records omitted final/best validation,
completed epoch, checkpoint paths and hashes, and artifact references. The
study manifest also retained null timestamps and a planned status.

The runtime fix:

- validates successful child summaries, finite validation metrics, deferred
  ID-test state, Git identity, single-visible-GPU metadata, and expected versus
  actual GPU UUID;
- loads `last.pt` and `best_val.pt` through the repository loader;
- records completed epoch, validation records, checkpoint paths/SHA256, and
  run-artifact paths/SHA256 in attempt and trial records;
- records study created/finished timestamps, final status, terminal accounting,
  and GPU-hours;
- preserves a child-artifact validation failure as an implementation failure
  instead of publishing it as completed;
- adds focused regression coverage for both a complete child result and an
  exit-zero child with missing artifacts.

No smoke was executed at the defective starting SHA. Tests passed, the fix was
committed, and dry-run/smoke were executed from the clean fix SHA above. There
was no failed GPU smoke and no smoke rerun.

## Environment

| item | observed value |
| --- | --- |
| host | `curie` |
| OS/kernel | Ubuntu kernel `6.17.0-35-generic`, x86_64 |
| Conda environment | `/home/ghjin/miniconda3/envs/oge-issue6` |
| Python | `3.13.11` |
| `oge` import | `/home/ghjin/0707_exp/2026-_0707-issue22-server/src/oge/__init__.py` |
| PyTorch | `2.11.0+cu130` |
| TorchVision | `0.26.0+cu130` |
| PyTorch CUDA runtime | `13.0` |
| cuDNN | `91900` |
| NVIDIA driver | `580.119.02` |
| filesystem before smoke | 734 GiB free; 119,156,375 free inodes |

The checkout-local import was enforced with `PYTHONPATH="$PWD/src"`. No new
Conda environment or dependency was installed.

## Data membership

Actual data root:

```text
/home/ghjin/datasets/openood-v1.5-3c35632e
```

| split | count | imglist SHA256 |
| --- | ---: | --- |
| ID train | 50,000 | `9317e18980f1b11df74bba9d64223e57e69bd08c2943adbc75aec393a4976d1b` |
| ID validation | 1,000 | `6f64c4370bf3c61dab3308f7e8c5c94938b112b20d38cca77c1dfa0bcc5c1d89` |
| ID test | 9,000 | `84559ec5225425e7f7363058d5f9442ed21a39c7c66b63c3f1bb8b1ddb935e13` |

`resolve_training_config` recomputed these counts and hashes from the selected
imglists. They match the protocol and the retained Issue #14 validation record.
The two attempt records contain the same three membership hashes.

## GPU inventory and selection

Immediately before smoke, all four GPUs reported 0% utilization and no compute
process. Xorg occupied 4 MiB on each device; GPU 2 additionally hosted desktop
graphics, so GPUs 0 and 1 were selected.

| index | UUID | model | used/free before smoke |
| ---: | --- | --- | ---: |
| 0 | `GPU-b7682b36-2ce0-802b-8bb1-e7668087ecde` | NVIDIA RTX A5000 | 15 / 24,098 MiB |
| 1 | `GPU-8f45a2e7-59dd-831b-efc4-1dc6e10e4fee` | NVIDIA RTX A5000 | 15 / 24,098 MiB |
| 2 | `GPU-aceccd92-a0bb-ac69-5b7f-61c913da2bad` | NVIDIA RTX A5000 | 235 / 23,875 MiB |
| 3 | `GPU-bed51d32-f978-d07a-c020-01d7a36e63ea` | NVIDIA RTX A5000 | 15 / 24,098 MiB |

No process was terminated. Post-smoke, GPUs 0 and 1 again showed 15 MiB used,
0% utilization, and no compute process.

## Commands and test results

### Clean execution state

```bash
git rev-parse HEAD
git status --short
```

Observed HEAD was `674acf23011570cd28fcc4e8144f6c5df937f786`; status was empty.

### Server tests

```bash
PYTHONPATH="$PWD/src" conda run -n oge-issue6 pytest -q tests/test_study_*.py
PYTHONPATH="$PWD/src" conda run -n oge-issue6 pytest -q tests/test_training_resume.py
PYTHONPATH="$PWD/src" conda run -n oge-issue6 python -m compileall -q src tests scripts
PYTHONPATH="$PWD/src" conda run -n oge-issue6 pytest -q
```

Results:

- focused study tests: 50 passed in 1.86s;
- training resume/deferred-ID-test tests: 36 passed in 2.55s;
- compileall: passed with no output;
- complete regression: 167 passed in 5.68s.

Before the runtime-fix commit, the combined focused command also passed 86
tests and the complete suite passed 167 tests. The clean-HEAD results above are
the acceptance evidence.

### CLI help and frozen bundle validation

```bash
PYTHONPATH="$PWD/src" conda run -n oge-issue6 \
  python scripts/run_optimizer_study.py --help
```

The CLI exposed the required phase, data/artifact roots, explicit GPU selectors,
concurrency, dry-run, smoke-only, two-smoke-trial, and retry options.

The frozen bundle was regenerated in memory and checked with
`verify_materialized_discovery_bundle(...)`. The observed bundle contained 16
rows for each of `sgd`, `sgdw`, `adam`, and `adamw`, 64 total rows, and manifest
hash:

```text
5b2f915d2337924cbb67077216bbcc4a49835f7927cea96c45004f0b76576b54
```

### Dry-run

External dry-run root:

```text
/home/ghjin/0707_exp/issue22_artifacts/674acf230115/dry_run
```

Exact command:

```bash
PYTHONPATH="$PWD/src" conda run --no-capture-output -n oge-issue6 \
  python scripts/run_optimizer_study.py \
  --study-config configs/studies/wrn28_10_optimizer_hpo_v1_1/study.yaml \
  --phase discovery \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e \
  --artifact-root /home/ghjin/0707_exp/issue22_artifacts/674acf230115/dry_run \
  --gpus 0,1 \
  --concurrency 2 \
  --smoke-only \
  --smoke-trials 2 \
  --dry-run
```

Result: `smoke_only_dry_run_completed`, effective concurrency 2, two ordered
trials, zero terminal attempts, zero GPU-hours, and no attempt directory. The
root contains only the ordered plan, study manifest, study summary, and their
valid SHA256 sidecars. Size was 32 KiB.

### Bounded two-GPU smoke

External smoke root:

```text
/home/ghjin/0707_exp/issue22_artifacts/674acf230115/smoke_2gpu
```

Exact command:

```bash
PYTHONPATH="$PWD/src" conda run --no-capture-output -n oge-issue6 \
  python scripts/run_optimizer_study.py \
  --study-config configs/studies/wrn28_10_optimizer_hpo_v1_1/study.yaml \
  --phase discovery \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e \
  --artifact-root /home/ghjin/0707_exp/issue22_artifacts/674acf230115/smoke_2gpu \
  --gpus 0,1 \
  --concurrency 2 \
  --smoke-only \
  --smoke-trials 2
```

The smoke root is outside the Git checkout and separate from the dry-run and
all production roots. Its retained size is 1.1 GiB.

## Smoke results

Study ID:

```text
wrn28_10_optimizer_hpo_v1_1__smoke_only__674acf230115
```

Study status was `smoke_only_completed`. Accounting was two planned, two
terminal, two completed, zero failed, and zero production slots assigned.
Recorded total GPU-hours were `0.05093352050636895`.

| trial | parent GPU / actual UUID | UTC interval | elapsed | final ID-validation |
| --- | --- | --- | ---: | --- |
| `smoke-00-sgd` | 0 / `GPU-b7682b36-2ce0-802b-8bb1-e7668087ecde` | 08:17:21.285–08:18:52.996 | 91.708s | accuracy 0.624, NLL 1.1611894631 |
| `smoke-01-sgd` | 1 / `GPU-8f45a2e7-59dd-831b-efc4-1dc6e10e4fee` | 08:17:21.287–08:18:52.941 | 91.652s | accuracy 0.605, NLL 1.1496942420 |

Both metrics were finite. Both `best_validation` and `final_validation` were at
completed epoch 1. The attempt intervals overlapped for 91.653468 seconds.
Each child recorded exactly one visible GPU, local device `cuda:0`, the parent
`CUDA_VISIBLE_DEVICES` selector, expected physical UUID, and PyTorch-observed
visible UUID. Expected and actual UUIDs matched for both attempts.

A mid-run `nvidia-smi` process snapshot was not retained. Concurrency evidence
comes from overlapping attempt timestamps plus two distinct child-observed GPU
UUIDs and independent subprocess artifacts.

## Artifact, checkpoint, and checksum validation

The repository `load_torch_artifact(..., map_location="cpu")` loaded all four
checkpoints. Every checkpoint had the expected role, `completed_epoch: 1`, and
execution Git SHA.

| trial | artifact | SHA256 |
| --- | --- | --- |
| `smoke-00-sgd` | `last.pt` | `778b5985e9429f362abdad5e4adb6f43c80322284d89edbe1613105f7cbe0743` |
| `smoke-00-sgd` | `best_val.pt` | `48f794dd6340e600a85f65dc6bd1e5f297475eec89cc31d254b3c4f8bea235d9` |
| `smoke-01-sgd` | `last.pt` | `3ecf759afe2d47a6aea369b69dd24ddfe077f1ee0f26129421d0cbe8bd5240b5` |
| `smoke-01-sgd` | `best_val.pt` | `7f6d27c3859b0a4abef150bae788f8ab8bc60aec14d40130993b989d99a8506a` |

Recorded SHA256 values were independently recomputed for each child summary,
history, resolved config, run metadata, environment, and checkpoint. All
matched. `sha256sum -c` passed for `ordered_plan.json`, `study_manifest.json`,
`study_summary.json`, both `trial.json` files, and both `attempt.json` files.
No checkpoint, log, history, attempt, trial, or study runtime artifact is tracked
by Git.

## Deferred ID-test boundary

The focused sentinel unit test proves that the deferred code path does not call
the ID-test evaluator and does not iterate the sentinel ID-test loader. That is
code-path evidence, not a claim inferred from missing files.

The actual smoke separately observed:

- child commands contained `--defer-id-test`;
- run metadata recorded `id_test_evaluation: deferred`;
- summaries recorded `id_test.status: deferred`, `metrics_available: false`,
  and `artifacts_created: false`;
- summaries contained no final/best ID-test metric fields or artifact paths;
- no `evaluation/final_id_test.json` or
  `evaluation/best_val_id_test.json` existed anywhere under the smoke root;
- console logs only reported completion and artifact paths, with no ID-test
  metric output.

## Failures, reruns, and remaining limitations

- Pre-smoke implementation defect: fixed and tested at `674acf2`.
- GPU smoke failures: none.
- Infrastructure retries or replacement trials: none.
- Scientific/config failures: none.
- Selective or result-driven reruns: none.
- Only two one-epoch SGD anchor smoke trials were run. This does not validate
  200-epoch behavior, the competitive search region, ranking outcomes,
  confirmation/final phases, or scientific conclusions.
- The code-enforced sentinel test establishes deferred loader/evaluator
  non-use. Actual smoke contributes metadata and artifact observations but is
  not treated as an independent runtime trace of every loader call.
- No active-process `nvidia-smi` snapshot was retained during the overlapping
  interval; the distinct child UUID and timestamp evidence is retained instead.

Actual HPO was not started.
