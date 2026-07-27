# Issue 37 practical three-server runtime status

## Scope

Issue #37 now validates only the runtime and actual-data smoke conditions
needed before a separate production-execution Issue. It does not require
benchmark-grade Conda/cache provenance, DataLoader throughput studies,
cross-GPU numerical parity, immutable GPU UUID shards, or runner-side freeze
artifacts.

No 200-epoch production cell ran. No protected ID-test, OOD, geometry, Neural
Collapse, or detector result was accessed.

## Required checks

Each approved host must report:

- Python 3.11, with the exact patch and executable recorded;
- the hash-locked runtime/test distribution set, including
  PyTorch `2.5.1+cu121` and TorchVision `0.20.1+cu121`;
- `pip check`, the complete regression suite, and bounded CUDA/cuDNN probes;
- the effective FP32, TF32, float32-matmul, cuDNN, AMP/BF16, and
  deterministic-algorithm values;
- a clean common Git SHA and the frozen dataset membership;
- an actual-data one-epoch smoke using the existing study runner.

Python patch, driver, installation prefix, GPU UUID, and GPU model are recorded
metadata. Runtime distributions and the effective numerical policy must agree.
The production execution plan must place every optimizer on every host, but
Issue #37 does not generate or freeze a UUID-based shard.

The existing protocol settings remain:

```text
num_workers=0
persistent_workers=false
```

## Current observations

- `curie`: preliminary runtime candidate PASS at Git
  `300783d820df318380f2a84c6ebdba939fbf724b`; actual-data one-epoch smoke
  remains NOT_RUN.
- `precision_medicine`: runtime candidate PASS, committed holdout verification
  PASS, bounded actual-data loader PASS, and actual-data one-epoch smoke PASS.
  The data/smoke checks ran from clean execution Git SHA
  `e9bfde43bb40f3ea2a6a11da9da86178049ecc40`.
- `lise`: runtime, complete-suite, CUDA/cuDNN, committed-holdout path/loader,
  and actual-data one-epoch smoke PASS at clean Git
  `e9bfde43bb40f3ea2a6a11da9da86178049ecc40`.

The detailed external attempt logs and wheel bundles remain outside Git.

## Consolidated three-host result

The server reports support two complete host-level practical validations, not
a completed three-host gate:

The chronology matters when interpreting that result. `curie` was validated
first under the earlier benchmark-grade environment plan and produced the
candidate runtime and offline bundle. Repeated transfer and recovery failures
on `precision_medicine` showed that matching Conda prefixes, cache and
repodata state, installation ownership, paths, and other host-local details
was unnecessary and brittle. Issue #37 and PR #38 were then narrowed to the
portable scientific minimum: the hash-locked runtime/test distributions,
effective numerical policy, clean code/data identity, bounded CUDA/cuDNN
behavior, actual-data loaders, and one existing-runner epoch.

`precision_medicine` and `lise` were made equivalent only on that required
surface. They are not claimed to be exact replicas of the `curie` system
environment. Python patch, driver, GPU model and UUID, installation prefix,
installer/bootstrap mechanism, Conda/cache state, and filesystem layout may
differ as recorded metadata or excluded infrastructure.

| host | runtime, full suite, CUDA/cuDNN | committed holdout and bounded loader | actual-data one-epoch smoke | execution Git SHA | current verdict |
| --- | --- | --- | --- | --- | --- |
| `curie` | preliminary PASS on the earlier branch state (`182 passed`; bounded CUDA probes passed) | preliminary regeneration/hash and loader PASS on the earlier branch state; current-scope rerun required | NOT_RUN | `300783d820df318380f2a84c6ebdba939fbf724b` | INCOMPLETE |
| `precision_medicine` | PASS (`175 passed` after the preserved fixture correction; bounded CUDA/cuDNN probes passed) | PASS | PASS (`smoke_only_completed`, 1/1) | `e9bfde43bb40f3ea2a6a11da9da86178049ecc40` | PASS |
| `lise` | PASS (`175 passed`; bounded CUDA/cuDNN probes passed on idle GPU 0) | PASS | PASS (`smoke_only_completed`, 1/1) | `e9bfde43bb40f3ea2a6a11da9da86178049ecc40` | PASS |

Within the deliberately narrow required surface, the observed
`precision_medicine` and `lise` runtime distributions and effective numerical
policies agree: Python `3.11.9`, PyTorch
`2.5.1+cu121`, TorchVision `0.20.1+cu121`, CUDA runtime `12.1`, cuDNN
`90100`, FP32 with AMP/BF16 disabled, float32 matmul precision `highest`, CUDA
matmul TF32 disabled, cuDNN TF32 enabled, cuDNN benchmark/deterministic
disabled, and deterministic algorithms disabled. Driver versions, GPU models,
GPU UUIDs, and installation prefixes differ only in record-only fields.

The `curie` report used the same Python, PyTorch, TorchVision, CUDA-runtime, and
cuDNN versions, but it predates the clean execution SHA used by the other two
hosts and does not contain the current-scope one-epoch smoke. Therefore the
three-host common-SHA and effective-policy gate is not yet established.

### Acceptance-criteria ledger

| Issue #37 acceptance criterion | status | evidence or remaining work |
| --- | --- | --- |
| all three hosts report runtime PASS or an explicit blocker | PARTIAL | `precision_medicine` and `lise` PASS; `curie` has only a preliminary earlier-SHA PASS |
| exact runtime/test distributions and effective numerical policy agree | PARTIAL | agreement is confirmed for `precision_medicine` and `lise`; current-scope `curie` reconciliation remains |
| complete regression suite and bounded CUDA/cuDNN probes pass on every host | PARTIAL | current-scope PASS on two hosts; the `curie` result is from the earlier branch state |
| committed CIFAR-10 holdout paths and loaders pass on every training host | PARTIAL | PASS on two hosts; current-scope `curie` path/loader validation remains |
| one actual-data epoch completes on every approved host | PARTIAL | PASS on two hosts; `curie` is NOT_RUN |
| repository records commands, outcomes, NOT_RUN work, and external artifact locations | PARTIAL | recorded for completed reports; the final `curie` commands and artifacts remain |
| no 200-epoch production cell or protected-evidence access | PASS | no such run or access was reported on any host |

Overall Issue #37 status is **INCOMPLETE**. Production execution remains
unauthorized until the `curie` current-scope runtime, numerical-policy,
actual-data loader, and one-epoch smoke report is added and the three-host
comparison is rerun.

## `curie` preliminary runtime observation

The Issue comment for `curie` records a preliminary candidate at clean Git
`300783d820df318380f2a84c6ebdba939fbf724b`, before the later
`precision_medicine` fixture correction and documentation commits. The
observed candidate was:

- Python `3.11.9`;
- PyTorch `2.5.1+cu121` and TorchVision `0.20.1+cu121`;
- CUDA runtime `12.1` and cuDNN `90100`;
- four NVIDIA RTX A5000 GPUs;
- `pip check` PASS;
- bounded FP32 matmul probes PASS on all four GPUs and a bounded cuDNN
  convolution PASS on GPU 0;
- the then-current complete suite PASS with `182 passed in 11.45s`;
- deterministic committed-holdout regeneration/hash and bounded-loader checks
  reported PASS.

The first external inventory invocation expected `torch==2.5.1` instead of
the approved local version string `torch==2.5.1+cu121`. The expectation was
corrected and the inventory was rerun; both the original harness failure and
the corrected evidence were preserved.

The external export root was
`/home/ghjin/0707_exp/issue37_artifacts/300783d/exports/`. No benchmark,
cross-GPU parity run, model smoke, one-epoch training smoke, 200-epoch run, or
protected-evidence access was reported. Because this preliminary evidence is
not the current common-SHA practical run, it does not replace the remaining
`curie` validation.

## `precision_medicine` runtime observation

The approved host resolved as `math-SYS-740GP-TNRT` with four NVIDIA RTX A6000
GPUs. The observed runtime was:

- Python `3.11.9`;
- PyTorch `2.5.1+cu121` and TorchVision `0.20.1+cu121`;
- CUDA runtime `12.1`, cuDNN `90100`, and NVIDIA driver `535.183.01`;
- NumPy `1.26.4`, Pillow `10.4.0`, PyYAML `6.0.2`, and scikit-learn `1.5.2`;
- FP32 parameters, activations, and storage with AMP and BF16 disabled;
- float32 matmul precision `highest`;
- CUDA matmul TF32 disabled and cuDNN TF32 enabled;
- cuDNN benchmark and deterministic modes disabled and deterministic
  algorithms disabled.

The existing offline bundle and hash locks were reused without modification.
The bootstrap tools were installed with `--no-deps`; the runtime and test
locks were then installed in one hash-checked transaction so the test lock's
explicit `packaging==26.0` satisfied Wheel's dependency. `pip check`, package
imports, a bounded FP32 matmul on all four GPUs, a bounded cuDNN convolution,
and the complete regression suite passed.

The first full-suite run reported `174 passed, 1 failed`: a CPU fixture assumed
that NVIDIA driver metadata must be absent even on a CUDA-capable host. The
runner correctly recorded the host driver as required by this Issue. Local
commit `d56762d0bdc7d8fbac173cbec94a347a3586b518` makes the test conditional on
CUDA availability. The clean-commit rerun reported `175 passed`.

The principal validation commands were:

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  PIP_NO_CACHE_DIR=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  -m pip install --no-index \
  --find-links /mnt/drive/lab1/oge/staging/issue37/300783d/precision_candidate_a2d_runtime/curie_candidate_a2_pip_offline_bundle/wheelhouse \
  --only-binary=:all: --require-hashes --no-deps \
  -r /mnt/drive/lab1/oge/staging/issue37/300783d/precision_candidate_a2d_runtime/curie_candidate_a2_pip_offline_bundle/locks/requirements-bootstrap-tools.lock

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  PIP_NO_CACHE_DIR=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  -m pip install --no-index \
  --find-links /mnt/drive/lab1/oge/staging/issue37/300783d/precision_candidate_a2d_runtime/curie_candidate_a2_pip_offline_bundle/wheelhouse \
  --only-binary=:all: --require-hashes \
  -r /mnt/drive/lab1/oge/staging/issue37/300783d/precision_candidate_a2d_runtime/curie_candidate_a2_pip_offline_bundle/locks/requirements-runtime.lock \
  -r /mnt/drive/lab1/oge/staging/issue37/300783d/precision_candidate_a2d_runtime/curie_candidate_a2_pip_offline_bundle/locks/requirements-test.lock

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  -m pip install --no-index --no-deps --no-build-isolation \
  -e /home/lab1/ghjin/2026-_0707-issue37

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  -m pip check

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  /home/lab1/precision_a2d_cuda_probe.py

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  -m pytest -q -p no:cacheprovider
```

The external reports are under
`/mnt/drive/lab1/oge/artifacts/issue37/f7558fb/precision_runtime_recovery/`.
They contain install reports, `pip check`, the package inventory, the effective
numerical environment, the CUDA/cuDNN probe, the preserved failed test run, and
the clean-commit passing test run. They contain no dataset, checkpoint, model,
or protected-evidence artifact.

## `precision_medicine` actual-data validation

The copied OpenOOD data root
`/home/lab1/ghjin/data/openood-v1.5-3c35632e` was validated on 2026-07-28
KST from clean Git SHA
`e9bfde43bb40f3ea2a6a11da9da86178049ecc40`. The pre-run and pre-document
fetches both left the local and remote task branch at that exact SHA with no
intervening code or configuration change.

The bounded validation results were:

- holdout validator: PASS. Deterministic 45k/5k/10k regeneration and the
  60,001-row manifest were byte-identical to the committed files. All three
  splits had zero missing images, zero duplicate sample IDs, and labels in
  `[0, 9]`;
- loader check: PASS with `id_max_samples=8`, `ood_max_samples=8`,
  `batch_size=8`, `num_workers=0`, and `persistent_workers=false`.
  `id_train`, `id_validation`, and `id_test` each produced one
  `[8, 3, 32, 32]` finite `float32` batch with ID labels in `[0, 9]` and eight
  unique sample IDs. Re-reading item 0 from validation and test produced an
  identical tensor and sample ID;
- one-epoch smoke: PASS with study status `smoke_only_completed`, one planned
  and one completed trial, 352 optimizer steps, completed epoch 1, and valid
  `last.pt` and `best_val.pt` checkpoints. The study runner and an independent
  artifact verifier both confirmed that ID-test evaluation was deferred, no
  protected ID-test metric or artifact was created, and the evaluation
  directory was empty.

The pre-run compute-process query returned no rows for any of the four GPUs.
GPU 0 had 16 MiB reported memory use and 0% utilization, so the smoke selected
parent-visible GPU index 0:

- GPU UUID `GPU-372d0a23-9fd2-d5cb-7708-192a7527f1dd`;
- NVIDIA RTX A6000 with driver `535.183.01`;
- host `math-SYS-740GP-TNRT`;
- Python `3.11.9` at
  `/mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python`;
- PyTorch `2.5.1+cu121`, TorchVision `0.20.1+cu121`, CUDA runtime `12.1`, and
  cuDNN `90100`;
- FP32 parameters, activations, and storage; AMP and BF16 disabled; float32
  matmul precision `highest`; CUDA matmul TF32 disabled; cuDNN TF32 enabled;
  cuDNN benchmark, cuDNN deterministic, and deterministic algorithms disabled.

The exact data and smoke validation commands were:

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  -m pip check

set -o pipefail
env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  scripts/verify_cifar10_holdout.py \
  --data-root /home/lab1/ghjin/data/openood-v1.5-3c35632e \
  | tee /mnt/drive/lab1/oge/artifacts/issue37/e9bfde4/precision_data_smoke/holdout_validator.json

set -o pipefail
env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  /mnt/drive/lab1/oge/artifacts/issue37/e9bfde4/precision_data_smoke/loader_check.py \
  --repository-root /home/lab1/ghjin/2026-_0707-issue37 \
  --data-root /home/lab1/ghjin/data/openood-v1.5-3c35632e \
  | tee /mnt/drive/lab1/oge/artifacts/issue37/e9bfde4/precision_data_smoke/loader_check.json

nvidia-smi \
  --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  scripts/run_optimizer_study.py \
  --study-config configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml \
  --phase grid \
  --data-root /home/lab1/ghjin/data/openood-v1.5-3c35632e \
  --artifact-root /mnt/drive/lab1/oge/artifacts/issue37/e9bfde4/precision_data_smoke \
  --gpus 0 \
  --concurrency 1 \
  --smoke-only \
  --smoke-trials 1

set -o pipefail
env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /mnt/drive/lab1/oge/envs/oge-wrn-v1.2-pm-bootstrap/candidate-venv/bin/python \
  /mnt/drive/lab1/oge/artifacts/issue37/e9bfde4/precision_data_smoke/smoke_artifact_verify.py \
  --study-root /mnt/drive/lab1/oge/artifacts/issue37/e9bfde4/precision_data_smoke/wrn28_10_optimizer_hpo_v1_2__smoke_only__e9bfde43bb40 \
  | tee /mnt/drive/lab1/oge/artifacts/issue37/e9bfde4/precision_data_smoke/smoke_artifact_verification.json
```

All generated evidence remains outside Git under
`/mnt/drive/lab1/oge/artifacts/issue37/e9bfde4/precision_data_smoke/`. It
includes the holdout and loader JSON reports, the two small validation scripts,
the independent smoke verification JSON, the study records, the child console
log, and the epoch-1 checkpoints. No raw log, dataset, checkpoint, or large
artifact was added to the repository.

The existing runtime was reused. No environment creation, package
reinstallation, Conda/network/ToS/cache-parity procedure, DataLoader
throughput benchmark, or complete regression-suite rerun was performed during
this data/smoke validation. The 200-epoch canary and production grid remain
NOT_RUN. Protected ID-test, OOD, geometry, Neural Collapse, and detector
evidence remain NOT_RUN.

## `lise` runtime and actual-data observation

The approved host resolved as `lise` with two NVIDIA RTX A5000 GPUs. Validation
used physical GPU 0 only. Physical GPU 1 had another user's compute processes,
was not interrupted or probed, and was not required by the host-level practical
runtime gate.

The run used clean Git
`e9bfde43bb40f3ea2a6a11da9da86178049ecc40`. The transferred Curie inputs were
kept outside Git and verified before use:

- Python 3.11.9 conda-pack bootstrap SHA-256
  `2094a5936a13d2935f8fbed7df07a94c3dbc5d83ffcdd9139433dcab992f4c30`;
- pip offline bundle SHA-256
  `315901b8f870b0ff9b6f06520262fdaf9901d9755b06b0afca7e1f688a19abf9`;
- every bundle-internal lock, validator, metadata file, and wheel matched
  `PIP_BUNDLE_SHA256SUMS`.

The bootstrap was extracted to a separate prefix, relocated with
`conda-unpack`, and used to create an internal `candidate-venv`. The bootstrap
lock was installed offline with `--require-hashes --no-deps`. The runtime and
test locks were installed together in one offline hash-checked transaction,
and the repository was installed with
`--no-deps --no-build-isolation -e`. The nested-venv isolation check passed
without running Conda identity or ownership-parity validation.

The observed runtime was:

- Python `3.11.9`, executable
  `/home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python`;
- PyTorch `2.5.1+cu121` and TorchVision `0.20.1+cu121`;
- CUDA runtime `12.1`, cuDNN `90100`, and NVIDIA driver `580.159.03`;
- NumPy `1.26.4`, Pillow `10.4.0`, PyYAML `6.0.2`, and scikit-learn `1.5.2`;
- FP32 parameters, activations, and storage with AMP and BF16 disabled;
- float32 matmul precision `highest`;
- CUDA matmul TF32 disabled and cuDNN TF32 enabled;
- cuDNN benchmark and deterministic modes disabled and deterministic
  algorithms disabled.

`pip check` passed and the complete suite reported `175 passed in 7.90s`.
On idle physical GPU 0, the bounded FP32 matrix multiplication and bounded
cuDNN convolution both produced finite FP32 outputs. The driver, installation
prefix, and GPU UUID
`GPU-65d2f656-103f-0b29-f925-677903576efc` are record-only metadata.

`scripts/verify_cifar10_holdout.py` regenerated the committed holdout
byte-identically and verified all 45,000 train, 5,000 validation, and 10,000
test image paths with zero missing images and zero duplicate sample IDs. The
bounded actual-data loader check passed for all three ID roles with
`num_workers=0`, `persistent_workers=false`, FP32 finite tensors, and eight
unique sample IDs per role.

The existing study runner completed one SGD seed-0 actual-data epoch with
`--smoke-only --smoke-trials 1 --concurrency 1` on GPU 0. The study status was
`smoke_only_completed`, one of one planned trials completed, and the child
recorded the expected Git SHA, membership hashes, GPU UUID, runtime, and
numerical policy. ID-test evaluation remained `deferred`; no ID-test metrics
or evaluation artifacts were created. The epoch-1 validation accuracy
`0.522` and NLL `1.4128568719863892` are infrastructure-smoke observations,
not research evidence.

The principal validation commands were:

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  PIP_NO_CACHE_DIR=1 \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python \
  -m pip install --no-index \
  --find-links /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/bundle/curie_candidate_a2_pip_offline_bundle/wheelhouse \
  --only-binary=:all: --require-hashes --no-deps \
  -r /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/bundle/curie_candidate_a2_pip_offline_bundle/locks/requirements-bootstrap-tools.lock

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  PIP_NO_CACHE_DIR=1 \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python \
  -m pip install --no-index \
  --find-links /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/bundle/curie_candidate_a2_pip_offline_bundle/wheelhouse \
  --only-binary=:all: --require-hashes \
  -r /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/bundle/curie_candidate_a2_pip_offline_bundle/locks/requirements-runtime.lock \
  -r /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/bundle/curie_candidate_a2_pip_offline_bundle/locks/requirements-test.lock

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  PIP_NO_CACHE_DIR=1 \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python \
  -m pip install --no-index --no-deps --no-build-isolation \
  -e /home/ghjin/0727ICLR실험/2026-_0707-issue37

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python \
  -m pip check

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python \
  -m pytest -q -p no:cacheprovider

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python \
  scripts/verify_cifar10_holdout.py \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/bundle/curie_candidate_a2_pip_offline_bundle/validators/issue37_loader_check.py \
  --repository-root /home/ghjin/0727ICLR실험/2026-_0707-issue37 \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES=0 \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/bundle/curie_candidate_a2_pip_offline_bundle/validators/issue37_cuda_probe.py \
  --physical-index 0 \
  --physical-uuid GPU-65d2f656-103f-0b29-f925-677903576efc

env -u LD_LIBRARY_PATH -u PYTHONPATH PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES=0 \
  /home/ghjin/0727ICLR실험/issue37_lise_runtime/e9bfde43/python3119-bootstrap/candidate-venv/bin/python \
  scripts/run_optimizer_study.py \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e \
  --artifact-root /home/ghjin/0727ICLR실험/issue37_lise_artifacts/e9bfde43/lise_runtime_smoke_20260728/study \
  --gpus 0 --concurrency 1 --smoke-only --smoke-trials 1
```

The external reports and smoke artifacts are under
`/home/ghjin/0727ICLR실험/issue37_lise_artifacts/e9bfde43/lise_runtime_smoke_20260728/`.
They contain the transfer and bundle checksum results, pip install reports,
package inventory, `pip check`, numerical-policy record, complete-suite log,
CUDA/cuDNN probe, holdout and loader reports, and the existing runner's
attempt/checkpoint artifacts. A first install-report-summary `jq` expression
had a post-install array-scope error; the corrected summary passed and the
error did not affect installation, tests, data validation, or the smoke.

## Setup-error boundary

A command typo or missing parent directory before environment/package creation
is a recoverable setup error in the same attempt. Preserve a short note, fix
the command, and continue. A new preserved attempt is required only after a
material environment installation or scientific run has begun.

## Remaining authorization boundary

After all three runtime candidates, dataset loaders, and one-epoch smokes pass,
record the observed values in this report and the Pull Request. A separate
execution Issue must still authorize the 200-epoch canary and production grid.
