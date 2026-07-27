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
- `lise`: runtime installation and actual-data smoke remain NOT_RUN.

The detailed external attempt logs and wheel bundles remain outside Git.

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

## Setup-error boundary

A command typo or missing parent directory before environment/package creation
is a recoverable setup error in the same attempt. Preserve a short note, fix
the command, and continue. A new preserved attempt is required only after a
material environment installation or scientific run has begun.

## Remaining authorization boundary

After all three runtime candidates, dataset loaders, and one-epoch smokes pass,
record the observed values in this report and the Pull Request. A separate
execution Issue must still authorize the 200-epoch canary and production grid.
