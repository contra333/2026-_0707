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
- `precision_medicine`: runtime candidate PASS at clean local validation commit
  `d56762d0bdc7d8fbac173cbec94a347a3586b518`; the dataset root was not
  available under the inspected user and project storage, so loader validation
  and the actual-data one-epoch smoke remain NOT_RUN.
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

## Setup-error boundary

A command typo or missing parent directory before environment/package creation
is a recoverable setup error in the same attempt. Preserve a short note, fix
the command, and continue. A new preserved attempt is required only after a
material environment installation or scientific run has begun.

## Remaining authorization boundary

After all three runtime candidates, dataset loaders, and one-epoch smokes pass,
record the observed values in this report and the Pull Request. A separate
execution Issue must still authorize the 200-epoch canary and production grid.
