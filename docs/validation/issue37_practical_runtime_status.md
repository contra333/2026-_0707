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
- `precision_medicine`: runtime installation is in progress; actual-data smoke
  remains NOT_RUN.
- `lise`: runtime installation and actual-data smoke remain NOT_RUN.

The detailed external attempt logs and wheel bundles remain outside Git.

## Setup-error boundary

A command typo or missing parent directory before environment/package creation
is a recoverable setup error in the same attempt. Preserve a short note, fix
the command, and continue. A new preserved attempt is required only after a
material environment installation or scientific run has begun.

## Remaining authorization boundary

After all three runtime candidates, dataset loaders, and one-epoch smokes pass,
record the observed values in this report and the Pull Request. A separate
execution Issue must still authorize the 200-epoch canary and production grid.
