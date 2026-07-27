# Issue 37 three-server preflight: blocked external validation

## Scope

This report records the desktop-side implementation and the current external
blocker for Issue #37. It is not a CUDA validation report, throughput result,
parity result, host-shard freeze, one-epoch smoke, or production authorization.

No 200-epoch production cell ran. No ID-test, OOD, geometry, Neural Collapse,
or detector result was accessed.

## Implemented locally

- versioned per-host environment capture with dependency-lock, Git, software,
  CUDA/cuDNN/driver, GPU UUID/process, storage/inode, numerical-policy, and
  backup-path fields;
- fail-closed three-host comparison across `curie`, `lise`, and
  `precision_medicine`;
- explicit rejection of RTX PRO 6000/Blackwell inventory;
- PyTorch legacy `allow_tf32` and PyTorch 2.9+ `fp32_precision` observation,
  active-family identification, single-family application, and runtime
  revalidation;
- full-local-GPU concurrent WRN one-epoch DataLoader benchmark tooling for
  workers 0/4/8 and the predeclared common 15% gate;
- deterministic largest-remainder and LR/WD-rank-balanced host-shard
  generation;
- synthetic same-initial-state WRN forward and one-step parity probe with
  operational `atol=1e-5`, `rtol=1e-4`;
- study-runner gates for the preflight freeze, dependency-lock identity,
  throughput decision, immutable host manifest, and exact approved GPU UUIDs.

The current training path remains `num_workers=0`,
`persistent_workers=false`. A multi-worker path is not implemented or adopted
without the required measurements and view-equivalence evidence.

## External access attempt

From the desktop WSL environment on 2026-07-27, this non-interactive command
was attempted:

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 curie 'hostname && id -un'
```

Observed result:

```text
ssh: Could not resolve hostname curie: Temporary failure in name resolution
```

The desktop has no configured `curie`/`lise` SSH aliases, resolvable department
DNS, or known hostname for the `precision_medicine` host. Therefore no remote
environment installation or measurement was performed.

## Intentionally unresolved gates

- `configs/environments/wrn_v1_2_common.yaml` remains
  `pending_server_measurement`; it is not an installable frozen lock.
- `precision_medicine.expected_hostname` remains `UNSPECIFIED` and makes
  comparison fail closed until the actual hostname is recorded.
- no common Python/PyTorch/TorchVision CUDA wheel is claimed installed;
- no TF32 or `fp32_precision` server value is claimed;
- no DataLoader wall time or 15% decision is claimed;
- no A5000/A6000 parity is claimed;
- no current GPU UUID inventory or host shard is frozen;
- no department-server holdout path/loader smoke is claimed;
- no one-epoch host smoke is claimed.

## Required continuation

The server-side Codex CLI must continue from the same Issue #37 branch. It must
first read `AGENTS.md`, the active Issue, and cards 04, 05, and 07. The
following order is mandatory.

1. Record the real `precision_medicine` hostname in `infrastructure.yaml`.
2. Install one identical Python/PyTorch/TorchVision CUDA environment on all
   three approved hosts. Fill `configs/environments/wrn_v1_2_common.yaml` with
   the observed exact versions and change its status to `frozen`.
3. Commit those two files, synchronize that clean Git SHA to all three hosts,
   and confirm that no approved GPU has an active compute process.
4. On each host, capture one preflight record:

```bash
python scripts/capture_server_preflight.py \
  --host-id <curie|lise|precision_medicine> \
  --dependency-lock configs/environments/wrn_v1_2_common.yaml \
  --artifact-root <absolute-artifact-root> \
  --backup-root <absolute-backup-root> \
  --output <host-id>_preflight.json
```

5. Bring the three small records into one clean checkout and compare them:

```bash
python scripts/compare_server_preflights.py \
  --report curie_preflight.json \
  --report lise_preflight.json \
  --report precision_medicine_preflight.json \
  --output preflight_freeze.json
```

   A hostname, Git SHA, lock hash, package, CUDA runtime, cuDNN, numerical
   policy, GPU count/model, process-state, or forbidden-GPU failure is a stop
   condition. Do not edit captured JSON to make it pass.

6. On each host, benchmark workers 0/4/8 under full local-GPU concurrency.
   Use every approved GPU index: `0,1,2,3` on `curie`, `0,1` on `lise`, and
   `0,1,2,3` on `precision_medicine`.

```bash
python scripts/benchmark_dataloader_workers.py \
  --host-id <host-id> \
  --data-root <absolute-cifar10-root> \
  --gpus <comma-separated-local-indices> \
  --repeats 2 \
  --output <host-id>_throughput.json
```

7. Decide the common worker policy:

```bash
python scripts/decide_dataloader_workers.py \
  --measurement curie_throughput.json \
  --measurement lise_throughput.json \
  --measurement precision_medicine_throughput.json \
  --output throughput_decision.json
```

   If the status is `multiworker_pending_equivalence`, stop. The current runner
   intentionally supports only the measured `workers_zero_frozen` outcome;
   stateless augmentation and resume/view-equivalence work must be implemented
   and validated before a multi-worker result can become production-ready.

8. Generate the immutable 36-cell host manifest:

```bash
python scripts/generate_host_shards.py \
  --preflight-freeze preflight_freeze.json \
  --throughput-decision throughput_decision.json \
  --output host_manifest.json
```

9. Run one parity probe on each host, then compare all three:

```bash
python scripts/run_gpu_parity_probe.py \
  --host-id <host-id> \
  --device cuda:0 \
  --output <host-id>_parity.json

python scripts/compare_gpu_parity.py \
  --report curie_parity.json \
  --report lise_parity.json \
  --report precision_medicine_parity.json \
  --output parity_comparison.json
```

10. After the comparison passes, run `run_optimizer_study.py --dry-run` and
    then `--smoke-only --smoke-trials 1` separately on each host. Supply that
    host's exact full approved GPU UUID set from `host_manifest.json`.
    Smoke-only records consume zero production slots.

The server-side PR update must include the small JSON evidence, the exact
commands actually run, pass/fail output, and any still-unverified assumption.
It must not start the 200-epoch canary or production grid. Those runs require a
separate execution authorization after Issue #37 is fully accepted.
