# Issue 10 CIFAR-10 Training Server Validation

## Validation scope and evidence boundary

- Validation date: 2026-07-13
- Hostname: `curie`
- Checkout: `/home/ghjin/0707_exp/2026-_0707`
- Repository branch: `feat/issue-10-cifar-training`
- Validation commit: `b67c0122aa341c97956b287d87b2d89cc3b4d512`
- Expected and observed remote validation HEAD:
  `b67c0122aa341c97956b287d87b2d89cc3b4d512`
- Conda environment: `oge-issue6`
- OpenOOD data root: `/home/ghjin/datasets/openood-v1.5-3c35632e`
- Artifact root: `/home/ghjin/0707_exp/issue10_artifacts/b67c012`

All test, CUDA, SGD, checkpoint, resume, Adam, and AdamW evidence in this
report was generated sequentially from the single validation commit above.
The validation-report and status updates were made afterward as documentation
only. No checkpoint from the validation commit was resumed after a repository
commit change. The repository working tree remained clean throughout runtime
validation, and no implementation source, test, configuration, pre-existing
dataset, Issue #6 artifact, or pre-existing Issue #10 artifact was changed.
Only the fresh, dedicated Issue #10 artifacts named in this report were
created and then updated by the required sequential resume.

This is infrastructure-only validation. The bounded losses and accuracies in
this report are not optimizer comparisons or research results.

## Repository and branch gate

The repository gate was checked before any training artifact was created.

```bash
git fetch origin
git switch --track origin/feat/issue-10-cifar-training
git rev-parse HEAD
git rev-parse origin/feat/issue-10-cifar-training
git merge-base --is-ancestor origin/main HEAD
git rev-list --left-right --count origin/main...HEAD
git status --short --branch
git diff --name-status origin/main...HEAD
git diff --check origin/main...HEAD
```

Observed results:

- local and remote task-branch HEAD both equaled the expected validation SHA;
- `origin/main` was an ancestor of the validation commit;
- `origin/main...HEAD` reported `0 1`, meaning behind 0 and ahead 1;
- the working tree was clean;
- `git diff --check origin/main...HEAD` exited 0 with no output;
- the 13 changed files were confined to Issue #10 training configuration,
  training/data source, training tests, and protocol/status documentation;
- no architecture, optimizer-semantics, OOD-evaluation, geometry, or unrelated
  refactor was present.

## Runtime environment

| component | observed value |
| --- | --- |
| Hostname | `curie` |
| Python | `3.13.11` (Anaconda build, GCC 14.3.0) |
| NumPy | `2.4.3` |
| PyTorch | `2.11.0+cu130` |
| TorchVision | `0.26.0+cu130` |
| CUDA runtime reported by PyTorch | `13.0` |
| cuDNN reported by PyTorch | `91900` (cuDNN 9.19.0) |
| NVIDIA driver | `580.119.02` |
| GPUs | 4 x NVIDIA RTX A5000, 24,564 MiB each |
| Training device | `cuda:0` |
| `oge` import path | `/home/ghjin/0707_exp/2026-_0707/src/oge/__init__.py` |
| Editable install source | `file:///home/ghjin/0707_exp/2026-_0707` |
| Editable install current-checkout check | passed |

The GPU inventory command was:

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
```

Before the fresh SGD run, GPU 0 reported 0% utilization and 15 MiB used.
During the run it reported active utilization and approximately 3,285 MiB
used.

The exact environment and bounded CUDA probe command was:

```bash
conda run -n oge-issue6 python -c 'import importlib.metadata as md, json, os, pathlib, platform, socket, subprocess, sys; import numpy as np; import torch; import torchvision; import oge; from oge.models import make_model; dist=md.distribution("oge"); raw=dist.read_text("direct_url.json"); direct=json.loads(raw) if raw else None; expected=(pathlib.Path.cwd()/"src/oge/__init__.py").resolve(); result={"hostname":socket.gethostname(),"python":sys.version,"python_executable":sys.executable,"numpy":np.__version__,"torch":torch.__version__,"torchvision":torchvision.__version__,"cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version() if torch.cuda.is_available() else None,"cuda_available":torch.cuda.is_available(),"cuda_device_count":torch.cuda.device_count(),"cuda_devices":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"oge_import_path":str(pathlib.Path(oge.__file__).resolve()),"expected_oge_path":str(expected),"oge_import_is_current_checkout":pathlib.Path(oge.__file__).resolve()==expected,"direct_url":direct}; a=torch.randn(256,256,device="cuda:0"); b=torch.randn(256,256,device="cuda:0"); c=a@b; result["cuda_matrix"]={"shape":list(c.shape),"finite":bool(torch.isfinite(c).all()),"device":str(c.device)}; model=make_model({"name":"wrn28_10","num_classes":10,"depth":28,"widen_factor":10,"dropout_rate":0.0}).to("cuda:0").eval(); x=torch.randn(2,3,32,32,device="cuda:0"); torch.cuda.synchronize(); import time; started=time.perf_counter(); exec_ctx=torch.inference_mode(); exec_ctx.__enter__(); logits,features=model(x,return_features=True); exec_ctx.__exit__(None,None,None); torch.cuda.synchronize(); result["wrn_forward"]={"input_shape":list(x.shape),"logits_shape":list(logits.shape),"features_shape":list(features.shape),"logits_finite":bool(torch.isfinite(logits).all()),"features_finite":bool(torch.isfinite(features).all()),"device":str(logits.device),"elapsed_seconds":time.perf_counter()-started}; print(json.dumps(result,indent=2,sort_keys=True))'
```

The probe ran through the editable install, recorded the versions and paths
above, performed a 256 x 256 CUDA matrix multiplication, and included this
bounded model operation:

```python
model = make_model(
    {
        "name": "wrn28_10",
        "num_classes": 10,
        "depth": 28,
        "widen_factor": 10,
        "dropout_rate": 0.0,
    }
).to("cuda:0").eval()
logits, features = model(
    torch.randn(2, 3, 32, 32, device="cuda:0"),
    return_features=True,
)
```

The matrix result was finite. The WRN forward produced finite logits
`[2, 10]` and finite penultimate features `[2, 640]`, and took 0.1065
seconds after synchronization.

## Test validation

The required complete suite was run exactly as:

```bash
conda run -n oge-issue6 pytest -q
```

Result: `116 passed in 5.30s`; command exit code 0. The command process wall
time observed outside pytest was 7.56 seconds.

The required CLI and diff checks were:

```bash
conda run -n oge-issue6 python scripts/train_cifar10.py --help
git diff --check
```

Both exited 0. CLI help completed in 2.98 seconds and displayed the required
`--config`, `--data-root`, `--run-dir`, `--device`, `--resume`, and
`--max-epochs` options. `git diff --check` produced no output; the execution
harness recorded it below 0.001 seconds (approximately 0.000007 seconds).

The complete suite included and passed the specifically requested checks:

| requested behavior | exercised test or path | result |
| --- | --- | --- |
| Full checkpoint readable with `torch.load(..., weights_only=True)` | `test_run_artifacts_checkpoint_schema_and_reload_logits` and actual `load_torch_artifact()` calls | passed |
| Scheduler boundaries | `test_multistep_scheduler_matches_one_based_epoch_boundaries` | passed for epochs 1–200 |
| CPU continuous 3 epochs vs 2+1 resume | `test_continuous_three_epochs_match_two_plus_one_resumed_epoch` | exact model, optimizer, scheduler, RNG, and metric equality passed |
| Atomic checkpoint failure | `test_atomic_checkpoint_failure_preserves_previous_complete_file` | prior file preserved and temporary file removed |
| Invalid resume rejection | parameterized incompatible-resume tests | passed |
| Artifact schemas and reload logits | `test_run_artifacts_checkpoint_schema_and_reload_logits` | passed |

The actual CUDA checkpoints produced below were also loaded through
`oge.training.load_torch_artifact`, whose implementation passes
`weights_only=True`, so the server environment independently exercised the
full optimizer, scheduler, RNG, resolved-config, and history containers.

## Actual data membership

The resolved run configuration recorded the following immutable membership.
The post-run validators reconstructed the loaders and asserted their dataset
lengths.

| split | sample count | imglist SHA256 |
| --- | ---: | --- |
| ID train | 50,000 | `9317e18980f1b11df74bba9d64223e57e69bd08c2943adbc75aec393a4976d1b` |
| ID validation | 1,000 | `6f64c4370bf3c61dab3308f7e8c5c94938b112b20d38cca77c1dfa0bcc5c1d89` |
| ID test | 9,000 | `84559ec5225425e7f7363058d5f9442ed21a39c7c66b63c3f1bb8b1ddb935e13` |

Train and validation `sample_count` are returned by the common engine but are
not persisted in history or summary. Their validation evidence is therefore
the recorded membership, reconstructed loader lengths, `drop_last=false`, and
391 train-loader batches per epoch. ID-test `sample_count=9000` is persisted
directly in both evaluation artifacts.

## Fresh actual-data CUDA SGD epoch

The requested fresh run directory did not exist before this command:

```bash
conda run -n oge-issue6 python scripts/train_cifar10.py \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e \
  --run-dir /home/ghjin/0707_exp/issue10_artifacts/b67c012/sgd_epoch_resume_smoke \
  --device cuda:0 \
  --max-epochs 1
```

The command exited 0 and printed `Completed epoch 1`. The history-recorded
training-plus-validation epoch time was 87.6237 seconds.

| quantity | epoch-1 observation |
| --- | ---: |
| Completed epoch | 1 |
| Global step | 391 |
| Train sample count | 50,000 |
| Train loss | 1.3851166911 |
| Train accuracy | 0.4958 |
| Validation sample count | 1,000 |
| Validation NLL | 1.1556532862 |
| Validation accuracy | 0.615 |
| Learning rate, both parameter groups | 0.1 |
| Final ID-test sample count | 9,000 |
| Final ID-test NLL | 1.1587340945 |
| Final ID-test accuracy | 0.6155555556 |

Every loss and accuracy was finite. History contained exactly one row, and
the row was epoch 1. The following required artifacts existed:

- `resolved_config.yaml`
- `run_metadata.json`
- `environment.json`
- `history.jsonl`
- `summary.json`
- `checkpoints/last.pt`
- `checkpoints/best_val.pt`
- `checkpoints/snapshots/epoch_0000.pt`
- `evaluation/final_id_test.json`
- `evaluation/best_val_id_test.json`

At this boundary, both full checkpoints identified epoch 1. The snapshot
identified initialized epoch 0. The full checkpoint recorded four CUDA RNG
states, a train-DataLoader generator state, nonempty SGD momentum state,
`MultiStepLR` state with `last_epoch=1`, the validation Git SHA, and the full
resolved configuration. `run_metadata.json` recorded `git_dirty=false` and no
resume events. The environment artifact contained one CUDA execution.

## Fresh checkpoint and reload validation

The post-run validator command was:

```bash
conda run -n oge-issue6 python /tmp/issue10_validate_sgd.py
```

It completed in 17.04 seconds and asserted all fresh-run counts, metrics,
history, schemas, membership hashes, Git metadata, scheduler/optimizer state,
CUDA RNG state, and artifact paths. It strictly reloaded both the final and
best-validation model states on `cuda:0`, produced finite logits with shape
`[128, 10]`, reran each checkpoint over all 9,000 ID-test samples, and matched
the stored NLL and accuracy within `1e-12`. This also directly confirmed that
`weights_only=True` could read both actual full checkpoints.

The epoch-1 checkpoint hashes observed immediately before resume were:

- `last.pt`: `a62a2aa065a4ab620b9f459923f78b8426c8634505cbbe4fa1a4fc6547c5ed6c`
- `best_val.pt`: `56c6ba6358a0df0210c91ab42d1519f14d187fde5c0314c8d36001dcc5447936`
- `epoch_0000.pt`: `096f35960db9320fa8fdc5a6442fba9411fe7d6e134350a40d985a6b37045bb2`

`last.pt` and `best_val.pt` were subsequently atomically updated by the
required epoch-2 resume. The initialized snapshot remained unchanged.

## Epoch-boundary CUDA resume

Only after the fresh run and its reload validation completed, the same run
directory and its epoch-1 `last.pt` were used:

```bash
conda run -n oge-issue6 python scripts/train_cifar10.py \
  --data-root /home/ghjin/datasets/openood-v1.5-3c35632e \
  --run-dir /home/ghjin/0707_exp/issue10_artifacts/b67c012/sgd_epoch_resume_smoke \
  --device cuda:0 \
  --resume /home/ghjin/0707_exp/issue10_artifacts/b67c012/sgd_epoch_resume_smoke/checkpoints/last.pt \
  --max-epochs 2
```

The command exited 0 and printed `Completed epoch 2`. The second
training-plus-validation epoch took 87.8779 seconds according to history.

| history epoch | global step | LR | train loss | train accuracy | validation NLL | validation accuracy | best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 391 | 0.1 | 1.3851166911 | 0.4958 | 1.1556532862 | 0.615 | yes |
| 2 | 782 | 0.1 | 0.8249335752 | 0.71084 | 0.7838229184 | 0.744 | yes |

The epoch-1 row was value-equivalent to the row captured before resume and
appeared exactly once. Epochs were exactly `[1, 2]`, steps were exactly
`[391, 782]`, and both epochs used LR 0.1 in both optimizer groups.

The final `last.pt` pointed to epoch 2 and global step 782. `best_val.pt` was a
valid epoch-2 checkpoint because epoch 2 won the documented validation
ordering. The scheduler state had `last_epoch=2`; both optimizer groups had LR
0.1, weight decays `[0.0005, 0.0]`, and nonempty momentum buffers.

`run_metadata.json` contained exactly one resume event from completed epoch 1.
`environment.json` contained exactly two CUDA executions. The top-level
resolved configuration recorded `max_epochs=2`.

Final and best-validation ID-test artifacts were both updated from epoch 2:

| checkpoint role | sample count | NLL | accuracy |
| --- | ---: | ---: | ---: |
| final | 9,000 | 0.7943944230 | 0.7425555556 |
| best validation | 9,000 | 0.7943944230 | 0.7425555556 |

The successful resume validator rerun strictly reconstructed and restored the
model, optimizer, scheduler, Python RNG, NumPy RNG, PyTorch CPU RNG, all four
CUDA RNG states, and the train-DataLoader generator state from the final full
checkpoint. Exact nested state comparisons passed. Final and best checkpoints
again produced finite `[128, 10]` logits, repeated evaluation logits were
bitwise equal, and full 9,000-sample reevaluation exactly matched the stored
evaluation artifacts.

Final full-checkpoint SHA256 values are:

- `last.pt`: `502baad5cfc0e43ee131a28de31194c9c9b388f4be6dcbfae418ba6d781ddeb1`
- `best_val.pt`: `50ac25a67439edbad5ca474c2c5e1e847e40020b39841af35974e09984c82a27`

CUDA bitwise equivalence between a continuous two-epoch run and this resumed
run was not required and was not claimed. The required deterministic CPU
equivalence is covered by the passing unit test.

## Actual-data CUDA Adam and AdamW one-batch smokes

After SGD resume validation was complete, Adam and AdamW were run sequentially
by a one-time helper:

```bash
conda run -n oge-issue6 python /tmp/issue10_adaptive_one_batch.py
```

The helper used the repository `load_training_config`,
`resolve_training_config`, `build_openood_cifar10_loaders(id_max_samples=128)`,
`make_model`, `make_optimizer`, and `train_one_epoch` paths. Both optimizers
used the exact requested configuration: LR 0.001, betas 0.9/0.999, epsilon
`1e-8`, weight decay 0.0005, and
`weights_only_no_bias_norm` parameter groups.

| optimizer | class | actual samples | steps | loss | finite | changed trainable parameter |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Adam | `torch.optim.adam.Adam` | 128 | 1 | 2.4999084473 | yes | `classifier.weight`, 6,400 changed elements |
| AdamW | `torch.optim.adamw.AdamW` | 128 | 1 | 2.4999084473 | yes | `classifier.weight`, 6,400 changed elements |

For each optimizer, the model and batch were on `cuda:0`, the loader contained
exactly one batch, optimizer state was created for 80 parameter tensors, and
the shared groups covered every trainable parameter exactly once. The decay
group contained 29 weight tensors with decay 0.0005; the no-decay group
contained 51 tensors with decay 0.0. Classifier weight was decayed,
classifier bias was not, all biases were excluded from decay, and all
BatchNorm parameters were excluded from decay.

The machine-readable artifact is:

`/home/ghjin/0707_exp/issue10_artifacts/b67c012/adaptive_one_batch_smoke.json`

It was created with exclusive-write mode, then parsed and asserted again. Its
size is 5,065 bytes and SHA256 is
`8318cd128befc366ef468ffc65b10c70ea13c9cb777f60178d128f7ed9c514b1`.
It records `smoke_only=true`, `infrastructure_only=true`, and
`research_result=false`. The adaptive losses and accuracies are not optimizer
performance evidence.

## Failures, retries, and implementation changes

No repository test, training command, checkpoint load, resume command, or
adaptive smoke failed. No implementation source, test, or configuration fix
was required.

The first post-resume validation-helper invocation exited 1 because the helper
compared a restored `cuda:0` model tensor directly with the CPU-mapped
checkpoint tensor. This was a validator-only device assertion, not a training
or checkpoint failure. It wrote no repository or run artifact. The exact
failing command was:

```bash
conda run -n oge-issue6 python /tmp/issue10_validate_resume.py
```

Two attempts to patch the `/tmp` helper encountered the execution sandbox's
`bwrap` loopback restriction. The successful device-normalizing retry command
was:

```bash
conda run -n oge-issue6 python -c 'import runpy, torch; original=torch.testing.assert_close; normalize=lambda value: value.detach().cpu() if isinstance(value, torch.Tensor) else value; torch.testing.assert_close=lambda left,right,*args,**kwargs: original(normalize(left),normalize(right),*args,**kwargs); runpy.run_path("/tmp/issue10_validate_resume.py",run_name="__main__")'
```

The rerun exited 0 in 17.39 seconds and all resume assertions passed. Because
the failed helper did not invoke the training runner, the run retained exactly
one resume event and two environment executions.

All three one-time helpers were removed after validation:

```bash
rm /tmp/issue10_validate_sgd.py /tmp/issue10_validate_resume.py /tmp/issue10_adaptive_one_batch.py
```

## Retained artifacts

The retained artifact root occupied approximately 697 MiB at validation time.
It contains:

- `/home/ghjin/0707_exp/issue10_artifacts/b67c012/sgd_epoch_resume_smoke/`
- `/home/ghjin/0707_exp/issue10_artifacts/b67c012/sgd_epoch_resume_smoke/checkpoints/last.pt`
- `/home/ghjin/0707_exp/issue10_artifacts/b67c012/sgd_epoch_resume_smoke/checkpoints/best_val.pt`
- `/home/ghjin/0707_exp/issue10_artifacts/b67c012/sgd_epoch_resume_smoke/checkpoints/snapshots/epoch_0000.pt`
- `/home/ghjin/0707_exp/issue10_artifacts/b67c012/sgd_epoch_resume_smoke/evaluation/final_id_test.json`
- `/home/ghjin/0707_exp/issue10_artifacts/b67c012/sgd_epoch_resume_smoke/evaluation/best_val_id_test.json`
- `/home/ghjin/0707_exp/issue10_artifacts/b67c012/adaptive_one_batch_smoke.json`

The existing Issue #6 dataset, archives, artifacts, and Conda environment were
not deleted or modified.

## Acceptance criteria addressed

The complete suite and server validation together covered the Issue #10
acceptance criteria: unchanged OpenOOD membership and preprocessing; common
WRN/model and optimizer/parameter-group factories; finite parameter-changing
training; no-grad evaluation; `none` and `MultiStepLR` boundaries; full and
snapshot checkpoint schemas; reload logits; best-validation ordering; exact
CPU deterministic resume; CUDA epoch-boundary state restoration; incompatible
resume rejection; run artifacts; existing regressions; and actual-data CUDA
SGD, Adam, and AdamW smokes.

## Checks not run and remaining limitations

- No 200-epoch training was run. A 200-epoch baseline must be a separate
  GitHub Issue with its own artifacts and validation.
- No HPO, multi-seed run, feature extraction, geometry or Neural Collapse
  metric, Mahalanobis/kNN/GMM detector, expanded OOD evaluation, dropout 0.3
  ablation, cosine scheduler, AMP, distributed training, or optimizer-result
  comparison was run.
- No PR was created and no merge was performed. PR review and merge remain.
- `environment.json` does not itself record hostname or NVIDIA driver; both are
  recorded in this report from direct server probes.
- Train and validation processed counts are not fields in history/summary;
  they were verified from immutable membership plus reconstructed loaders and
  full-loader traversal as described above.
- Only history's per-epoch training-plus-validation time is persisted. Exact
  end-to-end command wall time was not written as a run artifact.
- The bounded two-epoch and one-batch values validate infrastructure only and
  must not be treated as reproducible long-run performance or research
  evidence.
