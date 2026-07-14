# Issue 14 WRN-200 SGD Seed-0 Server Validation

## Validation scope and evidence boundary

This report records the completed and independently revalidated canonical
Issue #14 classifier run: WRN-28-10 trained for 200 epochs with SGD and seed
0. Here, **WRN-200 means WRN-28-10 trained for 200 epochs**, not a 200-layer
Wide ResNet.

The retained run covers one optimizer and one seed. It is a classifier
baseline and fixed-snapshot trajectory, not an optimizer comparison,
hyperparameter-optimization result, learned-feature geometry or Neural
Collapse result, or OOD-detector evaluation. The ID validation and ID-test
metrics below are reported without broader scientific interpretation.

The canonical training run was already complete before this post-run task.
No training, resume, additional seed, additional optimizer, feature
extraction, geometry metric, or OOD detector was run during this validation.

## Repository and branch gate

| item | observed value |
| --- | --- |
| Checkout | `/home/ghjin/0707_exp/2026-_0707` |
| Branch | `exp/issue-14-wrn200-sgd-seed0` |
| Pinned training SHA | `d3fb1db222e755fe721c78efd0eb52915dcef7fd` |
| Run metadata Git SHA | `d3fb1db222e755fe721c78efd0eb52915dcef7fd` |
| Run metadata `git_dirty` | `false` |
| Pre-documentation working tree | clean |
| Run ID | `d2a0993a-4e44-4e73-a038-523c4cb093b6` |

The post-run gate rechecked the branch, exact HEAD, and empty
`git status --porcelain`. No training process remained. The retained tmux
session `issue14-wrn200-sgd0` had a dead pane with exit status 0 and captured
output ending with `Completed epoch 200`. The external exit-code file also
contained `0`.

The pinned training SHA is distinct from the later documentation commit that
contains this report. Checkpoints record the pinned training SHA, not the
documentation commit.

## Runtime environment

The retained evidence under
`/home/ghjin/0707_exp/issue14_artifacts/d3fb1db/environment/`, the run
environment artifact, and the independent validator recorded:

| component | observed value |
| --- | --- |
| Validation date | 2026-07-14 |
| Hostname | `curie` |
| Conda environment | `oge-issue6` |
| Python | `3.13.11` (Anaconda build, GCC 14.3.0) |
| NumPy | `2.4.3` |
| PyTorch | `2.11.0+cu130` |
| TorchVision | `0.26.0+cu130` |
| CUDA runtime reported by PyTorch | `13.0` |
| cuDNN reported by PyTorch | `91900` (cuDNN 9.19.0) |
| NVIDIA driver | `580.119.02` |
| Selected physical GPU | index 0, NVIDIA RTX A5000 |
| Selected GPU UUID | `GPU-b7682b36-2ce0-802b-8bb1-e7668087ecde` |
| Training and validator device | `cuda:0` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| Visible CUDA device count | 1 |
| `oge` import path | `/home/ghjin/0707_exp/2026-_0707/src/oge/__init__.py` |
| Editable-install source | `file:///home/ghjin/0707_exp/2026-_0707` |

The editable `oge` import resolved to the current checkout. Immediately before
post-run recomputation, GPU 0 had 15 MiB used by the Xorg graphics context, 0%
utilization, and no compute process. Its current UUID matched the retained
selection exactly. No process was killed and the validator did not switch GPUs.

## Data membership

The validator compared the resolved config and reconstructed loaders against
the existing OpenOOD data root and the imglist SHA256 values in the Issue #6
server validation report.

| split | samples | imglist SHA256 |
| --- | ---: | --- |
| ID train | 50,000 | `9317e18980f1b11df74bba9d64223e57e69bd08c2943adbc75aec393a4976d1b` |
| ID validation | 1,000 | `6f64c4370bf3c61dab3308f7e8c5c94938b112b20d38cca77c1dfa0bcc5c1d89` |
| ID test | 9,000 | `84559ec5225425e7f7363058d5f9442ed21a39c7c66b63c3f1bb8b1ddb935e13` |

The retained manifest had an empty error list, zero missing images, and zero
duplicate sample IDs for these splits. Reconstructed loader lengths were
50,000, 1,000, and 9,000. The train loader had 391 batches per epoch;
validation and test both had `drop_last=false`.

## Canonical command

The persistent tmux pane launched:

```bash
exec bash '/home/ghjin/0707_exp/issue14_artifacts/d3fb1db/run_canonical.sh'
```

The script selected `GPU_INDEX=0` from `selected_gpu.txt` and executed this
canonical command without `--resume` or `--max-epochs`:

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n oge-issue6 \
  python scripts/train_cifar10.py \
    --config configs/training/cifar10_wrn28_10.yaml \
    --data-root /home/ghjin/datasets/openood-v1.5-3c35632e \
    --run-dir /home/ghjin/0707_exp/issue14_artifacts/d3fb1db/wrn28_10_sgd_seed0_200ep \
    --device cuda:0
```

The retained launcher gated execution on the pinned HEAD, a clean working
tree, and a previously nonexistent run directory and console log.

## Runtime and execution history

| item | observed value |
| --- | --- |
| Run metadata creation | 2026-07-14 15:44:01.676628 +09:00 |
| Exit-code file completion | 2026-07-14 20:38:30.141982 +09:00 |
| End-to-end artifact interval | 17,668.465 seconds (4:54:28.465) |
| Sum of 200 history epoch times | 17,606.791 seconds (4:53:26.791) |
| Exit code | 0 |
| Environment executions | 1 |
| Resume events | none (`[]`) |
| Resume console log | none |

The canonical console log contains one completion line. `run_metadata.json`
contains an empty `resume_events` list, and no resume log exists. The tmux pane
is dead with status 0; the retained session entry is not an active job.

The independent validator was run once as:

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n oge-issue6 \
  python /home/ghjin/0707_exp/issue14_artifacts/d3fb1db/tools/issue14_validate_wrn200_sgd0.py
```

It exited 0 and wrote `postrun_validation.json`; observed process wall time was
18.29 seconds.

## Training trajectory integrity

The resolved experiment-defining fields matched the unchanged reference
config exactly. Resolved-only fields recorded the actual data root, loaded
dataset definition and membership, and runtime device `cuda:0`. The external
config audit also recorded `exact_match=true`.

The independent validator observed:

- `summary.json`: `status=completed`, `completed_epoch=200`, and
  `global_step=78200`;
- exactly 200 history rows with epochs 1 through 200 and no gaps or duplicates;
- a 391-step increment per epoch and final global step 78,200;
- finite train loss, train accuracy, validation NLL, and validation accuracy
  in every row, with all accuracies in `[0, 1]`;
- two optimizer parameter groups following the same recorded LR schedule;
- checkpoint weight-decay groups `[0.0005, 0.0]`;
- best-validation ordering and all `is_best` flags consistent with accuracy,
  then NLL, then earliest epoch.

| epochs | learning rate used by both groups |
| --- | ---: |
| 1–60 | 0.1 |
| 61–120 | 0.02 |
| 121–160 | 0.004 |
| 161–200 | 0.0008 |

The best validation record was epoch 128 with accuracy `0.957` and NLL
`0.21887339928746224`. The final epoch validation record was accuracy `0.952`
and NLL `0.22305755364894866`.

## Checkpoints and snapshots

`last.pt`, `best_val.pt`, and all eight fixed snapshots loaded through the
repository API using `torch.load(..., weights_only=True)`. Full checkpoints
contained the required model, optimizer, scheduler, best-validation, embedded
history, resolved config, run ID, Git SHA, and RNG state. RNG state contained
Python, NumPy, PyTorch CPU, one visible CUDA device, and the train-DataLoader
generator.

Both full checkpoints loaded into fresh WRN-28-10 dropout-0.0 models with
`strict=True`, zero missing keys, and zero unexpected keys. Repeated evaluation
forwards were exact and produced finite logits `[4, 10]` and finite penultimate
features `[4, 640]`.

Every snapshot had `checkpoint_type=snapshot`, the expected completed epoch,
canonical run ID, pinned training SHA, protocol name, and model name. No
checkpoint-schema conflict with the training protocol was found.

| checkpoint | epoch | SHA256 |
| --- | ---: | --- |
| `last.pt` | 200 | `34c4556f6fe95bef9fccc927cb307c97ea9314d5dab962e34f39c1704dec9f86` |
| `best_val.pt` | 128 | `cb8aa818437dc5b55d34ec12bc50ad22a171bc7dba8a29fdd8afe8442a10c083` |
| `epoch_0000.pt` | 0 | `7996db358450c783c03a9d550b83afaa4508f19401e99d2f4d23b6a72e854f74` |
| `epoch_0060.pt` | 60 | `0009b250b1ca34f42c9835feb550509d108731d18512523caf7bcaedadca312e` |
| `epoch_0061.pt` | 61 | `5a0f27de6f9b5ccffc32a8e9a44ea6f3386640866841e24453b227f4df197b08` |
| `epoch_0120.pt` | 120 | `b0d67ac329cf6234e170df851cbbce8a29ab9f99c953b82f6861eaeef2a29bf5` |
| `epoch_0121.pt` | 121 | `18d3302be031e28f678153ba1f07b7fb9a77e48ecc7e21446f690f9224fe12ee` |
| `epoch_0160.pt` | 160 | `1dd95d081a034fe33414cf9ebaee25bc17de1235b94eb4530fa22f1e413e30de` |
| `epoch_0161.pt` | 161 | `86d14e3d199cc43e628d4bc6f8de5975e6b5989b90677abd5b5375b05896f619` |
| `epoch_0200.pt` | 200 | `25aba3b1e2e84d4ced2faf7672dabebef6707ea68099030f344abf76d9323835` |

## ID validation and ID-test results

The validator independently traversed the full 9,000-sample ID-test split and
full 1,000-sample ID-validation split for both checkpoints. Accuracy had to
match exactly. NLL had to match within absolute tolerance `1e-6` because the
FP32 run used `deterministic=false`.

| role | epoch | split | samples | stored/history NLL | recomputed NLL | NLL delta | stored/history accuracy | recomputed accuracy | accuracy delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final | 200 | ID test | 9,000 | 0.2023320838080512 | 0.2023320838080512 | 0.0 | 0.9554444444444444 | 0.9554444444444444 | 0.0 |
| best validation | 128 | ID test | 9,000 | 0.20186231083340114 | 0.20186231083340114 | 0.0 | 0.9524444444444444 | 0.9524444444444444 | 0.0 |
| final | 200 | ID validation | 1,000 | 0.22305755364894866 | 0.22305755364894866 | 0.0 | 0.952 | 0.952 | 0.0 |
| best validation | 128 | ID validation | 1,000 | 0.21887339928746224 | 0.21887339928746224 | 0.0 | 0.957 | 0.957 | 0.0 |

All metric deltas were exactly zero and therefore passed the tolerance. The
recomputed validation results confirmed the history-selected checkpoint.

## Accuracy anomaly review

Final ID-test accuracy was `0.9554444444444444`, above the Issue #14 review
trigger of `0.93`. The anomaly-review trigger did **not** fire. `0.93` is a
review trigger, not a baseline acceptance threshold.

## Failures, retries, and validator corrections

No repository preflight, canonical training command, checkpoint load, schema
check, strict reload, full recomputation, or validator assertion failed. The
validator passed on its first execution. No validator correction or retry was
required, and no implementation or run artifact was changed.

The first Phase 6 changed-file assertion exited 1 because the newly created
report was still untracked and therefore omitted by `git diff --name-only`.
The report was marked intent-to-add, without staging its contents, and the
two-file assertion then passed. That retry found one trailing blank line at
the end of this report through `git diff --check`; the blank line was removed
and the next diff gate passed. These were documentation-gate corrections, not
training, checkpoint, validator, source, config, or test failures.

## External artifacts

All run evidence remains outside Git under
`/home/ghjin/0707_exp/issue14_artifacts/d3fb1db`. The machine-readable result
is `postrun_validation.json`; the source is
`tools/issue14_validate_wrn200_sgd0.py`, with SHA256
`b1e8d198acc6f159d24e8d68106cb7fbc586b3b1272036e1b790508d7ea95e43`.

| artifact | SHA256 |
| --- | --- |
| `resolved_config.yaml` | `dfd84609f673afbbfed3d02137fe3b66d3f21f6918d373da18f38c41196f6aff` |
| `run_metadata.json` | `37b0fe822f2e126863ae3e874de2c631136be9b9a5e50f8b9ec7e133e4e07012` |
| `environment.json` | `5613a7cb2ee04a5858ada8ba22b870f02d63fb22885aa358b324b9e2098699b1` |
| `history.jsonl` | `869b5eadf9a69c07dac7a8a07b98b8467ddd2121b915c736e34d4463d83561ed` |
| `summary.json` | `c4c01b42ff657d41e165acfad6a24656435b87501f111a579e28268a0ebe8b35` |
| `evaluation/final_id_test.json` | `f409e71baa507fd2b8357bfa483a1fe4deea9c9fca0195ff5e56a9232f85e50a` |
| `evaluation/best_val_id_test.json` | `c543da3656a305fad5e77c0451d10f5ba2d23702c4a661c65a82fb132893de98` |
| OpenOOD manifest | `b4370a2c27235987676d8ff631f3cdd906e074b583fc8a09a8e41c179cf9da8c` |
| Canonical console log | `6bc82a8094e0b63632a1d118774e0a5145bf0c0d4671489d0909dc0d35539959` |

Checkpoints, datasets, logs, validator files, and large binaries are not
committed to the repository.

## Acceptance criteria addressed

### Repository and preflight

- [x] Exact pinned task-branch SHA and clean canonical-run working tree.
- [x] Complete server pytest preflight with retained count and runtime.
- [x] Training CLI help preflight.
- [x] Runtime versions, driver, GPU, hostname, Git SHA, and current-checkout
  editable `oge` import recorded.
- [x] ID counts and Issue #6 imglist hashes matched.
- [x] Fresh run-directory gate retained in the successful launcher.
- [x] One physical GPU exposed consistently as `CUDA_VISIBLE_DEVICES=0`.

### Canonical training run

- [x] Fresh command had no `--resume` and no `--max-epochs`.
- [x] Reference config was explicit and resolved fields matched.
- [x] Data root, definition, membership, and runtime device were checked.
- [x] Pinned Git SHA and `git_dirty=false` were recorded.
- [x] Epoch 200, summary completion, and global step 78,200 were checked.
- [x] Exactly 200 ordered, finite rows and LR boundaries were checked.
- [x] No resume event occurred.

### Artifacts, evaluation, and reload

- [x] All required run artifacts and eight snapshots existed.
- [x] All checkpoints loaded with `weights_only=True`.
- [x] Schema, run ID, pinned SHA, RNG state, optimizer groups, and hashes were
  checked.
- [x] Fresh strict reload passed for both full checkpoints.
- [x] Full ID-test recomputation passed for both checkpoints.
- [x] Full validation traversal confirmed checkpoint selection.
- [x] Metric deltas and tolerance are reported.
- [x] The 0.93 trigger did not fire and was not treated as a threshold.

### Documentation and delivery

- [x] This server-validation report was added.
- [x] `docs/STATUS.md` was updated only after validation passed.
- [x] The diff is restricted to this report and `docs/STATUS.md`.
- [x] No external artifact or binary is in the repository diff.
- [x] No interpretation beyond observed ID metrics is claimed.

The documentation commit, push, and PR are delivery actions after the final
diff and regression gates. The PR uses `Closes #14` and is not merged here.

## Checks not run and remaining limitations

- This task did not rerun or resume training.
- It did not run another seed, another optimizer, HPO, feature extraction,
  geometry or Neural Collapse metrics, OOD detection, or a 200-to-350 epoch
  extension.
- One SGD seed does not establish optimizer differences, seed stability,
  representation geometry, or OOD reliability.
- CUDA bitwise reproducibility across different stacks is not claimed. The
  actual same-server recomputation observed zero metric delta within the
  specified tolerance.
- The local WSL checkout lacked `oge-issue6`; its supplemental `.venv` result
  is not acceptance evidence. Required Conda validation ran on this server.

Post-documentation Phase 6 regression results:

- `conda run -n oge-issue6 pytest -q`: exit 0; `116 passed in 5.19s`.
- `conda run -n oge-issue6 python scripts/train_cifar10.py --help`: exit 0;
  required config, data-root, run-dir, device, resume, and max-epochs options
  displayed; execution-wrapper wall time 5.6 seconds.
