# Reference Card 05: Reproducible Classifier Training Protocol

## Purpose and authority boundary

This card defines the durable training, checkpoint, resume, and artifact
contract for the first classifier-training vertical slice. The reference path
is OpenOOD v1.5-aligned CIFAR-10 with WRN-28-10, but the training loop must use
the common model and optimizer factories rather than optimizer-specific loops.

The following cards remain authoritative for the inputs to this protocol:

- [`01_optimizers.md`](01_optimizers.md) defines optimizer semantics and the
  shared `weights_only_no_bias_norm` parameter-group policy.
- [`02_architectures.md`](02_architectures.md) defines WRN-28-10 and its
  penultimate-feature and dropout semantics.
- [`04_openood_v1_5_protocol.md`](04_openood_v1_5_protocol.md) defines CIFAR-10
  split membership and preprocessing.

Training must not regenerate OpenOOD splits, alter preprocessing, modify WRN
architecture behavior, or bypass `make_model()` or `make_optimizer()`.

## Reference configuration

[`configs/training/cifar10_wrn28_10.yaml`](../../configs/training/cifar10_wrn28_10.yaml)
is the first supported reference configuration. It selects:

- OpenOOD CIFAR-10 `id_train`, `id_validation`, and `id_test`;
- WRN-28-10 with `dropout_rate: 0.0`;
- unsmoothed cross entropy;
- SGD with learning rate `0.1`, momentum `0.9`, Nesterov enabled, and weight
  decay `0.0005` under the shared parameter-group policy;
- end-of-epoch `MultiStepLR` with milestones `[60, 120, 160]` and gamma `0.2`;
- 200 maximum epochs, batch size 128, seed 0, FP32, `num_workers: 0`, and
  strict deterministic-algorithm enforcement disabled;
- fixed snapshots at epochs 0, 60, 61, 120, 121, 160, 161, and 200.

These values are configuration, not training-engine constants. The resolved
configuration records all applied defaults, the actual data root, and the
loaded dataset definition rather than only the source config path.

The repository entry point is:

```bash
python scripts/train_cifar10.py \
  --data-root /path/to/openood-v1.5-root \
  --run-dir /path/to/run \
  --device cuda:0
```

`--max-epochs` is an explicit resolved-config override for bounded validation
and later extension. Resume uses the same run directory and its atomic
`last.pt`, for example `--resume /path/to/run/checkpoints/last.pt`. The
override value and every other effective setting are written to
`resolved_config.yaml`.

## DataLoader contract

The existing OpenOOD CIFAR-10 loader and transforms are reused. The train
loader shuffles with an explicit, seeded `torch.Generator`. The validation and
test loaders do not shuffle.

The first protocol uses these explicit settings:

```yaml
num_workers: 0
pin_memory: false
drop_last: false
persistent_workers: false
```

`training.drop_last` applies only to the train loader. Validation and test
always use `drop_last=False` so every member of the fixed split is evaluated.
The stateful generator belongs only to the shuffled train loader; validation
and test loaders do not require generator state for resume.

`persistent_workers=True` is not supported by this first protocol. In
particular, it is invalid with `num_workers=0`. Supporting worker processes in
a future protocol requires an explicit worker-seeding and resume contract.

## Training and evaluation semantics

The model is created by `make_model()`, moved to the configured device, and
only then passed to `make_optimizer(model, config)`. This ordering preserves
the optimizer's references to the device-resident parameters. A single
optimizer-independent training loop performs the following for every batch:

1. set training mode;
2. move images and class labels to the selected device;
3. clear gradients;
4. compute logits through the common model API;
5. compute mean `CrossEntropyLoss`;
6. backpropagate and take one optimizer step;
7. increment `global_step` once.

The first protocol is FP32 only. It does not use autocast, `GradScaler`, AMP,
or distributed training.

Validation runs after every completed training epoch with `model.eval()` and
without gradient creation. Validation NLL is the sample-weighted mean cross
entropy and accuracy is the number of correct predictions divided by the full
validation-split size. Final ID-test evaluation is performed for both the
final (`last.pt`) state and the selected `best_val.pt` state. Test data never
selects a checkpoint or changes training.

## Epoch, global-step, and scheduler semantics

Persisted and reported training epochs are 1-based completed-epoch numbers:

- epoch 1 is the first pass over the train loader;
- `completed_epoch: 0` means no training epoch has completed;
- `global_step` is the total number of completed optimizer steps;
- `epoch_0000.pt` is the initialized model before any optimizer update;
- `epoch_NNNN.pt` for `NNNN > 0` is the model after that epoch completed.

For `scheduler.name: none`, no learning-rate scheduler is stepped and the
checkpoint scheduler state is `null`.

For `scheduler.name: multistep`, construct PyTorch `MultiStepLR` from the
resolved milestones and gamma. Validation completes first, then
`scheduler.step()` is called exactly once at the end of each completed epoch.
The checkpoint boundary is after that scheduler step, so a resumed run is
ready to start the next epoch at the correct learning rate.

The learning rate written to a history row is the rate actually used by the
optimizer during that row's epoch, not the post-step rate prepared for the
next epoch. The reference schedule is therefore:

| training epoch | learning rate used |
| --- | ---: |
| 1 through 60 | 0.1 |
| 61 through 120 | 0.02 |
| 121 through 160 | 0.004 |
| 161 through 200 | 0.0008 |

## Randomness and deterministic boundary

At a fresh run, `training.seed` seeds all of the following before model and
loader construction:

- Python `random`;
- NumPy;
- PyTorch CPU;
- every available PyTorch CUDA device;
- the dedicated train DataLoader generator.

The full checkpoint stores the corresponding Python, NumPy, PyTorch CPU,
available PyTorch CUDA, and train-generator states. Resume reconstructs the
model, optimizer, scheduler, and loaders, loads their state, and restores all
RNG states before creating the next train-loader iterator or performing any
other stochastic operation.

`training.deterministic` controls PyTorch strict deterministic-algorithm
enforcement. The reference WRN CUDA configuration sets it to `false` because
PyTorch documents CUDA backward for `AdaptiveAvgPool2d` and CUDA `NLLLoss` as
operations that raise when strict deterministic algorithms are enabled. This
does not disable seeding, RNG checkpointing, or the fixed
`torch.backends.cudnn.benchmark = false` setting.

The required CPU resume test uses strict deterministic algorithms and must show
that a continuous three-epoch run and a two-epoch run plus one resumed epoch
have equivalent model, optimizer, scheduler, and metric results in its
supported fixture. Seeds and saved states do not imply bitwise equality across
different PyTorch, CUDA, cuDNN, driver, device, or platform versions; those
versions belong in `environment.json`.

## Checkpoint files and write behavior

Full checkpoints use schema version `1.0`. `last.pt` is atomically replaced
after every completed epoch. `best_val.pt` is atomically replaced only when
the just-completed epoch becomes the best validation epoch. A practical atomic
write means saving a temporary file in the destination directory and replacing
the destination with `os.replace()` only after the save succeeds. A failed
write must not delete the previous complete checkpoint.

`last.pt` is the atomic epoch commit marker. If interruption occurs after it is
committed but before the same epoch's best or fixed-snapshot artifact is
written, resume reconciles that committed epoch's artifacts before creating
the next train-loader iterator.

`last.pt` and `best_val.pt` contain at least:

```text
schema_version
checkpoint_type
completed_epoch
global_step
model_state
optimizer_state
scheduler_state
best_validation
rng_state
resolved_config
oge_git_sha
```

`checkpoint_type` is `last` or `best_val`. `best_validation` contains its
epoch, validation accuracy, and validation NLL. `rng_state` contains the
`python`, `numpy`, `torch_cpu`, `torch_cuda`, and
`train_dataloader_generator` states. `torch_cuda` is the per-device state when
CUDA is available and `null` for a CPU-only run. `scheduler_state` is `null`
for scheduler mode `none`.

Best validation is selected lexicographically by:

1. highest validation accuracy;
2. if accuracy is tied, lowest validation NLL;
3. if both are tied, the earliest epoch.

An exact double tie therefore keeps the existing checkpoint rather than
replacing it with a later epoch.

Configured fixed snapshots are written under `checkpoints/snapshots/` as
`epoch_NNNN.pt`. They contain at least:

```text
schema_version
checkpoint_type
completed_epoch
model_state
protocol_name
model_name
oge_git_sha
```

Their `checkpoint_type` is `snapshot`. Snapshots are inference/reload
artifacts, not full resume checkpoints. `epoch_0000.pt` is written after model
initialization and before any optimizer step; all other snapshots are written
at their completed-epoch boundary.

## Reload and epoch-boundary resume

Checkpoint reload must restore model weights such that logits match the saved
model on the same input in evaluation mode. Full resume is supported only from
the end-of-epoch `last.pt`; `best_val.pt` is reloaded for selected-model ID-test
evaluation but is not a resume source. Mid-epoch position, partially consumed
sample order, and partial gradient accumulation are not represented and cannot
be resumed.

Resume accepts an unchanged resolved experiment configuration and only these
changes:

- keep or increase `training.max_epochs`;
- keep the configured snapshot set or add snapshot epochs strictly later than
  the checkpoint's `completed_epoch`.

Existing snapshot epochs may not be removed or changed. A new snapshot at or
before the completed epoch is rejected because the missing historical model
state cannot be reconstructed. Decreasing `max_epochs` is rejected.

Every other resolved experiment field must match. Validation must reject with
a clear field-specific error at least:

- optimizer name or optimizer hyperparameter changes, including learning rate,
  weight decay, momentum, Nesterov, betas, or epsilon;
- scheduler type, gamma, step timing, or milestone changes;
- model name, shape, class count, or dropout changes;
- dataset protocol, selected split, loaded definition, or membership changes;
- seed, loss, precision, deterministic flag, batch size, or DataLoader-setting
  changes.

After compatibility validation, resume restores model, optimizer, scheduler,
best-validation state, `completed_epoch`, `global_step`, every saved RNG state,
and the train DataLoader generator state. The next loop iteration is epoch
`completed_epoch + 1`. Existing `history.jsonl` rows are preserved and the next
epoch is appended exactly once.

## Run artifact contract

A run directory has this stable structure:

```text
run_dir/
├── resolved_config.yaml
├── run_metadata.json
├── environment.json
├── history.jsonl
├── summary.json
├── checkpoints/
│   ├── last.pt
│   ├── best_val.pt
│   └── snapshots/
└── evaluation/
    ├── final_id_test.json
    └── best_val_id_test.json
```

- `resolved_config.yaml` is the complete resolved configuration used by the
  run, including the actual data root, loaded dataset definition, and explicit
  DataLoader settings.
- `run_metadata.json` identifies the schema, run, protocol, model, repository
  Git SHA, start/resume context, and artifact role.
- `environment.json` records Python, NumPy, PyTorch, TorchVision, platform,
  device, and available CUDA/cuDNN information.
- `history.jsonl` contains one row per completed epoch in ascending order.
- `summary.json` records completion state, final epoch/global step, final and
  best-validation metrics and epoch, and the ID-test artifact locations.
- Each ID-test JSON identifies the source checkpoint and completed epoch and
  records sample count, ID-test NLL, and ID-test accuracy.

Every history row contains at least:

```text
epoch
global_step
learning_rate
train_loss
train_accuracy
validation_nll
validation_accuracy
is_best
elapsed_seconds
```

Losses are sample-weighted means, accuracies are fractions in `[0, 1]`, and
`elapsed_seconds` is wall-clock duration for the epoch. Timing is diagnostic
metadata and is not expected to match across continuous and resumed runs.

## Validation and evidence boundary

Local validation covers focused training tests, scheduler boundaries,
checkpoint save/reload, best-validation tie handling, deterministic CPU resume
equivalence, invalid resume rejection, artifact schemas, `git diff --check`,
and the complete regression suite.

Department-server validation for the Issue #10 implementation completed with
actual OpenOOD CIFAR-10 data and CUDA. WRN-28-10 with dropout 0.0 and SGD
completed the required bounded batch, one epoch, last/best save, strict reload,
validation, and one resumed epoch; Adam and AdamW each exercised one actual-data
training batch through the same common engine. The environment, commands, and
artifacts are recorded in
[`issue10_cifar_training_server_validation.md`](../validation/issue10_cifar_training_server_validation.md).
These smokes validate infrastructure only; they are not research evidence.

Future changes to training, checkpoint, resume, loader-state, or device
semantics require the local and environment-dependent validation specified by
their own Issue before those changes join the validated foundation.

This protocol does not authorize a 200-epoch run, HPO, multi-seed
orchestration, feature extraction, geometry or Neural Collapse metrics, OOD
detectors, cosine scheduling, AMP, distributed training, or optimizer-result
interpretation.
