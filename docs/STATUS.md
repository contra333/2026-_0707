# Project Status

Last updated: 2026-07-13

## Current phase

The first optimizer, model, OpenOOD v1.5-aligned CIFAR-10 data/MSP, and
reproducible classifier-training foundations are merged and validated. Pull
Request #11 merged the Issue #10 training, checkpoint, reload, and
epoch-boundary resume vertical slice after bounded actual-data CUDA validation.

## Validated or implemented

- Optimizer reference semantics are documented.
- Shared weight-decay parameter-group policy is implemented.
- Optimizer and parameter-group tests exist.
- Common model API for logits and penultimate features is documented.
- `toy_cifar_cnn` is implemented as an API smoke-test fixture only.
- `resnet18` is implemented as a research backbone.
- `wrn28_10` is implemented as a research backbone.
- The OpenOOD v1.5-aligned CIFAR-10 dataset and evaluation contract is documented and implemented.
- Official released OpenOOD imglists and all eight required archives were validated on the department server.
- Actual-data manifest validation confirmed fixed split counts, image-path existence, sample-ID uniqueness, and label ranges.
- A bounded CUDA WRN-28-10 MSP vertical slice completed with random-model output marked as infrastructure-only.
- Pull Request #11 merged the Issue #10 WRN-28-10 training protocol,
  deterministic loader state, scheduler boundaries, checkpoints, resume, and
  run artifacts. Complete tests and bounded actual-data CUDA SGD, resume,
  Adam, and AdamW validation passed on the department server.

## Still missing

- A 200-epoch WRN-28-10 SGD seed-0 baseline
- Hyperparameter optimization
- Penultimate feature extraction pipeline
- Geometry and Neural Collapse metrics
- Feature-based OOD detectors
- Multi-seed experiment orchestration

## Active next phase

Create a new bounded Issue for the 200-epoch WRN-28-10 SGD seed-0 baseline,
then run it with its own configuration, environment, artifact, and validation
evidence. The bounded Issue #10 CUDA runs remain infrastructure validation and
must not be interpreted as the baseline research result.

## Known workflow maintenance

- `docs/reference_cards/03_architecture_implementation_checklist.md` describes a historical first implementation task whose listed models are now implemented. Treat its durable API and validation rules as useful context, but do not treat its one-time scope as the current active task.
- New one-time implementation tasks should be created as GitHub Issues rather than new permanent checklists under `docs/reference_cards/`.

## Blockers and unknowns

- No 200-epoch baseline has been run. That long run requires a separate Issue
  and is not evidence from the bounded Issue #10 infrastructure smoke.
- No active GitHub Issue currently authorizes that baseline or another
  long-running research experiment.

## Update rule

Update this file only when the project phase, validated foundation, major blocker, or next-phase priority changes. Do not use it as a daily log or duplicate Pull Request descriptions.
