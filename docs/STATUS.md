# Project Status

Last updated: 2026-07-13

## Current phase

Foundation implementation is in progress. Optimizer semantics, the first research backbones, and the OpenOOD v1.5-aligned CIFAR-10 dataset and OOD-evaluation vertical slice are validated. Training, experiment orchestration, feature extraction, and geometry analysis are not yet part of the validated foundation.

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

## Still missing

- Training loop
- Checkpoint and resume support
- Penultimate feature extraction pipeline
- Geometry and Neural Collapse metrics
- Feature-based OOD detectors
- Multi-seed experiment orchestration

## Active next phase

Define the minimum reproducible training protocol before adding full experiments.

A future training-protocol task should decide at least:

- dataset and transform semantics;
- seed and determinism policy;
- scheduler and epoch semantics;
- checkpoint and resume behavior;
- required run metadata;
- smoke-test acceptance criteria;
- server environment reporting.

## Known workflow maintenance

- `docs/reference_cards/03_architecture_implementation_checklist.md` describes a historical first implementation task whose listed models are now implemented. Treat its durable API and validation rules as useful context, but do not treat its one-time scope as the current active task.
- New one-time implementation tasks should be created as GitHub Issues rather than new permanent checklists under `docs/reference_cards/`.

## Blockers and unknowns

- No standard training run directory, checkpoint policy, or resume schema is defined yet.
- Training seed, determinism, scheduler, and epoch semantics remain to be specified.

## Update rule

Update this file only when the project phase, validated foundation, major blocker, or next-phase priority changes. Do not use it as a daily log or duplicate Pull Request descriptions.
