# Project Status

Last updated: 2026-07-10

## Current phase

Foundation implementation is in progress. Optimizer semantics and the first research backbones are implemented. Training, experiment orchestration, geometry analysis, and OOD evaluation are not yet part of the validated foundation.

## Validated or implemented

- Optimizer reference semantics are documented.
- Shared weight-decay parameter-group policy is implemented.
- Optimizer and parameter-group tests exist.
- Common model API for logits and penultimate features is documented.
- `toy_cifar_cnn` is implemented as an API smoke-test fixture only.
- `resnet18` is implemented as a research backbone.
- `wrn28_10` is implemented as a research backbone.

## Not yet validated as project infrastructure

- Dataset loaders
- Training and evaluation loops
- Checkpoint and run-metadata policy
- Reproducible experiment configuration
- Penultimate feature dump pipeline
- Geometry metrics
- OOD datasets, detectors, and evaluation
- Multi-seed experiment orchestration
- GPU training scripts

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

- Exact department-server Python, PyTorch, TorchVision, CUDA, and GPU versions are not recorded in the repository.
- No standard experiment run directory or metadata schema is defined yet.
- No active GitHub Issue is identified in this document. Agents must obtain the current Issue from the user or GitHub before editing for a task.

## Update rule

Update this file only when the project phase, validated foundation, major blocker, or next-phase priority changes. Do not use it as a daily log or duplicate Pull Request descriptions.