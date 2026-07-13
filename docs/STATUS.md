# Project Status

Last updated: 2026-07-13

## Current phase

Foundation implementation is in progress. Optimizer semantics, the first research backbones, and the OpenOOD v1.5-aligned CIFAR-10 dataset and OOD-evaluation vertical slice are validated. The first reproducible CIFAR-10 training, checkpoint, and epoch-boundary resume vertical slice is implemented on the Issue #10 task branch and locally tested; actual-data CUDA validation is still required before it joins the validated foundation.

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
- The Issue #10 branch documents and locally implements the first WRN-28-10 training protocol, including deterministic loader state, scheduler boundaries, checkpoints, resume, and run artifacts.

## Still missing

- Actual-data CUDA validation of the Issue #10 training, checkpoint reload, and resume path
- Penultimate feature extraction pipeline
- Geometry and Neural Collapse metrics
- Feature-based OOD detectors
- Multi-seed experiment orchestration

## Active next phase

Validate the Issue #10 training vertical slice on the department server using the same task branch and actual OpenOOD CIFAR-10 data. The bounded validation must cover WRN-28-10 with SGD on CUDA through one epoch, checkpoint reload, ID validation, and one resumed epoch, plus one actual-data batch each with Adam and AdamW through the same common engine.

## Known workflow maintenance

- `docs/reference_cards/03_architecture_implementation_checklist.md` describes a historical first implementation task whose listed models are now implemented. Treat its durable API and validation rules as useful context, but do not treat its one-time scope as the current active task.
- New one-time implementation tasks should be created as GitHub Issues rather than new permanent checklists under `docs/reference_cards/`.

## Blockers and unknowns

- The local environment reports CUDA unavailable because its NVIDIA driver is older than the installed PyTorch CUDA runtime; local CUDA behavior is unverified.
- Actual-data training artifact paths and CUDA results for Issue #10 have not yet been recorded from the department server.

## Update rule

Update this file only when the project phase, validated foundation, major blocker, or next-phase priority changes. Do not use it as a daily log or duplicate Pull Request descriptions.
