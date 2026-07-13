# Project Status

Last updated: 2026-07-13

## Current phase

Foundation implementation is in progress. Optimizer semantics, the first research backbones, and the OpenOOD v1.5-aligned CIFAR-10 dataset and OOD-evaluation vertical slice are validated. The first reproducible CIFAR-10 training, checkpoint, and epoch-boundary resume vertical slice is implemented and has completed actual-data CUDA validation on the Issue #10 task branch; Pull Request review and merge are still required before it joins the validated foundation.

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
- The Issue #10 branch documents and implements the first WRN-28-10 training protocol, including deterministic loader state, scheduler boundaries, checkpoints, resume, and run artifacts. Complete tests and bounded actual-data CUDA SGD, resume, Adam, and AdamW validation passed on the department server.

## Still missing

- Pull Request review and merge of the validated Issue #10 task branch
- Penultimate feature extraction pipeline
- Geometry and Neural Collapse metrics
- Feature-based OOD detectors
- Multi-seed experiment orchestration

## Active next phase

Open and review the Issue #10 Pull Request, then merge the validated training vertical slice. A 200-epoch baseline is not part of Issue #10 and must be run under a separate Issue with its own bounded specification and evidence.

## Known workflow maintenance

- `docs/reference_cards/03_architecture_implementation_checklist.md` describes a historical first implementation task whose listed models are now implemented. Treat its durable API and validation rules as useful context, but do not treat its one-time scope as the current active task.
- New one-time implementation tasks should be created as GitHub Issues rather than new permanent checklists under `docs/reference_cards/`.

## Blockers and unknowns

- Issue #10 remains unmerged; its validated implementation is still task-branch state.
- No 200-epoch baseline has been run. That long run requires a separate Issue and is not evidence from the bounded Issue #10 infrastructure smoke.

## Update rule

Update this file only when the project phase, validated foundation, major blocker, or next-phase priority changes. Do not use it as a daily log or duplicate Pull Request descriptions.
