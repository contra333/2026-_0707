# Optimizer Geometry Experiments

Initial PyTorch codebase for ICLR 2027 optimizer-geometry experiments.

The repository currently contains:

- optimizer semantics and factory code;
- optimizer and parameter-group unit tests;
- a model API contract for logits and penultimate features;
- implemented `toy_cifar_cnn`, `resnet18`, and `wrn28_10` model endpoints;
- an OpenOOD v1.5-aligned CIFAR-10 loader, fixed split manifests, and
  preprocessing contract;
- bounded MSP OOD inference and metric infrastructure;
- reproducible CIFAR-10 classifier training with scheduling, atomic
  checkpoints, strict reload, epoch-boundary resume, and stable run artifacts;
- a CIFAR-10 training CLI validated with bounded actual-data CUDA runs;
- deterministic optimizer-study orchestration with versioned provenance,
  frozen 64-row discovery tables, deferred ID-test selection mode,
  failure/retry accounting, and independent single-GPU trial scheduling;
- an AI-assisted research and implementation workflow;
- durable DDU fitting, adaptive-jitter, score, naming, and post-hoc variant
  semantics documented for a later implementation task.
- a documented WRN-28-10 four-optimizer HPO and comparison protocol whose
  orchestration foundation has passed bounded department-server validation.

The `toy_cifar_cnn` model is only for API smoke testing. `resnet18` and `wrn28_10` are implemented research backbones. VGG-16 and ConvNeXt-Tiny remain documented as planned backbones and should not be treated as implemented until code and tests add them.

This repository does **not** yet include learned-feature extraction, geometry
or Neural Collapse metrics, feature-based OOD detector implementations,
production discovery HPO, confirmation/final multi-seed execution, or completed
optimizer-comparison research runs. The current MSP, training, and orchestration
CUDA runs validate infrastructure only; they are not optimizer-comparison
research results. The DDU reference card fixes future semantics but does not
mean DDU or any DDU variant is implemented or validated.

## Start here

Repository-aware humans and AI agents should read in this order:

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/WORKFLOW.md`
4. `docs/STATUS.md`
5. the active GitHub Issue
6. task-specific reference cards

The repository is the source of truth. ChatGPT Projects, Work sessions, Codex sessions, and copied Markdown files are temporary interfaces or snapshots, not independent masters.

## Document roles

- Project context: `docs/PROJECT_CONTEXT.md`
- End-to-end workflow: `docs/WORKFLOW.md`
- Current validated state: `docs/STATUS.md`
- Optimizer semantics: `docs/reference_cards/01_optimizers.md`
- Architecture API and planning: `docs/reference_cards/02_architectures.md`
- Historical first architecture checklist: `docs/reference_cards/03_architecture_implementation_checklist.md`
- OpenOOD v1.5-aligned CIFAR-10 protocol: `docs/reference_cards/04_openood_v1_5_protocol.md`
- Classifier training, checkpoint, and resume protocol: `docs/reference_cards/05_training_protocol.md`
- Feature-based OOD detector and DDU semantics: `docs/reference_cards/06_feature_ood_detectors.md`
- WRN-28-10 optimizer-comparison and HPO protocol: `docs/reference_cards/07_optimizer_comparison_hpo_protocol.md`
- OpenOOD dataset/MSP server validation: `docs/validation/issue6_openood_cifar10_server_validation.md`
- CIFAR-10 training server validation: `docs/validation/issue10_cifar_training_server_validation.md`
- Optimizer-HPO orchestration server validation: `docs/validation/issue22_optimizer_hpo_orchestration_server_validation.md`
- New task template: `.github/ISSUE_TEMPLATE/research_task.md`
- Pull Request template: `.github/pull_request_template.md`

## Test

```bash
pytest -q
```
