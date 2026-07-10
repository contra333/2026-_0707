# Optimizer Geometry Experiments

Initial PyTorch codebase for ICLR 2027 optimizer-geometry experiments.

The repository currently contains:

- optimizer semantics and factory code;
- optimizer and parameter-group unit tests;
- a model API contract for logits and penultimate features;
- implemented `toy_cifar_cnn`, `resnet18`, and `wrn28_10` model endpoints;
- an AI-assisted research and implementation workflow.

The `toy_cifar_cnn` model is only for API smoke testing. `resnet18` and `wrn28_10` are implemented research backbones. VGG-16 and ConvNeXt-Tiny remain documented as planned backbones and should not be treated as implemented until code and tests add them.

This repository intentionally does **not** yet include dataset loaders, training loops, checkpointing, OOD detectors, CIFAR training scripts, or GPU training scripts.

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
- New task template: `.github/ISSUE_TEMPLATE/research_task.md`
- Pull Request template: `.github/pull_request_template.md`

## Test

```bash
pytest tests/test_model_api.py tests/test_optimizers.py tests/test_param_groups.py -q
```
