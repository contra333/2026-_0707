# AGENTS.md

## Project scope
This repository is for optimizer-geometry experiments. Use `docs/reference_cards/01_optimizers.md` as the implementation reference for optimizer semantics.

## Safety rules
- If PyTorch official optimizer semantics, the reference card, and implementation requirements conflict, stop and report instead of resolving silently.
- Do not add architecture, dataset, training loop, CIFAR, or GPU training code until optimizer endpoint parity tests pass.
- All optimizers must use the shared parameter-group builder.
