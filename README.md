# Optimizer Geometry Experiments

PyTorch research code for studying how training rules shape representation geometry and
fixed-readout OOD behavior.

## Start here

1. [`AGENTS.md`](AGENTS.md) — repository working rules
2. [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) — stable research objective
3. [`docs/STATUS.md`](docs/STATUS.md) — current state and next task
4. [`docs/reference_cards/13_active_paper_protocol.md`](docs/reference_cards/13_active_paper_protocol.md)
   — active paper experiment contract
5. [`docs/paper/intervention_supporting_theory_outline.md`](docs/paper/intervention_supporting_theory_outline.md)
   — human-readable manuscript skeleton

Read a component reference card only when changing that component. Past protocols,
validation logs, and external artifact pointers are under
[`docs/history/`](docs/history/README.md).

## Implemented foundation

- SGD/SGDW and Adam/AdamW semantics with shared parameter-group policy
- `toy_cifar_cnn`, CIFAR `resnet18`, and `wrn28_10`
- deterministic CIFAR-10 membership and OpenOOD-compatible dataset roles
- classifier training, checkpoint, resume, and optional `fork_from_prefix`
- Metric Contract v1.2 feature, geometry, calibration, and OOD evaluation code
- MD/RMD/Marginal component, pair-transition, size/stretch, and low-rank diagnostics

Fresh paired-trajectory training, multi-depth WRN extraction, CIFAR-100 replication,
DenseNet, ConvNeXt, and ImageNet-200 are not yet implemented or run. See STATUS for the
current boundary.

## Development

Install the project environment, then run focused tests for the component being changed.
Use the full CPU suite for shared runtime/API changes or before a production launch.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Generated tables, figures, checkpoints, feature arrays, and score arrays do not belong in
Git. Write them to ignored `artifacts/` or an external path and publish durable results to
a hash-addressed HF artifact.
