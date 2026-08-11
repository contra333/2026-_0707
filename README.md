# Optimizer Geometry Experiments

PyTorch research codebase for ICLR 2027 optimizer-geometry experiments.

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
  frozen protocol-v1.2 grid tables, deferred ID-test selection mode,
  failure/retry accounting, and independent single-GPU trial scheduling;
- a practical three-server runtime, numerical-policy, committed-data,
  bounded-loader, and actual-data one-epoch smoke gate completed by Issue #37;
- an AI-assisted research and implementation workflow;
- implemented Metric Contract v1.2 raw-feature extraction, geometry,
  calibration, feature/logit OOD detectors, checkpoint-centric bundles, and
  validation oracles;
- completed 36-cell WRN-28-10/CIFAR-10 seed-0 grid, C1--C4 freeze, fresh-seed
  role replication, coupled/decoupled pair controls, and protected evaluation;
- a checksum-verified C1--C4 descriptive analysis package with all six OOD
  datasets kept separate;
- Research Contract v5 for a fixed-readout mechanism study using paired
  from-scratch training trajectories, theory-constrained geometry diagnostics,
  exact component/pair-order accounting, and ordered replication.

The `toy_cifar_cnn` model is only for API smoke testing. `resnet18` and
`wrn28_10` are implemented research backbones. The active v5 main is
WRN-28-10/CIFAR-10, followed by ResNet-18/CIFAR-10 and CIFAR-100 replication;
DenseNet-BC/CIFAR-10 and ConvNeXt-Tiny/ImageNet-200 are planned appendices and
are not implemented.

The completed v1.2 population and v3 component analysis are
descriptive/discovery evidence; they are not from-scratch paired confirmation.
The `fork_from_prefix` runtime exists only as optional follow-up infrastructure.
Still missing are the v5 training/trajectory implementation and execution,
fresh paired confirmation, replication data/architecture contracts, and any
resulting causal or cross-regime conclusion. `GDA-ClassDensity` is implemented
and evaluated, while full DDU remains reserved for a future
spectral-normalization training ablation.

## Start here

Repository-aware humans and AI agents should read in this order:

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/WORKFLOW.md`
4. `docs/STATUS.md`
5. the active GitHub Issue, or the explicit bounded task when `AGENTS.md` fast path applies
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
- Raw checkpoint-feature artifact contract: `docs/reference_cards/08_raw_feature_artifact_contract.md`
- Frozen core representation metrics: `docs/reference_cards/09_core_representation_metrics.md`
- Optimizer grid literature anchors: `docs/reference_cards/10_optimizer_grid_literature_anchors.md`
- Metric Contract v1.2 dictionary and completed historical scope: `docs/reference_cards/11_metric_contract_v1_2.md`
- Historical failed radial intervention protocol v2: `docs/reference_cards/12_fixed_readout_intervention_protocol_v2.md`
- Active theory-constrained paired-trajectory protocol v5: `docs/reference_cards/13_paired_trajectory_component_attribution_protocol_v5.md`
- Current manuscript skeleton: `docs/paper/intervention_supporting_theory_outline.md`
- Local historical/superseded material manifest: `docs/history/local_research_draft_manifest.md`
- OpenOOD dataset/MSP server validation: `docs/validation/issue6_openood_cifar10_server_validation.md`
- CIFAR-10 training server validation: `docs/validation/issue10_cifar_training_server_validation.md`
- Optimizer-HPO orchestration server validation: `docs/validation/issue22_optimizer_hpo_orchestration_server_validation.md`
- Practical three-server runtime/data/smoke validation: `docs/validation/issue37_practical_runtime_status.md`
- New task template: `.github/ISSUE_TEMPLATE/research_task.md`
- Pull Request template: `.github/pull_request_template.md`

## Test

```bash
pytest -q
```
