# Issue #63: Component Attribution and Shared-Prefix Fork Foundation

Date: 2026-08-11

Issue: [#63](https://github.com/contra333/2026-_0707/issues/63)

Implementation commit: `28ba5a067c55ba1f7a57d8265f55b57057d54762`

Post-review hardening commit: `7fc32b7398b1206326612c544cfc8e4aacf88a10`

## Outcome

`PASS` for the Contract-v3 implementation foundation and historical discovery
analysis. `NOT_RUN` for fresh shared-prefix training, protected-OOD
confirmation, and all replication arms.

The implementation adds two independent paths:

1. A read-only MD--Marginal--RMD score and pair-order attribution analysis over
   the existing 30 `last.pt` bundles.
2. A `fork_from_prefix` training operation that creates a new run from an
   exact zero-decay prefix while transferring model state, optimizer dynamic
   tensors, scheduler position, RNG state, and DataLoader generator state.

Ordinary same-run resume remains strict and unchanged. The fork path rejects a
non-zero-decay prefix and an Adam/SGD cross-family branch.

Post-review hardening adds explicit Adam/AdamW transfer coverage for populated
`step`, `exp_avg`, and `exp_avg_sq` states; source/branch run, canonical config,
and seed identities in the fork manifest; randomized brute-force pair and
Shapley oracles; and the interpretation boundaries for computational hybrid
readouts, nominal decay, and normalized-network effective-step dynamics.

## Evidence boundary

The discovery input is the previously checksum-verified Stage-2 reuse tree:

```text
/home/contra333/2026여름방학실험코드/fixed_readout_stage2_reuse/fixed_readout_stage2_reuse_manifest_v1
```

The exact reuse-manifest SHA-256 is:

```text
f814f1ab3070a2dc4f0d541746bbd05c3ad74d761bd5c8fa88bc859f308318dc
```

The analysis used:

- 30 `last.pt` bundles;
- CIFAR-100, TinyImageNet, MNIST, SVHN, Textures, and Places365 OOD scores;
- raw and L2 versions of MD, Marginal, and RMD;
- nine role-frozen, seed-matched historical pair definitions, yielding 108
  pair-attribution rows (`9 pairs x 6 OOD datasets x 2 transforms`).

It did not reevaluate checkpoints, load protected images, train a model,
select a detector after seeing the result, or modify the failed v2 radial gate.
The historical pairs contain independently trained configurations and remain
descriptive/noncausal.

## Result artifacts

The no-overwrite output directory is:

```text
/home/contra333/2026여름방학실험코드/fixed_readout_component_attribution_v3/28ba5a067c55ba1f7a57d8265f55b57057d54762
```

It contains:

- `summary.json`;
- `bundle_component_metrics.jsonl` with 360 rows;
- `pair_component_attribution.jsonl` with 108 rows;
- `score_source_manifest.jsonl` binding the source arrays;
- `checksums.sha256`.

The summary reports `status=PASS`, `selection_performed=false`, and
`fresh_shared_prefix_confirmation=NOT_RUN`. All four payload checksums passed.
The historical cache CLI publishes score/component attribution only. The
size--stretch API and reconstruction oracle are implemented, but actual
size--stretch branch artifacts remain `NOT_RUN` until fresh forked quadratic
inputs exist.

## Historical discovery result

The following values summarize the 30-model historical population. AUROC range
is computed across the 30 models separately for each OOD dataset and then
averaged across the six datasets.

| Transform | MD range | Marginal range | RMD range |
| --- | ---: | ---: | ---: |
| Raw | 0.4176 | 0.4488 | 0.0502 |
| L2 | 0.1171 | 0.1814 | 0.0528 |

For each prespecified left-to-right historical pair, the exact symmetric
two-component attribution reconstructs the MD AUROC change from Marginal and
RMD contributions.

| Transform | Pair rows | Mean MD change | Mean Marginal attribution | Mean RMD attribution | Median absolute Marginal share | Marginal larger in absolute value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 54 | -0.1033 | -0.0591 | -0.0442 | 0.633 | 43/54 |
| L2 | 54 | 0.0134 | 0.0218 | -0.0084 | 0.753 | 43/54 |

The maximum attribution reconstruction residual was zero for raw and
`5.55e-17` for L2. The result supports advancing the Marginal/global-reference
component as the focal *confirmation question*. It does not establish that
coupled versus decoupled decay caused the historical gap, and the signed means
must not be interpreted as an optimizer ranking.

L2 reduces the raw MD range in this historical population, but it does not
make the Marginal contribution disappear. That observation justifies retaining
L2 as a radial control rather than treating radial geometry as the complete
mechanism.

## Commands and validation

The production analysis was run from the clean tracked implementation commit:

```bash
PYTHONPATH=src python scripts/analyze_fixed_readout_component_attribution.py \
  --reuse-manifest configs/evaluation/fixed_readout_stage2/reuse_manifest.json \
  --retrieval-root /home/contra333/2026여름방학실험코드/fixed_readout_stage2_reuse/fixed_readout_stage2_reuse_manifest_v1 \
  --pair-manifest configs/evaluation/fixed_readout_component_attribution_v3/historical_pairs.json \
  --output-directory /home/contra333/2026여름방학실험코드/fixed_readout_component_attribution_v3/28ba5a067c55ba1f7a57d8265f55b57057d54762
```

Result: `PASS`, 30 bundles, 360 bundle rows, 108 pair rows.

The published payload was independently rehashed:

```bash
cd /home/contra333/2026여름방학실험코드/fixed_readout_component_attribution_v3/28ba5a067c55ba1f7a57d8265f55b57057d54762
sha256sum -c checksums.sha256
```

Result: four of four payload files `OK`.

The complete repository test suite was run on Curie from a temporary Git
snapshot of the implementation commit:

```bash
git archive HEAD | ssh curie 'issue63_dir=$(mktemp -d /tmp/oge_issue63_validation.XXXXXX) && tar -xf - -C "$issue63_dir" && cd "$issue63_dir" && git init -q && git config user.name codex-validation && git config user.email codex-validation@example.invalid && git add -A && git commit -qm validation-snapshot && /home/ghjin/miniconda3/bin/python -m pytest -q; test_rc=$?; case "$issue63_dir" in /tmp/oge_issue63_validation.*) rm -rf -- "$issue63_dir" ;; *) echo "refusing cleanup: $issue63_dir" >&2; exit 97 ;; esac; exit "$test_rc"'
```

Result: `472 passed in 77.48s`.

### Post-review hardening validation

The modified training, attribution, and Contract-v3 test files were run in the
repository `.venv` with capture disabled because the default capture path
failed before collection in the local WSL filesystem environment:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -s \
  tests/test_training_resume.py \
  tests/test_fixed_readout_component_attribution.py \
  tests/test_research_contract_v3_docs.py
```

Result: `57 passed, 1 warning in 9.01s`. The warning was the local PyTorch
build reporting that its CUDA build is newer than the installed WSL driver;
the relevant fixtures ran on CPU.

An attempted local full-suite run was stopped after `14 passed in 246.96s`
because the process remained in uninterruptible filesystem-I/O wait. This is
an environment-limited interrupted check, not a passing full-suite result.

The exact pushed code-bearing hardening commit was then validated on Curie:

```bash
git archive 7fc32b7 | ssh curie 'issue63_dir=$(mktemp -d /tmp/oge_issue63_postreview.XXXXXX) && tar -xf - -C "$issue63_dir" && cd "$issue63_dir" && git init -q && git config user.name codex-validation && git config user.email codex-validation@example.invalid && git add -A && git commit -qm validation-snapshot && /home/ghjin/miniconda3/bin/python -m pytest -q; test_rc=$?; case "$issue63_dir" in /tmp/oge_issue63_postreview.*) rm -rf -- "$issue63_dir" ;; *) echo "refusing cleanup: $issue63_dir" >&2; exit 97 ;; esac; exit "$test_rc"'
```

Result: `477 passed in 78.36s`.

The Issue's requested command
`python scripts/validate_research_contract_v2_docs.py` is `NOT_RUN` because
that script does not exist in the repository. Its intended v2/v3 document
checks are present in the passing pytest suite. The component-attribution CLI
`--help`, Python compilation, JSON/YAML parsing, and `git diff --check` passed.

The repository `.venv` provides PyTorch and pytest, but the full local suite
was not a valid completion because of the WSL filesystem-I/O stall. Curie
therefore supplies the complete-suite result. The analysis itself ran locally
because it only requires the verified cached score arrays and NumPy.

## Not run

- fresh prefix training or any GPU job;
- coupled/decoupled/zero branch execution;
- frozen ID-equivalence and practical-margin classification;
- protected-OOD confirmation;
- size--stretch attribution between actual forked branches;
- ResNet-18/CIFAR-10, ResNet-18/CIFAR-100, DenseNet-BC, or
  ConvNeXt-Tiny/ImageNet-200 replication;
- Hugging Face upload.

These remain separate execution tasks. No strong causal or replication claim
is unlocked by this validation record.
