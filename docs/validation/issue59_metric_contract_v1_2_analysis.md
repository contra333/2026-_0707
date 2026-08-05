# Issue #59 C1–C4 Metric Contract v1.2 local analysis

Date: 2026-08-05

Issue: [#59](https://github.com/contra333/2026-_0707/issues/59)

## Outcome

The checksum-verified Issue #57 central aggregate was analyzed locally without
opening a checkpoint, traversing a dataset, starting a server evaluation, or
mutating Hugging Face. The completed package is
[`docs/analysis/metric_contract_v1_2_c1_c4/`](../analysis/metric_contract_v1_2_c1_c4/README.md).

The analysis keeps `last.pt` as the confirmatory primary and `best_val.pt` as
an independent deployment control. It contains ten unique configurations,
seeds `{0,1,2}`, all 31,720 central three-seed scalar aggregates, exact seed-0
audit values, 5,700 seed-matched descriptive optimizer deltas, and 8,448
dataset-specific geometry–OOD association rows. CIFAR-100, TinyImageNet,
MNIST, SVHN, Textures, and Places365 remain in separate files and figures.
Near/far/overall macro rows are isolated in a cross-dataset audit and are not
used for dataset-specific conclusions.

The package also contains
[`notion_c1_c4_primary_comparison.md`](../analysis/metric_contract_v1_2_c1_c4/notion_c1_c4_primary_comparison.md),
a UTF-8, plain-Markdown import document for Notion `Text & Markdown`. It keeps
`last.pt` only, places optimizers on rows, reports mean ± `ddof=1` sample SD,
and partitions all six OOD datasets before displaying the canonical 11-detector
AUROC/FPR@95 panel. `best_val`, seed-0 audit rows, AUPR, and the eight appendix
detectors are intentionally absent from this focused view.

## Frozen input identity

| Item | Value |
| --- | --- |
| Definition | `metric_contract_v1.2` |
| Scientific evaluator SHA | `c38b09694be88aa74de0741b39e9d3ba0d6ff61a` |
| Inventory hash | `35a1a66f78dfd6255c4d804586ee6863969be78d21acec5cd0ccf00476ddaa5c` |
| Dataset-policy hash | `f80984d28ef9a7550633fb3aea514f783f3db1fa655a6d3f49850b10857e6afc` |
| Aggregate manifest SHA-256 | `2fd313f751d56059b2ed7054bc62389b167dd5c17a7c61800f29c5b46b477a24` |
| `aggregate.json` SHA-256 | `d9e51acedadaf532e5cd37fe06d745f962bc036d61234ce1c819de0d855e4cac` |
| `per_checkpoint_records.jsonl` SHA-256 | `c845ba8734d03ba170a48da121fec7a4fd0ccbbebb5e54c3442bc32156ad85cc` |
| `seed_aggregates.jsonl` SHA-256 | `8d0891051fc719455364602652932a8cadd2308552ec266963cf9e74b54197d2` |
| `detector_rank_concordance.jsonl` SHA-256 | `3f22c203f41bbb5e2dc90d0fda1653b504a6e8dbfc8af23d89f39f06f2234be1` |

The analysis reran the manifest and payload hash checks before reading scalar
records. It independently enforced 60 checkpoint jobs, 30 authorized training
identities, ten unique configurations, three seeds, 95,160 successful
per-checkpoint scalars, 31,720 successful seed aggregates, and zero non-success
aggregates. Every aggregate mean and `ddof=1` sample SD was recomputed from
`seed_values`; every seed-0 value and checkpoint SHA matched its corresponding
per-checkpoint record.

## Analysis contract

- C1–C4 LR/WD and labels are read from the frozen inventory and checked against
  the protocol matrix. AdamW C2 remains `NA — protocol absent`.
- Adam C1 and C3 display in both roles but enter cross-configuration analysis
  once because they share one configuration hash.
- The paper panel contains 11 canonical detectors. The appendix tables contain
  all 19 detector identities present in the aggregate.
- Every three-seed table uses the arithmetic mean and sample SD (`ddof=1`).
  Paired deltas are seed-matched `optimizer_a - optimizer_b` descriptions and
  are not significance tests.
- Recalculation, rank statistics, bootstrap storage, and comparison checks use
  float64; the output manifest records `atol=rtol=1e-12` identity tolerances.
- Primary geometry–OOD coefficients are Spearman correlations across ten
  unique configuration means, calculated separately for every dataset,
  checkpoint, endpoint, geometry metric, and canonical detector.
- AUROC is the primary association endpoint and performance-aligned FPR@95 is
  the sensitivity endpoint. AUPR-In and AUPR-Out use the same procedure but
  remain appendix CSVs rather than paper heatmaps.
- Robustness uses a separate 30-observation seed-level Spearman coefficient.
  Each of 10,000 configuration-block bootstrap replicates resamples ten
  configurations, retains all three seeds in a block, reranks the resampled
  raw values, and forms a percentile 95% interval.
- Two-sided p-values use 10,000 permutations of the ten configuration-mean
  blocks. Benjamini–Hochberg FDR is applied within each
  dataset/checkpoint/endpoint family of 16×11 tests.
- FPR association uses `-FPR@95` so all endpoints have the same
  performance-up direction. No dataset-pooled correlation is calculated.

## Descriptive result landmarks

For `last.pt`, C1 ID-test accuracy is:

| Optimizer | Mean ± sample SD |
| --- | --- |
| SGD | `0.9541 ± 0.0013` |
| Adam | `0.9385 ± 0.0017` |
| AdamW | `0.9525 ± 0.0018` |

The highest observed C1 canonical-detector AUROC within each dataset is a
descriptive locator, not a post-hoc primary-detector selection:

| OOD dataset | Optimizer | Detector | AUROC mean ± sample SD |
| --- | --- | --- | --- |
| CIFAR-100 | SGD | `kNN-L2` | `0.905050 ± 0.001195` |
| TinyImageNet | SGD | `kNN-L2` | `0.923844 ± 0.001346` |
| MNIST | SGD | `NECO` | `0.961736 ± 0.004277` |
| SVHN | SGD | `Mahalanobis++` | `0.984284 ± 0.002994` |
| Textures | SGD | `Mahalanobis++` | `0.966999 ± 0.002912` |
| Places365 | AdamW | `Relative Mahalanobis raw` | `0.929226 ± 0.002549` |

The report identifies the largest absolute cell in each `last.pt` AUROC
association matrix only to navigate the full heatmaps. Three of those six
dataset-specific cells have BH-FDR `q<0.10` (MNIST, SVHN, and Textures), but
all remain exploratory because there are only ten unique configurations,
seed 0 contributed to role selection, and the hypotheses share common
checkpoint provenance. No causal or universal-detector conclusion is made.

## Output inventory and reproducibility

The generated tree contains 206 files and 49,115,007 bytes. Its manifest lists
204 payloads plus the manifest and SHA-256 sidecar. The output-manifest
SHA-256 is:

```text
844d206b6246a0c1073e06259fcc1bef4495b8083828566c51060c7f3efd16f2
```

A second complete 10,000-resample run was written to an independent temporary
directory. `diff -qr` reported no differences and both output manifests had
the SHA-256 above. Matplotlib output is deterministic through a fixed SVG hash
salt and date-free PDF metadata.

The committed package provides:

- a Korean technical report and Methods-ready English text;
- a Notion-import-safe C1–C4 `last.pt` primary comparison in one Markdown file;
- numeric CSV plus Markdown/LaTeX paper tables;
- all central scalar seed-0 and three-seed tables;
- independent OOD dataset directories and a separate macro audit;
- seed-matched optimizer-delta tables;
- dataset/checkpoint-specific association CSVs and SVG/PDF heatmaps;
- formula/artifact/checkpoint/split/direction/tier/Methods crosswalks;
- source hashes, analysis settings, and checksums for every output payload.

Raw detector scores, feature tensors, full spectra, class-pair matrices, and
other non-central intermediates are **not included** and were not recomputed.

## Validation evidence

| Check | Result |
| --- | --- |
| Analysis-focused and affected-contract tests | `18 passed in 4.32s` |
| Complete local CPU suite | `277 passed, 1 warning in 12.78s` |
| Deterministic full rerun | `PASS`; byte-identical tree and manifest |
| Notion import/value parity | `PASS`; generated text equals committed aggregate-derived reference; 64 tables with consistent rows; C2 AdamW remains absent |
| Markdown links | `PASS`; 29 generated-package local links resolved |
| LaTeX table structure | `PASS`; 16 longtables, no non-finite cells |
| PDF render structure | `PASS`; role figures and six primary AUROC heatmaps are single-page |
| Visual inspection | `PASS`; role summary and MNIST AUROC heatmap rendered and inspected |
| `git diff --check` | `PASS` |

The first focused pytest invocation omitted `TMPDIR=/tmp` and failed before
test collection while pytest cleaned a capture file under the Korean worktree
path. Repeating the exact suite with the repository's established `TMPDIR`
setting passed. The single full-suite warning is the existing local PyTorch CUDA-driver
initialization warning in the CPU temperature-scaling test; it did not fail or
skip the test. No CUDA, server, protected checkpoint, dataset, or Hugging Face
operation was requested or run for Issue #59.

## Commands actually run

```bash
TMPDIR=/tmp PYTHONPATH=src \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  scripts/analyze_metric_contract_v1_2.py \
  --aggregate-dir /home/contra333/2026여름방학실험코드/issue57_metric_results/c38b09694be88aa74de0741b39e9d3ba0d6ff61a/aggregate \
  --inventory configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json \
  --output-dir docs/analysis/metric_contract_v1_2_c1_c4 \
  --resamples 10000

TMPDIR=/tmp PYTHONPATH=src \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  scripts/analyze_metric_contract_v1_2.py \
  --aggregate-dir /home/contra333/2026여름방학실험코드/issue57_metric_results/c38b09694be88aa74de0741b39e9d3ba0d6ff61a/aggregate \
  --inventory configs/evaluation/wrn28_10_cifar10_metric_v1_2/checkpoint_inventory.json \
  --output-dir /tmp/issue59_analysis_determinism_notion_final \
  --resamples 10000

diff -qr \
  docs/analysis/metric_contract_v1_2_c1_c4 \
  /tmp/issue59_analysis_determinism_notion_final

TMPDIR=/tmp \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  -m pytest -q \
  tests/test_metric_contract_v1_2_analysis.py \
  tests/test_metric_contract_docs.py \
  tests/test_metric_evaluation_aggregation_v1_2.py

TMPDIR=/tmp \
  /home/contra333/2026여름방학실험코드/2026-_0707/.venv/bin/python \
  -m pytest -q

git diff --check
```

## Claim boundary

This work completes bounded local analysis and paper-supporting result
organization. It does not authorize another protected population, checkpoint
reevaluation, server run, Hugging Face upload, confirmatory causal model,
detector reselection, or DDU/SN ablation.
