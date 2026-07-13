# Reference Card 04: OpenOOD v1.5-Aligned CIFAR-10 Protocol

## Purpose and authority boundary

This card defines the reusable CIFAR-10 data and OOD-evaluation protocol for
optimizer-geometry experiments in this repository. It adopts OpenOOD v1.5
dataset identities, released imglist membership, preprocessing, near/far
grouping, and per-dataset evaluation structure.

It does not adopt OpenOOD trainer or optimizer behavior, all-parameter weight
decay, model implementations, APS hyperparameter search, string `eval()`
dataset construction, or the complete Config/Pipeline framework. OGE remains
authoritative for optimizer semantics, parameter groups, model and feature API,
geometry definitions, score naming, and artifact policy.

## Pinned external sources

- Paper: `OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection`,
  `arXiv:2306.09301v5`.
- Repository: `Jingkang50/OpenOOD`.
- Commit: `3c35632ee91b54b09d1f085d04f94744cece7d0b`.
- Dataset configs: `configs/datasets/cifar10/cifar10.yml` and
  `configs/datasets/cifar10/cifar10_ood.yml`.
- Preprocessing: `openood/preprocessors/base_preprocessor.py`,
  `test_preprocessor.py`, and `transform.py`.
- Dataset behavior: `openood/datasets/imglist_dataset.py` and `utils.py`.
- Evaluation behavior: `openood/evaluators/metrics.py` and `ood_evaluator.py`.
- Exclusion reference: `openood/trainers/base_trainer.py`.

## Split and dataset contract

Official released imglists define membership. Never regenerate these splits.

| role | dataset | imglist |
| --- | --- | --- |
| ID train | CIFAR-10 | `benchmark_imglist/cifar10/train_cifar10.txt` |
| ID validation | CIFAR-10 | `benchmark_imglist/cifar10/val_cifar10.txt` |
| ID test | CIFAR-10 | `benchmark_imglist/cifar10/test_cifar10.txt` |
| compatibility-only OOD validation | TinyImageNet | `benchmark_imglist/cifar10/val_tin.txt` |
| near-OOD | CIFAR-100 | `benchmark_imglist/cifar10/test_cifar100.txt` |
| near-OOD | TinyImageNet | `benchmark_imglist/cifar10/test_tin.txt` |
| far-OOD | MNIST | `benchmark_imglist/cifar10/test_mnist.txt` |
| far-OOD | SVHN | `benchmark_imglist/cifar10/test_svhn.txt` |
| far-OOD | Textures | `benchmark_imglist/cifar10/test_texture.txt` |
| far-OOD | Places365 | `benchmark_imglist/cifar10/test_places365.txt` |

Use released lists unchanged. Do not perform runtime overlap filtering or
hidden deduplication. OOD validation has role `compatibility_only` and is not
used for model, detector, or hyperparameter selection in the main protocol.

## Validated department-server observations

The pinned released archives and imglists were validated on 2026-07-13 at
repository commit `68e6eaf408124468384c5f5df118a5dc8426462e`. The assembled
data root was `/home/ghjin/datasets/openood-v1.5-3c35632e`.

Observed released-imglist counts were:

| role | dataset | count |
| --- | --- | ---: |
| ID train | CIFAR-10 | 50,000 |
| ID validation | CIFAR-10 | 1,000 |
| ID test | CIFAR-10 | 9,000 |
| compatibility-only OOD validation | TinyImageNet | 1,000 |
| near-OOD | CIFAR-100 | 9,000 |
| near-OOD | TinyImageNet | 7,793 |
| far-OOD | MNIST | 70,000 |
| far-OOD | SVHN | 26,032 |
| far-OOD | Textures | 5,640 |
| far-OOD | Places365 | 35,195 |

Every referenced path existed, no duplicate sample IDs were found, ID labels
were within `[0, 9]`, and every OOD label was `-1`. A bounded random WRN-28-10
CUDA MSP vertical slice completed successfully. Exact archive and imglist
checksums, runtime versions, commands, and artifact locations are recorded in
the [Issue 6 department-server validation report](../validation/issue6_openood_cifar10_server_validation.md).

## Preprocessing

CIFAR-10 normalization is:

```text
mean = [0.4914, 0.4822, 0.4465]
std  = [0.2470, 0.2435, 0.2616]
```

Training transform:

```text
Convert RGB
Resize(32, bilinear)
CenterCrop(32)
RandomHorizontalFlip
RandomCrop(32, padding=4)
ToTensor
Normalize(CIFAR-10 mean, std)
```

ID validation/test and all OOD evaluation transforms are deterministic:

```text
Convert RGB
Resize(32, bilinear)
CenterCrop(32)
ToTensor
Normalize(CIFAR-10 mean, std)
```

All OOD inputs use CIFAR-10 normalization.

## Dataset sample identity

Every item exposes:

```python
{
    "image": Tensor,
    "class_label": int,
    "sample_id": str,
    "dataset_name": str,
    "split": str,
    "is_id": bool,
}
```

`sample_id` is exactly
`f"{dataset_name}:{normalized_relative_posix_path}"`. The absolute data root
is excluded and `split` remains a separate field. The original integer imglist
label is preserved as `class_label`, including `-1` for OOD. Missing label
tokens and non-integer labels are malformed. Binary OOD evaluation uses
`is_id`, never `class_label`. Data roots come from config or CLI and dataset
selection uses an explicit registry or factory.

## Score and metric definitions

All project-facing detector scores are ID-like: higher means more ID-like,
ID binary label is `1`, and OOD binary label is `0`.

MSP is the infrastructure score:

```text
id_like_score = max(softmax(logits))
```

The required metrics have no ambiguous aliases:

- `fpr95_id_tpr`: sort unique ID-like thresholds from highest to lowest and
  choose the first threshold whose inclusive rule `score >= threshold` reaches
  ID TPR at least 0.95. Report the fraction of OOD scores satisfying the same
  inclusive rule. Equal scores form one threshold group; ties are never split.
- `fpr95_openood_ood_tpr`: reproduce pinned OpenOOD `metrics.py` by treating
  OOD as positive, passing `-id_like_score` to
  `sklearn.metrics.roc_curve`, and choosing the first returned point whose OOD
  TPR is at least 0.95. This reports the corresponding ID false-positive rate.
- `aupr_in_ap`: `sklearn.metrics.average_precision_score` with ID positive and
  the ID-like score.
- `aupr_in_openood_auc`: `sklearn.metrics.precision_recall_curve` with ID
  positive and the ID-like score, followed by `sklearn.metrics.auc(recall,
  precision)`, matching pinned OpenOOD behavior.
- `auroc`: ID positive with the ID-like score. It is invariant to the equivalent
  OpenOOD OOD-positive/negated-score representation.

All metric functions reject empty ID or OOD arrays and non-finite scores.
OpenOOD parity tests target only the explicitly OpenOOD-compatible metrics.

## Evaluation, aggregation, and artifacts

Evaluate each OOD dataset against the same ID-test scores. Compute near/far
summaries as arithmetic means of per-dataset metrics; never pool group samples.

An output directory contains at least:

```text
resolved_config.yaml
run_metadata.json
metrics.json
scores/<dataset>.npz
```

`run_metadata.json` contains `schema_version`, `protocol_name`,
`openood_source_commit`, `oge_git_sha`, `model_name`,
`model_is_random_or_untrained`, `score_name`, and `device`.

Each score artifact contains `sample_id`, `prediction`, `class_label`, `is_id`,
and `id_like_score`. `metrics.json` contains per-dataset metrics, near mean,
far mean, metric-definition metadata, and `smoke_only`.

A random or untrained model is permitted only for bounded infrastructure smoke
validation. Its metrics are not research evidence.

## Manifest and real-environment validation

For every imglist, record path, SHA256, line count, missing image count,
duplicate sample-ID count, label range, and class histogram where applicable.
The Issue 6 server validation established these observations for the pinned
artifact and also exercised real-data loaders and bounded CUDA evaluation.

## Remaining limitations

- No independently published upstream checksum was available for the pinned
  Google Drive archives; local SHA256 values establish immutability after
  download but not independent upstream authentication.
- ZIP integrity was checked for every entry, but only representative images
  were decoded rather than every image in the release.
- Released TinyImageNet and Places365 list membership was used unchanged. The
  semantic-overlap-removal generation process and exact removed-image set were
  not reconstructed.
- The random-model CUDA smoke did not fix an initialization seed and is
  infrastructure validation only, not research evidence.
