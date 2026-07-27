# Reference Card 08: Raw Feature Artifact Contract

## Purpose and authority boundary

This card defines the durable checkpoint-to-feature artifact contract used by
later geometry, calibration, and feature-based OOD measurement tasks. It fixes
what is extracted, which dataset view is permitted, how sample identity is
preserved, and which provenance must accompany a cache.

This card does not implement extraction, authorize protected-split access, or
turn a smoke artifact into research evidence. Later implementation Issues must
also follow:

- [`02_architectures.md`](02_architectures.md) for the model and penultimate
  feature API;
- [`04_openood_v1_5_protocol.md`](04_openood_v1_5_protocol.md) for dataset
  membership, deterministic evaluation preprocessing, and sample identity;
- [`05_training_protocol.md`](05_training_protocol.md) for checkpoint and run
  provenance;
- [`06_feature_ood_detectors.md`](06_feature_ood_detectors.md) for detector fit
  splits and downstream transformations.

The audit handoff informed this contract, but the repository API overrides its
stale suggestions to use a forward hook and `model.fc`. Project code must use
`model(x, return_features=True)` and `model.classifier`.

## Artifact identity and feature endpoint

One artifact unit is identified by:

```text
checkpoint_sha256 x dataset_name x split
```

For the primary path, `wrn28_10` exposes a raw 640-dimensional penultimate
vector immediately before `model.classifier`:

```python
logits, features = model(images, return_features=True)
```

Extraction must verify that `features.shape[1] == model.feature_dim`, that the
classifier input width matches the feature width, and that logits from the
feature-returning path equal the ordinary forward logits for the same input.
No forward hook, hidden layer lookup, projection, implicit normalization, or
architecture-specific `fc` attribute is part of this contract.

## Deterministic extraction views

Detector and geometry fit statistics use `id_train`, but feature extraction
must not reuse the stochastic training view. Every extracted split uses:

```text
evaluation transform
shuffle = false
drop_last = false
model.eval()
torch.inference_mode()
```

The released OpenOOD imglist order and stable `sample_id` are preserved. No
sample may be silently skipped, duplicated, re-ordered, or augmented. A later
implementation must reject a mismatch between the observed sample sequence and
the configured manifest.

The permitted roles are:

| split | role |
| --- | --- |
| `id_train` | fit detector and representation statistics |
| `id_validation` | ID-only validation when a later reference card explicitly permits it |
| `id_test` | protected final ID evaluation after authorization |
| near/far OOD test | protected detector evaluation after authorization |
| OOD validation | compatibility only; never metric or detector selection in the main protocol |

## Protected-split authorization

`id_test` and OOD-test extraction is allowed only after the evaluated
configuration, training seed, checkpoint role, and checkpoint identity are
frozen by the active experiment Issue. Discovery and confirmation HPO trials
must not extract or traverse protected splits.

A fixed random/untrained checkpoint or the historical SGD seed-0 checkpoint may
be authorized by a bounded validation Issue to test the pipeline. Such output
must set `smoke_only: true` and is never an optimizer comparison or research
result.

## Stored arrays

Cache raw values only. L2 normalization, centering, standardization, PCA,
covariance estimation, and detector-specific dtype conversion are downstream
operations and must not create ambiguous replacement caches.

Each dataset/split directory contains memmap-compatible `.npy` arrays:

| file | dtype | shape | meaning |
| --- | --- | --- | --- |
| `features.npy` | `float32` | `[N, D]` | raw penultimate vectors |
| `logits.npy` | `float32` | `[N, C]` | raw classifier logits |
| `class_labels.npy` | `int64` | `[N]` | original class labels, including `-1` for OOD |
| `predictions.npy` | `int64` | `[N]` | `argmax` classifier predictions |
| `is_id.npy` | `bool` | `[N]` | binary ID/OOD identity |
| `sample_ids.npy` | Unicode string | `[N]` | stable project sample identifiers |

Checkpoint-level arrays are stored once per checkpoint artifact:

| file | dtype | shape | meaning |
| --- | --- | --- | --- |
| `classifier_weight.npy` | `float32` | `[C, D]` | `model.classifier.weight` |
| `classifier_bias.npy` | `float32` | `[C]` | classifier bias; manifest records `null` when absent |

Downstream covariance, eigendecomposition, and DDU paths cast the raw cache to
float64 as required by their own reference cards. They must not reinterpret the
stored extraction dtype as the fit dtype.

## Layout and manifest

The minimum artifact layout is:

```text
<artifact_root>/<checkpoint_sha256>/
  manifest.json
  checksums.sha256
  classifier_weight.npy
  classifier_bias.npy                 # only when the classifier has bias
  <dataset_name>/<split>/
    features.npy
    logits.npy
    class_labels.npy
    predictions.npy
    is_id.npy
    sample_ids.npy
```

`manifest.json` records at least:

```text
schema_version
checkpoint path and SHA256
checkpoint completed epoch and role (last, best_val, or snapshot)
training run/config identity and seed
model name, feature endpoint, feature dimension, class count
classifier bias presence
OGE Git SHA and dirty state
dataset protocol and membership-manifest hashes
dataset name, split, sample count, and ordered sample-ID digest
extraction dtype, array shapes, and array filenames
device and extraction command
protected-split authorization reference
smoke_only
```

Absolute data roots are runtime provenance and must not leak into
`sample_ids.npy`. A completed artifact must not be silently overwritten; a new
checkpoint or changed provenance produces a new artifact identity.

`checksums.sha256` covers every `.npy` file and `manifest.json`. The manifest and
checksum set are published only after every array has been written and
validated.

## Later implementation validation requirements

A later code Issue must test:

- ordinary-forward and `return_features=True` logits parity;
- deterministic repeated extraction and stable sample ordering;
- exact array dtypes, shapes, labels, predictions, and sample IDs;
- strict checkpoint loading and classifier width checks;
- rejection of missing, duplicate, non-finite, or reordered samples;
- manifest schema and SHA256 verification;
- raw-cache preservation across downstream transformations;
- protected-split refusal without explicit authorization;
- clear `smoke_only` boundaries for random or historical validation artifacts.

This card is documentation only. No raw-feature cache has been implemented or
validated by this Issue.
