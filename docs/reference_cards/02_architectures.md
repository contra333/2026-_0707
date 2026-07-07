# Reference Card 02: Architectures

This card defines the model API contract for architectures in this repository.

## Forward API

Models must support two forward-call forms:

- `model(x)` returns only `logits`.
- `model(x, return_features=True)` returns `(logits, features)`.

## Tensor shapes

For a batch size `B`, configured class count `num_classes`, and configured feature size `feature_dim`:

- `logits.shape == [B, num_classes]`
- `features.shape == [B, feature_dim]`

## Feature semantics

`features` is the penultimate representation immediately before the final classifier.

The `return_features=True` code path must not change the computed logits values. Both forward-call forms must compute logits from the same penultimate representation using the same final classifier operation.

## Classifier exposure

When practical, architectures should expose the final classifier as `model.classifier`.
