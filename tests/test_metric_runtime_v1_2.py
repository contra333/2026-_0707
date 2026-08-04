import json

import numpy as np

from oge.evaluation.extraction import sha256_file
from oge.evaluation.runtime import FittedMetricRuntime, write_bounded_smoke_result


def _write_split(root, relative, *, features, logits, labels, sample_prefix):
    directory = root / relative
    directory.mkdir(parents=True)
    arrays = {
        "features": features.astype(np.float32),
        "logits": logits.astype(np.float32),
        "class_labels": labels.astype(np.int64),
        "predictions": np.argmax(logits, axis=1).astype(np.int64),
        "is_id": np.ones(len(features), dtype=np.bool_),
        "sample_ids": np.asarray([f"{sample_prefix}-{index:04d}" for index in range(len(features))]),
    }
    for name, array in arrays.items():
        np.save(directory / f"{name}.npy", array, allow_pickle=False)
    return arrays


def _raw_cache(tmp_path):
    root = tmp_path / ("a" * 64)
    root.mkdir()
    rng = np.random.default_rng(42)
    feature_dim = 100
    class_count = 10
    weight = rng.normal(size=(class_count, feature_dim))
    bias = rng.normal(size=class_count)
    labels = np.repeat(np.arange(class_count), 12)
    class_offsets = rng.normal(scale=0.5, size=(class_count, feature_dim))
    train_features = rng.normal(size=(len(labels), feature_dim)) + class_offsets[labels]
    train_logits = train_features @ weight.T + bias
    validation_labels = np.repeat(np.arange(class_count), 2)
    validation_features = rng.normal(size=(len(validation_labels), feature_dim)) + class_offsets[validation_labels]
    validation_logits = validation_features @ weight.T + bias
    train = _write_split(
        root,
        "cifar10/train",
        features=train_features,
        logits=train_logits,
        labels=labels,
        sample_prefix="cifar10:train",
    )
    validation = _write_split(
        root,
        "cifar10/validation",
        features=validation_features,
        logits=validation_logits,
        labels=validation_labels,
        sample_prefix="cifar10:validation",
    )
    np.save(root / "classifier_weight.npy", weight.astype(np.float32), allow_pickle=False)
    np.save(root / "classifier_bias.npy", bias.astype(np.float32), allow_pickle=False)
    manifest = {
        "checkpoint": {
            "sha256": "a" * 64,
            "role": "last",
            "completed_epoch": 200,
            "training_seed": 1,
            "config_hash": "b" * 64,
        },
        "dataset": {
            "splits": {
                "id_train": {
                    "relative_directory": "cifar10/train",
                    "sample_count": len(train_features),
                },
                "id_validation": {
                    "relative_directory": "cifar10/validation",
                    "sample_count": len(validation_features),
                },
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n")
    return root, train, validation


def test_fit_once_runtime_scores_every_canonical_detector_family(tmp_path):
    root, _, validation = _raw_cache(tmp_path)
    _, _, runtime = FittedMetricRuntime.fit(root, include_gda=False, include_neco=True)

    scores = runtime.score(validation)

    expected = {
        "ood_score/msp",
        "ood_score/max_logit",
        "ood_score/energy_t1",
        "detector/mahalanobis_raw",
        "detector/marginal_mahalanobis_raw",
        "detector/relative_mahalanobis_raw",
        "detector/mahalanobis_pp",
        "detector/relative_mahalanobis_pp",
        "detector/knn_l2_k50",
        "detector/ncp_euclidean_raw",
        "detector/ctm_prototype_cosine",
        "detector/weight_cosine_appendix",
        "detector/vim_author_dim",
        "detector/neco_author_std_dim100",
        "diagnostics",
    }
    assert set(scores) == expected
    for key in expected - {"diagnostics"}:
        assert scores[key].shape == (len(validation["features"]),)
        assert np.isfinite(scores[key]).all()


def test_bounded_cache_smoke_writes_nonprotected_metrics_and_score_checksums(tmp_path):
    root, _, _ = _raw_cache(tmp_path)
    output = write_bounded_smoke_result(
        raw_artifact_path=root,
        output_dir=tmp_path / "smoke",
    )

    payload = json.loads((output / "metrics.json").read_text())
    assert payload["smoke_only"] is True
    assert payload["protected_result"] is False
    assert payload["query_split"] == "id_validation"
    assert payload["fit_diagnostics"]["gda_executed"] is False
    assert payload["fit_diagnostics"]["neco_executed"] is True
    assert "detector/vim_author_dim" in payload["score_index"]
    assert "metrics.json" in (output / "checksums.sha256").read_text()
