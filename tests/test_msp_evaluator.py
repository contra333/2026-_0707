import json

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from oge.evaluation import evaluate_msp_protocol, infer_msp
from oge.models import make_model


class ScoreFixture(Dataset):
    def __init__(self, dataset_name, values, *, is_id, label):
        self.dataset_name = dataset_name
        self.values = values
        self.is_id = is_id
        self.label = label

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {
            "image": torch.tensor([self.values[index]], dtype=torch.float32),
            "class_label": self.label,
            "sample_id": f"{self.dataset_name}:sample_{index}.png",
            "dataset_name": self.dataset_name,
            "split": "test",
            "is_id": self.is_id,
        }


class TwoClassFixtureModel(nn.Module):
    def forward(self, x):
        value = x[:, 0]
        return torch.stack([value, -value], dim=1)


class ImageFixture(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 32, 32),
            "class_label": index,
            "sample_id": f"cifar10:image_{index}.png",
            "dataset_name": "cifar10",
            "split": "test",
            "is_id": True,
        }


def _loader(name, values, *, is_id):
    return DataLoader(
        ScoreFixture(name, values, is_id=is_id, label=0 if is_id else -1),
        batch_size=2,
    )


def test_msp_inference_uses_existing_oge_model_logits_api():
    model = make_model({"name": "toy_cifar_cnn", "num_classes": 10})

    scores = infer_msp(model, DataLoader(ImageFixture(), batch_size=2), device="cpu")

    assert scores["sample_id"].tolist() == ["cifar10:image_0.png", "cifar10:image_1.png"]
    assert scores["prediction"].shape == (2,)
    assert np.all(scores["id_like_score"] >= 0)
    assert np.all(scores["id_like_score"] <= 1)


def test_msp_evaluator_writes_minimum_artifacts_and_averages_dataset_metrics(tmp_path):
    loaders = {
        "id": {"id_test": _loader("cifar10", [4.0, 3.0, 2.0, 1.0], is_id=True)},
        "ood_validation": {},
        "near": {
            "cifar100": _loader("cifar100", [0.3, 0.2], is_id=False),
            "tin": _loader("tin", [1.5, 0.1], is_id=False),
        },
        "far": {
            "mnist": _loader("mnist", [0.1, 0.0], is_id=False),
            "svhn": _loader("svhn", [0.2, 0.1], is_id=False),
            "texture": _loader("texture", [0.4, 0.3], is_id=False),
            "places365": _loader("places365", [0.6, 0.5], is_id=False),
        },
    }

    payload = evaluate_msp_protocol(
        TwoClassFixtureModel(),
        loaders,
        resolved_config={"fixture": True},
        output_dir=tmp_path,
        model_name="fixture",
        model_is_random_or_untrained=True,
        device="cpu",
        oge_git_sha="abc123",
    )

    assert yaml.safe_load((tmp_path / "resolved_config.yaml").read_text()) == {"fixture": True}
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["model_is_random_or_untrained"] is True
    assert metadata["score_name"] == "msp"
    assert payload["smoke_only"] is True
    metrics_file = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics_file == payload
    near_auroc = np.mean(
        [payload["per_dataset"][name]["auroc"] for name in ("cifar100", "tin")]
    )
    assert payload["near_mean"]["auroc"] == near_auroc

    expected_score_files = {
        "cifar10.npz",
        "cifar100.npz",
        "tin.npz",
        "mnist.npz",
        "svhn.npz",
        "texture.npz",
        "places365.npz",
    }
    assert {path.name for path in (tmp_path / "scores").iterdir()} == expected_score_files
    with np.load(tmp_path / "scores/cifar100.npz") as scores:
        assert set(scores.files) == {
            "sample_id",
            "prediction",
            "class_label",
            "is_id",
            "id_like_score",
        }
        assert scores["class_label"].tolist() == [-1, -1]
        assert scores["is_id"].tolist() == [False, False]
