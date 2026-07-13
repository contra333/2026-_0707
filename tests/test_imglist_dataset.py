from pathlib import Path

import pytest
import numpy as np
import torch
from PIL import Image

from oge.data import ImglistDataset, build_openood_cifar10_loaders, parse_imglist_entry


def _make_image(path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (32, 32), color=1).save(path)


def test_imglist_dataset_preserves_identity_and_integer_label(tmp_path):
    _make_image(tmp_path / "images/nested/example.png", mode="L")
    imglist = tmp_path / "list.txt"
    imglist.write_text("nested\\example.png -1\n", encoding="utf-8")
    dataset = ImglistDataset(
        dataset_name="tin",
        split="test",
        is_id=False,
        imglist_path=imglist,
        data_root=tmp_path / "images",
        transform=lambda image: torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1),
    )

    sample = dataset[0]

    assert sample["sample_id"] == "tin:nested/example.png"
    assert sample["dataset_name"] == "tin"
    assert sample["split"] == "test"
    assert sample["class_label"] == -1
    assert sample["is_id"] is False
    assert sample["image"].shape == (3, 32, 32)


@pytest.mark.parametrize(
    "line,match",
    [
        ("image.png", "expected"),
        ("image.png label", "integer"),
        ("/absolute.png 0", "relative"),
        ("../escape.png 0", "relative"),
        ("too many tokens 0", "expected"),
    ],
)
def test_parse_imglist_rejects_malformed_entries(line, match):
    with pytest.raises(ValueError, match=match):
        parse_imglist_entry(line, line_number=3)


def test_dataset_name_cannot_break_sample_id_namespace(tmp_path):
    imglist = tmp_path / "list.txt"
    imglist.write_text("image.png 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_name"):
        ImglistDataset(
            dataset_name="bad:name",
            split="test",
            is_id=True,
            imglist_path=imglist,
            data_root=tmp_path,
            transform=lambda image: image,
        )


def test_explicit_factory_builds_hierarchical_loaders_from_imglists(tmp_path):
    _make_image(tmp_path / "images/id.png")
    _make_image(tmp_path / "images/ood.png")
    (tmp_path / "id.txt").write_text("id.png 3\n", encoding="utf-8")
    (tmp_path / "ood.txt").write_text("ood.png -1\n", encoding="utf-8")
    config = {
        "dataset_class": "imglist",
        "image_root": "images",
        "batch_size": 1,
        "num_workers": 0,
        "datasets": {
            "id_test": {
                "dataset_name": "cifar10",
                "split": "test",
                "is_id": True,
                "group": "id",
                "imglist": "id.txt",
            },
            "cifar100": {
                "dataset_name": "cifar100",
                "split": "test",
                "is_id": False,
                "group": "near",
                "imglist": "ood.txt",
            },
        },
    }

    loaders = build_openood_cifar10_loaders(config, data_root=tmp_path)

    id_loader = loaders["id"]["id_test"]
    id_batch = next(iter(id_loader))
    ood_batch = next(iter(loaders["near"]["cifar100"]))
    assert id_batch["sample_id"] == ["cifar10:id.png"]
    assert id_batch["is_id"].tolist() == [True]
    assert ood_batch["class_label"].tolist() == [-1]
    assert ood_batch["is_id"].tolist() == [False]
    assert id_loader.batch_size == 1
    assert id_loader.num_workers == 0
    assert id_loader.pin_memory is False
    assert id_loader.drop_last is False
    assert id_loader.persistent_workers is False
    assert id_loader.generator is None


def test_factory_rejects_unregistered_dataset_class(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset_class"):
        build_openood_cifar10_loaders(
            {"dataset_class": "NotEvaluated", "datasets": {}}, data_root=tmp_path
        )


def test_factory_applies_explicit_loader_settings_and_train_generator(tmp_path):
    imglist = tmp_path / "id.txt"
    imglist.write_text("image.png 0\n", encoding="utf-8")
    config = {
        "datasets": {
            "id_train": {
                "dataset_name": "cifar10",
                "split": "train",
                "is_id": True,
                "group": "id",
                "imglist": "id.txt",
            },
            "id_validation": {
                "dataset_name": "cifar10",
                "split": "validation",
                "is_id": True,
                "group": "id",
                "imglist": "id.txt",
            },
        }
    }
    train_generator = torch.Generator().manual_seed(7)

    loaders = build_openood_cifar10_loaders(
        config,
        data_root=tmp_path,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        train_generator=train_generator,
    )

    train_loader = loaders["id"]["id_train"]
    validation_loader = loaders["id"]["id_validation"]
    for loader in (train_loader, validation_loader):
        assert loader.batch_size == 4
        assert loader.num_workers == 1
        assert loader.pin_memory is True
        assert loader.persistent_workers is True
    assert train_loader.drop_last is True
    assert validation_loader.drop_last is False
    assert train_loader.generator is train_generator
    assert validation_loader.generator is None


def test_factory_rejects_persistent_workers_without_workers(tmp_path):
    with pytest.raises(ValueError, match="requires num_workers > 0"):
        build_openood_cifar10_loaders(
            {
                "num_workers": 0,
                "persistent_workers": True,
                "datasets": {},
            },
            data_root=tmp_path,
        )
