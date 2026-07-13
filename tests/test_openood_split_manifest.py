import hashlib
from pathlib import Path

from PIL import Image

from oge.data import inspect_imglist, load_dataset_config


def test_protocol_config_contains_fixed_id_near_and_far_lists():
    config_path = Path(__file__).parents[1] / "configs/datasets/openood_cifar10_v1_5.yaml"
    config = load_dataset_config(config_path)

    assert config["datasets"]["id_train"]["expected_count"] == 50000
    assert config["datasets"]["id_validation"]["expected_count"] == 1000
    assert config["datasets"]["id_test"]["expected_count"] == 9000
    assert {
        key for key, item in config["datasets"].items() if item["group"] == "near"
    } == {"cifar100", "tin"}
    assert {
        key for key, item in config["datasets"].items() if item["group"] == "far"
    } == {"mnist", "svhn", "texture", "places365"}
    assert config["datasets"]["ood_validation_tin"]["role"] == "compatibility_only"


def test_manifest_records_checksum_missing_duplicates_and_labels(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (4, 4)).save(image_root / "present.png")
    content = "present.png 2\npresent.png 2\nmissing.png -1\n"
    imglist = tmp_path / "list.txt"
    imglist.write_text(content, encoding="utf-8")

    report = inspect_imglist(
        imglist_path=imglist,
        data_root=image_root,
        dataset_name="fixture",
    )

    assert report["sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert report["line_count"] == 3
    assert report["missing_image_count"] == 1
    assert report["duplicate_sample_id_count"] == 1
    assert report["label_min"] == -1
    assert report["label_max"] == 2
    assert report["class_histogram"] == {"-1": 1, "2": 2}
