"""Deterministic CIFAR-10 45k/5k holdout generation and validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .imglist_dataset import parse_imglist_entry

PROTOCOL_NAME = "oge_cifar10_holdout_v1"
SPLIT_SEED = "2027"
DIGEST_NAMESPACE = "oge-cifar10-holdout-v1"
CLASS_LABELS = tuple(range(10))
TRAIN_PER_CLASS = 4_500
VALIDATION_PER_CLASS = 500
TEST_PER_CLASS = 1_000

TRAIN_FILENAME = "train_cifar10_45k.txt"
VALIDATION_FILENAME = "val_cifar10_5k.txt"
TEST_FILENAME = "test_cifar10_10k.txt"
MANIFEST_FILENAME = "split_manifest.jsonl"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def membership_digest(sample_id: str) -> str:
    payload = (
        DIGEST_NAMESPACE.encode("utf-8")
        + b"\0"
        + SPLIT_SEED.encode("utf-8")
        + b"\0"
        + sample_id.encode("utf-8")
    )
    return _sha256_bytes(payload)


def _read_source(
    path: str | Path,
    *,
    source_list: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = Path(path)
    content = source_path.read_bytes()
    lines = content.decode("utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for source_row, line in enumerate(lines, start=1):
        relative_path, label = parse_imglist_entry(line, line_number=source_row)
        records.append(
            {
                "source_list": source_list,
                "source_row": source_row,
                "relative_path": relative_path,
                "sample_id": f"cifar10:{relative_path}",
                "label": label,
            }
        )
    return records, {
        "logical_name": source_list,
        "filename": source_path.name,
        "sha256": _sha256_bytes(content),
        "row_count": len(records),
    }


def _validate_source(
    records: list[dict[str, Any]],
    *,
    expected_per_class: int,
    source_list: str,
) -> None:
    labels = Counter(record["label"] for record in records)
    expected = {label: expected_per_class for label in CLASS_LABELS}
    if labels != expected:
        raise ValueError(
            f"{source_list} class counts must be "
            f"{dict(sorted(expected.items()))}, got {dict(sorted(labels.items()))}"
        )
    sample_ids = [record["sample_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{source_list} contains duplicate sample IDs")


def _render_imglist(records: list[dict[str, Any]]) -> bytes:
    return "".join(
        f"{record['relative_path']} {record['label']}\n" for record in records
    ).encode("utf-8")


def _class_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["label"] for record in records)
    return {str(label): counts[label] for label in CLASS_LABELS}


def _write_manifest(
    path: Path,
    *,
    header: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    lines = [
        json.dumps(
            {"record_type": "header", **header},
            sort_keys=True,
            separators=(",", ":"),
        )
    ]
    lines.extend(
        json.dumps(
            {"record_type": "membership", **record},
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_cifar10_holdout(
    *,
    official_train_imglist: str | Path,
    openood_id_validation_imglist: str | Path,
    openood_id_test_imglist: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate the frozen project ID lists and a row-level JSONL manifest."""
    train_source, train_source_meta = _read_source(
        official_train_imglist,
        source_list="openood_official_train_50k",
    )
    openood_validation, validation_source_meta = _read_source(
        openood_id_validation_imglist,
        source_list="openood_id_validation_1k",
    )
    openood_test, test_source_meta = _read_source(
        openood_id_test_imglist,
        source_list="openood_id_test_9k",
    )

    _validate_source(
        train_source,
        expected_per_class=TRAIN_PER_CLASS + VALIDATION_PER_CLASS,
        source_list=train_source_meta["logical_name"],
    )
    _validate_source(
        openood_validation,
        expected_per_class=100,
        source_list=validation_source_meta["logical_name"],
    )
    _validate_source(
        openood_test,
        expected_per_class=900,
        source_list=test_source_meta["logical_name"],
    )

    validation_ids: set[str] = set()
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in train_source:
        by_class[record["label"]].append(record)
    for label in CLASS_LABELS:
        ranked = sorted(
            by_class[label],
            key=lambda record: (
                membership_digest(record["sample_id"]),
                record["sample_id"],
            ),
        )
        validation_ids.update(
            record["sample_id"] for record in ranked[:VALIDATION_PER_CLASS]
        )

    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for record in train_source:
        role = (
            "id_validation"
            if record["sample_id"] in validation_ids
            else "id_train"
        )
        annotated = {
            **record,
            "membership_digest": membership_digest(record["sample_id"]),
            "role": role,
        }
        manifest_records.append(annotated)
        if role == "id_validation":
            validation_records.append(record)
        else:
            train_records.append(record)

    official_test = openood_validation + openood_test
    validation_test_ids = {record["sample_id"] for record in openood_validation}
    openood_test_ids = {record["sample_id"] for record in openood_test}
    overlap = validation_test_ids.intersection(openood_test_ids)
    if overlap:
        raise ValueError("OpenOOD ID-validation 1k and ID-test 9k overlap")
    if len(validation_test_ids.union(openood_test_ids)) != 10_000:
        raise ValueError("OpenOOD ID-validation 1k and ID-test 9k do not form 10k")
    _validate_source(
        official_test,
        expected_per_class=TEST_PER_CLASS,
        source_list="official_test_union_10k",
    )
    for record in official_test:
        manifest_records.append({**record, "role": "id_test"})

    expected_counts = {
        "id_train": TRAIN_PER_CLASS * len(CLASS_LABELS),
        "id_validation": VALIDATION_PER_CLASS * len(CLASS_LABELS),
        "id_test": TEST_PER_CLASS * len(CLASS_LABELS),
    }
    actual_counts = Counter(record["role"] for record in manifest_records)
    if actual_counts != expected_counts:
        raise ValueError(
            f"generated role counts must be {expected_counts}, got {dict(actual_counts)}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered = {
        TRAIN_FILENAME: _render_imglist(train_records),
        VALIDATION_FILENAME: _render_imglist(validation_records),
        TEST_FILENAME: _render_imglist(official_test),
    }
    for filename, content in rendered.items():
        (output / filename).write_bytes(content)

    role_records = {
        "id_train": train_records,
        "id_validation": validation_records,
        "id_test": official_test,
    }
    header = {
        "schema_version": "1.0",
        "protocol_name": PROTOCOL_NAME,
        "split_seed": SPLIT_SEED,
        "membership_rule": (
            'per class sort by '
            '(SHA256("oge-cifar10-holdout-v1\\0" + "2027\\0" + sample_id), '
            "sample_id); first 500 are id_validation"
        ),
        "sample_id_rule": '"cifar10:" + canonical_relative_path',
        "sources": {
            meta["logical_name"]: meta
            for meta in (
                train_source_meta,
                validation_source_meta,
                test_source_meta,
            )
        },
        "outputs": {
            role: {
                "filename": filename,
                "sha256": _sha256_bytes(rendered[filename]),
                "row_count": len(role_records[role]),
                "class_counts": _class_counts(role_records[role]),
            }
            for role, filename in (
                ("id_train", TRAIN_FILENAME),
                ("id_validation", VALIDATION_FILENAME),
                ("id_test", TEST_FILENAME),
            )
        },
        "openood_test_union": {
            "intersection_count": 0,
            "union_count": len(official_test),
            "class_counts": _class_counts(official_test),
        },
    }
    manifest_path = output / MANIFEST_FILENAME
    _write_manifest(manifest_path, header=header, records=manifest_records)
    return {
        **header,
        "manifest": {
            "filename": MANIFEST_FILENAME,
            "sha256": _sha256_bytes(manifest_path.read_bytes()),
            "row_count": 1 + len(manifest_records),
        },
    }


def read_holdout_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and minimally validate a generated JSONL split manifest."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("split manifest is empty")
    records = [json.loads(line) for line in lines]
    header = records[0]
    if header.pop("record_type", None) != "header":
        raise ValueError("split manifest first record must be a header")
    memberships: list[dict[str, Any]] = []
    for record in records[1:]:
        if record.pop("record_type", None) != "membership":
            raise ValueError("split manifest contains a non-membership row")
        memberships.append(record)
    return header, memberships
