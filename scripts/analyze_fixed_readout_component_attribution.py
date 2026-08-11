#!/usr/bin/env python3
"""Analyze MD--Marginal--RMD components from the verified Stage-2 reuse tree.

This is a versioned discovery analysis, not a continuation of the failed v2
radial gate.  It reads only score/sample-identity files named in the existing
allowlist, verifies each file immediately before use, never traverses a
protected dataset, and publishes into a new no-overwrite output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from oge.analysis.fixed_readout_component_attribution import (
    SCHEMA_VERSION,
    pair_outcome_summary,
    paired_component_attribution,
    reconstruction_record,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REUSE_SCHEMA = "fixed_readout_stage2_reuse_manifest_v1"
PAIR_SCHEMA = "fixed_readout_component_pairs_v3"
OOD_KEYS = ("cifar100", "tin", "mnist", "svhn", "texture", "places365")
TRANSFORMS = ("raw", "l2")
SCORE_PATHS = {
    "raw": {
        "md": "detector__mahalanobis_raw.npy",
        "rmd": "detector__relative_mahalanobis_raw.npy",
        "marginal": "diagnostics/raw_gaussian/global_distances.npy",
    },
    "l2": {
        "md": "detector__mahalanobis_pp.npy",
        "rmd": "detector__relative_mahalanobis_pp.npy",
        "marginal": "diagnostics/normalized_gaussian/marginal_mahalanobis.npy",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON mapping")
    return payload


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"unsafe allowlisted path: {value!r}")
    return value


def _assert_external_root(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the Git worktree")
    return resolved


class BundleReader:
    def __init__(self, root: Path, specification: Mapping[str, Any]) -> None:
        self.bundle_id = str(specification["bundle_id"])
        self.specification = specification
        self.root = (root / self.bundle_id).resolve()
        if root != self.root and root not in self.root.parents:
            raise ValueError("bundle root escapes retrieval root")
        files = specification.get("remote_integrity", {}).get("files")
        if not isinstance(files, list):
            raise ValueError("bundle is missing its file allowlist")
        self.files = {str(row["logical_path"]): dict(row) for row in files}

    def verified_path(self, logical_path: str) -> Path:
        logical = _safe_relative(logical_path)
        if logical not in self.files:
            raise PermissionError(f"file is outside the frozen reuse allowlist: {logical}")
        path = (self.root / logical).resolve()
        if self.root not in path.parents:
            raise PermissionError("allowlisted file escapes bundle root")
        record = self.files[logical]
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise ValueError(f"allowlisted file is absent or has wrong size: {logical}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"allowlisted file checksum mismatch: {logical}")
        return path

    def array(self, logical_path: str) -> np.ndarray:
        value = np.load(self.verified_path(logical_path), allow_pickle=False)
        if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"score array is not a finite vector: {logical_path}")
        return np.asarray(value, dtype=np.float64)

    def source_record(self, logical_path: str) -> dict[str, Any]:
        logical = _safe_relative(logical_path)
        record = self.files[logical]
        return {
            "bundle_id": self.bundle_id,
            "logical_path": logical,
            "sha256": record["sha256"],
            "size": int(record["size"]),
        }


def _score_logical_path(split: str, filename: str) -> str:
    return f"metrics/arrays/scores/{split}/{filename}"


def _sample_id_logical_path(split: str) -> str:
    dataset = "cifar10" if split == "id_test" else split
    return f"{dataset}/test/sample_ids.npy"


def load_component_scores(
    reader: BundleReader, split: str, transform: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    if transform not in SCORE_PATHS:
        raise ValueError(f"unsupported transform: {transform}")
    scores: dict[str, np.ndarray] = {}
    sources = []
    for component, filename in SCORE_PATHS[transform].items():
        logical = _score_logical_path(split, filename)
        values = reader.array(logical)
        # Raw global_distances is a distance; all other selected arrays already
        # follow the larger-is-more-ID-like score convention.
        if transform == "raw" and component == "marginal":
            values = -values
        scores[component] = values
        sources.append(reader.source_record(logical))
    check = reconstruction_record(scores["md"], scores["rmd"], scores["marginal"])
    if not check["pass"]:
        raise ValueError(
            f"MD component reconstruction failed for {reader.bundle_id}:{split}:{transform}: {check}"
        )
    return scores, sources, check


def _validate_reuse_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != REUSE_SCHEMA:
        raise ValueError("reuse manifest schema mismatch")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("reuse manifest is missing authorization")
    expected = {
        "mode": "reuse_only",
        "destination_overwrite_allowed": False,
        "new_protected_split_traversal_allowed": False,
        "checkpoint_or_feature_recompute_allowed": False,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ValueError("reuse authorization does not permit this read-only analysis")
    bundles = payload.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 30:
        raise ValueError("component discovery requires the frozen 30 last.pt bundles")
    if any(row.get("identity", {}).get("checkpoint_role") != "last" for row in bundles):
        raise ValueError("component discovery accepts only last.pt bundles")


def _load_pairs(
    path: Path | None, bundles: Mapping[str, BundleReader]
) -> list[dict[str, str]]:
    if path is None:
        return []
    payload = _read_mapping(path, label="pair manifest")
    if payload.get("schema_version") != PAIR_SCHEMA:
        raise ValueError("pair manifest schema mismatch")
    rows = payload.get("pairs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("pair manifest must contain at least one pair")
    output = []
    seen = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("each pair must be a mapping")
        required = {"pair_id", "left_bundle_id", "right_bundle_id", "left_label", "right_label"}
        if set(raw) != required:
            raise ValueError("pair rows have an unexpected schema")
        pair = {key: str(raw[key]) for key in required}
        if not pair["pair_id"] or pair["pair_id"] in seen:
            raise ValueError("pair IDs must be non-empty and unique")
        seen.add(pair["pair_id"])
        if pair["left_bundle_id"] not in bundles or pair["right_bundle_id"] not in bundles:
            raise ValueError("pair references a bundle outside the frozen population")
        left = bundles[pair["left_bundle_id"]].specification
        right = bundles[pair["right_bundle_id"]].specification
        if int(left["training_seed"]) != int(right["training_seed"]):
            raise ValueError("descriptive pairs must be seed-matched")
        output.append(pair)
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_checksums(root: Path) -> None:
    rows = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_analysis(
    *,
    reuse_manifest_path: Path,
    retrieval_root: Path,
    output_directory: Path,
    pair_manifest_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _read_mapping(reuse_manifest_path, label="reuse manifest")
    _validate_reuse_manifest(manifest)
    root = _assert_external_root(retrieval_root, label="retrieval root")
    if not root.is_dir():
        raise FileNotFoundError(f"retrieval root is absent: {root}")
    output = _assert_external_root(output_directory, label="output directory")
    if output.exists():
        raise FileExistsError("component-attribution output is no-overwrite")
    readers = {
        str(row["bundle_id"]): BundleReader(root, row) for row in manifest["bundles"]
    }
    pairs = _load_pairs(pair_manifest_path, readers)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    if temporary.exists():
        raise FileExistsError("temporary output path unexpectedly exists")
    temporary.mkdir(parents=True)
    try:
        component_cache: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
        bundle_rows: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        for bundle_id in sorted(readers):
            reader = readers[bundle_id]
            sample_id_hashes = {}
            for split in ("id_test", *OOD_KEYS):
                sample_path = _sample_id_logical_path(split)
                reader.verified_path(sample_path)
                sample_id_hashes[split] = reader.files[sample_path]["sha256"]
                for transform in TRANSFORMS:
                    scores, sources, check = load_component_scores(reader, split, transform)
                    component_cache[(bundle_id, split, transform)] = scores
                    source_rows.extend(sources)
                    if split == "id_test":
                        continue
                    id_scores = component_cache[(bundle_id, "id_test", transform)]
                    bundle_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "analysis_role": "historical_discovery_only",
                            "bundle_id": bundle_id,
                            "config_hash": reader.specification["config_hash"],
                            "training_seed": int(reader.specification["training_seed"]),
                            "optimizer_family": reader.specification["optimizer_family"],
                            "labels": reader.specification.get("labels", []),
                            "ood_dataset": split,
                            "transform": transform,
                            "score_reconstruction": check,
                            "pair_outcomes": {
                                component: pair_outcome_summary(
                                    id_scores[component], scores[component]
                                )
                                for component in ("md", "marginal", "rmd")
                            },
                            "ordered_sample_id_sha256": {
                                "id_test": sample_id_hashes["id_test"],
                                split: sample_id_hashes[split],
                            },
                        }
                    )

        pair_rows: list[dict[str, Any]] = []
        for pair in pairs:
            left = readers[pair["left_bundle_id"]]
            right = readers[pair["right_bundle_id"]]
            for split in OOD_KEYS:
                left_sample = left.files[_sample_id_logical_path(split)]["sha256"]
                right_sample = right.files[_sample_id_logical_path(split)]["sha256"]
                left_id = left.files[_sample_id_logical_path("id_test")]["sha256"]
                right_id = right.files[_sample_id_logical_path("id_test")]["sha256"]
                if left_sample != right_sample or left_id != right_id:
                    raise ValueError("paired bundles have different ordered sample identities")
                for transform in TRANSFORMS:
                    branch_0 = {
                        "id": component_cache[(pair["left_bundle_id"], "id_test", transform)],
                        "ood": component_cache[(pair["left_bundle_id"], split, transform)],
                    }
                    branch_1 = {
                        "id": component_cache[(pair["right_bundle_id"], "id_test", transform)],
                        "ood": component_cache[(pair["right_bundle_id"], split, transform)],
                    }
                    result = paired_component_attribution(branch_0, branch_1)
                    if not result["pass"]:
                        raise AssertionError(
                            f"pair attribution reconstruction failed: {pair['pair_id']}:{split}:{transform}"
                        )
                    pair_rows.append(
                        {
                            **pair,
                            "schema_version": SCHEMA_VERSION,
                            "analysis_role": "historical_descriptive_pair_not_causal",
                            "ood_dataset": split,
                            "transform": transform,
                            "result": result,
                        }
                    )

        unique_sources = {
            (row["bundle_id"], row["logical_path"]): row for row in source_rows
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "analysis_role": "historical_discovery_only",
            "claim_boundary": (
                "independently trained historical configurations; component localization "
                "is descriptive and is not a shared-prefix causal effect"
            ),
            "reuse_manifest": {
                "path": str(reuse_manifest_path.resolve()),
                "sha256": _sha256(reuse_manifest_path),
            },
            "bundle_count": len(readers),
            "bundle_metric_row_count": len(bundle_rows),
            "descriptive_pair_count": len(pairs),
            "pair_attribution_row_count": len(pair_rows),
            "ood_datasets": list(OOD_KEYS),
            "transforms": list(TRANSFORMS),
            "selection_performed": False,
            "fresh_shared_prefix_confirmation": "NOT_RUN",
        }
        _write_json(temporary / "summary.json", summary)
        _write_jsonl(temporary / "bundle_component_metrics.jsonl", bundle_rows)
        _write_jsonl(temporary / "pair_component_attribution.jsonl", pair_rows)
        _write_jsonl(
            temporary / "score_source_manifest.jsonl",
            [unique_sources[key] for key in sorted(unique_sources)],
        )
        _write_checksums(temporary)
        os.replace(temporary, output)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        help="Optional prespecified seed-matched descriptive pairs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_analysis(
        reuse_manifest_path=args.reuse_manifest,
        retrieval_root=args.retrieval_root,
        output_directory=args.output_directory,
        pair_manifest_path=args.pair_manifest,
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
