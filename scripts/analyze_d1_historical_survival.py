#!/usr/bin/env python3
"""Reproduce the checksum-frozen historical D1 survival analysis.

This command only reuses existing score and sample-ID arrays. It performs no
checkpoint inference and publishes an output directory only after every input
and expected-output identity check has passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from oge.analysis.fixed_readout_component_attribution import (
    pair_outcome_summary,
    pair_transition_summary,
)


SCHEMA_VERSION = "d1_historical_survival_input_v1"
STATE_UTILITY = {"incorrect": 0.0, "tie": 0.5, "correct": 1.0}
RESULT_FILENAMES = ("data_quality.json", "pair_metrics.csv", "summary.json")


class D1ValidationError(ValueError):
    """Raised when a frozen input or expected output identity does not match."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transition_rates(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return tie-aware Gain, Loss, and PairOrderChurn rates.

    The transition table itself is supplied by the existing exact
    ``pair_transition_summary`` primitive. Pair utility is 0, 0.5, or 1 for
    incorrect, tied, or correct ID--OOD ordering, respectively.
    """

    pair_count = int(summary["pair_count"])
    if pair_count <= 0:
        raise D1ValidationError("pair transition summary has no pairs")
    gain = 0.0
    loss = 0.0
    transitions = summary["transitions"]
    for source, targets in transitions.items():
        if source not in STATE_UTILITY:
            raise D1ValidationError(f"unknown pair state: {source!r}")
        for target, count in targets.items():
            if target not in STATE_UTILITY:
                raise D1ValidationError(f"unknown pair state: {target!r}")
            delta = STATE_UTILITY[target] - STATE_UTILITY[source]
            gain += max(delta, 0.0) * int(count)
            loss += max(-delta, 0.0) * int(count)
    return gain / pair_count, loss / pair_count, (gain + loss) / pair_count


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise D1ValidationError(f"cannot read input manifest: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise D1ValidationError(f"input manifest must use {SCHEMA_VERSION!r}")
    for key in ("analysis", "logical_files", "bundles", "expected_output_identity"):
        if key not in value:
            raise D1ValidationError(f"input manifest is missing {key!r}")
    if not isinstance(value["bundles"], list) or len(value["bundles"]) != 12:
        raise D1ValidationError("input manifest must contain exactly 12 bundles")
    return value


def _safe_input_path(
    retrieval_root: Path,
    bundle_id: str,
    logical_path: str,
) -> Path:
    logical = PurePosixPath(logical_path)
    if logical.is_absolute() or ".." in logical.parts or not logical.parts:
        raise D1ValidationError(f"unsafe logical path: {logical_path!r}")
    bundle_root = (retrieval_root / bundle_id).resolve()
    candidate = (bundle_root / Path(*logical.parts)).resolve()
    if not candidate.is_relative_to(bundle_root):
        raise D1ValidationError(f"logical path escapes its bundle: {logical_path!r}")
    return candidate


def _validate_file_identity(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    if not path.is_file():
        raise D1ValidationError(f"missing input file for {label}: {path}")
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise D1ValidationError(
            f"size mismatch for {label}: expected {expected_size}, got {observed_size}"
        )
    observed_sha256 = sha256(path)
    if observed_sha256 != expected_sha256:
        raise D1ValidationError(
            f"hash mismatch for {label}: expected {expected_sha256}, "
            f"got {observed_sha256}"
        )


def _load_array(path: Path, *, label: str, score: bool) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise D1ValidationError(f"cannot load {label}: {error}") from error
    if array.ndim != 1 or array.size == 0:
        raise D1ValidationError(f"shape failure for {label}: expected non-empty 1D")
    if score and not np.isfinite(array).all():
        raise D1ValidationError(f"non-finite score values for {label}")
    return array


def _validated_inputs(
    retrieval_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, str, bool], np.ndarray], dict[str, Any]]:
    logical_files = manifest["logical_files"]
    datasets = manifest["analysis"]["datasets"]
    bundle_ids: set[str] = set()
    reference_ids: dict[str, np.ndarray] = {}
    array_cache: dict[tuple[str, str], np.ndarray] = {}
    score_cache: dict[tuple[str, str, str, bool], np.ndarray] = {}
    quality = {
        "bundle_count": len(manifest["bundles"]),
        "score_files_checked": 0,
        "score_sha_mismatches": [],
        "sample_id_mismatches": [],
        "finite_failures": [],
    }

    def load_key(bundle: Mapping[str, Any], key: str, *, score: bool) -> np.ndarray:
        bundle_id = bundle["bundle_id"]
        if key not in logical_files or key not in bundle["files"]:
            raise D1ValidationError(
                f"bundle {bundle_id} is missing logical file identity {key!r}"
            )
        spec = logical_files[key]
        path = _safe_input_path(retrieval_root, bundle_id, spec["logical_path"])
        _validate_file_identity(
            path,
            expected_size=int(spec["size"]),
            expected_sha256=bundle["files"][key],
            label=f"{bundle_id}:{spec['logical_path']}",
        )
        cache_key = (bundle_id, key)
        if cache_key not in array_cache:
            array_cache[cache_key] = _load_array(
                path,
                label=f"{bundle_id}:{spec['logical_path']}",
                score=score,
            )
        return array_cache[cache_key]

    for bundle in manifest["bundles"]:
        bundle_id = bundle.get("bundle_id")
        if (
            not isinstance(bundle_id, str)
            or len(bundle_id) != 64
            or any(character not in "0123456789abcdef" for character in bundle_id)
        ):
            raise D1ValidationError(f"invalid bundle ID: {bundle_id!r}")
        if bundle_id in bundle_ids:
            raise D1ValidationError(f"duplicate bundle ID: {bundle_id}")
        bundle_ids.add(bundle_id)
        for dataset, dataset_spec in datasets.items():
            ids_by_side = {
                True: load_key(bundle, dataset_spec["id_sample_ids"], score=False),
                False: load_key(bundle, dataset_spec["ood_sample_ids"], score=False),
            }
            for id_side, ids in ids_by_side.items():
                split = "id_test" if id_side else dataset
                if split not in reference_ids:
                    reference_ids[split] = ids
                elif not np.array_equal(reference_ids[split], ids):
                    raise D1ValidationError(
                        f"sample-ID mismatch for bundle {bundle_id}, split {split}"
                    )
            for detector, score_keys in dataset_spec["scores"].items():
                for id_side, side in ((True, "id"), (False, "ood")):
                    scores = load_key(bundle, score_keys[side], score=True)
                    quality["score_files_checked"] += 1
                    ids = ids_by_side[id_side]
                    if scores.shape != ids.shape:
                        split = "id_test" if id_side else dataset
                        raise D1ValidationError(
                            f"shape failure for bundle {bundle_id}, {split}, {detector}: "
                            f"scores {scores.shape} versus sample IDs {ids.shape}"
                        )
                    score_cache[(bundle_id, dataset, detector, id_side)] = scores

    if quality["score_files_checked"] != 96:
        raise D1ValidationError(
            "frozen scope must validate exactly 96 score-file uses, got "
            f"{quality['score_files_checked']}"
        )
    return score_cache, quality


def _grouped_bundles(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    groups: dict[str, list[tuple[int, str]]] = {}
    for bundle in manifest["bundles"]:
        groups.setdefault(bundle["group"], []).append(
            (int(bundle["seed"]), bundle["bundle_id"])
        )
    expected = ("adam_c3", "adamw_c3", "adam_c4", "adamw_c4")
    if tuple(groups) != expected:
        raise D1ValidationError(f"unexpected bundle group order or scope: {tuple(groups)}")
    output = {}
    for group, members in groups.items():
        members.sort()
        if [seed for seed, _ in members] != [0, 1, 2]:
            raise D1ValidationError(f"group {group} must contain seeds 0, 1, and 2")
        output[group] = [bundle_id for _, bundle_id in members]
    return output


def _pair_rows(
    manifest: Mapping[str, Any],
    cache: Mapping[tuple[str, str, str, bool], np.ndarray],
) -> list[dict[str, Any]]:
    groups = _grouped_bundles(manifest)
    pairs = []
    for role in ("c3", "c4"):
        left = groups[f"adam_{role}"]
        right = groups[f"adamw_{role}"]
        for seed, (bundle_0, bundle_1) in enumerate(zip(left, right, strict=True)):
            pairs.append(("cross_policy", role, f"seed{seed}", bundle_0, bundle_1))
    for group_name, group in groups.items():
        for (seed_0, bundle_0), (seed_1, bundle_1) in combinations(
            enumerate(group), 2
        ):
            pairs.append(
                (
                    "same_policy_seed",
                    group_name,
                    f"seed{seed_0}_seed{seed_1}",
                    bundle_0,
                    bundle_1,
                )
            )

    rows = []
    for pair_type, group, pair_label, bundle_0, bundle_1 in pairs:
        for dataset, dataset_spec in manifest["analysis"]["datasets"].items():
            for detector in dataset_spec["scores"]:
                id_0 = cache[(bundle_0, dataset, detector, True)]
                ood_0 = cache[(bundle_0, dataset, detector, False)]
                id_1 = cache[(bundle_1, dataset, detector, True)]
                ood_1 = cache[(bundle_1, dataset, detector, False)]
                transition = pair_transition_summary(id_0, ood_0, id_1, ood_1)
                outcome_0 = pair_outcome_summary(id_0, ood_0)
                outcome_1 = pair_outcome_summary(id_1, ood_1)
                gain, loss, churn = transition_rates(transition)
                delta = (
                    outcome_1["auroc_id_positive"]
                    - outcome_0["auroc_id_positive"]
                )
                rows.append(
                    {
                        "pair_type": pair_type,
                        "group": group,
                        "pair_label": pair_label,
                        "bundle_0": bundle_0,
                        "bundle_1": bundle_1,
                        "dataset": dataset,
                        "detector": detector,
                        "auroc_0": outcome_0["auroc_id_positive"],
                        "auroc_1": outcome_1["auroc_id_positive"],
                        "delta_auroc": delta,
                        "abs_delta_auroc": abs(delta),
                        "gain": gain,
                        "loss": loss,
                        "pair_order_churn": churn,
                        "hidden_churn_gap": churn - abs(delta),
                        "balance_residual": delta - (gain - loss),
                        "pair_count": transition["pair_count"],
                    }
                )
    return rows


def _summary(
    manifest: Mapping[str, Any],
    quality: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    datasets = manifest["analysis"]["datasets"]
    summary_rows = []
    for dataset, dataset_spec in datasets.items():
        for detector in dataset_spec["scores"]:
            for pair_type in ("cross_policy", "same_policy_seed"):
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["detector"] == detector
                    and row["pair_type"] == pair_type
                ]
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "detector": detector,
                        "pair_type": pair_type,
                        "n_pairs": len(selected),
                        "median_abs_delta_auroc": float(
                            np.median([row["abs_delta_auroc"] for row in selected])
                        ),
                        "median_pair_order_churn": float(
                            np.median(
                                [row["pair_order_churn"] for row in selected]
                            )
                        ),
                        "median_hidden_churn_gap": float(
                            np.median([row["hidden_churn_gap"] for row in selected])
                        ),
                        "min_pair_order_churn": min(
                            row["pair_order_churn"] for row in selected
                        ),
                        "max_pair_order_churn": max(
                            row["pair_order_churn"] for row in selected
                        ),
                    }
                )

    role_churn_rows = []
    for role in ("c3", "c4"):
        for dataset, dataset_spec in datasets.items():
            detector_medians = {}
            for detector in dataset_spec["scores"]:
                cross = [
                    row["pair_order_churn"]
                    for row in rows
                    if row["pair_type"] == "cross_policy"
                    and row["group"] == role
                    and row["dataset"] == dataset
                    and row["detector"] == detector
                ]
                seed = [
                    row["pair_order_churn"]
                    for row in rows
                    if row["pair_type"] == "same_policy_seed"
                    and row["group"] in (f"adam_{role}", f"adamw_{role}")
                    and row["dataset"] == dataset
                    and row["detector"] == detector
                ]
                cross_median = float(np.median(cross))
                seed_median = float(np.median(seed))
                detector_medians[detector] = cross_median
                role_churn_rows.append(
                    {
                        "role": role,
                        "dataset": dataset,
                        "detector": detector,
                        "cross_policy_n": len(cross),
                        "same_policy_seed_n": len(seed),
                        "cross_policy_median_churn": cross_median,
                        "same_policy_seed_median_churn": seed_median,
                        "r_churn": cross_median / seed_median,
                    }
                )
            role_churn_rows.append(
                {
                    "role": role,
                    "dataset": dataset,
                    "detector": "rmd_vs_md_attenuation",
                    "cross_policy_rmd_over_md": (
                        detector_medians["raw_rmd"] / detector_medians["raw_md"]
                    ),
                    "cross_policy_churn_reduction_fraction": 1.0
                    - detector_medians["raw_rmd"] / detector_medians["raw_md"],
                }
            )

    return {
        "scope": {
            "checkpoint": "last.pt",
            "datasets": list(datasets),
            "detectors": list(next(iter(datasets.values()))["scores"]),
            "cross_policy_pairs": manifest["analysis"]["cross_policy_pairs"],
            "same_policy_seed_pairs": manifest["analysis"][
                "same_policy_seed_pairs"
            ],
            "evidence_grade": "historical_descriptive_discovery",
        },
        "quality": dict(quality),
        "summary_rows": summary_rows,
        "role_churn_rows": role_churn_rows,
        "max_abs_balance_residual": max(
            abs(row["balance_residual"]) for row in rows
        ),
    }


def _write_results(
    directory: Path,
    quality: Mapping[str, Any],
    rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    with (directory / "pair_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "data_quality.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, indent=2, sort_keys=True)
    with (directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _verify_expected_outputs(
    directory: Path,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    expected = manifest["expected_output_identity"]
    observed = {}
    for filename in RESULT_FILENAMES:
        observed[filename] = sha256(directory / filename)
        if observed[filename] != expected[filename]:
            raise D1ValidationError(
                f"reproduced output hash mismatch for {filename}: "
                f"expected {expected[filename]}, got {observed[filename]}"
            )
    return observed


def reproduce(
    *,
    retrieval_root: Path,
    input_manifest: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if os.path.lexists(output_directory):
        raise D1ValidationError(
            f"output directory already exists; refusing overwrite: {output_directory}"
        )
    if not retrieval_root.is_dir():
        raise D1ValidationError(f"retrieval root is not a directory: {retrieval_root}")
    manifest = _load_manifest(input_manifest)
    cache, quality = _validated_inputs(retrieval_root, manifest)
    rows = _pair_rows(manifest, cache)
    summary = _summary(manifest, quality, rows)

    output_parent = output_directory.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.tmp-", dir=output_parent)
    )
    try:
        _write_results(temporary, quality, rows, summary)
        output_hashes = _verify_expected_outputs(temporary, manifest)
        if os.path.lexists(output_directory):
            raise D1ValidationError(
                "output directory appeared during execution; refusing overwrite: "
                f"{output_directory}"
            )
        os.replace(temporary, output_directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "PASS",
        "analysis_role": manifest["analysis_role"],
        "claim_boundary": manifest["claim_boundary"],
        "score_file_uses_verified": quality["score_files_checked"],
        "output_directory": str(output_directory),
        "output_sha256": output_hashes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = reproduce(
            retrieval_root=arguments.retrieval_root,
            input_manifest=arguments.input_manifest,
            output_directory=arguments.output_directory,
        )
    except D1ValidationError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
