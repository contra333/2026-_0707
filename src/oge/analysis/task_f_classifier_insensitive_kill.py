"""Fast ID-only classifier-insensitive geometry kill test.

This module only consumes existing Task F ``id_validation`` feature, bridge,
and affine-alignment artifacts.  It deliberately contains no checkpoint,
training, detector-fit, or protected-data path.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from oge.analysis.task_f_fresh_id import (
    ID_EQUIVALENCE_MARGINS,
    _verify_alignment_directory,
)
from oge.evaluation.classification import (
    expected_calibration_error,
    negative_log_likelihood,
    softmax_probabilities,
    top1_accuracy,
)
from oge.evaluation.task_f_fresh import verify_bridge_artifact
from oge.feature_export import verify_task_f_artifact
from oge.studies.hashing import canonical_json_bytes, canonical_sha256


SCHEMA_VERSION = "task_f_classifier_insensitive_kill_v1"
PRIMARY_ANCHOR_CELL = "adam_lr1e-3_wd1e-4_anchor"
ALL_ID_GUARDRAIL_PASS_CELL = "adam_lr3e-4_wd1e-4"
HIGH_WD_CELLS = ("adam_lr1e-3_wd1e-3", "adam_lr3e-4_wd1e-3")
EXPECTED_SEEDS = {
    PRIMARY_ANCHOR_CELL: (0, 1, 2, 3, 4),
    "adam_lr1e-3_wd1e-3": (0, 1, 2),
    ALL_ID_GUARDRAIL_PASS_CELL: (0, 1, 2),
    "adam_lr3e-4_wd1e-3": (0, 1, 2),
}
EXPECTED_CONTEXTS = {
    (cell, seed) for cell, seeds in EXPECTED_SEEDS.items() for seed in seeds
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_canonical_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Atomically write canonical JSON and a SHA-256 sidecar."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(payload) + b"\n"
    digest = hashlib.sha256(content).hexdigest()
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, destination)
    sidecar = destination.with_name(f"{destination.name}.sha256")
    sidecar.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return digest


def load_canonical_json(path: str | Path, *, require_sidecar: bool = True) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected = canonical_json_bytes(payload) + b"\n"
    if source.read_bytes() != expected:
        raise ValueError(f"JSON is not canonical: {source}")
    if require_sidecar:
        fields = source.with_name(f"{source.name}.sha256").read_text(
            encoding="utf-8"
        ).strip().split()
        if len(fields) != 2 or fields[0] != _sha256_file(source) or fields[1] != source.name:
            raise ValueError(f"JSON checksum sidecar mismatch: {source}")
    return payload


def centered_classifier_basis(weight: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an orthonormal basis for the common-logit-centered rowspace."""

    value = np.asarray(weight, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("classifier weight must be a finite rank-2 array")
    centered = value - np.mean(value, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = sigma_max * max(centered.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    dimension = int(centered.shape[1])
    if not 1 <= rank < dimension:
        raise ValueError(
            f"centered classifier rowspace rank must satisfy 1 <= r < d; got {rank}/{dimension}"
        )
    basis = np.asarray(vh[:rank].T, dtype=np.float64)
    orthogonality_error = float(
        np.linalg.norm(basis.T @ basis - np.eye(rank), ord="fro")
    )
    return basis, {
        "dimension": dimension,
        "rank": rank,
        "svd_tolerance": tolerance,
        "orthogonality_error": orthogonality_error,
        "singular_values": [float(value) for value in singular_values],
    }


def _finalize_energies(
    *,
    total: float,
    parallel: float,
    perpendicular: float,
    target_energy: float,
    count: int,
    dimension: int,
    rank: int,
    machine_epsilon: float,
) -> dict[str, Any]:
    scale = max(abs(total), abs(parallel) + abs(perpendicular), 1.0)
    reconstruction = abs(total - parallel - perpendicular) / scale
    floor = 100.0 * machine_epsilon * max(target_energy, 1.0)
    if total <= floor:
        return {
            "status": "RESIDUAL_FLOOR",
            "rho": None,
            "log_rho": None,
            "e_parallel": None,
            "e_perpendicular": None,
            "total_energy": total,
            "parallel_energy": parallel,
            "perpendicular_energy": perpendicular,
            "residual_floor": floor,
            "reconstruction_relative_error": reconstruction,
        }
    e_parallel = parallel / (count * rank)
    e_perpendicular = perpendicular / (count * (dimension - rank))
    if e_parallel <= 0.0 or e_perpendicular < 0.0:
        raise ValueError("projection energy is non-positive")
    rho = e_perpendicular / e_parallel
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho is non-finite or non-positive")
    return {
        "status": "PASS" if reconstruction <= 1.0e-5 else "NUMERICAL_FAIL",
        "rho": float(rho),
        "log_rho": float(math.log(rho)),
        "e_parallel": float(e_parallel),
        "e_perpendicular": float(e_perpendicular),
        "total_energy": float(total),
        "parallel_energy": float(parallel),
        "perpendicular_energy": float(perpendicular),
        "residual_floor": float(floor),
        "reconstruction_relative_error": float(reconstruction),
    }


def decompose_residual_numpy(
    residual: Any,
    basis: Any,
    *,
    target_energy: float | None = None,
    dtype: Any = np.float64,
) -> dict[str, Any]:
    """Decompose a materialized residual; primarily used by focused tests."""

    value = np.asarray(residual, dtype=dtype)
    q = np.asarray(basis, dtype=dtype)
    if value.ndim != 2 or q.ndim != 2 or value.shape[1] != q.shape[0]:
        raise ValueError("residual and basis shapes differ")
    projection = value @ q
    reconstructed = projection @ q.T
    orthogonal = value - reconstructed
    total = float(np.sum(value * value, dtype=np.float64))
    parallel = float(np.sum(projection * projection, dtype=np.float64))
    perpendicular = float(np.sum(orthogonal * orthogonal, dtype=np.float64))
    return _finalize_energies(
        total=total,
        parallel=parallel,
        perpendicular=perpendicular,
        target_energy=float(total if target_energy is None else target_energy),
        count=len(value),
        dimension=value.shape[1],
        rank=q.shape[1],
        machine_epsilon=float(np.finfo(np.dtype(dtype)).eps),
    )


def _energy_numpy(
    *,
    coupled: np.ndarray,
    decoupled: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
    basis: np.ndarray,
    dtype: Any,
    chunk_size: int,
) -> dict[str, Any]:
    computation_dtype = np.dtype(dtype)
    transform = np.asarray(matrix, dtype=computation_dtype)
    offset = np.asarray(bias, dtype=computation_dtype)
    q = np.asarray(basis, dtype=computation_dtype)
    total = parallel = perpendicular = target_energy = 0.0
    for start in range(0, len(coupled), chunk_size):
        stop = min(start + chunk_size, len(coupled))
        target = np.asarray(coupled[start:stop], dtype=computation_dtype)
        source = np.asarray(decoupled[start:stop], dtype=computation_dtype)
        residual = source @ transform + offset - target
        projected = residual @ q
        orthogonal = residual - projected @ q.T
        total += float(np.sum(residual * residual, dtype=np.float64))
        parallel += float(np.sum(projected * projected, dtype=np.float64))
        perpendicular += float(np.sum(orthogonal * orthogonal, dtype=np.float64))
        target_energy += float(np.sum(target * target, dtype=np.float64))
    return _finalize_energies(
        total=total,
        parallel=parallel,
        perpendicular=perpendicular,
        target_energy=target_energy,
        count=len(coupled),
        dimension=coupled.shape[1],
        rank=basis.shape[1],
        machine_epsilon=float(np.finfo(computation_dtype).eps),
    )


def _energy_torch(
    *,
    coupled: np.ndarray,
    decoupled: np.ndarray,
    matrix: np.ndarray,
    bias: np.ndarray,
    basis: np.ndarray,
    device_index: int,
    chunk_size: int,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but torch.cuda is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    transform = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    offset = torch.as_tensor(bias, dtype=torch.float32, device=device)
    q = torch.as_tensor(basis, dtype=torch.float32, device=device)
    total = parallel = perpendicular = target_energy = 0.0
    for start in range(0, len(coupled), chunk_size):
        stop = min(start + chunk_size, len(coupled))
        target = torch.tensor(
            np.asarray(coupled[start:stop]), dtype=torch.float32, device=device
        )
        source = torch.tensor(
            np.asarray(decoupled[start:stop]), dtype=torch.float32, device=device
        )
        residual = source @ transform + offset - target
        projected = residual @ q
        orthogonal = residual - projected @ q.T
        total += float(torch.sum(residual * residual).item())
        parallel += float(torch.sum(projected * projected).item())
        perpendicular += float(torch.sum(orthogonal * orthogonal).item())
        target_energy += float(torch.sum(target * target).item())
    torch.cuda.synchronize(device)
    return _finalize_energies(
        total=total,
        parallel=parallel,
        perpendicular=perpendicular,
        target_energy=target_energy,
        count=len(coupled),
        dimension=coupled.shape[1],
        rank=basis.shape[1],
        machine_epsilon=float(np.finfo(np.float32).eps),
    )


def output_similarity(logits_c: Any, logits_d: Any, labels: Any) -> dict[str, Any]:
    coupled = np.asarray(logits_c, dtype=np.float64)
    decoupled = np.asarray(logits_d, dtype=np.float64)
    targets = np.asarray(labels)
    if coupled.shape != decoupled.shape or coupled.ndim != 2:
        raise ValueError("C/D logits must have the same rank-2 shape")
    if targets.shape != (len(coupled),) or not np.isfinite(coupled).all() or not np.isfinite(decoupled).all():
        raise ValueError("logits/labels are invalid")
    probability_c = softmax_probabilities(coupled)
    probability_d = softmax_probabilities(decoupled)
    tiny = np.finfo(np.float64).tiny
    probability_c = np.maximum(probability_c, tiny)
    probability_d = np.maximum(probability_d, tiny)
    midpoint = 0.5 * (probability_c + probability_d)
    js = 0.5 * np.sum(
        probability_c * (np.log(probability_c) - np.log(midpoint)), axis=1
    ) + 0.5 * np.sum(
        probability_d * (np.log(probability_d) - np.log(midpoint)), axis=1
    )
    top_c = np.sort(np.partition(coupled, -2, axis=1)[:, -2:], axis=1)
    top_d = np.sort(np.partition(decoupled, -2, axis=1)[:, -2:], axis=1)
    margin_c = top_c[:, 1] - top_c[:, 0]
    margin_d = top_d[:, 1] - top_d[:, 0]
    metrics: dict[str, dict[str, float]] = {}
    for name, values in (("coupled", coupled), ("decoupled", decoupled)):
        metrics[name] = {
            "accuracy": float(top1_accuracy(values, targets)["metric"]["value"]),
            "nll": float(negative_log_likelihood(values, targets)["metric"]["value"]),
            "ece": float(expected_calibration_error(values, targets)["metric"]["value"]),
        }
    differences = {
        name: abs(metrics["coupled"][name] - metrics["decoupled"][name])
        for name in ID_EQUIVALENCE_MARGINS
    }
    return {
        "top1_prediction_disagreement": float(
            np.mean(np.argmax(coupled, axis=1) != np.argmax(decoupled, axis=1))
        ),
        "predictive_js_divergence_mean": float(np.mean(js)),
        "logit_margin_absolute_difference_mean": float(np.mean(np.abs(margin_c - margin_d))),
        "id_metrics": metrics,
        "id_metric_absolute_differences": differences,
        "existing_guardrail": {
            "margins": dict(ID_EQUIVALENCE_MARGINS),
            "passes": {
                name: differences[name] <= margin
                for name, margin in ID_EQUIVALENCE_MARGINS.items()
            },
        },
    }


def _bridge_index(root: str | Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        identity = str(manifest.get("feature_output_identity_sha256", ""))
        if not identity:
            continue
        if identity in result:
            raise ValueError(f"duplicate bridge feature identity: {identity}")
        result[identity] = path.parent
    return result


def discover_contexts(
    *, alignment_root: str | Path, feature_root: str | Path, bridge_root: str | Path
) -> list[dict[str, str]]:
    """Discover this host's endpoint C-D contexts without opening large arrays."""

    bridges = _bridge_index(bridge_root)
    bindings: list[dict[str, str]] = []
    observed: set[tuple[str, int]] = set()
    for path in sorted(Path(alignment_root).glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        key = (str(manifest.get("cell_id")), int(manifest.get("training_seed", -1)))
        if (
            key not in EXPECTED_CONTEXTS
            or manifest.get("pair_direction") != "coupled_minus_decoupled"
            or manifest.get("checkpoint_role") != "last"
            or int(manifest.get("checkpoint_epoch", -1)) != 200
            or manifest.get("depth_tap") != "penultimate"
        ):
            continue
        if key in observed:
            raise ValueError(f"duplicate endpoint alignment context: {key}")
        left_identity = str(manifest["left_validation_feature_identity"])
        right_identity = str(manifest["right_validation_feature_identity"])
        left_feature = Path(feature_root) / left_identity
        right_feature = Path(feature_root) / right_identity
        if not left_feature.is_dir() or not right_feature.is_dir():
            raise FileNotFoundError(f"endpoint feature artifact is absent for {key}")
        if left_identity not in bridges or right_identity not in bridges:
            raise FileNotFoundError(f"endpoint bridge artifact is absent for {key}")
        bindings.append(
            {
                "alignment": str(path.parent),
                "coupled_feature": str(left_feature),
                "decoupled_feature": str(right_feature),
                "coupled_bridge": str(bridges[left_identity]),
                "decoupled_bridge": str(bridges[right_identity]),
            }
        )
        observed.add(key)
    return bindings


def _verify_binding(binding: Mapping[str, str]) -> dict[str, Any]:
    alignment = _verify_alignment_directory(binding["alignment"])["manifest"]
    coupled_feature = verify_task_f_artifact(binding["coupled_feature"])["manifest"]
    decoupled_feature = verify_task_f_artifact(binding["decoupled_feature"])["manifest"]
    coupled_bridge = verify_bridge_artifact(binding["coupled_bridge"])["manifest"]
    decoupled_bridge = verify_bridge_artifact(binding["decoupled_bridge"])["manifest"]
    checks = {
        "alignment_id_only": alignment["protected_data_access"] is False,
        "endpoint": alignment["checkpoint_role"] == "last"
        and int(alignment["checkpoint_epoch"]) == 200,
        "penultimate": alignment["depth_tap"] == "penultimate",
        "direction": alignment["pair_direction"] == "coupled_minus_decoupled",
        "coupled_feature_identity": coupled_feature["output_identity_sha256"]
        == alignment["left_validation_feature_identity"]
        == coupled_bridge["feature_output_identity_sha256"],
        "decoupled_feature_identity": decoupled_feature["output_identity_sha256"]
        == alignment["right_validation_feature_identity"]
        == decoupled_bridge["feature_output_identity_sha256"],
        "coupled_run": coupled_feature["run_id"]
        == coupled_bridge["run_id"]
        == alignment["left_run_id"],
        "decoupled_run": decoupled_feature["run_id"]
        == decoupled_bridge["run_id"]
        == alignment["right_run_id"],
        "same_sibling": coupled_bridge["sibling_group_id"]
        == decoupled_bridge["sibling_group_id"]
        == alignment["sibling_group_id"],
        "same_initialization": coupled_bridge["initialization_sha256"]
        == decoupled_bridge["initialization_sha256"]
        == alignment["initialization_sha256"],
        "same_data_stream": coupled_bridge["data_stream_sha256"]
        == decoupled_bridge["data_stream_sha256"]
        == alignment["data_stream_sha256"],
        "same_sample_order": coupled_feature["ordered_sample_id_sha256"]
        == decoupled_feature["ordered_sample_id_sha256"]
        == coupled_bridge["ordered_sample_id_sha256"]
        == decoupled_bridge["ordered_sample_id_sha256"],
        "validation_only": coupled_feature["dataset_split"]
        == decoupled_feature["dataset_split"]
        == coupled_bridge["dataset_split"]
        == decoupled_bridge["dataset_split"]
        == "id_validation",
        "classifier_ready": coupled_bridge["classifier"]["status"] == "READY"
        and decoupled_bridge["classifier"]["status"] == "READY",
        "raw_affine_pass": alignment["transforms"]["raw"]["affine"]["status"] == "PASS",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"kill-test identity checks failed: {failed}")
    return {
        "alignment": alignment,
        "coupled_feature": coupled_feature,
        "decoupled_feature": decoupled_feature,
        "coupled_bridge": coupled_bridge,
        "decoupled_bridge": decoupled_bridge,
        "checks": checks,
    }


def analyze_context(
    binding: Mapping[str, str], *, device_index: int | None, chunk_size: int
) -> dict[str, Any]:
    verified = _verify_binding(binding)
    alignment = verified["alignment"]
    with np.load(Path(binding["alignment"]) / "alignment_state.npz", allow_pickle=False) as state:
        matrix = np.asarray(state["raw__matrix"])
        bias = np.asarray(state["raw__bias"])
    coupled = np.load(Path(binding["coupled_feature"]) / "features.npy", mmap_mode="r")
    decoupled = np.load(Path(binding["decoupled_feature"]) / "features.npy", mmap_mode="r")
    coupled_weight = np.load(
        Path(binding["coupled_bridge"]) / "classifier_weight.npy", allow_pickle=False
    )
    coupled_logits = np.load(
        Path(binding["coupled_bridge"]) / "logits.npy", allow_pickle=False
    )
    decoupled_logits = np.load(
        Path(binding["decoupled_bridge"]) / "logits.npy", allow_pickle=False
    )
    coupled_labels = np.load(
        Path(binding["coupled_bridge"]) / "labels.npy", allow_pickle=False
    )
    decoupled_labels = np.load(
        Path(binding["decoupled_bridge"]) / "labels.npy", allow_pickle=False
    )
    if coupled.shape != decoupled.shape or not np.array_equal(coupled_labels, decoupled_labels):
        raise ValueError("paired validation feature/label arrays differ")
    basis, basis_record = centered_classifier_basis(coupled_weight)
    if device_index is None:
        energy = _energy_numpy(
            coupled=coupled,
            decoupled=decoupled,
            matrix=matrix,
            bias=bias,
            basis=basis,
            dtype=np.float32,
            chunk_size=chunk_size,
        )
        backend = "cpu_float32"
    else:
        energy = _energy_torch(
            coupled=coupled,
            decoupled=decoupled,
            matrix=matrix,
            bias=bias,
            basis=basis,
            device_index=device_index,
            chunk_size=chunk_size,
        )
        backend = f"cuda:{device_index}_float32_tf32_off"
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_id": alignment["cell_id"],
        "training_seed": int(alignment["training_seed"]),
        "sibling_group_id": alignment["sibling_group_id"],
        "backend": backend,
        "identity_status": "PASS",
        "identity_checks": verified["checks"],
        "classifier_basis": basis_record,
        "energy": energy,
        "output_similarity": output_similarity(
            coupled_logits, decoupled_logits, coupled_labels
        ),
        "source_artifacts": {
            "alignment_output_identity_sha256": alignment["output_identity_sha256"],
            "alignment_state_sha256": _sha256_file(
                Path(binding["alignment"]) / "alignment_state.npz"
            ),
            "coupled_feature_output_identity_sha256": verified["coupled_feature"][
                "output_identity_sha256"
            ],
            "decoupled_feature_output_identity_sha256": verified["decoupled_feature"][
                "output_identity_sha256"
            ],
            "coupled_bridge_identity_sha256": canonical_sha256(
                verified["coupled_bridge"]
            ),
            "decoupled_bridge_identity_sha256": canonical_sha256(
                verified["decoupled_bridge"]
            ),
        },
        "binding": dict(binding),
    }


def _confirm_cpu_float64(
    row: Mapping[str, Any], binding: Mapping[str, str], *, chunk_size: int
) -> dict[str, Any]:
    with np.load(Path(binding["alignment"]) / "alignment_state.npz", allow_pickle=False) as state:
        matrix = np.asarray(state["raw__matrix"])
        bias = np.asarray(state["raw__bias"])
    coupled = np.load(Path(binding["coupled_feature"]) / "features.npy", mmap_mode="r")
    decoupled = np.load(Path(binding["decoupled_feature"]) / "features.npy", mmap_mode="r")
    weight = np.load(
        Path(binding["coupled_bridge"]) / "classifier_weight.npy", allow_pickle=False
    )
    basis, _ = centered_classifier_basis(weight)
    reference = _energy_numpy(
        coupled=coupled,
        decoupled=decoupled,
        matrix=matrix,
        bias=bias,
        basis=basis,
        dtype=np.float64,
        chunk_size=chunk_size,
    )
    primary_rho = row["energy"]["rho"]
    reference_rho = reference["rho"]
    if primary_rho is None or reference_rho is None:
        return {
            "status": "FAIL",
            "reason_codes": ["residual_floor_in_confirmation"],
            "cpu_float64": reference,
            "rho_relative_difference": None,
            "side_preserved": False,
        }
    relative = abs(float(primary_rho) - float(reference_rho)) / abs(float(reference_rho))
    side_preserved = (float(primary_rho) > 1.0) == (float(reference_rho) > 1.0)
    return {
        "status": "PASS" if relative <= 1.0e-4 and side_preserved else "FAIL",
        "reason_codes": [],
        "cpu_float64": reference,
        "rho_relative_difference": float(relative),
        "side_preserved": side_preserved,
    }


def _run_queue(
    bindings: Sequence[Mapping[str, str]], device_index: int | None, chunk_size: int
) -> list[dict[str, Any]]:
    return [
        analyze_context(binding, device_index=device_index, chunk_size=chunk_size)
        for binding in bindings
    ]


def parse_idle_gpus(
    inventory_text: str, compute_text: str, *, min_free_mib: int = 2048
) -> list[int]:
    busy = {line.strip() for line in compute_text.splitlines() if line.strip()}
    eligible: list[int] = []
    for line in inventory_text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"invalid nvidia-smi inventory row: {line}")
        index, uuid, free_mib = int(fields[0]), fields[1], int(fields[2])
        if uuid not in busy and free_mib >= min_free_mib:
            eligible.append(index)
    return sorted(eligible)


def idle_gpu_indices(*, min_free_mib: int = 2048) -> list[int]:
    try:
        inventory = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        compute = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_idle_gpus(inventory, compute, min_free_mib=min_free_mib)


def run_host_analysis(
    *,
    host_id: str,
    alignment_root: str | Path,
    feature_root: str | Path,
    bridge_root: str | Path,
    analysis_git_sha: str,
    cpu_workers: int,
    chunk_size: int = 2048,
    min_free_mib: int = 2048,
    force_cpu: bool = False,
) -> dict[str, Any]:
    bindings = discover_contexts(
        alignment_root=alignment_root, feature_root=feature_root, bridge_root=bridge_root
    )
    if not bindings:
        raise ValueError("no endpoint kill-test contexts were discovered")
    devices = [] if force_cpu else idle_gpu_indices(min_free_mib=min_free_mib)
    workers: list[int | None] = devices or [None] * max(1, int(cpu_workers))
    workers = workers[: len(bindings)]
    queues: list[list[Mapping[str, str]]] = [[] for _ in workers]
    for index, binding in enumerate(bindings):
        queues[index % len(queues)].append(binding)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=len(workers), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_run_queue, queue, device, chunk_size)
            for queue, device in zip(queues, workers, strict=True)
            if queue
        ]
        for future in futures:
            rows.extend(future.result())
    by_key = {(row["cell_id"], int(row["training_seed"])): row for row in rows}
    binding_by_key: dict[tuple[str, int], Mapping[str, str]] = {}
    for binding in bindings:
        manifest = json.loads(
            (Path(binding["alignment"]) / "manifest.json").read_text(encoding="utf-8")
        )
        binding_by_key[(str(manifest["cell_id"]), int(manifest["training_seed"]))] = binding
    for key, row in by_key.items():
        reasons: list[str] = []
        if key == (PRIMARY_ANCHOR_CELL, 0):
            reasons.append("anchor_seed0_parity")
        log_rho = row["energy"]["log_rho"]
        if log_rho is not None and abs(float(log_rho)) < 0.05:
            reasons.append("boundary_context")
        if row["energy"]["reconstruction_relative_error"] > 1.0e-5:
            reasons.append("projection_reconstruction")
        if reasons:
            row["cpu_float64_confirmation"] = {
                **_confirm_cpu_float64(row, binding_by_key[key], chunk_size=chunk_size),
                "trigger_reason_codes": sorted(set(reasons)),
            }
        else:
            row["cpu_float64_confirmation"] = {
                "status": "NOT_REQUIRED",
                "reason_codes": [],
                "trigger_reason_codes": [],
            }
        confirmation = row["cpu_float64_confirmation"]
        row["numerical_status"] = (
            "PASS"
            if row["energy"]["status"] == "PASS"
            and confirmation["status"] in {"PASS", "NOT_REQUIRED"}
            else "FAIL"
        )
    ordered = sorted(rows, key=lambda row: (row["cell_id"], row["training_seed"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "host",
        "host_id": host_id,
        "analysis_git_sha": analysis_git_sha,
        "protected_data_access": False,
        "checkpoint_load": False,
        "detector_refit": False,
        "scheduler": {
            "backend": "gpu" if devices else "cpu_fallback",
            "eligible_idle_gpu_indices": devices,
            "worker_count": len(workers),
            "min_free_mib": min_free_mib,
        },
        "context_count": len(ordered),
        "contexts": ordered,
    }


def adjudicate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    duplicate: list[list[Any]] = []
    for row in rows:
        key = (str(row["cell_id"]), int(row["training_seed"]))
        if key in by_key:
            duplicate.append([key[0], key[1]])
        by_key[key] = row
    observed = set(by_key)
    missing = sorted(EXPECTED_CONTEXTS - observed)
    unexpected = sorted(observed - EXPECTED_CONTEXTS)
    invalid = sorted(
        key
        for key, row in by_key.items()
        if row.get("identity_status") != "PASS"
        or row.get("numerical_status") != "PASS"
        or row.get("energy", {}).get("rho") is None
    )
    if duplicate or missing or unexpected or invalid:
        return {
            "decision": "BLOCKED" if missing or invalid else "FAIL",
            "reason_codes": sorted(
                {
                    *(("duplicate_contexts",) if duplicate else ()),
                    *(("missing_contexts",) if missing else ()),
                    *(("unexpected_contexts",) if unexpected else ()),
                    *(("invalid_contexts",) if invalid else ()),
                }
            ),
            "coverage": {
                "expected": 14,
                "observed": len(observed),
                "missing": [[cell, seed] for cell, seed in missing],
                "unexpected": [[cell, seed] for cell, seed in unexpected],
                "duplicate": duplicate,
                "invalid": [[cell, seed] for cell, seed in invalid],
            },
        }
    counts = {
        cell: sum(
            float(by_key[(cell, seed)]["energy"]["rho"]) > 1.0 for seed in seeds
        )
        for cell, seeds in EXPECTED_SEEDS.items()
    }
    confirmations_preserve_side = all(
        row["cpu_float64_confirmation"]["status"] in {"PASS", "NOT_REQUIRED"}
        for row in rows
    )
    criteria = {
        "coverage_14_of_14": len(observed) == 14,
        "primary_anchor_at_least_4_of_5": counts[PRIMARY_ANCHOR_CELL] >= 4,
        "all_id_guardrail_pass_cell_at_least_2_of_3": counts[
            ALL_ID_GUARDRAIL_PASS_CELL
        ]
        >= 2,
        "at_least_one_high_wd_cell_at_least_2_of_3": any(
            counts[cell] >= 2 for cell in HIGH_WD_CELLS
        ),
        "required_confirmations_preserve_side": confirmations_preserve_side,
    }
    failed = [name for name, passed in criteria.items() if not passed]
    return {
        "decision": "GO" if not failed else "FAIL",
        "reason_codes": [] if not failed else [f"criterion_failed:{name}" for name in failed],
        "coverage": {
            "expected": 14,
            "observed": 14,
            "missing": [],
            "unexpected": [],
            "duplicate": [],
            "invalid": [],
        },
        "rho_above_one_counts": counts,
        "criteria": criteria,
    }


def merge_host_results(
    host_payloads: Sequence[Mapping[str, Any]], *, analysis_git_sha: str | None = None
) -> dict[str, Any]:
    if len(host_payloads) != 3:
        raise ValueError("merge requires exactly three host payloads")
    host_ids = [str(payload["host_id"]) for payload in host_payloads]
    if len(set(host_ids)) != 3:
        raise ValueError("host payload IDs must be unique")
    shas = {str(payload["analysis_git_sha"]) for payload in host_payloads}
    if len(shas) != 1 or (analysis_git_sha is not None and shas != {analysis_git_sha}):
        raise ValueError("host analysis Git SHAs differ")
    rows = [row for payload in host_payloads for row in payload["contexts"]]
    cells: dict[str, Any] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cell_id"])].append(row)
    for cell, values in sorted(grouped.items()):
        rho = [float(row["energy"]["rho"]) for row in values if row["energy"]["rho"] is not None]
        cells[cell] = {
            "seed_count": len(values),
            "rho_above_one_count": sum(value > 1.0 for value in rho),
            "rho_median": float(np.median(rho)) if rho else None,
            "seeds": sorted(int(row["training_seed"]) for row in values),
        }
    decision = adjudicate(rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "mode": "merge",
        "analysis_git_sha": next(iter(shas)),
        "protected_data_access": False,
        "host_ids": sorted(host_ids),
        "host_payload_sha256": {
            str(value["host_id"]): canonical_sha256(value) for value in host_payloads
        },
        "context_count": len(rows),
        "cell_summary": cells,
        "decision": decision,
        "contexts": sorted(rows, key=lambda row: (row["cell_id"], row["training_seed"])),
    }
    return {**payload, "payload_sha256": canonical_sha256(payload)}
