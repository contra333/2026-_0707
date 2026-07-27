"""Fail-closed server preflight, throughput, parity, and host-shard rules."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torchvision
import yaml

from .hashing import canonical_sha256
from .protocol import OPTIMIZER_ORDER, validate_grid_bundle

PREFLIGHT_SCHEMA_VERSION = "1.0"
PREFLIGHT_COMPARISON_SCHEMA_VERSION = "1.0"
HOST_MANIFEST_SCHEMA_VERSION = "1.0"
THROUGHPUT_DECISION_SCHEMA_VERSION = "1.0"
ASSIGNMENT_SEED = 2027
WORKER_CANDIDATES = [0, 4, 8]
MINIMUM_WORKER_IMPROVEMENT = 0.15
PARITY_ATOL = 1e-5
PARITY_RTOL = 1e-4


def load_infrastructure_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("infrastructure config must contain a mapping")
    validate_infrastructure_config(value)
    return value


def validate_infrastructure_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported infrastructure config schema_version")
    hosts = config.get("hosts")
    if not isinstance(hosts, list) or len(hosts) != 3:
        raise ValueError("exactly three approved hosts are required")
    host_ids = [host.get("host_id") for host in hosts if isinstance(host, Mapping)]
    if len(host_ids) != 3 or len(set(host_ids)) != 3:
        raise ValueError("approved host IDs must be three unique strings")
    for host in hosts:
        if not isinstance(host, Mapping):
            raise ValueError("host configuration must be a mapping")
        if not isinstance(host.get("expected_gpu_count"), int) or int(
            host["expected_gpu_count"]
        ) <= 0:
            raise ValueError("host expected_gpu_count must be positive")
        if not isinstance(host.get("expected_gpu_name"), str):
            raise ValueError("host expected_gpu_name is required")
        if not isinstance(host.get("expected_hostname"), str):
            raise ValueError("host expected_hostname is required")
    if config.get("forbidden_gpu_name_substrings") != [
        "RTX PRO 6000",
        "Blackwell",
    ]:
        raise ValueError("the RTX PRO 6000/Blackwell exclusion must be explicit")
    throughput = config.get("throughput")
    if throughput != {
        "worker_candidates": WORKER_CANDIDATES,
        "minimum_improvement": MINIMUM_WORKER_IMPROVEMENT,
        "require_full_gpu_concurrency": True,
    }:
        raise ValueError("throughput gate does not match the frozen policy")
    parity = config.get("parity")
    if parity != {
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "initialization_seed": ASSIGNMENT_SEED,
    }:
        raise ValueError("parity tolerance does not match the frozen policy")


def _safe_backend_value(path: str) -> dict[str, Any]:
    value: Any = torch
    try:
        for name in path.split("."):
            value = getattr(value, name)
        return {"available": True, "value": value}
    except (AttributeError, RuntimeError, TypeError) as exc:
        return {
            "available": False,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def capture_numerical_policy() -> dict[str, Any]:
    """Read both API families without mutating either one."""
    new_api = {
        path: _safe_backend_value(path)
        for path in (
            "backends.fp32_precision",
            "backends.cuda.matmul.fp32_precision",
            "backends.cudnn.fp32_precision",
            "backends.cudnn.conv.fp32_precision",
            "backends.cudnn.rnn.fp32_precision",
        )
    }
    new_available = all(item["available"] for item in new_api.values())
    legacy_api = {
        path: _safe_backend_value(path)
        for path in (
            "backends.cuda.matmul.allow_tf32",
            "backends.cudnn.allow_tf32",
        )
    }
    return {
        "active_api_family": (
            "fp32_precision" if new_available else "allow_tf32_legacy"
        ),
        "new_api": new_api,
        "legacy_api": legacy_api,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "precision_contract": {
            "parameter_activation_storage": "fp32",
            "amp": False,
            "bf16": False,
        },
        "environment_overrides": {
            key: os.environ.get(key)
            for key in (
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "NVIDIA_TF32_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }


def apply_numerical_policy(policy: Mapping[str, Any]) -> None:
    """Apply exactly one PyTorch TF32 API family, then verify the result."""
    if policy.get("precision_contract") != {
        "parameter_activation_storage": "fp32",
        "amp": False,
        "bf16": False,
    }:
        raise ValueError("cannot apply a non-FP32 production policy")
    torch.set_float32_matmul_precision(str(policy["float32_matmul_precision"]))
    family = policy.get("active_api_family")
    if family == "fp32_precision":
        for path, item in policy.get("new_api", {}).items():
            if not item.get("available"):
                raise ValueError("frozen fp32_precision policy contains unavailable API")
            target: Any = torch
            components = str(path).split(".")
            for component in components[:-1]:
                target = getattr(target, component)
            setattr(target, components[-1], item["value"])
    elif family == "allow_tf32_legacy":
        for path, item in policy.get("legacy_api", {}).items():
            if not item.get("available"):
                raise ValueError("frozen legacy TF32 policy contains unavailable API")
            target = torch
            components = str(path).split(".")
            for component in components[:-1]:
                target = getattr(target, component)
            setattr(target, components[-1], bool(item["value"]))
    else:
        raise ValueError("unknown frozen TF32 API family")
    torch.backends.cudnn.benchmark = bool(policy["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(policy["cudnn_deterministic"])
    torch.use_deterministic_algorithms(bool(policy["deterministic_algorithms"]))
    observed = capture_numerical_policy()
    if observed != policy:
        raise ValueError("applied numerical policy does not match the frozen values")


def _command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _gpu_inventory() -> list[dict[str, Any]]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,"
            "utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return []
    result = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=6)]
        if len(parts) != 7:
            raise ValueError("unexpected nvidia-smi inventory row")
        result.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": int(parts[4]),
                "memory_used_mib": int(parts[5]),
                "utilization_percent": int(parts[6]),
            }
        )
    return result


def _gpu_processes() -> list[dict[str, str]]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    result = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) == 4:
            result.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": parts[1],
                    "process_name": parts[2],
                    "used_gpu_memory_mib": parts[3],
                }
            )
    return result


def _filesystem_record(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    usage = shutil.disk_usage(target)
    stats = os.statvfs(target)
    return {
        "path": str(target),
        "bytes_total": usage.total,
        "bytes_free": usage.free,
        "inodes_total": stats.f_files,
        "inodes_free": stats.f_favail,
    }


def _package_versions() -> dict[str, str | None]:
    names = (
        "numpy",
        "pillow",
        "pyyaml",
        "scikit-learn",
        "torch",
        "torchvision",
        "oge",
    )
    values: dict[str, str | None] = {}
    for name in names:
        try:
            values[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            values[name] = None
    return values


def capture_server_preflight(
    *,
    host_id: str,
    repository_root: str | Path,
    dependency_lock: str | Path,
    artifact_root: str | Path,
    backup_root: str | Path,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    lock_path = Path(dependency_lock).resolve()
    lock_content = lock_path.read_bytes()
    lock_payload = yaml.safe_load(lock_content.decode("utf-8"))
    git_sha = _command_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
    git_status = _command_output(
        ["git", "-C", str(repository), "status", "--porcelain"]
    )
    inventory = _gpu_inventory()
    driver_versions = sorted({gpu["driver_version"] for gpu in inventory})
    record = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "host_id": host_id,
        "hostname": socket.gethostname(),
        "git": {
            "sha": git_sha,
            "dirty": bool(git_status),
        },
        "dependency_lock": {
            "path": str(lock_path),
            "sha256": hashlib.sha256(lock_content).hexdigest(),
            "status": (
                lock_payload.get("status")
                if isinstance(lock_payload, Mapping)
                else None
            ),
        },
        "software": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "driver_versions": driver_versions,
        },
        "numerical_policy": capture_numerical_policy(),
        "gpu_inventory": inventory,
        "gpu_processes": _gpu_processes(),
        "storage": {
            "artifact": _filesystem_record(artifact_root),
            "backup": _filesystem_record(backup_root),
        },
    }
    record["record_hash"] = canonical_sha256(record)
    return record


def _hash_without(payload: Mapping[str, Any], field: str) -> str:
    value = copy.deepcopy(dict(payload))
    value.pop(field, None)
    return canonical_sha256(value)


def validate_preflight_record(
    record: Mapping[str, Any],
    infrastructure: Mapping[str, Any],
) -> None:
    if record.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported preflight schema_version")
    if record.get("record_hash") != _hash_without(record, "record_hash"):
        raise ValueError("preflight record hash mismatch")
    host_specs = {
        host["host_id"]: host for host in infrastructure["hosts"]
    }
    host_id = record.get("host_id")
    if host_id not in host_specs:
        raise ValueError("preflight record host is not approved")
    spec = host_specs[host_id]
    if spec["expected_hostname"] == "UNSPECIFIED":
        raise ValueError("approved host expected hostname is still UNSPECIFIED")
    if record.get("hostname") != spec["expected_hostname"]:
        raise ValueError("preflight hostname differs from the approved host")
    inventory = record.get("gpu_inventory")
    if not isinstance(inventory, list) or len(inventory) != spec["expected_gpu_count"]:
        raise ValueError("approved host GPU count differs from expectation")
    forbidden = infrastructure["forbidden_gpu_name_substrings"]
    for gpu in inventory:
        name = str(gpu.get("name", ""))
        if name != spec["expected_gpu_name"]:
            raise ValueError("approved host GPU model differs from expectation")
        if any(marker.lower() in name.lower() for marker in forbidden):
            raise ValueError("RTX PRO 6000/Blackwell GPU is forbidden")
        if not str(gpu.get("uuid", "")).startswith("GPU-"):
            raise ValueError("preflight GPU UUID is missing or invalid")
    if record.get("git", {}).get("dirty") is not False:
        raise ValueError("server preflight requires a clean Git checkout")
    software = record.get("software")
    if not isinstance(software, Mapping) or not software.get("cuda_available"):
        raise ValueError("server preflight requires CUDA availability")
    if int(software.get("cuda_device_count", -1)) != len(inventory):
        raise ValueError("PyTorch and nvidia-smi GPU counts differ")
    if record.get("dependency_lock", {}).get("status") != "frozen":
        raise ValueError("dependency lock is not frozen")
    if record.get("gpu_processes"):
        raise ValueError("approved GPUs have active compute processes")
    numerical = record.get("numerical_policy")
    if not isinstance(numerical, Mapping):
        raise ValueError("preflight numerical policy is missing")
    if numerical.get("cudnn_benchmark") is not False:
        raise ValueError("cudnn.benchmark must be false")
    if numerical.get("deterministic_algorithms") is not False:
        raise ValueError("strict deterministic algorithms must be false")
    if numerical.get("precision_contract") != {
        "parameter_activation_storage": "fp32",
        "amp": False,
        "bf16": False,
    }:
        raise ValueError("FP32/AMP/BF16 contract mismatch")


def common_environment_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    software = record["software"]
    numerical = record["numerical_policy"]
    return {
        "git_sha": record["git"]["sha"],
        "dependency_lock_sha256": record["dependency_lock"]["sha256"],
        "python": software["python"],
        "packages": software["packages"],
        "torch": software["torch"],
        "torchvision": software["torchvision"],
        "cuda_runtime": software["cuda_runtime"],
        "cudnn": software["cudnn"],
        "numerical_policy": numerical,
    }


def current_common_environment_identity(
    *,
    repository_root: str | Path,
    dependency_lock: str | Path,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    lock_content = Path(dependency_lock).resolve().read_bytes()
    return {
        "git_sha": _command_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"]
        ),
        "dependency_lock_sha256": hashlib.sha256(lock_content).hexdigest(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numerical_policy": capture_numerical_policy(),
    }


def validate_runtime_environment(
    preflight_freeze: Mapping[str, Any],
    *,
    repository_root: str | Path,
    dependency_lock: str | Path,
) -> None:
    if preflight_freeze.get("status") != "ready":
        raise ValueError("runtime validation requires a ready preflight freeze")
    expected = preflight_freeze.get("common_environment_identity")
    actual = current_common_environment_identity(
        repository_root=repository_root,
        dependency_lock=dependency_lock,
    )
    if actual != expected:
        raise ValueError("current runtime differs from the frozen common environment")


def compare_server_preflights(
    records: Sequence[Mapping[str, Any]],
    infrastructure: Mapping[str, Any],
) -> dict[str, Any]:
    validate_infrastructure_config(infrastructure)
    expected_hosts = [host["host_id"] for host in infrastructure["hosts"]]
    by_host = {str(record.get("host_id")): record for record in records}
    if len(by_host) != len(records) or set(by_host) != set(expected_hosts):
        raise ValueError("preflight comparison requires exactly all approved hosts")
    for record in records:
        validate_preflight_record(record, infrastructure)
    identities = {
        host_id: common_environment_identity(by_host[host_id])
        for host_id in expected_hosts
    }
    reference = identities[expected_hosts[0]]
    mismatches = {
        host_id: identities[host_id]
        for host_id in expected_hosts[1:]
        if identities[host_id] != reference
    }
    if mismatches:
        raise ValueError(
            "common environment or numerical policy differs across approved hosts"
        )
    freeze = {
        "schema_version": PREFLIGHT_COMPARISON_SCHEMA_VERSION,
        "status": "ready",
        "host_order": expected_hosts,
        "preflight_record_hashes": {
            host_id: by_host[host_id]["record_hash"] for host_id in expected_hosts
        },
        "common_environment_identity": reference,
        "common_environment_hash": canonical_sha256(reference),
        "gpu_inventory": {
            host_id: copy.deepcopy(by_host[host_id]["gpu_inventory"])
            for host_id in expected_hosts
        },
    }
    freeze["freeze_hash"] = canonical_sha256(freeze)
    return freeze


def decide_num_workers(
    measurements: Sequence[Mapping[str, Any]],
    infrastructure: Mapping[str, Any],
) -> dict[str, Any]:
    validate_infrastructure_config(infrastructure)
    expected_hosts = [host["host_id"] for host in infrastructure["hosts"]]
    by_host = {str(item.get("host_id")): item for item in measurements}
    if len(by_host) != len(measurements) or set(by_host) != set(expected_hosts):
        raise ValueError("throughput decision requires all approved hosts")
    medians: dict[str, dict[int, float]] = {}
    host_specs = {
        host["host_id"]: host for host in infrastructure["hosts"]
    }
    for host_id in expected_hosts:
        item = by_host[host_id]
        if item.get("full_gpu_concurrency") is not True:
            raise ValueError("throughput measurement was not full-GPU concurrent")
        selectors = item.get("gpu_selectors")
        if (
            not isinstance(selectors, list)
            or len(selectors) != host_specs[host_id]["expected_gpu_count"]
            or len(set(selectors)) != len(selectors)
        ):
            raise ValueError("throughput measurement did not use every host GPU")
        candidates = item.get("candidates")
        if not isinstance(candidates, Mapping):
            raise ValueError("throughput candidate measurements are missing")
        host_values: dict[int, float] = {}
        for workers in WORKER_CANDIDATES:
            raw = candidates.get(str(workers), candidates.get(workers))
            if not isinstance(raw, list) or not raw:
                raise ValueError("every worker candidate needs epoch measurements")
            values = [float(value) for value in raw]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("epoch wall times must be finite and positive")
            host_values[workers] = statistics.median(values)
        medians[host_id] = host_values

    qualifying: list[tuple[float, float, int]] = []
    for workers in (4, 8):
        improvements = [
            1.0 - medians[host_id][workers] / medians[host_id][0]
            for host_id in expected_hosts
        ]
        if min(improvements) >= MINIMUM_WORKER_IMPROVEMENT:
            qualifying.append(
                (
                    min(improvements),
                    statistics.fmean(improvements),
                    workers,
                )
            )
    if qualifying:
        _, _, selected = max(
            qualifying,
            key=lambda item: (item[0], item[1], -item[2]),
        )
        stateless_required = True
    else:
        selected = 0
        stateless_required = False
    decision = {
        "schema_version": THROUGHPUT_DECISION_SCHEMA_VERSION,
        "minimum_improvement": MINIMUM_WORKER_IMPROVEMENT,
        "host_median_epoch_seconds": medians,
        "selected_num_workers": selected,
        "persistent_workers": selected > 0,
        "stateless_augmentation_required": stateless_required,
        "status": (
            "multiworker_pending_equivalence"
            if stateless_required
            else "workers_zero_frozen"
        ),
    }
    decision["decision_hash"] = canonical_sha256(decision)
    return decision


def _largest_remainder_quotas(
    capacities: Sequence[int],
    total: int,
) -> list[int]:
    capacity_sum = sum(capacities)
    exact = [total * capacity / capacity_sum for capacity in capacities]
    quotas = [math.floor(value) for value in exact]
    remaining = total - sum(quotas)
    order = sorted(
        range(len(capacities)),
        key=lambda index: (-(exact[index] - quotas[index]), index),
    )
    for index in order[:remaining]:
        quotas[index] += 1
    return quotas


def _partition_score(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    quotas: Sequence[int],
) -> int:
    score = 0
    for group, quota in zip(groups, quotas):
        for rank_key, rank_count in (("lr_rank", 3), ("weight_decay_rank", 4)):
            total_per_rank = 12 // rank_count
            for rank in range(rank_count):
                observed = sum(int(row[rank_key]) == rank for row in group)
                difference = observed * 12 - quota * total_per_rank
                score += difference * difference
    return score


def _balanced_partition(
    rows: Sequence[Mapping[str, Any]],
    quotas: Sequence[int],
    *,
    optimizer: str,
) -> list[list[dict[str, Any]]]:
    if len(rows) != 12 or sum(quotas) != 12 or len(quotas) != 3:
        raise ValueError("balanced partition requires 12 rows and three host quotas")
    indexed = list(range(12))
    best_key = None
    best_groups = None
    for first_indices in itertools.combinations(indexed, quotas[0]):
        remaining = [index for index in indexed if index not in first_indices]
        for second_indices in itertools.combinations(remaining, quotas[1]):
            third_indices = [
                index for index in remaining if index not in second_indices
            ]
            index_groups = [first_indices, second_indices, third_indices]
            groups = [
                [rows[index] for index in group]
                for group in index_groups
            ]
            score = _partition_score(groups, quotas)
            tie_hash = hashlib.sha256(
                (
                    f"{ASSIGNMENT_SEED}\0{optimizer}\0"
                    + "|".join(
                        ",".join(str(index) for index in group)
                        for group in index_groups
                    )
                ).encode("utf-8")
            ).hexdigest()
            key = (score, tie_hash)
            if best_key is None or key < best_key:
                best_key = key
                best_groups = [tuple(group) for group in index_groups]
    if best_groups is None:
        raise RuntimeError("failed to construct a balanced host partition")
    return [
        [copy.deepcopy(dict(rows[index])) for index in group]
        for group in best_groups
    ]


def generate_host_shards(
    grid_bundle: Mapping[str, Any],
    preflight_freeze: Mapping[str, Any],
    infrastructure: Mapping[str, Any],
    throughput_decision: Mapping[str, Any],
) -> dict[str, Any]:
    validate_infrastructure_config(infrastructure)
    validate_grid_bundle(grid_bundle)
    if preflight_freeze.get("schema_version") != PREFLIGHT_COMPARISON_SCHEMA_VERSION:
        raise ValueError("host shards require a valid preflight comparison")
    if preflight_freeze.get("status") != "ready":
        raise ValueError("host shards require a ready preflight comparison")
    if preflight_freeze.get("freeze_hash") != _hash_without(
        preflight_freeze, "freeze_hash"
    ):
        raise ValueError("preflight comparison freeze hash mismatch")
    if throughput_decision.get("decision_hash") != _hash_without(
        throughput_decision, "decision_hash"
    ):
        raise ValueError("throughput decision hash mismatch")
    if throughput_decision.get("status") not in {
        "workers_zero_frozen",
        "multiworker_equivalence_passed",
    }:
        raise ValueError("throughput decision is not production-ready")
    host_order = [host["host_id"] for host in infrastructure["hosts"]]
    if preflight_freeze.get("host_order") != host_order:
        raise ValueError("preflight host order differs from infrastructure config")
    capacities = [
        len(preflight_freeze["gpu_inventory"][host_id])
        for host_id in host_order
    ]
    quotas = _largest_remainder_quotas(capacities, 12)
    shards = {
        host_id: {
            "host_id": host_id,
            "approved_gpu_uuids": [
                gpu["uuid"]
                for gpu in preflight_freeze["gpu_inventory"][host_id]
            ],
            "optimizer_quota": quotas[index],
            "rows": [],
        }
        for index, host_id in enumerate(host_order)
    }
    for optimizer in OPTIMIZER_ORDER:
        rows = grid_bundle["tables"][optimizer]["rows"]
        groups = _balanced_partition(rows, quotas, optimizer=optimizer)
        for host_id, group in zip(host_order, groups):
            shards[host_id]["rows"].extend(
                {
                    "trial_id": row["trial_id"],
                    "optimizer_family": optimizer,
                    "config_hash": row["config_hash"],
                    "row_hash": row["row_hash"],
                    "lr_rank": row["lr_rank"],
                    "weight_decay_rank": row["weight_decay_rank"],
                }
                for row in group
            )
    for host_id in host_order:
        shards[host_id]["rank_histograms"] = {}
        for optimizer in OPTIMIZER_ORDER:
            rows = [
                row
                for row in shards[host_id]["rows"]
                if row["optimizer_family"] == optimizer
            ]
            shards[host_id]["rank_histograms"][optimizer] = {
                "lr_rank": {
                    str(rank): sum(row["lr_rank"] == rank for row in rows)
                    for rank in range(3)
                },
                "weight_decay_rank": {
                    str(rank): sum(
                        row["weight_decay_rank"] == rank for row in rows
                    )
                    for rank in range(4)
                },
            }
    manifest = {
        "schema_version": HOST_MANIFEST_SCHEMA_VERSION,
        "assignment_seed": ASSIGNMENT_SEED,
        "grid_manifest_hash": grid_bundle["manifest"]["manifest_hash"],
        "preflight_freeze_hash": preflight_freeze["freeze_hash"],
        "throughput_decision_hash": throughput_decision["decision_hash"],
        "host_order": host_order,
        "capacities": dict(zip(host_order, capacities)),
        "optimizer_quotas": dict(zip(host_order, quotas)),
        "shards": shards,
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    validate_host_shards(
        manifest,
        grid_bundle,
        preflight_freeze,
        throughput_decision,
    )
    return manifest


def validate_host_shards(
    manifest: Mapping[str, Any],
    grid_bundle: Mapping[str, Any],
    preflight_freeze: Mapping[str, Any],
    throughput_decision: Mapping[str, Any],
) -> None:
    if manifest.get("schema_version") != HOST_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported host-manifest schema_version")
    if manifest.get("manifest_hash") != _hash_without(manifest, "manifest_hash"):
        raise ValueError("host-manifest hash mismatch")
    if manifest.get("grid_manifest_hash") != grid_bundle["manifest"]["manifest_hash"]:
        raise ValueError("host manifest and grid identities differ")
    if manifest.get("preflight_freeze_hash") != preflight_freeze["freeze_hash"]:
        raise ValueError("host manifest and preflight identities differ")
    if (
        manifest.get("throughput_decision_hash")
        != throughput_decision["decision_hash"]
    ):
        raise ValueError("host manifest and throughput identities differ")
    expected = {
        row["trial_id"]: row
        for optimizer in OPTIMIZER_ORDER
        for row in grid_bundle["tables"][optimizer]["rows"]
    }
    actual: dict[str, Mapping[str, Any]] = {}
    shards = manifest.get("shards")
    if not isinstance(shards, Mapping):
        raise ValueError("host manifest is missing shards")
    quotas = manifest["optimizer_quotas"]
    for host_id in manifest["host_order"]:
        shard = shards[host_id]
        expected_uuids = [
            gpu["uuid"] for gpu in preflight_freeze["gpu_inventory"][host_id]
        ]
        if shard["approved_gpu_uuids"] != expected_uuids:
            raise ValueError("host shard GPU UUIDs differ from preflight")
        counts = {optimizer: 0 for optimizer in OPTIMIZER_ORDER}
        for row in shard["rows"]:
            trial_id = row["trial_id"]
            if trial_id in actual:
                raise ValueError("host shards contain a duplicate grid cell")
            actual[trial_id] = row
            counts[row["optimizer_family"]] += 1
        if any(count != quotas[host_id] for count in counts.values()):
            raise ValueError("optimizer host quotas are not equal")
        expected_histograms = {}
        for optimizer in OPTIMIZER_ORDER:
            rows = [
                row
                for row in shard["rows"]
                if row["optimizer_family"] == optimizer
            ]
            expected_histograms[optimizer] = {
                "lr_rank": {
                    str(rank): sum(row["lr_rank"] == rank for row in rows)
                    for rank in range(3)
                },
                "weight_decay_rank": {
                    str(rank): sum(
                        row["weight_decay_rank"] == rank for row in rows
                    )
                    for rank in range(4)
                },
            }
        if shard.get("rank_histograms") != expected_histograms:
            raise ValueError("host shard rank histograms are inconsistent")
    if set(actual) != set(expected):
        raise ValueError("host shards omit or add grid cells")
    for trial_id, row in actual.items():
        source = expected[trial_id]
        for key in (
            "optimizer_family",
            "config_hash",
            "row_hash",
            "lr_rank",
            "weight_decay_rank",
        ):
            if row[key] != source[key]:
                raise ValueError("host shard row identity differs from frozen grid")


def compare_parity_arrays(
    reports: Sequence[Mapping[str, Any]],
    *,
    atol: float = PARITY_ATOL,
    rtol: float = PARITY_RTOL,
) -> dict[str, Any]:
    if len(reports) != 3:
        raise ValueError("parity comparison requires all three hosts")
    host_ids = [str(report.get("host_id")) for report in reports]
    if len(set(host_ids)) != 3:
        raise ValueError("parity comparison requires three unique hosts")
    initial_hashes = {report["initial_state_hash"] for report in reports}
    if len(initial_hashes) != 1:
        raise ValueError("parity reports do not share the same initial state")
    reference = reports[0]
    comparisons = []
    for report in reports[1:]:
        fields = {}
        for key in (
            "loss",
            "gradient_norm",
            "logits_before",
            "logits_after",
            "features_after",
        ):
            left = np.asarray(reference[key], dtype=np.float64)
            right = np.asarray(report[key], dtype=np.float64)
            if left.shape != right.shape:
                raise ValueError("parity tensor shapes differ")
            max_absolute = float(np.max(np.abs(left - right)))
            denominator = np.maximum(np.abs(left), atol)
            max_relative = float(np.max(np.abs(left - right) / denominator))
            passed = bool(np.allclose(left, right, atol=atol, rtol=rtol))
            fields[key] = {
                "max_absolute_difference": max_absolute,
                "max_relative_difference": max_relative,
                "passed": passed,
            }
            if not passed:
                raise ValueError("A5000/A6000 parity is outside tolerance")
        comparisons.append(
            {
                "reference_host": reference["host_id"],
                "other_host": report["host_id"],
                "fields": fields,
            }
        )
    result = {
        "status": "passed",
        "atol": atol,
        "rtol": rtol,
        "initial_state_hash": next(iter(initial_hashes)),
        "comparisons": comparisons,
    }
    result["comparison_hash"] = canonical_sha256(result)
    return result
