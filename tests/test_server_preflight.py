import copy
from pathlib import Path

import numpy as np
import pytest

from oge.studies.hashing import canonical_sha256
from oge.studies.preflight import (
    apply_numerical_policy,
    capture_numerical_policy,
    compare_parity_arrays,
    compare_server_preflights,
    current_common_environment_identity,
    decide_num_workers,
    generate_host_shards,
    load_infrastructure_config,
    validate_host_shards,
    validate_preflight_record,
    validate_runtime_environment,
)
from oge.studies.protocol import (
    generate_grid_bundle,
    load_study_config,
)
from oge.training import load_training_config, resolve_training_config

ROOT = Path(__file__).parents[1]
INFRASTRUCTURE_PATH = (
    ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/infrastructure.yaml"
)
STUDY_PATH = ROOT / "configs/studies/wrn28_10_optimizer_hpo_v1_2/study.yaml"


def _infrastructure():
    return load_infrastructure_config(INFRASTRUCTURE_PATH)


def _ready_infrastructure():
    value = copy.deepcopy(_infrastructure())
    value["hosts"][-1]["expected_hostname"] = "precision_medicine"
    return value


def _grid_bundle():
    study = load_study_config(STUDY_PATH)
    base = load_training_config(ROOT / study["base_training_config"])
    resolved = resolve_training_config(base, data_root=ROOT, device="cpu")
    return generate_grid_bundle(study, resolved)


def _numerical_policy(*, conv="tf32"):
    return {
        "active_api_family": "fp32_precision",
        "new_api": {
            "backends.fp32_precision": {"available": True, "value": "ieee"},
            "backends.cuda.matmul.fp32_precision": {
                "available": True,
                "value": "ieee",
            },
            "backends.cudnn.fp32_precision": {
                "available": True,
                "value": "ieee",
            },
            "backends.cudnn.conv.fp32_precision": {
                "available": True,
                "value": conv,
            },
            "backends.cudnn.rnn.fp32_precision": {
                "available": True,
                "value": "ieee",
            },
        },
        "legacy_api": {
            "backends.cuda.matmul.allow_tf32": {
                "available": True,
                "value": False,
            },
            "backends.cudnn.allow_tf32": {
                "available": True,
                "value": True,
            },
        },
        "float32_matmul_precision": "highest",
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "deterministic_algorithms": False,
        "precision_contract": {
            "parameter_activation_storage": "fp32",
            "amp": False,
            "bf16": False,
        },
        "environment_overrides": {
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": None,
            "NVIDIA_TF32_OVERRIDE": None,
            "CUBLAS_WORKSPACE_CONFIG": None,
        },
    }


def _report(host, *, lock_status="frozen", conv="tf32"):
    count = host["expected_gpu_count"]
    record = {
        "schema_version": "1.0",
        "host_id": host["host_id"],
        "hostname": host["host_id"],
        "git": {"sha": "a" * 40, "dirty": False},
        "dependency_lock": {
            "path": "/lock",
            "sha256": "b" * 64,
            "status": lock_status,
        },
        "software": {
            "python": "3.11.9",
            "python_executable": "/env/python",
            "platform": "linux",
            "packages": {
                "numpy": "2",
                "pillow": "11",
                "pyyaml": "6",
                "scikit-learn": "1.7",
                "torch": "2.5.1+cu121",
                "torchvision": "0.20.1+cu121",
                "oge": "0.1.0",
            },
            "torch": "2.5.1+cu121",
            "torchvision": "0.20.1+cu121",
            "cuda_runtime": "12.1",
            "cudnn": 90100,
            "cuda_available": True,
            "cuda_device_count": count,
            "driver_versions": ["535.0"],
        },
        "numerical_policy": _numerical_policy(conv=conv),
        "gpu_inventory": [
            {
                "index": index,
                "uuid": f"GPU-{host['host_id']}-{index}",
                "name": host["expected_gpu_name"],
                "driver_version": "535.0",
                "memory_total_mib": 24000,
                "memory_used_mib": 0,
                "utilization_percent": 0,
            }
            for index in range(count)
        ],
        "gpu_processes": [],
        "storage": {
            "artifact": {"bytes_free": 1, "inodes_free": 1},
            "backup": {"bytes_free": 1, "inodes_free": 1},
        },
    }
    record["record_hash"] = canonical_sha256(record)
    return record


def _reports():
    return [_report(host) for host in _ready_infrastructure()["hosts"]]


def test_capture_numerical_policy_records_new_and_legacy_apis_without_mutation():
    policy = capture_numerical_policy()
    assert policy["active_api_family"] in {"fp32_precision", "allow_tf32_legacy"}
    assert set(policy["legacy_api"]) == {
        "backends.cuda.matmul.allow_tf32",
        "backends.cudnn.allow_tf32",
    }
    assert policy["cudnn_benchmark"] is False
    assert policy["precision_contract"] == {
        "parameter_activation_storage": "fp32",
        "amp": False,
        "bf16": False,
    }
    apply_numerical_policy(policy)


def test_preflight_rejects_pending_lock_blackwell_and_wrong_gpu_count():
    infrastructure = _infrastructure()
    report = _report(infrastructure["hosts"][0], lock_status="pending")
    with pytest.raises(ValueError, match="lock"):
        validate_preflight_record(report, infrastructure)

    report = _report(infrastructure["hosts"][0])
    report["gpu_inventory"][0]["name"] = "NVIDIA RTX PRO 6000 Blackwell"
    report["record_hash"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "record_hash"}
    )
    with pytest.raises(ValueError, match="model|forbidden"):
        validate_preflight_record(report, infrastructure)

    report = _report(infrastructure["hosts"][0])
    report["gpu_inventory"].pop()
    report["software"]["cuda_device_count"] -= 1
    report["record_hash"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "record_hash"}
    )
    with pytest.raises(ValueError, match="count"):
        validate_preflight_record(report, infrastructure)


def test_common_preflight_freeze_requires_exact_tf32_and_environment_identity():
    infrastructure = _ready_infrastructure()
    freeze = compare_server_preflights(_reports(), infrastructure)
    assert freeze["status"] == "ready"
    assert set(freeze["gpu_inventory"]) == {
        "curie",
        "lise",
        "precision_medicine",
    }

    mismatched = _reports()
    mismatched[-1]["numerical_policy"] = _numerical_policy(conv="ieee")
    mismatched[-1]["record_hash"] = canonical_sha256(
        {
            key: value
            for key, value in mismatched[-1].items()
            if key != "record_hash"
        }
    )
    with pytest.raises(ValueError, match="differs"):
        compare_server_preflights(mismatched, infrastructure)


def test_runtime_must_still_match_the_frozen_environment_and_lock(tmp_path):
    lock = tmp_path / "lock.yaml"
    lock.write_text("status: frozen\n", encoding="utf-8")
    identity = current_common_environment_identity(
        repository_root=ROOT,
        dependency_lock=lock,
    )
    freeze = {
        "status": "ready",
        "common_environment_identity": identity,
    }
    validate_runtime_environment(
        freeze,
        repository_root=ROOT,
        dependency_lock=lock,
    )
    lock.write_text("status: frozen\nchanged: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        validate_runtime_environment(
            freeze,
            repository_root=ROOT,
            dependency_lock=lock,
        )


def _measurements(times):
    return [
        {
            "host_id": host["host_id"],
            "full_gpu_concurrency": True,
            "gpu_selectors": [
                str(gpu) for gpu in range(host["expected_gpu_count"])
            ],
            "candidates": {
                "0": [100.0, 102.0],
                "4": [times[index], times[index] + 1.0],
                "8": [95.0, 96.0],
            },
        }
        for index, host in enumerate(_infrastructure()["hosts"])
    ]


def _workers_zero_decision():
    return decide_num_workers(
        _measurements([90.0, 90.0, 90.0]),
        _ready_infrastructure(),
    )


def test_worker_gate_requires_common_15_percent_improvement_or_freezes_zero():
    infrastructure = _ready_infrastructure()
    adopted = decide_num_workers(_measurements([80.0, 81.0, 82.0]), infrastructure)
    assert adopted["selected_num_workers"] == 4
    assert adopted["status"] == "multiworker_pending_equivalence"

    fallback = decide_num_workers(_measurements([80.0, 90.0, 82.0]), infrastructure)
    assert fallback["selected_num_workers"] == 0
    assert fallback["persistent_workers"] is False
    assert fallback["status"] == "workers_zero_frozen"


def test_host_shards_are_deterministic_complete_and_use_5_2_5_per_optimizer():
    infrastructure = _ready_infrastructure()
    reports = _reports()
    freeze = compare_server_preflights(reports, infrastructure)
    grid = _grid_bundle()
    throughput = _workers_zero_decision()
    first = generate_host_shards(grid, freeze, infrastructure, throughput)
    second = generate_host_shards(grid, freeze, infrastructure, throughput)
    assert first == second
    assert first["optimizer_quotas"] == {
        "curie": 5,
        "lise": 2,
        "precision_medicine": 5,
    }
    assert sum(len(shard["rows"]) for shard in first["shards"].values()) == 36
    validate_host_shards(first, grid, freeze, throughput)
    for shard in first["shards"].values():
        counts = {
            optimizer: sum(
                row["optimizer_family"] == optimizer for row in shard["rows"]
            )
            for optimizer in ("sgd", "adam", "adamw")
        }
        assert len(set(counts.values())) == 1
        for optimizer in ("sgd", "adam", "adamw"):
            rows = [
                row
                for row in shard["rows"]
                if row["optimizer_family"] == optimizer
            ]
            assert len({row["lr_rank"] for row in rows}) >= min(2, len(rows))
            assert len({row["weight_decay_rank"] for row in rows}) >= min(
                2, len(rows)
            )


def test_parity_comparison_uses_initial_state_and_fails_outside_tolerance():
    base = {
        "host_id": "curie",
        "initial_state_hash": "a" * 64,
        "loss": 2.0,
        "gradient_norm": 3.0,
        "logits_before": [[1.0, 2.0]],
        "logits_after": [[1.5, 2.5]],
        "features_after": [[0.1, 0.2]],
    }
    reports = [
        {**copy.deepcopy(base), "host_id": host}
        for host in ("curie", "lise", "precision_medicine")
    ]
    reports[-1]["features_after"] = (
        np.asarray(reports[-1]["features_after"]) + 1e-7
    ).tolist()
    result = compare_parity_arrays(reports)
    assert result["status"] == "passed"

    reports[-1]["features_after"] = [[0.1, 0.3]]
    with pytest.raises(ValueError, match="outside tolerance"):
        compare_parity_arrays(reports)
