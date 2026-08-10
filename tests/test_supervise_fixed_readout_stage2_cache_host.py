import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "supervise_fixed_readout_stage2_cache_host.py"
)
SPEC = importlib.util.spec_from_file_location("stage2_cache_supervisor", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SUPERVISOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUPERVISOR
SPEC.loader.exec_module(SUPERVISOR)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _payload(bundle_id, logical_path):
    return f"{bundle_id}:{logical_path}".encode("utf-8")


def _source_validation(manifest_relative):
    extractor_paths = (
        *SUPERVISOR.EXTRACTOR_CODE_SOURCE_PATHS,
        manifest_relative,
        SUPERVISOR.CANDIDATE_REGISTRY_RELATIVE_PATH,
        SUPERVISOR.GATE_POLICY_RELATIVE_PATH,
    )
    all_paths = (
        SUPERVISOR.SUPERVISOR_RELATIVE_PATH,
        SUPERVISOR.MATERIALIZER_RELATIVE_PATH,
        SUPERVISOR.REUSE_VALIDATOR_RELATIVE_PATH,
        *extractor_paths,
    )
    by_path = {
        path: {
            "path": path,
            "sha256": _digest(path.encode("utf-8")),
            "size": len(path),
        }
        for path in dict.fromkeys(all_paths)
    }
    return {
        "git_head": "1" * 40,
        "manifest_relative_path": manifest_relative,
        "sources": [by_path[path] for path in dict.fromkeys(all_paths)],
        "extractor_code_sources": [by_path[path] for path in extractor_paths],
        "status": "HEAD_BYTE_IDENTICAL",
    }


def _runtime(python_executable):
    return {
        "schema_version": SUPERVISOR.NUMERIC_RUNTIME_SCHEMA,
        "python": {
            "implementation": "CPython",
            "version": "3.test",
            "version_string": "synthetic",
            "executable": str(python_executable),
            "resolved_executable": str(Path(python_executable).resolve()),
            "prefix": "/synthetic/venv",
            "base_prefix": "/synthetic/base",
        },
        "packages": {
            "numpy": "synthetic",
            "scipy": "synthetic",
            "scikit_learn": "synthetic",
        },
        "blas": {"numpy_build_configuration": {}, "runtime_libraries": []},
        "thread_environment": {
            name: SUPERVISOR.FIXED_ENVIRONMENT[name]
            for name in SUPERVISOR.THREAD_ENVIRONMENT_NAMES
        },
        "python_import_environment": {
            "PYTHONPATH": SUPERVISOR.FIXED_ENVIRONMENT["PYTHONPATH"],
            "PYTHONNOUSERSITE": "1",
        },
        "oge_module_origins": {},
        "repository_runtime_locks": [],
        "repository_runtime_lock_sha256": "2" * 64,
        "canonical_lock_sha256": "3" * 64,
    }


def _synthetic_case(tmp_path, monkeypatch, *, concurrency=2):
    monkeypatch.setattr(SUPERVISOR, "EXPECTED_BUNDLE_COUNT", 3)
    monkeypatch.setattr(SUPERVISOR, "EXPECTED_FILE_COUNT", 2)
    monkeypatch.setattr(
        SUPERVISOR,
        "validate_reuse_manifest",
        lambda manifest, require_remote_catalog=True: None,
    )
    manifest_relative = "configs/evaluation/fixed_readout_stage2/reuse_manifest.json"
    source_validation = _source_validation(manifest_relative)
    monkeypatch.setattr(
        SUPERVISOR,
        "_validate_committed_sources",
        lambda path: source_validation,
    )
    monkeypatch.setattr(
        SUPERVISOR,
        "_probe_numeric_runtime",
        lambda python, environment: _runtime(python),
    )

    logical_paths = ["arrays/features.npy", "manifest.json"]
    config_hash = _digest(b"one-config")
    bundles = []
    for seed in range(3):
        bundle_id = _digest(f"bundle-{seed}".encode("utf-8"))
        files = [
            {
                "logical_path": logical_path,
                "sha256": _digest(_payload(bundle_id, logical_path)),
                "size": len(_payload(bundle_id, logical_path)),
            }
            for logical_path in logical_paths
        ]
        bundles.append(
            {
                "assigned_host": "synthetic",
                "bundle_id": bundle_id,
                "config_hash": config_hash,
                "identity": {
                    "checkpoint_sha256": bundle_id,
                    "training_seed": seed,
                },
                "remote_integrity": {
                    "allowed_payload_catalog_sha256": _digest(
                        f"catalog-{seed}".encode("utf-8")
                    ),
                    "files": files,
                    "selected_file_count": len(files),
                    "selected_total_size": sum(row["size"] for row in files),
                },
                "training_seed": seed,
            }
        )
    source_rows = {row["path"]: row for row in source_validation["sources"]}
    manifest = {
        "schema_version": SUPERVISOR.REUSE_SCHEMA,
        "retrieval_ready": True,
        "analysis_policy": {
            "candidate_registry": {
                "path": SUPERVISOR.CANDIDATE_REGISTRY_RELATIVE_PATH,
                "sha256": source_rows[
                    SUPERVISOR.CANDIDATE_REGISTRY_RELATIVE_PATH
                ]["sha256"],
            },
            "gate_policy": {
                "path": SUPERVISOR.GATE_POLICY_RELATIVE_PATH,
                "sha256": source_rows[SUPERVISOR.GATE_POLICY_RELATIVE_PATH][
                    "sha256"
                ],
            },
        },
        "path_contract": {
            "query_keys": ["id_test", "ood"],
            "required_logical_paths": logical_paths,
        },
        "bundles": bundles,
    }
    manifest_path = tmp_path / "reuse_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    root_names = ("source", "materialized", "receipts", "cache", "state")
    roots = {name: tmp_path / name for name in root_names}
    for root in roots.values():
        root.mkdir()
    for bundle in bundles:
        bundle_root = roots["source"] / bundle["bundle_id"]
        for row in bundle["remote_integrity"]["files"]:
            path = bundle_root.joinpath(*PurePosixPath(row["logical_path"]).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_payload(bundle["bundle_id"], row["logical_path"]))

    launcher = tmp_path / "venv-python"
    launcher.symlink_to(Path(sys.executable).resolve())
    plan_path = tmp_path / "plan.json"
    plan = SUPERVISOR.build_plan(
        reuse_manifest=manifest_path,
        source_root=roots["source"],
        materialized_roots_parent=roots["materialized"],
        receipts_parent=roots["receipts"],
        cache_root=roots["cache"],
        logs_state_root=roots["state"],
        python_executable=launcher,
        concurrency=concurrency,
        query_chunk_size=4,
        bank_chunk_size=64,
        output_plan=plan_path,
    )
    SUPERVISOR.write_plan(plan, plan_path)
    normalized = SUPERVISOR._normalize_manifest(manifest_path, source_validation)
    bundle_map = {row["bundle_id"]: row for row in normalized["bundles"]}
    return plan, plan_path, bundle_map, roots, launcher


def _materialize(plan, bundle_plan, bundle):
    root = Path(bundle_plan["materialized_root"])
    for row in bundle["files"]:
        path = root / bundle["bundle_id"] / PurePosixPath(row["logical_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_payload(bundle["bundle_id"], row["logical_path"]))
    expected_files = SUPERVISOR._expected_receipt_files(bundle)
    sources = {row["path"]: row for row in plan["source_validation"]["sources"]}
    receipt = {
        "schema_version": SUPERVISOR.RECEIPT_SCHEMA,
        "status": "CHECKSUM_VERIFIED_EXACT_ONE_BUNDLE_COPY",
        "bundle_id": bundle["bundle_id"],
        "bundle_catalog_sha256": bundle["source_allowlist_sha256"],
        "file_count": len(bundle["files"]),
        "total_size": sum(row["size"] for row in bundle["files"]),
        "files": expected_files,
        "copied_file_catalog_sha256": SUPERVISOR._sha256_bytes(
            SUPERVISOR._canonical_bytes(expected_files)
        ),
        "manifest": plan["reuse_manifest"],
        "destination": {
            "root": bundle_plan["materialized_root"],
            "bundle_directory": bundle_plan["materialized_bundle_directory"],
        },
        "source": {
            "artifact_identity": bundle["identity"],
            "bundle_directory": bundle_plan["source_bundle"],
        },
        "committed_source_validation": {
            "git_head": plan["git_head"],
            "sources": [
                sources[path]
                for path in (
                    SUPERVISOR.MATERIALIZER_RELATIVE_PATH,
                    SUPERVISOR.REUSE_VALIDATOR_RELATIVE_PATH,
                    plan["source_validation"]["manifest_relative_path"],
                )
            ],
            "status": "HEAD_MATCH",
        },
    }
    Path(bundle_plan["receipt"]).write_bytes(SUPERVISOR._canonical_bytes(receipt))


def _write_cache(plan, bundle):
    backend_sha = next(
        row["sha256"]
        for row in plan["source_validation"]["sources"]
        if row["path"] == "scripts/fixed_readout_stage2_raw_knn_backend.py"
    )
    for variant_id in SUPERVISOR.KNN_VARIANT_IDS:
        include = variant_id in SUPERVISOR.COMPONENT_VARIANTS
        root = (
            Path(plan["roots"]["cache_root"])
            / plan["analysis_id"]
            / bundle["bundle_id"]
            / variant_id
        )
        root.mkdir(parents=True)
        manifest = {
            "schema_version": SUPERVISOR.CACHE_SCHEMA,
            "analysis_id": plan["analysis_id"],
            "bundle_id": bundle["bundle_id"],
            "source_allowlist_sha256": bundle["source_allowlist_sha256"],
            "variant_id": variant_id,
            "include_top50_components": include,
            "backend": SUPERVISOR.RAW_KNN_BACKEND_ID,
            "k": SUPERVISOR.KNN_K,
            "query_keys": plan["reuse_query_keys"],
            "query_counts": {key: 2 for key in plan["reuse_query_keys"]},
            "train_sample_id_sha256": "4" * 64,
            "query_sample_id_sha256": {
                key: "5" * 64 for key in plan["reuse_query_keys"]
            },
            "backend_file_sha256": backend_sha,
            "requested_query_chunk_size": plan["query_chunk_size"],
            "requested_bank_chunk_size": plan["bank_chunk_size"],
            "requested_memory_budget_bytes": None,
            "numeric_runtime_fingerprint": plan["numeric_runtime_fingerprint"],
            "numeric_runtime_fingerprint_sha256": plan[
                "numeric_runtime_fingerprint_sha256"
            ],
            "neighbor_cache_encoding": (
                "int32_bank_indices_losslessly_mapped_through_bound_train_sample_ids"
            ),
            "backend_metadata": {
                "backend": SUPERVISOR.RAW_KNN_BACKEND_ID,
                "k": SUPERVISOR.KNN_K,
                "query_chunk_size": plan["query_chunk_size"],
                "bank_chunk_size": plan["bank_chunk_size"],
            },
        }
        (root / "manifest.json").write_bytes(SUPERVISOR._canonical_bytes(manifest))
        for query_key in plan["reuse_query_keys"]:
            split = root / "splits" / query_key
            split.mkdir(parents=True)
            (split / "score.npy").write_bytes(b"synthetic-score")
            if include:
                (split / "neighbor_squared_distances.npy").write_bytes(b"distance")
                (split / "neighbor_bank_indices.npy").write_bytes(b"indices")
        rows = [
            f"{SUPERVISOR._sha256_file(path)}  {path.relative_to(root).as_posix()}"
            for path in sorted(root.rglob("*"))
            if path.is_file()
        ]
        (root / "checksums.sha256").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )


def _report(plan, bundle_id):
    return {
        "schema_version": SUPERVISOR.PRECOMPUTE_SCHEMA,
        "analysis_id": plan["analysis_id"],
        "analysis_git_sha": plan["git_head"],
        "bundle_ids": [bundle_id],
        "bundle_count": 1,
        "variant_count_per_bundle": 7,
        "numeric_runtime_fingerprint": plan["numeric_runtime_fingerprint"],
        "bundles": [
            {
                "bundle_id": bundle_id,
                "variant_count": 7,
                "variants": [
                    {"variant_id": variant, "status": "PASS"}
                    for variant in SUPERVISOR.KNN_VARIANT_IDS
                ],
                "status": "PASS",
            }
        ],
        "status": "PASS",
        "scientific_gate": "NOT_RUN",
    }


def test_plan_is_deterministic_uses_lstat_only_and_preserves_launcher(
    tmp_path, monkeypatch
):
    plan, _, _, roots, launcher = _synthetic_case(tmp_path, monkeypatch)
    real_open = Path.open

    def reject_npy_open(path, *args, **kwargs):
        if path.suffix == ".npy":
            raise AssertionError("plan must not open an npy file")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_npy_open)
    repeated = SUPERVISOR.build_plan(
        reuse_manifest=Path(plan["reuse_manifest"]["path"]),
        source_root=roots["source"],
        materialized_roots_parent=roots["materialized"],
        receipts_parent=roots["receipts"],
        cache_root=roots["cache"],
        logs_state_root=roots["state"],
        python_executable=launcher,
        concurrency=2,
        query_chunk_size=4,
        bank_chunk_size=64,
        output_plan=tmp_path / "plan.json",
    )

    assert repeated == plan
    assert plan["roots"]["cache_staging_root"] == str(
        tmp_path / ".cache.stage2-knn-staging"
    )
    assert plan["plan_output_path"] == str(tmp_path / "plan.json")
    assert plan["python_executable"] == str(launcher)
    assert Path(plan["python_executable"]).is_symlink()
    assert plan["array_access_policy"] == "PLAN_USES_LSTAT_ONLY_AND_NEVER_LOADS_NPY"
    assert plan["fixed_environment"]["PYTHONNOUSERSITE"] == "1"
    assert plan["fixed_environment"]["PYTHONPATH"] == os.pathsep.join(
        (str(SUPERVISOR.REPOSITORY_ROOT), str(SUPERVISOR.REPOSITORY_ROOT / "src"))
    )
    assert "LD_LIBRARY_PATH" not in plan["fixed_environment"]
    assert all(
        command[0] == str(launcher)
        for row in plan["bundles"]
        for command in (row["materializer_command"], row["extractor_command"])
    )


def test_plan_rejects_derived_cache_staging_collisions(tmp_path, monkeypatch):
    plan, _, _, roots, launcher = _synthetic_case(tmp_path, monkeypatch)
    staging = Path(plan["roots"]["cache_staging_root"])
    staging.mkdir()
    common = {
        "reuse_manifest": Path(plan["reuse_manifest"]["path"]),
        "source_root": roots["source"],
        "materialized_roots_parent": roots["materialized"],
        "cache_root": roots["cache"],
        "logs_state_root": roots["state"],
        "python_executable": launcher,
        "concurrency": 2,
        "query_chunk_size": 4,
        "bank_chunk_size": 64,
    }

    with pytest.raises(
        ValueError, match="receipts_parent, cache_staging_root"
    ):
        SUPERVISOR.build_plan(
            **common,
            receipts_parent=staging,
            output_plan=roots["state"] / "collision-plan.json",
        )

    with pytest.raises(
        ValueError, match="cache_staging_root, plan_output_path"
    ):
        SUPERVISOR.build_plan(
            **common,
            receipts_parent=roots["receipts"],
            output_plan=staging / "collision-plan.json",
        )

    source_output = roots["source"] / "collision-plan.json"
    with pytest.raises(ValueError, match="source_root, plan_output_path"):
        SUPERVISOR.build_plan(
            **common,
            receipts_parent=roots["receipts"],
            output_plan=source_output,
        )
    assert not source_output.exists()


def test_head_byte_equality_rejects_worktree_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(SUPERVISOR, "REPOSITORY_ROOT", tmp_path)
    (tmp_path / "code.py").write_bytes(b"working")
    monkeypatch.setattr(SUPERVISOR, "_git_bytes", lambda relative: b"committed")

    with pytest.raises(ValueError, match="differs from HEAD"):
        SUPERVISOR._committed_source_row("code.py")


def test_bounded_run_resume_and_exact_array_free_status(tmp_path, monkeypatch):
    plan, plan_path, bundle_map, _, _ = _synthetic_case(
        tmp_path, monkeypatch, concurrency=2
    )
    lock = threading.Lock()
    active = 0
    maximum = 0

    def fake_child(command, *, log_handle, environment, phase):
        nonlocal active, maximum
        assert environment == plan["fixed_environment"]
        assert "LD_LIBRARY_PATH" not in environment
        assert isinstance(command, list)
        bundle_id = command[
            command.index("--bundle-id") + 1
            if phase == "materialize"
            else command.index("--precompute-bundle-id") + 1
        ]
        bundle_plan = next(row for row in plan["bundles"] if row["bundle_id"] == bundle_id)
        if phase == "materialize":
            _materialize(plan, bundle_plan, bundle_map[bundle_id])
            return 0, {"status": "PASS"}
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.04)
        _write_cache(plan, bundle_map[bundle_id])
        with lock:
            active -= 1
        return 0, _report(plan, bundle_id)

    monkeypatch.setattr(SUPERVISOR, "_execute_child", fake_child)
    result = SUPERVISOR.run_plan(plan_path)

    assert result["status"] == "PASS"
    assert len(result["completed_bundle_ids"]) == 3
    assert maximum == 2
    status = SUPERVISOR.status_plan(plan_path)
    assert status["status"] == "PASS"
    assert status["bundle_count"] == 3
    assert status["variant_count"] == 21

    monkeypatch.setattr(
        SUPERVISOR,
        "_execute_child",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume must not launch a child")
        ),
    )
    resumed = SUPERVISOR.run_plan(plan_path)
    assert resumed["completed_bundle_ids"] == []
    assert len(resumed["resumed_pass_bundle_ids"]) == 3

    extra = (
        Path(plan["roots"]["cache_root"])
        / plan["analysis_id"]
        / plan["bundles"][0]["bundle_id"]
        / "unexpected-variant"
    )
    extra.mkdir()
    with pytest.raises(ValueError, match="variant identity mismatch"):
        SUPERVISOR.status_plan(plan_path)


def test_run_rejects_unilateral_materialized_state_before_launch(tmp_path, monkeypatch):
    plan, plan_path, bundle_map, _, _ = _synthetic_case(tmp_path, monkeypatch)
    first = plan["bundles"][0]
    root = Path(first["materialized_root"])
    root.mkdir()
    monkeypatch.setattr(
        SUPERVISOR,
        "_execute_child",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unilateral state must fail before child launch")
        ),
    )

    with pytest.raises(ValueError, match="unilateral"):
        SUPERVISOR.run_plan(plan_path)
    assert not Path(first["completion"]).exists()
    assert bundle_map[first["bundle_id"]]["bundle_id"] == first["bundle_id"]


def test_run_rejects_mismatched_existing_receipt_before_launch(tmp_path, monkeypatch):
    plan, plan_path, bundle_map, _, _ = _synthetic_case(tmp_path, monkeypatch)
    first = plan["bundles"][0]
    _materialize(plan, first, bundle_map[first["bundle_id"]])
    receipt_path = Path(first["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["bundle_id"] = "f" * 64
    receipt_path.write_bytes(SUPERVISOR._canonical_bytes(receipt))
    monkeypatch.setattr(
        SUPERVISOR,
        "_execute_child",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("mismatched receipt must fail before child launch")
        ),
    )

    with pytest.raises(ValueError, match="receipt bundle identity mismatch"):
        SUPERVISOR.run_plan(plan_path)


def test_failed_extractor_preserves_attempt_log_without_completion(tmp_path, monkeypatch):
    plan, plan_path, bundle_map, _, _ = _synthetic_case(tmp_path, monkeypatch)

    def fake_child(command, *, log_handle, environment, phase):
        bundle_id = command[
            command.index("--bundle-id") + 1
            if phase == "materialize"
            else command.index("--precompute-bundle-id") + 1
        ]
        bundle_plan = next(row for row in plan["bundles"] if row["bundle_id"] == bundle_id)
        if phase == "materialize":
            _materialize(plan, bundle_plan, bundle_map[bundle_id])
            return 0, {"status": "PASS"}
        log_handle.write("synthetic extractor failure\n")
        return 17, None

    monkeypatch.setattr(SUPERVISOR, "_execute_child", fake_child)
    result = SUPERVISOR.run_plan(plan_path)

    assert result["status"] == "FAILED"
    assert len(result["failed_bundles"]) == 3
    for bundle in plan["bundles"]:
        logs = list(Path(bundle["log_directory"]).glob("attempt-*.log"))
        assert len(logs) == 1
        assert "synthetic extractor failure" in logs[0].read_text(encoding="utf-8")
        assert not Path(bundle["completion"]).exists()
