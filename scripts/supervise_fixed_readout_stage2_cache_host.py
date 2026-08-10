#!/usr/bin/env python3
"""Plan, run, and verify one host's complete Stage-2 kNN cache build.

The supervisor is deliberately host-local.  A plan freezes one committed
30-bundle reuse manifest, the exact numeric runtime, chunk sizes, commands,
and non-Git roots.  ``plan`` only inspects source artifacts with ``lstat``;
the committed materializer and extractor perform checksum and array-level
verification during ``run``.  ``status`` hashes cache files and validates
their manifests without loading any NumPy array.

Production entry points always require every scientific/code input to be
byte-identical to ``HEAD``.  There is no CLI bypass for this requirement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

try:
    from scripts.build_fixed_readout_stage2_reuse_manifest import (
        validate_reuse_manifest,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from build_fixed_readout_stage2_reuse_manifest import (  # type: ignore[no-redef]
        validate_reuse_manifest,
    )


REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
PLAN_SCHEMA = "fixed_readout_stage2_cache_host_plan_v1"
COMPLETION_SCHEMA = "fixed_readout_stage2_cache_host_completion_v1"
STATUS_SCHEMA = "fixed_readout_stage2_cache_host_status_v1"
REUSE_SCHEMA = "fixed_readout_stage2_reuse_manifest_v1"
RECEIPT_SCHEMA = "fixed_readout_stage2_one_bundle_copy_receipt_v1"
PRECOMPUTE_SCHEMA = "fixed_readout_stage2_knn_cache_precompute_v1"
CACHE_SCHEMA = "fixed_readout_stage2_knn_variant_cache_v1"
NUMERIC_RUNTIME_SCHEMA = "fixed_readout_stage2_numeric_runtime_v1"
ANALYSIS_CONTRACT_VERSION = "fixed_readout_training_rule_intervention_v2"
RAW_KNN_BACKEND_ID = "numpy_blas_exact_float64_lexicographic_v1"
EXPECTED_BUNDLE_COUNT = 30
EXPECTED_FILE_COUNT = 184
KNN_K = 50
ID_TRAIN_COUNT = 45_000
CACHE_STAGING_DIRECTORY_SUFFIX = ".stage2-knn-staging"

SUPERVISOR_RELATIVE_PATH = "scripts/supervise_fixed_readout_stage2_cache_host.py"
MATERIALIZER_RELATIVE_PATH = "scripts/materialize_fixed_readout_stage2_bundle.py"
REUSE_VALIDATOR_RELATIVE_PATH = "scripts/build_fixed_readout_stage2_reuse_manifest.py"
EXTRACTOR_RELATIVE_PATH = "scripts/extract_fixed_readout_stage2_model_metrics.py"
CANDIDATE_REGISTRY_RELATIVE_PATH = (
    "configs/evaluation/fixed_readout_stage2/candidate_registry.json"
)
GATE_POLICY_RELATIVE_PATH = "configs/evaluation/fixed_readout_stage2/gate_policy.json"
NUMERIC_RUNTIME_LOCK_PATHS = (
    "pyproject.toml",
    "configs/environments/wrn_v1_2_common.yaml",
)
EXTRACTOR_CODE_SOURCE_PATHS = (
    EXTRACTOR_RELATIVE_PATH,
    "scripts/analyze_fixed_readout_stage2_mechanism.py",
    "scripts/fixed_readout_stage2_raw_knn_backend.py",
    "src/oge/evaluation/classification.py",
    "src/oge/evaluation/common.py",
    "src/oge/evaluation/detectors.py",
    "src/oge/evaluation/extraction.py",
    "src/oge/evaluation/metrics.py",
    *NUMERIC_RUNTIME_LOCK_PATHS,
)
KNN_VARIANT_IDS = (
    "raw_identity",
    "l2_targeted",
    "signed_coordinate_permutation",
    "raw_common_positive_scale_2",
    "l2_common_positive_scale_2",
    "raw_nonuniform_samplewise_scale",
    "l2_after_nonuniform_samplewise_scale",
)
COMPONENT_VARIANTS = frozenset(("raw_identity", "l2_targeted"))
THREAD_COUNT_ENVIRONMENT_NAMES = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
THREAD_DYNAMIC_ENVIRONMENT = {
    "MKL_DYNAMIC": "FALSE",
    "OMP_DYNAMIC": "FALSE",
}
THREAD_ENVIRONMENT_NAMES = (
    *THREAD_COUNT_ENVIRONMENT_NAMES,
    *THREAD_DYNAMIC_ENVIRONMENT,
)
FIXED_ENVIRONMENT = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": os.pathsep.join(
        (str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src"))
    ),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    **THREAD_DYNAMIC_ENVIRONMENT,
    **{name: "1" for name in THREAD_COUNT_ENVIRONMENT_NAMES},
}
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_LOG = re.compile(r"attempt-[0-9]{4}\.log")


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _safe_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("logical path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"unsafe logical path: {value!r}")
    return value


def _assert_regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is absent: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    return path


def _assert_outside_git(path: Path, *, label: str) -> None:
    repository = REPOSITORY_ROOT.resolve()
    if path == repository or repository in path.parents:
        raise ValueError(f"{label} must be outside the repository worktree")
    for ancestor in (path, *path.parents):
        marker = ancestor / ".git"
        if marker.is_file() or (
            marker.is_dir() and (marker / "HEAD").is_file()
        ):
            raise ValueError(f"{label} must not be inside any Git worktree")


def _existing_non_git_directory(value: str | Path, *, label: str) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError(f"{label} must be absolute")
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is absent: {raw}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory")
    _assert_outside_git(resolved, label=label)
    return resolved


def _assert_disjoint_roots(roots: Mapping[str, Path]) -> None:
    rows = list(roots.items())
    for index, (left_name, left) in enumerate(rows):
        for right_name, right in rows[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    f"supervisor roots must be disjoint: {left_name}, {right_name}"
                )


def _cache_staging_root_path(cache_root: Path) -> Path:
    return cache_root.parent / f".{cache_root.name}{CACHE_STAGING_DIRECTORY_SUFFIX}"


def _planned_cache_staging_root(cache_root: Path) -> Path:
    """Validate the derived sibling staging namespace without creating it."""
    parent = cache_root.parent
    if parent.resolve(strict=True) != parent or parent.is_symlink():
        raise ValueError("cache staging parent chain must not contain symlinks")
    staging = _cache_staging_root_path(cache_root)
    if _lexists(staging):
        mode = staging.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError("cache staging root must be a real directory")
    _assert_outside_git(staging, label="cache staging root")
    return staging


def _planned_output_path(value: str | Path) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError("plan output path must be absolute")
    normalized = Path(os.path.abspath(os.fspath(raw)))
    try:
        parent = normalized.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"plan output parent is absent: {normalized.parent}"
        ) from exc
    if parent != normalized.parent or normalized.parent.is_symlink():
        raise ValueError("plan output parent chain must not contain symlinks")
    if not parent.is_dir():
        raise ValueError("plan output parent must be an existing real directory")
    output = parent / normalized.name
    if _lexists(output):
        mode = output.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError("existing plan output must be a real regular file")
    _assert_outside_git(output, label="plan output")
    return output


def _assert_path_disjoint(
    left: Path, right: Path, *, left_label: str, right_label: str
) -> None:
    if left == right or left in right.parents or right in left.parents:
        raise ValueError(
            f"supervisor paths must be disjoint: {left_label}, {right_label}"
        )


def _assert_output_disjoint_from_roots(
    roots: Mapping[str, Path], output: Path
) -> None:
    for name, root in roots.items():
        _assert_path_disjoint(
            root,
            output,
            left_label=name,
            right_label="plan_output_path",
        )


def _git_bytes(relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _git_head() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if _HEX40.fullmatch(head) is None:
        raise ValueError("Git HEAD is not a lowercase 40-character SHA-1")
    return head


def _committed_source_row(relative: str) -> dict[str, Any]:
    path = _assert_regular_file(
        REPOSITORY_ROOT / relative, label=f"required committed source {relative}"
    )
    working = path.read_bytes()
    try:
        committed = _git_bytes(relative)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"required source is not committed at HEAD: {relative}") from exc
    if working != committed:
        raise ValueError(f"required source differs from HEAD: {relative}")
    return {"path": relative, "sha256": _sha256_bytes(working), "size": len(working)}


def _validate_committed_sources(manifest_path: Path) -> dict[str, Any]:
    resolved_manifest = manifest_path.resolve()
    try:
        manifest_relative = resolved_manifest.relative_to(
            REPOSITORY_ROOT.resolve()
        ).as_posix()
    except ValueError as exc:
        raise ValueError("production reuse manifest must be inside the repository") from exc
    extractor_paths = (
        *EXTRACTOR_CODE_SOURCE_PATHS,
        manifest_relative,
        CANDIDATE_REGISTRY_RELATIVE_PATH,
        GATE_POLICY_RELATIVE_PATH,
    )
    all_paths = (
        SUPERVISOR_RELATIVE_PATH,
        MATERIALIZER_RELATIVE_PATH,
        REUSE_VALIDATOR_RELATIVE_PATH,
        *extractor_paths,
    )
    rows_by_path: dict[str, dict[str, Any]] = {}
    for relative in all_paths:
        if relative not in rows_by_path:
            rows_by_path[relative] = _committed_source_row(relative)
    return {
        "git_head": _git_head(),
        "manifest_relative_path": manifest_relative,
        "sources": [rows_by_path[path] for path in dict.fromkeys(all_paths)],
        "extractor_code_sources": [rows_by_path[path] for path in extractor_paths],
        "status": "HEAD_BYTE_IDENTICAL",
    }


def _normalize_manifest(
    manifest_path: Path, source_validation: Mapping[str, Any]
) -> dict[str, Any]:
    _assert_regular_file(manifest_path, label="reuse manifest")
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise ValueError("reuse manifest must contain a JSON mapping")
    validate_reuse_manifest(manifest, require_remote_catalog=True)
    if manifest.get("schema_version") != REUSE_SCHEMA:
        raise ValueError("reuse manifest schema mismatch")
    if manifest.get("retrieval_ready") is not True:
        raise ValueError("reuse manifest is not retrieval-ready")
    required_paths_raw = manifest.get("path_contract", {}).get(
        "required_logical_paths"
    )
    if not isinstance(required_paths_raw, list):
        raise ValueError("reuse manifest lacks the exact logical-path allowlist")
    required_paths = [_safe_logical_path(item) for item in required_paths_raw]
    if required_paths != sorted(set(required_paths)):
        raise ValueError("reuse logical-path allowlist must be sorted and unique")
    if len(required_paths) != EXPECTED_FILE_COUNT:
        raise ValueError(
            f"reuse manifest must select exactly {EXPECTED_FILE_COUNT} files per bundle"
        )
    query_keys = manifest.get("path_contract", {}).get("query_keys")
    if not isinstance(query_keys, list) or not query_keys or any(
        not isinstance(item, str) or not item for item in query_keys
    ):
        raise ValueError("reuse manifest query-key contract is invalid")
    bundles_raw = manifest.get("bundles")
    if not isinstance(bundles_raw, list) or len(bundles_raw) != EXPECTED_BUNDLE_COUNT:
        raise ValueError(
            f"reuse manifest must contain exactly {EXPECTED_BUNDLE_COUNT} bundles"
        )
    bundles: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    bundle_ids: set[str] = set()
    for raw in bundles_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("reuse bundle rows must be mappings")
        bundle_id = _require_hex64(raw.get("bundle_id"), "bundle_id")
        if bundle_id in bundle_ids:
            raise ValueError("reuse manifest repeats a bundle ID")
        bundle_ids.add(bundle_id)
        identity = raw.get("identity")
        if not isinstance(identity, Mapping) or identity.get(
            "checkpoint_sha256"
        ) != bundle_id:
            raise ValueError("reuse bundle/checkpoint identity mismatch")
        config_hash = _require_hex64(raw.get("config_hash"), "config_hash")
        seed = raw.get("training_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in (0, 1, 2):
            raise ValueError("reuse training seed must be one of 0, 1, 2")
        if identity.get("training_seed") != seed:
            raise ValueError("reuse bundle training-seed identity mismatch")
        if (config_hash, seed) in identities:
            raise ValueError("reuse manifest repeats a config/seed identity")
        identities.add((config_hash, seed))
        integrity = raw.get("remote_integrity")
        if not isinstance(integrity, Mapping):
            raise ValueError("reuse bundle lacks remote-integrity metadata")
        files_raw = integrity.get("files")
        if not isinstance(files_raw, list) or len(files_raw) != EXPECTED_FILE_COUNT:
            raise ValueError("reuse bundle selected-file count mismatch")
        files: list[dict[str, Any]] = []
        for file_row in files_raw:
            if not isinstance(file_row, Mapping):
                raise ValueError("reuse file catalog rows must be mappings")
            logical_path = _safe_logical_path(file_row.get("logical_path"))
            digest = _require_hex64(file_row.get("sha256"), "reuse file SHA-256")
            size = file_row.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("reuse file size must be a positive integer")
            files.append(
                {"logical_path": logical_path, "sha256": digest, "size": size}
            )
        if [row["logical_path"] for row in files] != required_paths:
            raise ValueError("reuse bundle file catalog differs from the allowlist")
        catalog_sha = _require_hex64(
            integrity.get("allowed_payload_catalog_sha256"),
            "source allowlist SHA-256",
        )
        if integrity.get("selected_file_count") != EXPECTED_FILE_COUNT:
            raise ValueError("reuse selected-file metadata mismatch")
        if integrity.get("selected_total_size") != sum(row["size"] for row in files):
            raise ValueError("reuse selected-byte metadata mismatch")
        bundles.append(
            {
                "assigned_host": raw.get("assigned_host"),
                "bundle_id": bundle_id,
                "config_hash": config_hash,
                "identity": dict(identity),
                "source_allowlist_sha256": catalog_sha,
                "training_seed": seed,
                "files": files,
            }
        )
    configs = {config_hash for config_hash, _ in identities}
    if len(configs) != EXPECTED_BUNDLE_COUNT // 3 or identities != {
        (config_hash, seed) for config_hash in configs for seed in (0, 1, 2)
    }:
        raise ValueError("reuse population must be an exact config x seed-{0,1,2} grid")
    source_rows = {
        row["path"]: row for row in source_validation["sources"]
    }
    policy = manifest.get("analysis_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("reuse manifest lacks analysis-policy bindings")
    for name, relative in (
        ("candidate_registry", CANDIDATE_REGISTRY_RELATIVE_PATH),
        ("gate_policy", GATE_POLICY_RELATIVE_PATH),
    ):
        binding = policy.get(name)
        if not isinstance(binding, Mapping) or binding.get("path") != relative:
            raise ValueError(f"reuse {name} path binding mismatch")
        if binding.get("sha256") != source_rows[relative]["sha256"]:
            raise ValueError(f"reuse {name} SHA-256 binding mismatch")
    bundles.sort(key=lambda row: row["bundle_id"])
    return {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256_bytes(payload),
        "query_keys": list(query_keys),
        "required_logical_paths": required_paths,
        "candidate_registry_sha256": policy["candidate_registry"]["sha256"],
        "gate_policy_sha256": policy["gate_policy"]["sha256"],
        "bundles": bundles,
    }


def _expected_directories(prefix: str, files: Sequence[Mapping[str, Any]]) -> set[str]:
    expected: set[str] = set()
    for row in files:
        path = PurePosixPath(prefix, str(row["logical_path"]))
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            expected.add(parent.as_posix())
    return expected


def _lstat_exact_tree(
    root: Path,
    *,
    prefix: str,
    files: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} must be a non-symlink directory")
    observed_files: dict[str, Path] = {}
    observed_directories: set[str] = set()
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        if not stat.S_ISDIR(current_path.lstat().st_mode):
            raise ValueError(f"{label} contains a non-directory traversal entry")
        for name in directories:
            candidate = current_path / name
            mode = candidate.lstat().st_mode
            relative = candidate.relative_to(root).as_posix()
            if stat.S_ISLNK(mode):
                raise ValueError(f"{label} contains a symlink: {relative}")
            if not stat.S_ISDIR(mode):
                raise ValueError(f"{label} contains a non-directory: {relative}")
            observed_directories.add(relative)
        for name in filenames:
            candidate = current_path / name
            mode = candidate.lstat().st_mode
            relative = candidate.relative_to(root).as_posix()
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError(f"{label} contains a non-regular file: {relative}")
            observed_files[relative] = candidate
    expected_files = {
        PurePosixPath(prefix, str(row["logical_path"])).as_posix(): row
        for row in files
    }
    if set(observed_files) != set(expected_files):
        unexpected = sorted(set(observed_files) - set(expected_files))
        missing = sorted(set(expected_files) - set(observed_files))
        detail = unexpected[0] if unexpected else missing[0]
        raise ValueError(f"{label} exact-file set mismatch: {detail}")
    expected_dirs = _expected_directories(prefix, files)
    if observed_directories != expected_dirs:
        unexpected = sorted(observed_directories - expected_dirs)
        missing = sorted(expected_dirs - observed_directories)
        detail = unexpected[0] if unexpected else missing[0]
        raise ValueError(f"{label} exact-directory set mismatch: {detail}")
    for relative, path in observed_files.items():
        expected_size = int(expected_files[relative]["size"])
        if path.lstat().st_size != expected_size:
            raise ValueError(f"{label} lstat-size mismatch: {relative}")
    return {
        "directory_count": len(observed_directories),
        "file_count": len(observed_files),
        "total_size": sum(path.lstat().st_size for path in observed_files.values()),
        "validation": "LSTAT_ONLY_NO_CONTENT_READ",
    }


def _validate_source_root(
    source_root: Path, bundles: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_ids = {str(bundle["bundle_id"]) for bundle in bundles}
    observed: set[str] = set()
    for entry in source_root.iterdir():
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError(f"source root contains a non-directory identity: {entry.name}")
        observed.add(entry.name)
    if observed != expected_ids:
        unexpected = sorted(observed - expected_ids)
        missing = sorted(expected_ids - observed)
        detail = unexpected[0] if unexpected else missing[0]
        raise ValueError(f"source root exact bundle-identity mismatch: {detail}")
    total_files = 0
    total_size = 0
    for bundle in bundles:
        bundle_id = str(bundle["bundle_id"])
        result = _lstat_exact_tree(
            source_root / bundle_id,
            prefix="",
            files=bundle["files"],
            label=f"source bundle {bundle_id}",
        )
        total_files += result["file_count"]
        total_size += result["total_size"]
    return {
        "bundle_count": len(bundles),
        "file_count": total_files,
        "total_size": total_size,
        "validation": "LSTAT_ONLY_NO_CONTENT_READ",
    }


def _validated_python_executable(value: str | Path) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError("Python executable must be absolute")
    launcher = Path(os.path.abspath(os.fspath(raw)))
    try:
        mode = launcher.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Python executable is absent: {launcher}") from exc
    if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
        raise ValueError("Python launcher must be a regular file or symlink")
    try:
        resolved = launcher.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Python launcher target is absent: {launcher}"
        ) from exc
    _assert_regular_file(resolved, label="resolved Python executable")
    if not os.access(launcher, os.X_OK):
        raise ValueError("Python launcher is not executable")
    # Keep the launcher path: resolving a venv's bin/python symlink before
    # invocation silently bypasses that venv's pyvenv.cfg and site-packages.
    return launcher


def _child_environment(overrides: Mapping[str, str]) -> dict[str, str]:
    # Deliberately do not inherit PYTHONPATH, LD_LIBRARY_PATH, thread settings,
    # or any other ambient runtime input.  Every child receives the exact
    # environment recorded in the plan.
    return {str(key): str(value) for key, value in overrides.items()}


def _probe_numeric_runtime(
    python_executable: Path, fixed_environment: Mapping[str, str]
) -> dict[str, Any]:
    probe = (
        "import importlib.util,json,sys;"
        "p=sys.argv[1];"
        "s=importlib.util.spec_from_file_location('stage2_runtime_probe',p);"
        "m=importlib.util.module_from_spec(s);"
        "sys.modules[s.name]=m;"
        "s.loader.exec_module(m);"
        "print(json.dumps(m._numeric_runtime_fingerprint(),sort_keys=True,allow_nan=False))"
    )
    result = subprocess.run(
        [str(python_executable), "-c", probe, str(REPOSITORY_ROOT / EXTRACTOR_RELATIVE_PATH)],
        cwd=REPOSITORY_ROOT,
        env=_child_environment(fixed_environment),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("numeric runtime probe did not emit exactly one JSON record")
    fingerprint = json.loads(lines[0])
    if not isinstance(fingerprint, dict):
        raise ValueError("numeric runtime fingerprint must be a mapping")
    return _validate_numeric_runtime_fingerprint(
        fingerprint, python_executable=python_executable, fixed_environment=fixed_environment
    )


def _validate_numeric_runtime_fingerprint(
    value: Mapping[str, Any],
    *,
    python_executable: Path,
    fixed_environment: Mapping[str, str],
) -> dict[str, Any]:
    payload = dict(value)
    supplied = payload.pop("canonical_lock_sha256", None)
    if payload.get("schema_version") != NUMERIC_RUNTIME_SCHEMA:
        raise ValueError("numeric runtime fingerprint schema mismatch")
    expected = _sha256_bytes(_canonical_bytes(payload))
    if supplied != expected:
        raise ValueError("numeric runtime fingerprint canonical lock mismatch")
    python = payload.get("python")
    if not isinstance(python, Mapping) or python.get("executable") != str(
        python_executable
    ):
        raise ValueError("numeric runtime Python executable mismatch")
    if python.get("resolved_executable") != str(python_executable.resolve()):
        raise ValueError("numeric runtime resolved Python executable mismatch")
    if not isinstance(python.get("prefix"), str) or not isinstance(
        python.get("base_prefix"), str
    ):
        raise ValueError("numeric runtime Python prefix identity is incomplete")
    observed_threads = payload.get("thread_environment")
    expected_threads = {
        name: fixed_environment[name] for name in THREAD_ENVIRONMENT_NAMES
    }
    if observed_threads != expected_threads:
        raise ValueError("numeric runtime thread environment is not the frozen one")
    if payload.get("python_import_environment") != {
        "PYTHONPATH": fixed_environment["PYTHONPATH"],
        "PYTHONNOUSERSITE": fixed_environment["PYTHONNOUSERSITE"],
    }:
        raise ValueError("numeric runtime Python import environment mismatch")
    module_origins = payload.get("oge_module_origins")
    expected_modules = {
        "oge": REPOSITORY_ROOT / "src/oge/__init__.py",
        "oge.evaluation.classification": REPOSITORY_ROOT
        / "src/oge/evaluation/classification.py",
        "oge.evaluation.common": REPOSITORY_ROOT / "src/oge/evaluation/common.py",
        "oge.evaluation.detectors": REPOSITORY_ROOT
        / "src/oge/evaluation/detectors.py",
        "oge.evaluation.extraction": REPOSITORY_ROOT
        / "src/oge/evaluation/extraction.py",
        "oge.evaluation.metrics": REPOSITORY_ROOT / "src/oge/evaluation/metrics.py",
    }
    if not isinstance(module_origins, Mapping) or set(module_origins) != set(
        expected_modules
    ):
        raise ValueError("numeric runtime OGE module-origin population mismatch")
    for module_name, expected_path in expected_modules.items():
        origin = module_origins[module_name]
        if not isinstance(origin, Mapping) or origin.get("resolved_path") != str(
            expected_path.resolve()
        ):
            raise ValueError(f"numeric runtime OGE module origin mismatch: {module_name}")
        if origin.get("sha256") != _sha256_file(expected_path):
            raise ValueError(f"numeric runtime OGE module SHA-256 mismatch: {module_name}")
    packages = payload.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != {
        "numpy",
        "scipy",
        "scikit_learn",
    }:
        raise ValueError("numeric runtime package fingerprint is incomplete")
    blas = payload.get("blas")
    runtime_libraries = blas.get("runtime_libraries") if isinstance(blas, Mapping) else None
    if not isinstance(runtime_libraries, list) or not runtime_libraries:
        raise ValueError("numeric runtime BLAS/threadpool fingerprint is empty")
    if any(
        not isinstance(row, Mapping) or row.get("num_threads") != 1
        for row in runtime_libraries
    ):
        raise ValueError("numeric runtime library thread count is not frozen to one")
    locks = payload.get("repository_runtime_locks")
    if not isinstance(locks, list) or [row.get("path") for row in locks] != list(
        NUMERIC_RUNTIME_LOCK_PATHS
    ):
        raise ValueError("numeric runtime repository-lock population mismatch")
    for row in locks:
        if row.get("sha256") != _sha256_file(REPOSITORY_ROOT / row["path"]):
            raise ValueError("numeric runtime repository-lock SHA-256 mismatch")
    lock_digest = _sha256_bytes(_canonical_bytes(locks))
    if payload.get("repository_runtime_lock_sha256") != lock_digest:
        raise ValueError("numeric runtime repository-lock digest mismatch")
    payload["canonical_lock_sha256"] = expected
    return payload


def _analysis_identity_payload(
    manifest: Mapping[str, Any],
    source_validation: Mapping[str, Any],
    numeric_runtime_fingerprint: Mapping[str, Any],
    *,
    query_chunk_size: int,
    bank_chunk_size: int,
) -> dict[str, Any]:
    return {
        "contract_version": ANALYSIS_CONTRACT_VERSION,
        "reuse_manifest_sha256": manifest["sha256"],
        "analysis_git_sha": source_validation["git_head"],
        "code_sources": list(source_validation["extractor_code_sources"]),
        "candidate_registry_sha256": manifest["candidate_registry_sha256"],
        "gate_policy_sha256": manifest["gate_policy_sha256"],
        "checkpoint_ids": sorted(
            str(row["bundle_id"]) for row in manifest["bundles"]
        ),
        "knn_backend": RAW_KNN_BACKEND_ID,
        "knn_k": KNN_K,
        "query_chunk_size": query_chunk_size,
        "bank_chunk_size": bank_chunk_size,
        "memory_budget_bytes": None,
        "numeric_runtime_fingerprint": dict(numeric_runtime_fingerprint),
    }


def _bundle_commands(
    *,
    python_executable: Path,
    manifest_path: Path,
    source_root: Path,
    materialized_parent: Path,
    receipts_parent: Path,
    cache_root: Path,
    bundle_id: str,
    query_chunk_size: int,
    bank_chunk_size: int,
) -> tuple[list[str], list[str]]:
    materialized_root = materialized_parent / bundle_id
    receipt = receipts_parent / f"{bundle_id}.json"
    materializer = [
        str(python_executable),
        str(REPOSITORY_ROOT / MATERIALIZER_RELATIVE_PATH),
        "--manifest",
        str(manifest_path),
        "--bundle-id",
        bundle_id,
        "--source-bundle",
        str(source_root / bundle_id),
        "--destination-root",
        str(materialized_root),
        "--receipt",
        str(receipt),
    ]
    extractor = [
        str(python_executable),
        str(REPOSITORY_ROOT / EXTRACTOR_RELATIVE_PATH),
        "--reuse-manifest",
        str(manifest_path),
        "--retrieval-root",
        str(materialized_root),
        "--cache-root",
        str(cache_root),
        "--precompute-bundle-id",
        bundle_id,
        "--query-chunk-size",
        str(query_chunk_size),
        "--bank-chunk-size",
        str(bank_chunk_size),
    ]
    return materializer, extractor


def build_plan(
    *,
    reuse_manifest: str | Path,
    source_root: str | Path,
    materialized_roots_parent: str | Path,
    receipts_parent: str | Path,
    cache_root: str | Path,
    logs_state_root: str | Path,
    python_executable: str | Path,
    concurrency: int,
    query_chunk_size: int,
    bank_chunk_size: int,
    output_plan: str | Path,
) -> dict[str, Any]:
    """Build a deterministic, array-free production plan."""
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    if concurrency > EXPECTED_BUNDLE_COUNT:
        raise ValueError("concurrency cannot exceed the bundle count")
    if (
        isinstance(query_chunk_size, bool)
        or not isinstance(query_chunk_size, int)
        or query_chunk_size <= 0
    ):
        raise ValueError("query chunk size must be a positive integer")
    if (
        isinstance(bank_chunk_size, bool)
        or not isinstance(bank_chunk_size, int)
        or bank_chunk_size < KNN_K
    ):
        raise ValueError(f"bank chunk size must be an integer >= {KNN_K}")
    manifest_path = Path(reuse_manifest).resolve()
    source_validation = _validate_committed_sources(manifest_path)
    manifest = _normalize_manifest(manifest_path, source_validation)
    roots = {
        "source_root": _existing_non_git_directory(source_root, label="source root"),
        "materialized_roots_parent": _existing_non_git_directory(
            materialized_roots_parent, label="materialized-roots parent"
        ),
        "receipts_parent": _existing_non_git_directory(
            receipts_parent, label="receipts parent"
        ),
        "cache_root": _existing_non_git_directory(cache_root, label="cache root"),
        "logs_state_root": _existing_non_git_directory(
            logs_state_root, label="logs/state root"
        ),
    }
    roots["cache_staging_root"] = _planned_cache_staging_root(
        roots["cache_root"]
    )
    _assert_disjoint_roots(roots)
    plan_output = _planned_output_path(output_plan)
    _assert_output_disjoint_from_roots(roots, plan_output)
    source_lstat = _validate_source_root(roots["source_root"], manifest["bundles"])
    python_path = _validated_python_executable(python_executable)
    numeric_runtime = _probe_numeric_runtime(python_path, FIXED_ENVIRONMENT)
    identity_payload = _analysis_identity_payload(
        manifest,
        source_validation,
        numeric_runtime,
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
    )
    analysis_id = _sha256_bytes(_canonical_bytes(identity_payload))
    bundles = []
    for bundle in manifest["bundles"]:
        bundle_id = str(bundle["bundle_id"])
        materializer, extractor = _bundle_commands(
            python_executable=python_path,
            manifest_path=manifest_path,
            source_root=roots["source_root"],
            materialized_parent=roots["materialized_roots_parent"],
            receipts_parent=roots["receipts_parent"],
            cache_root=roots["cache_root"],
            bundle_id=bundle_id,
            query_chunk_size=query_chunk_size,
            bank_chunk_size=bank_chunk_size,
        )
        bundles.append(
            {
                "bundle_id": bundle_id,
                "config_hash": bundle["config_hash"],
                "training_seed": bundle["training_seed"],
                "source_allowlist_sha256": bundle["source_allowlist_sha256"],
                "source_bundle": str(roots["source_root"] / bundle_id),
                "materialized_root": str(
                    roots["materialized_roots_parent"] / bundle_id
                ),
                "materialized_bundle_directory": str(
                    roots["materialized_roots_parent"] / bundle_id / bundle_id
                ),
                "receipt": str(roots["receipts_parent"] / f"{bundle_id}.json"),
                "completion": str(
                    roots["logs_state_root"]
                    / "completions"
                    / f"{bundle_id}.json"
                ),
                "log_directory": str(
                    roots["logs_state_root"] / "logs" / bundle_id
                ),
                "materializer_command": materializer,
                "extractor_command": extractor,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "purpose": "host_local_complete_30_bundle_stage2_knn_cache_production",
        "git_head": source_validation["git_head"],
        "source_validation": source_validation,
        "reuse_manifest": {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
        },
        "source_lstat_preflight": source_lstat,
        "roots": {name: str(path) for name, path in roots.items()},
        "plan_output_path": str(plan_output),
        "python_executable": str(python_path),
        "fixed_environment": dict(FIXED_ENVIRONMENT),
        "numeric_runtime_fingerprint": numeric_runtime,
        "numeric_runtime_fingerprint_sha256": numeric_runtime[
            "canonical_lock_sha256"
        ],
        "numeric_runtime_record_sha256": _sha256_bytes(
            _canonical_bytes(numeric_runtime)
        ),
        "analysis_identity_payload": identity_payload,
        "analysis_id": analysis_id,
        "reuse_query_keys": manifest["query_keys"],
        "concurrency": concurrency,
        "query_chunk_size": query_chunk_size,
        "bank_chunk_size": bank_chunk_size,
        "memory_budget_bytes": None,
        "expected_bundle_count": EXPECTED_BUNDLE_COUNT,
        "expected_selected_file_count_per_bundle": EXPECTED_FILE_COUNT,
        "expected_variant_ids": list(KNN_VARIANT_IDS),
        "bundles": bundles,
        "array_access_policy": "PLAN_USES_LSTAT_ONLY_AND_NEVER_LOADS_NPY",
        "scientific_gate": "NOT_RUN",
    }
    payload["plan_id"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _atomic_write_new(path: Path, payload: bytes) -> None:
    if _lexists(path):
        raise FileExistsError(f"no-overwrite output already exists: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError(f"output parent must be an existing non-symlink directory: {path.parent}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_plan(plan: Mapping[str, Any], output_path: str | Path) -> None:
    output = _planned_output_path(output_path)
    if plan.get("plan_output_path") != str(output):
        raise ValueError("plan output path differs from the frozen plan")
    roots = plan.get("roots")
    if not isinstance(roots, Mapping):
        raise ValueError("plan roots are invalid")
    root_names = (
        "source_root",
        "materialized_roots_parent",
        "receipts_parent",
        "cache_root",
        "logs_state_root",
        "cache_staging_root",
    )
    if set(roots) != set(root_names):
        raise ValueError("plan root identity set mismatch")
    validated_roots = {
        name: _existing_non_git_directory(roots[name], label=name)
        for name in root_names[:-1]
    }
    cache_root = validated_roots["cache_root"]
    expected_staging = _planned_cache_staging_root(cache_root)
    if roots.get("cache_staging_root") != str(expected_staging):
        raise ValueError("plan cache staging root is not the derived cache sibling")
    validated_roots["cache_staging_root"] = expected_staging
    _assert_disjoint_roots(validated_roots)
    _assert_output_disjoint_from_roots(validated_roots, output)
    _atomic_write_new(output, _canonical_bytes(plan))


def load_plan(plan_path: str | Path) -> dict[str, Any]:
    path = _assert_regular_file(Path(plan_path), label="supervisor plan")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("supervisor plan schema mismatch")
    supplied = plan.get("plan_id")
    payload = dict(plan)
    payload.pop("plan_id", None)
    if supplied != _sha256_bytes(_canonical_bytes(payload)):
        raise ValueError("supervisor plan canonical identity mismatch")
    return plan


def _manifest_bundle_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["bundle_id"]): dict(row) for row in manifest["bundles"]}


def _revalidate_plan(
    plan: Mapping[str, Any], *, plan_path: str | Path
) -> dict[str, Any]:
    if plan.get("expected_bundle_count") != EXPECTED_BUNDLE_COUNT:
        raise ValueError("plan bundle-count contract mismatch")
    if plan.get("expected_selected_file_count_per_bundle") != EXPECTED_FILE_COUNT:
        raise ValueError("plan selected-file-count contract mismatch")
    if plan.get("expected_variant_ids") != list(KNN_VARIANT_IDS):
        raise ValueError("plan variant identity contract mismatch")
    manifest_path = Path(plan["reuse_manifest"]["path"])
    source_validation = _validate_committed_sources(manifest_path)
    if source_validation != plan.get("source_validation"):
        raise ValueError("current committed source identity differs from the plan")
    if source_validation["git_head"] != plan.get("git_head"):
        raise ValueError("current Git HEAD differs from the plan")
    manifest = _normalize_manifest(manifest_path, source_validation)
    if {
        "path": manifest["path"],
        "sha256": manifest["sha256"],
    } != plan.get("reuse_manifest"):
        raise ValueError("current reuse manifest differs from the plan")
    root_payload = plan.get("roots")
    root_names = (
        "source_root",
        "materialized_roots_parent",
        "receipts_parent",
        "cache_root",
        "logs_state_root",
        "cache_staging_root",
    )
    if not isinstance(root_payload, Mapping) or set(root_payload) != set(root_names):
        raise ValueError("plan root identity set mismatch")
    roots = {
        name: _existing_non_git_directory(plan["roots"][name], label=name)
        for name in (
            "source_root",
            "materialized_roots_parent",
            "receipts_parent",
            "cache_root",
            "logs_state_root",
        )
    }
    expected_staging = _planned_cache_staging_root(roots["cache_root"])
    recorded_staging = Path(root_payload["cache_staging_root"])
    if recorded_staging != expected_staging:
        raise ValueError("plan cache staging root is not the derived cache sibling")
    roots["cache_staging_root"] = expected_staging
    _assert_disjoint_roots(roots)
    recorded_output = _planned_output_path(plan.get("plan_output_path"))
    current_output = _planned_output_path(plan_path)
    if recorded_output != current_output:
        raise ValueError("loaded plan path differs from the frozen plan output path")
    _assert_output_disjoint_from_roots(roots, recorded_output)
    source_lstat = _validate_source_root(roots["source_root"], manifest["bundles"])
    if source_lstat != plan.get("source_lstat_preflight"):
        raise ValueError("source-root lstat inventory differs from the plan")
    python_path = _validated_python_executable(plan["python_executable"])
    fixed_environment = plan.get("fixed_environment")
    if fixed_environment != FIXED_ENVIRONMENT:
        raise ValueError("plan fixed environment differs from the supervisor contract")
    runtime = _probe_numeric_runtime(python_path, fixed_environment)
    if runtime != plan.get("numeric_runtime_fingerprint"):
        raise ValueError("current numeric runtime differs from the plan")
    if runtime["canonical_lock_sha256"] != plan.get(
        "numeric_runtime_fingerprint_sha256"
    ):
        raise ValueError("plan numeric runtime SHA-256 mismatch")
    if _sha256_bytes(_canonical_bytes(runtime)) != plan.get(
        "numeric_runtime_record_sha256"
    ):
        raise ValueError("plan numeric runtime record SHA-256 mismatch")
    if plan.get("reuse_query_keys") != manifest["query_keys"]:
        raise ValueError("plan reuse query-key contract mismatch")
    query_chunk = plan.get("query_chunk_size")
    bank_chunk = plan.get("bank_chunk_size")
    identity = _analysis_identity_payload(
        manifest,
        source_validation,
        runtime,
        query_chunk_size=query_chunk,
        bank_chunk_size=bank_chunk,
    )
    if identity != plan.get("analysis_identity_payload"):
        raise ValueError("plan analysis identity payload mismatch")
    analysis_id = _sha256_bytes(_canonical_bytes(identity))
    if analysis_id != plan.get("analysis_id"):
        raise ValueError("plan analysis ID mismatch")
    expected_plan_rows = []
    for bundle in manifest["bundles"]:
        bundle_id = str(bundle["bundle_id"])
        materializer, extractor = _bundle_commands(
            python_executable=python_path,
            manifest_path=manifest_path,
            source_root=roots["source_root"],
            materialized_parent=roots["materialized_roots_parent"],
            receipts_parent=roots["receipts_parent"],
            cache_root=roots["cache_root"],
            bundle_id=bundle_id,
            query_chunk_size=query_chunk,
            bank_chunk_size=bank_chunk,
        )
        expected_plan_rows.append(
            {
                "bundle_id": bundle_id,
                "config_hash": bundle["config_hash"],
                "training_seed": bundle["training_seed"],
                "source_allowlist_sha256": bundle["source_allowlist_sha256"],
                "source_bundle": str(roots["source_root"] / bundle_id),
                "materialized_root": str(roots["materialized_roots_parent"] / bundle_id),
                "materialized_bundle_directory": str(
                    roots["materialized_roots_parent"] / bundle_id / bundle_id
                ),
                "receipt": str(roots["receipts_parent"] / f"{bundle_id}.json"),
                "completion": str(
                    roots["logs_state_root"] / "completions" / f"{bundle_id}.json"
                ),
                "log_directory": str(roots["logs_state_root"] / "logs" / bundle_id),
                "materializer_command": materializer,
                "extractor_command": extractor,
            }
        )
    if plan.get("bundles") != expected_plan_rows:
        raise ValueError("plan bundle paths or commands differ from reconstructed commands")
    concurrency = plan.get("concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or not (
        1 <= concurrency <= EXPECTED_BUNDLE_COUNT
    ):
        raise ValueError("plan concurrency is invalid")
    return {
        "manifest": manifest,
        "bundle_map": _manifest_bundle_map(manifest),
        "roots": roots,
        "python_executable": python_path,
        "numeric_runtime_fingerprint": runtime,
        "analysis_id": analysis_id,
    }


def _expected_receipt_files(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "copy_method": "shutil.copyfile(follow_symlinks=False)",
            "logical_path": row["logical_path"],
            "sha256": row["sha256"],
            "size": row["size"],
        }
        for row in bundle["files"]
    ]


def _preflight_materialized_pair(
    plan: Mapping[str, Any], bundle_plan: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(bundle_plan["materialized_root"])
    receipt_path = Path(bundle_plan["receipt"])
    root_exists = _lexists(root)
    receipt_exists = _lexists(receipt_path)
    if root_exists != receipt_exists:
        raise ValueError(
            f"unilateral materialized-root/receipt state: {bundle_plan['bundle_id']}"
        )
    if not root_exists:
        return {"status": "ABSENT"}
    tree = _lstat_exact_tree(
        root,
        prefix=str(bundle_plan["bundle_id"]),
        files=bundle["files"],
        label=f"materialized bundle {bundle_plan['bundle_id']}",
    )
    receipt_file = _assert_regular_file(receipt_path, label="materialization receipt")
    receipt_bytes = receipt_file.read_bytes()
    receipt = json.loads(receipt_bytes)
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("materialization receipt schema mismatch")
    if receipt.get("status") != "CHECKSUM_VERIFIED_EXACT_ONE_BUNDLE_COPY":
        raise ValueError("materialization receipt is not checksum-verified PASS")
    bundle_id = str(bundle_plan["bundle_id"])
    if receipt.get("bundle_id") != bundle_id:
        raise ValueError("materialization receipt bundle identity mismatch")
    if receipt.get("bundle_catalog_sha256") != bundle["source_allowlist_sha256"]:
        raise ValueError("materialization receipt allowlist identity mismatch")
    if receipt.get("file_count") != EXPECTED_FILE_COUNT or receipt.get(
        "total_size"
    ) != sum(row["size"] for row in bundle["files"]):
        raise ValueError("materialization receipt count/size mismatch")
    expected_files = _expected_receipt_files(bundle)
    if receipt.get("files") != expected_files:
        raise ValueError("materialization receipt exact file catalog mismatch")
    if receipt.get("copied_file_catalog_sha256") != _sha256_bytes(
        _canonical_bytes(expected_files)
    ):
        raise ValueError("materialization receipt copied-catalog digest mismatch")
    if receipt.get("manifest") != plan.get("reuse_manifest"):
        raise ValueError("materialization receipt reuse-manifest binding mismatch")
    if receipt.get("destination") != {
        "root": bundle_plan["materialized_root"],
        "bundle_directory": bundle_plan["materialized_bundle_directory"],
    }:
        raise ValueError("materialization receipt destination binding mismatch")
    source = receipt.get("source")
    if not isinstance(source, Mapping) or source.get("artifact_identity") != bundle[
        "identity"
    ]:
        raise ValueError("materialization receipt source identity mismatch")
    if source.get("bundle_directory") != bundle_plan["source_bundle"]:
        raise ValueError("materialization receipt source path mismatch")
    committed = receipt.get("committed_source_validation")
    if not isinstance(committed, Mapping) or committed.get("status") != "HEAD_MATCH":
        raise ValueError("materialization receipt lacks committed HEAD validation")
    if committed.get("git_head") != plan["git_head"]:
        raise ValueError("materialization receipt Git HEAD mismatch")
    expected_source_paths = (
        MATERIALIZER_RELATIVE_PATH,
        REUSE_VALIDATOR_RELATIVE_PATH,
        plan["source_validation"]["manifest_relative_path"],
    )
    source_by_path = {
        row["path"]: row for row in plan["source_validation"]["sources"]
    }
    if committed.get("sources") != [source_by_path[path] for path in expected_source_paths]:
        raise ValueError("materialization receipt committed-source set mismatch")
    return {
        "status": "VERIFIED_EXISTING",
        "receipt_sha256": _sha256_bytes(receipt_bytes),
        "tree": tree,
    }


def _cache_checksum_rows(root: Path) -> dict[str, str]:
    checksum_path = _assert_regular_file(
        root / "checksums.sha256", label="cache checksum inventory"
    )
    rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError("malformed cache checksum row") from exc
        safe = _safe_logical_path(relative)
        if _HEX64.fullmatch(digest) is None or safe in rows or safe == "checksums.sha256":
            raise ValueError("invalid cache checksum identity")
        path = _assert_regular_file(root / PurePosixPath(safe), label="cache payload")
        if _sha256_file(path) != digest:
            raise ValueError(f"cache checksum mismatch: {safe}")
        rows[safe] = digest
    actual: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("cache variant contains a symlink")
        if stat.S_ISREG(mode) and path.name != "checksums.sha256":
            actual.add(path.relative_to(root).as_posix())
        elif not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
            raise ValueError("cache variant contains a special filesystem entry")
    if actual != set(rows):
        raise ValueError("cache variant contains missing or unexpected files")
    return rows


def _validate_cache_variant(
    plan: Mapping[str, Any],
    bundle: Mapping[str, Any],
    variant_id: str,
    variant_root: Path,
) -> dict[str, Any]:
    if variant_root.is_symlink() or not variant_root.is_dir():
        raise ValueError("cache variant must be a non-symlink directory")
    checksum_rows = _cache_checksum_rows(variant_root)
    manifest_path = _assert_regular_file(
        variant_root / "manifest.json", label="cache manifest"
    )
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_manifest_keys = {
        "schema_version",
        "analysis_id",
        "bundle_id",
        "source_allowlist_sha256",
        "variant_id",
        "include_top50_components",
        "backend",
        "k",
        "query_keys",
        "query_counts",
        "train_sample_id_sha256",
        "query_sample_id_sha256",
        "backend_file_sha256",
        "requested_query_chunk_size",
        "requested_bank_chunk_size",
        "requested_memory_budget_bytes",
        "numeric_runtime_fingerprint",
        "numeric_runtime_fingerprint_sha256",
        "neighbor_cache_encoding",
        "backend_metadata",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_manifest_keys:
        raise ValueError("cache manifest exact field set mismatch")
    include_components = variant_id in COMPONENT_VARIANTS
    expected_fields = {
        "schema_version": CACHE_SCHEMA,
        "analysis_id": plan["analysis_id"],
        "bundle_id": bundle["bundle_id"],
        "source_allowlist_sha256": bundle["source_allowlist_sha256"],
        "variant_id": variant_id,
        "include_top50_components": include_components,
        "backend": RAW_KNN_BACKEND_ID,
        "k": KNN_K,
        "query_keys": plan["reuse_query_keys"],
        "backend_file_sha256": next(
            row["sha256"]
            for row in plan["source_validation"]["sources"]
            if row["path"] == "scripts/fixed_readout_stage2_raw_knn_backend.py"
        ),
        "requested_query_chunk_size": plan["query_chunk_size"],
        "requested_bank_chunk_size": plan["bank_chunk_size"],
        "requested_memory_budget_bytes": None,
        "numeric_runtime_fingerprint": plan["numeric_runtime_fingerprint"],
        "numeric_runtime_fingerprint_sha256": plan[
            "numeric_runtime_fingerprint_sha256"
        ],
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(f"cache manifest identity mismatch: {key}")
    query_counts = manifest.get("query_counts")
    query_digests = manifest.get("query_sample_id_sha256")
    if not isinstance(query_counts, Mapping) or set(query_counts) != set(
        plan["reuse_query_keys"]
    ):
        raise ValueError("cache query-count identity mismatch")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in query_counts.values()
    ):
        raise ValueError("cache query counts must be positive integers")
    if not isinstance(query_digests, Mapping) or set(query_digests) != set(
        plan["reuse_query_keys"]
    ) or any(_HEX64.fullmatch(str(value)) is None for value in query_digests.values()):
        raise ValueError("cache query sample-ID digest identity mismatch")
    _require_hex64(manifest.get("train_sample_id_sha256"), "train sample-ID digest")
    backend = manifest.get("backend_metadata")
    if (
        not isinstance(backend, Mapping)
        or set(backend) != {"backend", "k", "query_chunk_size", "bank_chunk_size"}
        or backend.get("backend") != RAW_KNN_BACKEND_ID
    ):
        raise ValueError("cache backend metadata mismatch")
    if backend.get("k") != KNN_K or backend.get("query_chunk_size") != plan[
        "query_chunk_size"
    ]:
        raise ValueError("cache backend K/query-chunk mismatch")
    resolved_bank = backend.get("bank_chunk_size")
    if resolved_bank != min(ID_TRAIN_COUNT, plan["bank_chunk_size"]):
        raise ValueError("cache resolved bank-chunk metadata mismatch")
    if manifest.get("neighbor_cache_encoding") != (
        "int32_bank_indices_losslessly_mapped_through_bound_train_sample_ids"
    ):
        raise ValueError("cache neighbor-component encoding mismatch")
    expected_payload_paths = {"manifest.json"}
    for query_key in plan["reuse_query_keys"]:
        expected_payload_paths.add(f"splits/{query_key}/score.npy")
        if include_components:
            expected_payload_paths.add(
                f"splits/{query_key}/neighbor_squared_distances.npy"
            )
            expected_payload_paths.add(f"splits/{query_key}/neighbor_bank_indices.npy")
    if set(checksum_rows) != expected_payload_paths:
        raise ValueError("cache variant exact payload-path set mismatch")
    inventory = [
        {
            "logical_path": relative,
            "sha256": digest,
            "size": (variant_root / PurePosixPath(relative)).lstat().st_size,
        }
        for relative, digest in sorted(checksum_rows.items())
    ]
    return {
        "variant_id": variant_id,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "checksums_sha256": _sha256_file(variant_root / "checksums.sha256"),
        "payload_file_count": len(inventory),
        "payload_catalog_sha256": _sha256_bytes(_canonical_bytes(inventory)),
        "status": "PASS",
    }


def _validate_cache_bundle(
    plan: Mapping[str, Any], bundle: Mapping[str, Any]
) -> dict[str, Any]:
    root = (
        Path(plan["roots"]["cache_root"])
        / plan["analysis_id"]
        / str(bundle["bundle_id"])
    )
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"cache bundle is absent: {bundle['bundle_id']}")
    observed = {entry.name for entry in root.iterdir()}
    if observed != set(KNN_VARIANT_IDS):
        unexpected = sorted(observed - set(KNN_VARIANT_IDS))
        missing = sorted(set(KNN_VARIANT_IDS) - observed)
        detail = unexpected[0] if unexpected else missing[0]
        raise ValueError(f"cache bundle exact variant identity mismatch: {detail}")
    variants = [
        _validate_cache_variant(plan, bundle, variant_id, root / variant_id)
        for variant_id in KNN_VARIANT_IDS
    ]
    summary = {
        "analysis_id": plan["analysis_id"],
        "bundle_id": bundle["bundle_id"],
        "variant_count": len(variants),
        "variants": variants,
        "status": "PASS",
    }
    summary["cache_inventory_sha256"] = _sha256_bytes(_canonical_bytes(summary))
    return summary


def _validate_partial_cache_state(
    plan: Mapping[str, Any], bundle_map: Mapping[str, Mapping[str, Any]]
) -> None:
    cache_root = Path(plan["roots"]["cache_root"])
    entries = list(cache_root.iterdir())
    if not entries:
        return
    if {entry.name for entry in entries} != {plan["analysis_id"]}:
        raise ValueError("cache root contains an unexpected analysis identity")
    analysis_root = entries[0]
    if analysis_root.is_symlink() or not analysis_root.is_dir():
        raise ValueError("cache analysis identity must be a directory")
    observed_bundles = {entry.name for entry in analysis_root.iterdir()}
    unexpected = observed_bundles - set(bundle_map)
    if unexpected:
        raise ValueError(f"cache root contains an unexpected bundle: {sorted(unexpected)[0]}")
    for bundle_id in sorted(observed_bundles):
        bundle_root = analysis_root / bundle_id
        if bundle_root.is_symlink() or not bundle_root.is_dir():
            raise ValueError("cache bundle identity must be a directory")
        observed_variants = {entry.name for entry in bundle_root.iterdir()}
        extra = observed_variants - set(KNN_VARIANT_IDS)
        if extra:
            raise ValueError(f"cache bundle contains an unexpected variant: {sorted(extra)[0]}")
        for variant_id in sorted(observed_variants):
            _validate_cache_variant(
                plan, bundle_map[bundle_id], variant_id, bundle_root / variant_id
            )


def _ensure_state_directories(plan: Mapping[str, Any]) -> tuple[Path, Path]:
    root = Path(plan["roots"]["logs_state_root"])
    logs = root / "logs"
    completions = root / "completions"
    for path in (logs, completions):
        if _lexists(path):
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"state path must be a non-symlink directory: {path}")
        else:
            path.mkdir()
    expected_ids = {row["bundle_id"] for row in plan["bundles"]}
    completion_names = {f"{bundle_id}.json" for bundle_id in expected_ids}
    extras = {entry.name for entry in completions.iterdir()} - completion_names
    if extras:
        raise ValueError(f"completion root contains an unexpected identity: {sorted(extras)[0]}")
    log_extras = {entry.name for entry in logs.iterdir()} - expected_ids
    if log_extras:
        raise ValueError(f"log root contains an unexpected identity: {sorted(log_extras)[0]}")
    return logs, completions


def _next_log_path(log_directory: Path) -> Path:
    if _lexists(log_directory):
        if log_directory.is_symlink() or not log_directory.is_dir():
            raise ValueError("bundle log path must be a non-symlink directory")
    else:
        log_directory.mkdir()
    names = {entry.name for entry in log_directory.iterdir()}
    if any(_ATTEMPT_LOG.fullmatch(name) is None for name in names):
        raise ValueError("bundle log directory contains an unexpected entry")
    for index in range(1, 10000):
        candidate = log_directory / f"attempt-{index:04d}.log"
        if not _lexists(candidate):
            return candidate
    raise RuntimeError("bundle attempt-log namespace is exhausted")


def _execute_child(
    command: Sequence[str],
    *,
    log_handle: Any,
    environment: Mapping[str, str],
    phase: str,
) -> tuple[int, dict[str, Any] | None]:
    marker = json.dumps(
        {"command": list(command), "phase": phase}, sort_keys=True, allow_nan=False
    )
    log_handle.write(f"SUPERVISOR_START {marker}\n")
    log_handle.flush()
    process = subprocess.Popen(
        list(command),
        cwd=REPOSITORY_ROOT,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=False,
    )
    report: dict[str, Any] | None = None
    assert process.stdout is not None
    for line in process.stdout:
        log_handle.write(line)
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                candidate = json.loads(stripped)
            except json.JSONDecodeError:
                candidate = None
            if isinstance(candidate, dict):
                report = candidate
    returncode = process.wait()
    log_handle.write(
        f"SUPERVISOR_END {json.dumps({'phase': phase, 'returncode': returncode}, sort_keys=True)}\n"
    )
    log_handle.flush()
    os.fsync(log_handle.fileno())
    return returncode, report


def _validate_precompute_report(
    plan: Mapping[str, Any], bundle_id: str, report: Mapping[str, Any] | None
) -> None:
    if not isinstance(report, Mapping):
        raise ValueError("extractor emitted no JSON precompute report")
    expected = {
        "schema_version": PRECOMPUTE_SCHEMA,
        "analysis_id": plan["analysis_id"],
        "analysis_git_sha": plan["git_head"],
        "bundle_ids": [bundle_id],
        "bundle_count": 1,
        "variant_count_per_bundle": len(KNN_VARIANT_IDS),
        "status": "PASS",
        "scientific_gate": "NOT_RUN",
        "numeric_runtime_fingerprint": plan["numeric_runtime_fingerprint"],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"extractor precompute report mismatch: {key}")
    bundles = report.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 1:
        raise ValueError("extractor precompute report bundle population mismatch")
    bundle = bundles[0]
    if bundle.get("bundle_id") != bundle_id or bundle.get("status") != "PASS":
        raise ValueError("extractor precompute bundle report mismatch")
    if bundle.get("variant_count") != len(KNN_VARIANT_IDS):
        raise ValueError("extractor precompute variant count mismatch")
    variants = bundle.get("variants")
    if not isinstance(variants, list) or [row.get("variant_id") for row in variants] != list(
        KNN_VARIANT_IDS
    ) or any(row.get("status") != "PASS" for row in variants):
        raise ValueError("extractor precompute variant report mismatch")


def _completion_payload(
    plan: Mapping[str, Any],
    bundle_plan: Mapping[str, Any],
    materialized: Mapping[str, Any],
    cache: Mapping[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": COMPLETION_SCHEMA,
        "status": "PASS",
        "plan_id": plan["plan_id"],
        "git_head": plan["git_head"],
        "analysis_id": plan["analysis_id"],
        "bundle_id": bundle_plan["bundle_id"],
        "source_allowlist_sha256": bundle_plan["source_allowlist_sha256"],
        "materialization_receipt_sha256": materialized["receipt_sha256"],
        "numeric_runtime_fingerprint_sha256": plan[
            "numeric_runtime_fingerprint_sha256"
        ],
        "extractor_command_sha256": _sha256_bytes(
            _canonical_bytes(bundle_plan["extractor_command"])
        ),
        "log_path": str(log_path),
        "cache_inventory_sha256": cache["cache_inventory_sha256"],
        "variant_count": cache["variant_count"],
        "variant_ids": [row["variant_id"] for row in cache["variants"]],
        "scientific_gate": "NOT_RUN",
    }


def _validate_completion(
    plan: Mapping[str, Any],
    bundle_plan: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    path = _assert_regular_file(Path(bundle_plan["completion"]), label="completion record")
    completion = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(completion, Mapping) or completion.get("schema_version") != COMPLETION_SCHEMA:
        raise ValueError("completion record schema mismatch")
    materialized = _preflight_materialized_pair(plan, bundle_plan, bundle)
    if materialized["status"] != "VERIFIED_EXISTING":
        raise ValueError("PASS completion lacks its materialized root and receipt")
    cache = _validate_cache_bundle(plan, bundle)
    log_path = Path(str(completion.get("log_path", "")))
    _assert_regular_file(log_path, label="completion log")
    expected_log_directory = Path(bundle_plan["log_directory"])
    if (
        log_path.parent != expected_log_directory
        or _ATTEMPT_LOG.fullmatch(log_path.name) is None
    ):
        raise ValueError("completion log identity is outside its bundle log directory")
    expected = _completion_payload(plan, bundle_plan, materialized, cache, log_path)
    if completion != expected:
        raise ValueError("completion record identity mismatch")
    return dict(completion)


def _run_one_bundle(
    plan: Mapping[str, Any],
    bundle_plan: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    log_path = _next_log_path(Path(bundle_plan["log_directory"]))
    environment = _child_environment(plan["fixed_environment"])
    with log_path.open("x", encoding="utf-8") as log_handle:
        materialized = _preflight_materialized_pair(plan, bundle_plan, bundle)
        if materialized["status"] == "ABSENT":
            returncode, _ = _execute_child(
                bundle_plan["materializer_command"],
                log_handle=log_handle,
                environment=environment,
                phase="materialize",
            )
            if returncode != 0:
                raise RuntimeError(
                    f"materializer failed for {bundle_plan['bundle_id']} with {returncode}"
                )
            materialized = _preflight_materialized_pair(plan, bundle_plan, bundle)
        returncode, report = _execute_child(
            bundle_plan["extractor_command"],
            log_handle=log_handle,
            environment=environment,
            phase="extract",
        )
        if returncode != 0:
            raise RuntimeError(
                f"extractor failed for {bundle_plan['bundle_id']} with {returncode}"
            )
        _validate_precompute_report(plan, str(bundle_plan["bundle_id"]), report)
        cache = _validate_cache_bundle(plan, bundle)
        completion = _completion_payload(
            plan, bundle_plan, materialized, cache, log_path
        )
        _atomic_write_new(Path(bundle_plan["completion"]), _canonical_bytes(completion))
        return completion


def run_plan(plan_path: str | Path) -> dict[str, Any]:
    plan = load_plan(plan_path)
    context = _revalidate_plan(plan, plan_path=plan_path)
    bundle_map = context["bundle_map"]
    _ensure_state_directories(plan)
    _validate_partial_cache_state(plan, bundle_map)
    pending: list[Mapping[str, Any]] = []
    skipped: list[str] = []
    # Preflight every unilateral state before launching any new subprocess.
    for bundle_plan in plan["bundles"]:
        bundle_id = str(bundle_plan["bundle_id"])
        bundle = bundle_map[bundle_id]
        _preflight_materialized_pair(plan, bundle_plan, bundle)
        completion_path = Path(bundle_plan["completion"])
        if _lexists(completion_path):
            _validate_completion(plan, bundle_plan, bundle)
            skipped.append(bundle_id)
        else:
            pending.append(bundle_plan)
    failures: dict[str, str] = {}
    completed: list[str] = []
    with ThreadPoolExecutor(max_workers=int(plan["concurrency"])) as executor:
        future_to_bundle = {
            executor.submit(
                _run_one_bundle,
                plan,
                bundle_plan,
                bundle_map[str(bundle_plan["bundle_id"])],
            ): str(bundle_plan["bundle_id"])
            for bundle_plan in pending
        }
        for future in as_completed(future_to_bundle):
            bundle_id = future_to_bundle[future]
            try:
                future.result()
            except Exception as exc:  # Preserve all other jobs and their logs.
                failures[bundle_id] = f"{type(exc).__name__}: {exc}"
            else:
                completed.append(bundle_id)
    result = {
        "schema_version": STATUS_SCHEMA,
        "plan_id": plan["plan_id"],
        "analysis_id": plan["analysis_id"],
        "completed_bundle_ids": sorted(completed),
        "resumed_pass_bundle_ids": sorted(skipped),
        "failed_bundles": dict(sorted(failures.items())),
        "status": "FAILED" if failures else "PASS",
        "scientific_gate": "NOT_RUN",
    }
    if not failures:
        status = status_plan(plan_path)
        if status["status"] != "PASS":
            raise AssertionError("post-run exact status validation did not pass")
    return result


def _exact_directory_names(path: Path, expected: set[str], *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a non-symlink directory")
    observed = {entry.name for entry in path.iterdir()}
    if observed != expected:
        unexpected = sorted(observed - expected)
        missing = sorted(expected - observed)
        detail = unexpected[0] if unexpected else missing[0]
        raise ValueError(f"{label} exact identity mismatch: {detail}")


def status_plan(plan_path: str | Path) -> dict[str, Any]:
    plan = load_plan(plan_path)
    context = _revalidate_plan(plan, plan_path=plan_path)
    bundle_map = context["bundle_map"]
    bundle_ids = {str(row["bundle_id"]) for row in plan["bundles"]}
    _exact_directory_names(
        Path(plan["roots"]["materialized_roots_parent"]),
        bundle_ids,
        label="materialized-roots parent",
    )
    receipts = Path(plan["roots"]["receipts_parent"])
    _exact_directory_names(
        receipts,
        {f"{bundle_id}.json" for bundle_id in bundle_ids},
        label="receipts parent",
    )
    cache_root = Path(plan["roots"]["cache_root"])
    _exact_directory_names(cache_root, {plan["analysis_id"]}, label="cache root")
    staging_root = Path(plan["roots"]["cache_staging_root"])
    if _lexists(staging_root):
        _exact_directory_names(
            staging_root, set(), label="cache staging root"
        )
    _exact_directory_names(
        cache_root / plan["analysis_id"], bundle_ids, label="cache analysis root"
    )
    logs_root = Path(plan["roots"]["logs_state_root"]) / "logs"
    completions_root = Path(plan["roots"]["logs_state_root"]) / "completions"
    _exact_directory_names(logs_root, bundle_ids, label="logs root")
    _exact_directory_names(
        completions_root,
        {f"{bundle_id}.json" for bundle_id in bundle_ids},
        label="completions root",
    )
    completion_rows = []
    for bundle_plan in plan["bundles"]:
        bundle_id = str(bundle_plan["bundle_id"])
        completion_rows.append(
            _validate_completion(plan, bundle_plan, bundle_map[bundle_id])
        )
    result = {
        "schema_version": STATUS_SCHEMA,
        "plan_id": plan["plan_id"],
        "git_head": plan["git_head"],
        "analysis_id": plan["analysis_id"],
        "numeric_runtime_fingerprint_sha256": plan[
            "numeric_runtime_fingerprint_sha256"
        ],
        "bundle_count": len(completion_rows),
        "variant_count": sum(row["variant_count"] for row in completion_rows),
        "bundle_ids": sorted(bundle_ids),
        "completion_catalog_sha256": _sha256_bytes(
            _canonical_bytes(completion_rows)
        ),
        "status": "PASS",
        "scientific_gate": "NOT_RUN",
    }
    if result["bundle_count"] != EXPECTED_BUNDLE_COUNT or result[
        "variant_count"
    ] != EXPECTED_BUNDLE_COUNT * len(KNN_VARIANT_IDS):
        raise AssertionError("complete cache status population mismatch")
    return result


def _add_plan_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--materialized-roots-parent", type=Path, required=True)
    parser.add_argument("--receipts-parent", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--logs-state-root", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--query-chunk-size", type=int, required=True)
    parser.add_argument("--bank-chunk-size", type=int, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="write one deterministic plan")
    _add_plan_inputs(plan_parser)
    run_parser = subparsers.add_parser("run", help="resume/run the frozen plan")
    run_parser.add_argument("--plan", type=Path, required=True)
    status_parser = subparsers.add_parser("status", help="verify all cache outputs")
    status_parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        plan = build_plan(
            reuse_manifest=args.reuse_manifest,
            source_root=args.source_root,
            materialized_roots_parent=args.materialized_roots_parent,
            receipts_parent=args.receipts_parent,
            cache_root=args.cache_root,
            logs_state_root=args.logs_state_root,
            python_executable=args.python_executable,
            concurrency=args.concurrency,
            query_chunk_size=args.query_chunk_size,
            bank_chunk_size=args.bank_chunk_size,
            output_plan=args.output_plan,
        )
        write_plan(plan, args.output_plan)
        print(json.dumps(plan, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "run":
        result = run_plan(args.plan)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "PASS" else 1
    result = status_plan(args.plan)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
