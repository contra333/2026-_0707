import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

import oge.analysis.discriminant_residual_diagnostic as diagnostic


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl(rows):
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def _record(bundle_id, transform, index):
    if index < 22:
        applicable = False
        rank = 639
        condition = "infinity"
        required_pass = False
    elif index < 30:
        applicable = False
        rank = 640
        condition = 2.0e8
        required_pass = False
    elif index < 51:
        applicable = True
        rank = 640
        condition = 1.0e6
        required_pass = False
    else:
        applicable = True
        rank = 640
        condition = 1.0e6
        required_pass = True
    split_checks = {
        "score_reconstruction": True,
        "marginal_reconstruction": True,
        "rmd_cancellation": required_pass or not applicable,
        "pair_margin_reconstruction": True,
        "component_covariance": True,
    }
    parity_pass = index >= 36
    parity_residuals = {
        "md": 1.0e-12,
        "rmd": 2.0e-8 if not parity_pass else 1.0e-12,
        "global_distance": 1.0e-12,
    }
    correlations = None
    if transform == "raw":
        correlations = {
            split: {
                component: {"spearman": 0.0, "pearson": 0.0}
                for component in ("s_perp", "s_parallel_marginal", "s_rmd")
            }
            for split in ("id_train", "id_test")
        }
    spectrum = {
        "rank": rank,
        "dimension": 640,
        "condition_number": condition,
    }
    return {
        "bundle_id": bundle_id,
        "config_hash": f"{index // 6:064x}",
        "optimizer_family": ("adam", "adamw", "sgd")[index % 3],
        "training_seed": index % 3,
        "transform": transform,
        "applicable": applicable,
        "required_numerical_pass": required_pass,
        "numerical": {
            "within_spectrum": spectrum,
            "global_spectrum": spectrum,
            "condition_ceiling": 1.0e8,
            "tau_alg": 1.0e-8,
            "checks": {"backend_parity": True},
        },
        "id_train": {"checks": split_checks},
        "id_test": {"checks": split_checks},
        "id_train_residual_energy": {"pass": True},
        "cached_metric_v1_2_id_test_parity": {
            "pass": parity_pass,
            "scaled_max_residuals": parity_residuals,
        },
        "raw_norm_component_correlations": correlations,
    }


def _fixture(tmp_path, monkeypatch):
    bundles = [f"{index + 1:064x}" for index in range(30)]
    rows = []
    for bundle_index, bundle_id in enumerate(bundles):
        for transform_offset, transform in enumerate(("raw", "l2")):
            rows.append(_record(bundle_id, transform, bundle_index * 2 + transform_offset))
    tails = [
        {
            "bundle_id": row["bundle_id"],
            "transform": row["transform"],
            "status": "PASS" if row["applicable"] else "INCONCLUSIVE_NUMERICAL",
        }
        for row in rows
    ]
    summary = {
        "schema_version": "discriminant_residual_preflight_v1",
        "status": "PASS",
        "gate_2": {"gate_status": "INCONCLUSIVE"},
    }
    payloads = {
        "bundle_records.jsonl": _jsonl(rows),
        "summary.json": json.dumps(summary, sort_keys=True).encode(),
        "tail_records.jsonl": _jsonl(tails),
    }
    checksums = "".join(
        f"{_sha256_bytes(payload)}  {name}\n"
        for name, payload in sorted(payloads.items())
    ).encode()
    payloads["checksums.sha256"] = checksums
    archive = tmp_path / "preflight.tar"
    with tarfile.open(archive, "w") as handle:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(
        diagnostic, "EXPECTED_PREFLIGHT_ARCHIVE_SHA256", archive_sha
    )

    nc1_root = tmp_path / "nc1"
    entries = []
    for index, bundle_id in enumerate(bundles):
        path = nc1_root / bundle_id / "trace_components.npy"
        path.parent.mkdir(parents=True)
        np.save(path, np.full(640, (index + 1) / 1_000_000.0, dtype=np.float64))
        entries.append(
            {
                "bundle_id": bundle_id,
                "local_path": f"{bundle_id}/trace_components.npy",
                "hf_uri": (
                    "hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract.v1.2/"
                    f"{bundle_id}/{diagnostic.NC1_LOGICAL_PATH}"
                ),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": "discriminant_residual_diagnostic_inputs_v1",
        "preflight_archive": {
            "sha256": archive_sha,
            "required_members": list(diagnostic.PREFLIGHT_MEMBERS),
        },
        "hf_base_uri": (
            "hf://buckets/contra333/ICLR_RUN/evaluations/metric_contract.v1.2"
        ),
        "nc1_array_contract": {
            "shape": [640],
            "dtype": "float64",
            "class_count": 10,
            "aggregation": "sum(trace_components)/class_count",
        },
        "nc1_hypothesis": {"predicted_spearman_sign": "negative"},
        "historical_pass_rate_may_select_scale": False,
        "v1_2_comparison": {
            "covariance_denominator": "biased_1_over_N_in_both",
            "centering": "class_residual_and_global_mean_in_both",
            "l2": "per_sample_normalization_then_refit_in_both",
            "backend": "sklearn_scipy_pinvh_vs_card13_numpy_eigh",
        },
        "nc1_entries": entries,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, archive, nc1_root, manifest


def test_forward_error_fixture_supports_operand_scale_without_changing_tau():
    result = diagnostic.cancellation_forward_error_fixture()

    assert result["criteria"]["high_precision_algebraic_equivalence"]
    assert result["criteria"]["all_injected_errors_within_condition_eps"]
    assert result["criteria"]["operand_scale_respects_tau_on_grid"]
    assert result["criteria"]["output_scale_understates_error_on_grid"]
    assert result["supports_operand_scale_clarification"]
    assert not result["historical_pass_rate_used"]


def test_generator_rederives_layers_and_is_byte_deterministic(tmp_path, monkeypatch):
    manifest, archive, nc1_root, _ = _fixture(tmp_path, monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"

    result = diagnostic.run_diagnostic(
        input_manifest_path=manifest,
        preflight_archive_path=archive,
        nc1_root=nc1_root,
        output_dir=first,
    )
    repeated = diagnostic.run_diagnostic(
        input_manifest_path=manifest,
        preflight_archive_path=archive,
        nc1_root=nc1_root,
        output_dir=second,
    )

    assert result == repeated
    assert result["status"] == "PASS"
    assert result["evidence"]["count_conservation"]["pass"]
    assert result["evidence"]["layer_a_applicability"]["failure_fit_count"] == 30
    assert result["evidence"]["layer_b_required_numerical"]["failure_fit_count"] == 21
    assert result["evidence"]["layer_c_cached_parity_diagnostic"]["failure_fit_count"] == 36
    assert result["decision"]["gate_2"]["status"] == "INCONCLUSIVE_IMMUTABLE"
    assert result["decision"]["rmd_cancellation"]["historical_pass_rate_used"] is False
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()
    assert (first / "diagnostic_records.jsonl").read_bytes() == (
        second / "diagnostic_records.jsonl"
    ).read_bytes()


def test_missing_nc1_stops_before_records_are_emitted(tmp_path, monkeypatch):
    manifest, archive, nc1_root, payload = _fixture(tmp_path, monkeypatch)
    missing = nc1_root / payload["nc1_entries"][0]["local_path"]
    missing.unlink()
    output = tmp_path / "output"

    result = diagnostic.run_diagnostic(
        input_manifest_path=manifest,
        preflight_archive_path=archive,
        nc1_root=nc1_root,
        output_dir=output,
    )

    assert result["status"] == "INCONCLUSIVE_INPUT"
    assert "required NC1 input is absent" in result["reason"]
    assert not (output / "diagnostic_records.jsonl").exists()


def test_checksum_mismatch_is_inconclusive(tmp_path, monkeypatch):
    manifest, archive, nc1_root, payload = _fixture(tmp_path, monkeypatch)
    payload["nc1_entries"][0]["sha256"] = "f" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = diagnostic.run_diagnostic(
        input_manifest_path=manifest,
        preflight_archive_path=archive,
        nc1_root=nc1_root,
        output_dir=tmp_path / "output",
    )

    assert result["status"] == "INCONCLUSIVE_INPUT"
    assert "NC1 checksum mismatch" in result["reason"]


def test_forbidden_remote_path_is_rejected_before_archive_access(
    tmp_path, monkeypatch
):
    manifest, archive, nc1_root, payload = _fixture(tmp_path, monkeypatch)
    payload["hf_base_uri"] = (
        "hf://buckets/contra333/ICLR_RUN/evaluations/ood/protected"
    )
    for entry in payload["nc1_entries"]:
        entry["hf_uri"] = (
            f"{payload['hf_base_uri']}/{entry['bundle_id']}/"
            f"{diagnostic.NC1_LOGICAL_PATH}"
        )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    archive_called = False

    def forbidden_archive_read(*args, **kwargs):
        nonlocal archive_called
        archive_called = True
        raise AssertionError("archive read must not start")

    monkeypatch.setattr(diagnostic, "_read_preflight_archive", forbidden_archive_read)
    result = diagnostic.run_diagnostic(
        input_manifest_path=manifest,
        preflight_archive_path=archive,
        nc1_root=nc1_root,
        output_dir=tmp_path / "output",
    )

    assert result["status"] == "INCONCLUSIVE_INPUT"
    assert "forbidden path" in result["reason"]
    assert not archive_called


def test_raw_norm_component_regression_requires_both_id_splits(
    tmp_path, monkeypatch
):
    manifest, archive, nc1_root, _ = _fixture(tmp_path, monkeypatch)
    with tarfile.open(archive, "r") as handle:
        bundle_payload = handle.extractfile("bundle_records.jsonl").read()
    rows = [json.loads(line) for line in bundle_payload.decode().splitlines()]
    rows[0]["raw_norm_component_correlations"].pop("id_test")
    evidence = None
    try:
        evidence, _ = diagnostic._analyze_rows(
            rows,
            [
                {
                    "bundle_id": row["bundle_id"],
                    "transform": row["transform"],
                    "status": "PASS",
                }
                for row in rows
            ],
            {entry["bundle_id"]: 0.1 for entry in json.loads(manifest.read_text())["nc1_entries"]},
        )
    except diagnostic.DiagnosticInputError as error:
        assert "correlations lack train/test coverage" in str(error)
    else:
        raise AssertionError(f"missing id_test correlations were accepted: {evidence}")
