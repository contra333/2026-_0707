import numpy as np
import pytest

from oge.analysis.task_f_classifier_insensitive_kill import (
    ALL_ID_GUARDRAIL_PASS_CELL,
    EXPECTED_SEEDS,
    HIGH_WD_CELLS,
    PRIMARY_ANCHOR_CELL,
    adjudicate,
    centered_classifier_basis,
    decompose_residual_numpy,
    load_canonical_json,
    parse_idle_gpus,
    write_canonical_json,
)


def _basis_fixture():
    weight = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    basis, record = centered_classifier_basis(weight)
    assert record["rank"] == 1
    return basis


def test_insensitive_residual_has_rho_above_one_and_reconstructs():
    basis = _basis_fixture()
    residual = np.tile(np.asarray([0.1, 2.0, -1.0, 0.5]), (16, 1))
    result = decompose_residual_numpy(residual, basis)
    assert result["status"] == "PASS"
    assert result["rho"] > 1.0
    assert result["reconstruction_relative_error"] <= 1.0e-5


def test_sensitive_residual_has_rho_below_one():
    basis = _basis_fixture()
    residual = np.tile(np.asarray([2.0, 0.1, -0.1, 0.1]), (16, 1))
    result = decompose_residual_numpy(residual, basis)
    assert result["status"] == "PASS"
    assert result["rho"] < 1.0


def test_affine_only_residual_floor_is_not_rescued_by_a_ratio():
    basis = _basis_fixture()
    result = decompose_residual_numpy(
        np.zeros((8, 4), dtype=np.float64), basis, target_energy=100.0
    )
    assert result["status"] == "RESIDUAL_FLOOR"
    assert result["rho"] is None


def _gate_rows(*, primary=4, guardrail=2, high_wd=(2, 0)):
    desired = {
        PRIMARY_ANCHOR_CELL: primary,
        ALL_ID_GUARDRAIL_PASS_CELL: guardrail,
        HIGH_WD_CELLS[0]: high_wd[0],
        HIGH_WD_CELLS[1]: high_wd[1],
    }
    rows = []
    for cell, seeds in EXPECTED_SEEDS.items():
        for position, seed in enumerate(seeds):
            rows.append(
                {
                    "cell_id": cell,
                    "training_seed": seed,
                    "identity_status": "PASS",
                    "numerical_status": "PASS",
                    "energy": {"rho": 2.0 if position < desired[cell] else 0.5},
                    "cpu_float64_confirmation": {"status": "NOT_REQUIRED"},
                }
            )
    return rows


def test_gate_requires_the_frozen_14_context_pattern():
    assert adjudicate(_gate_rows())["decision"] == "GO"
    assert adjudicate(_gate_rows(primary=3))["decision"] == "FAIL"
    missing = _gate_rows()[:-1]
    assert adjudicate(missing)["decision"] == "BLOCKED"


def test_checksum_or_identity_failure_is_rejected_by_gate():
    rows = _gate_rows()
    rows[0]["identity_status"] = "FAIL"
    result = adjudicate(rows)
    assert result["decision"] == "BLOCKED"
    assert "invalid_contexts" in result["reason_codes"]


def test_canonical_result_checksum_mismatch_is_rejected(tmp_path):
    output = tmp_path / "host.json"
    write_canonical_json(output, {"status": "PASS"})
    output.with_name("host.json.sha256").write_text(
        f"{'0' * 64}  host.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum sidecar"):
        load_canonical_json(output)


def test_idle_gpu_parser_requires_no_compute_process_and_two_gib_free():
    inventory = "0, GPU-a, 4096\n1, GPU-b, 8192\n2, GPU-c, 1024\n"
    assert parse_idle_gpus(inventory, "GPU-b\n") == [0]
    with pytest.raises(ValueError, match="inventory row"):
        parse_idle_gpus("broken", "")
