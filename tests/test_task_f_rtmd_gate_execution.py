import copy
import json

import numpy as np
import pytest

import oge.analysis.task_f_rtmd_gate_execution as execution
from oge.analysis.discriminant_residual_preflight import fit_discriminant_geometry
from oge.analysis.task_f_rtmd_gate import RTMD_GATE3_SPEC_SHA256
from oge.evaluation.task_f_fresh import EXPECTED_SPECIFICATION_SHA256
from oge.evaluation.task_f_fresh_orchestration import SOURCE_TRAINING_SHA
from oge.studies.hashing import canonical_json_bytes


EVALUATION_SHA = "e" * 40
EXECUTION_SHA = "a" * 40


def _identity(*, seed=0, role="coupled"):
    return {
        "cell_id": "adam_lr1e-3_wd1e-4_anchor",
        "checkpoint_role": "last",
        "checkpoint_epoch": 200,
        "depth_tap": "penultimate",
        "role": role,
        "training_seed": seed,
        "run_id": f"anchor-{seed}-{role}",
        "sibling_group_id": f"anchor-{seed}",
        "initialization_sha256": f"{seed + 1:064x}",
        "data_stream_sha256": f"{seed + 20:064x}",
        "checkpoint_sha256": f"{seed + 40:064x}",
        "geometry_output_identity_sha256": f"{seed + 60:064x}",
        "train_feature_output_identity_sha256": f"{seed + 80:064x}",
        "validation_feature_output_identity_sha256": f"{seed + 100:064x}",
    }


def _fixture(seed=7):
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            [1.0, 0.2, -0.1, 0.3, 0.1, -0.2],
            [-0.4, 1.1, 0.2, -0.2, 0.2, 0.3],
            [-0.6, -0.8, 0.5, 0.1, -0.3, 0.4],
        ]
    )
    labels = np.repeat(np.arange(3), 40)
    train = means[labels] + rng.normal(scale=0.35, size=(len(labels), 6))
    fit = fit_discriminant_geometry(train, labels)
    q = rng.chisquare(fit.residual_dim, size=60)
    return train, labels, np.asarray([f"sample-{i}" for i in range(len(train))]), q, fit


def test_fresh_gate3_twofold_record_is_deterministic_and_id_only():
    train, labels, sample_ids, q, fit = _fixture()
    first = execution.fit_gate3_record(
        train_features=train,
        train_labels=labels,
        train_sample_ids=sample_ids,
        validation_q_perp=q,
        residual_dim=fit.residual_dim,
        final_fit_applicable=True,
        identity=_identity(),
    )
    second = execution.fit_gate3_record(
        train_features=train,
        train_labels=labels,
        train_sample_ids=sample_ids,
        validation_q_perp=q,
        residual_dim=fit.residual_dim,
        final_fit_applicable=True,
        identity=_identity(),
    )
    assert first == second
    assert first["numerically_applicable"]
    assert first["dataset_split"] == "id_validation"
    assert first["transform"] == "raw"
    assert first["partition"]["fold_counts"] == [60, 60]
    assert first["rtmd_gate3_specification_sha256"] == RTMD_GATE3_SPEC_SHA256
    assert not first["protected_data_access"]


def test_fresh_gate3_inapplicability_is_preserved_without_fallback_estimator():
    train, labels, sample_ids, q, fit = _fixture()
    result = execution.fit_gate3_record(
        train_features=train,
        train_labels=labels,
        train_sample_ids=sample_ids,
        validation_q_perp=q,
        residual_dim=fit.residual_dim,
        final_fit_applicable=False,
        identity=_identity(),
    )
    assert not result["numerically_applicable"]
    assert not result["finite_t_selected"]
    assert result["nu_fit"] is None
    assert result["tail_statistic"] is None


def _gate_record(*, seed, role, effect=0.5):
    base = [0.00, 0.04, -0.03, 0.02, -0.01][seed]
    return {
        "schema_version": execution.RECORD_SCHEMA_VERSION,
        **_identity(seed=seed, role=role),
        "transform": "raw",
        "dataset_split": "id_validation",
        "numerically_applicable": True,
        "finite_t_selected": seed != 4,
        "tail_statistic": base + (effect if role == "coupled" else 0.0),
        "protected_data_access": False,
        "rtmd_gate3_specification_sha256": RTMD_GATE3_SPEC_SHA256,
    }


def _write_host(tmp_path, host, records):
    payload = {
        "schema_version": execution.HOST_SCHEMA_VERSION,
        "status": "PASS",
        "host_id": host,
        "source_training_sha": SOURCE_TRAINING_SHA,
        "evaluation_git_sha": EVALUATION_SHA,
        "execution_git_sha": EXECUTION_SHA,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "rtmd_gate3_specification_sha256": RTMD_GATE3_SPEC_SHA256,
        "record_count": len(records),
        "records": records,
        "protected_data_access": False,
    }
    return execution._write_directory(
        tmp_path / host, payloads={"HOST_GATE3_RECORDS.json": payload}
    )


def _host_outputs(tmp_path):
    allocation = {
        "curie": [(0, "coupled"), (0, "decoupled"), (1, "coupled"), (1, "decoupled")],
        "lise": [(3, "coupled"), (3, "decoupled"), (4, "coupled"), (4, "decoupled")],
        "precision_medicine": [(2, "coupled"), (2, "decoupled")],
    }
    return [
        _write_host(
            tmp_path,
            host,
            [_gate_record(seed=seed, role=role) for seed, role in rows],
        )
        for host, rows in allocation.items()
    ]


def test_three_host_collection_is_exact_checksummed_and_no_overwrite(tmp_path):
    outputs = _host_outputs(tmp_path / "hosts")
    destination = execution.collect_gate3(
        host_outputs=outputs, output_directory=tmp_path / "terminal"
    )
    terminal = execution.verify_gate3_terminal(destination)
    assert terminal["status"] == "PASS"
    assert terminal["record_count"] == 10
    assert terminal["gate3_verdict"]["status"] == "PASS"
    assert terminal["rtmd_included_in_protected_plan"]
    assert not terminal["protected_data_access"]
    assert execution.collect_gate3(
        host_outputs=outputs, output_directory=destination
    ) == destination

    changed = json.loads((destination / "GATE3_COMPLETE.json").read_text())
    changed["record_count"] = 9
    (destination / "GATE3_COMPLETE.json").write_bytes(
        canonical_json_bytes(changed) + b"\n"
    )
    with pytest.raises(ValueError, match="checksum"):
        execution.verify_gate3_terminal(destination)


def test_collection_rejects_missing_duplicate_or_mixed_execution(tmp_path):
    outputs = _host_outputs(tmp_path / "hosts")
    with pytest.raises(ValueError, match="exactly three"):
        execution.collect_gate3(
            host_outputs=outputs[:2], output_directory=tmp_path / "missing"
        )
    with pytest.raises(ValueError, match="duplicate"):
        execution.collect_gate3(
            host_outputs=[outputs[0], outputs[0], outputs[2]],
            output_directory=tmp_path / "duplicate",
        )

    mixed = tmp_path / "mixed"
    payload = copy.deepcopy(execution.verify_gate3_host_output(outputs[2]))
    payload["execution_git_sha"] = "b" * 40
    execution._write_directory(
        mixed, payloads={"HOST_GATE3_RECORDS.json": payload}
    )
    with pytest.raises(ValueError, match="identities differ"):
        execution.collect_gate3(
            host_outputs=[outputs[0], outputs[1], mixed],
            output_directory=tmp_path / "mixed-terminal",
        )


def test_host_selection_binds_only_exact_passed_geometry_jobs(tmp_path, monkeypatch):
    worker_root = tmp_path / "specs"
    worker_root.mkdir()
    jobs = {}
    manifests = {}
    for seed in (0, 1):
        for sibling, role in (("alpha_1", "coupled"), ("alpha_0", "decoupled")):
            job_id = f"geometry::{seed}::{role}"
            bridge_path = str(tmp_path / f"bridge-{seed}-{sibling}")
            manifests[bridge_path] = {
                "cell_id": "adam_lr1e-3_wd1e-4_anchor",
                "checkpoint_role": "last",
                "checkpoint_epoch": 200,
                "depth_tap": "penultimate",
                "dataset_split": "id_train",
                "sibling_role": sibling,
                "training_seed": seed,
            }
            spec = {
                "job_id": job_id,
                "stage": "geometry",
                "train_binding": {"bridge_path": bridge_path},
                "validation_binding": {},
            }
            path = worker_root / f"{seed}-{role}.json"
            path.write_bytes(canonical_json_bytes(spec) + b"\n")
            jobs[job_id] = {
                "status": "PASS",
                "result": {"artifact_path": str(tmp_path / f"geometry-{seed}-{role}")},
            }
    ledger = {
        "schema_version": "fixture",
        "status": "PASS",
        "identity": {
            "evaluation_git_sha": EVALUATION_SHA,
            "source_training_sha": SOURCE_TRAINING_SHA,
            "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        },
        "jobs": jobs,
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_bytes(canonical_json_bytes(ledger) + b"\n")
    monkeypatch.setattr(
        execution,
        "verify_bridge_artifact",
        lambda path: {"manifest": manifests[str(path)]},
    )
    monkeypatch.setattr(
        execution,
        "build_gate3_record_from_spec",
        lambda worker_spec, geometry_root: _gate_record(
            seed=int(worker_spec["job_id"].split("::")[1]),
            role=worker_spec["job_id"].split("::")[2],
        ),
    )
    destination = execution.run_gate3_host(
        host_id="curie",
        worker_spec_root=worker_root,
        ledger_path=ledger_path,
        expected_evaluation_git_sha=EVALUATION_SHA,
        execution_git_sha=EXECUTION_SHA,
        output_directory=tmp_path / "host-output",
    )
    assert execution.verify_gate3_host_output(destination)["record_count"] == 4

    protected = copy.deepcopy(next(iter(manifests.values())))
    protected["dataset_split"] = "id_test"
    manifests[next(iter(manifests))] = protected
    with pytest.raises(ValueError, match="selection differs"):
        execution.run_gate3_host(
            host_id="curie",
            worker_spec_root=worker_root,
            ledger_path=ledger_path,
            expected_evaluation_git_sha=EVALUATION_SHA,
            execution_git_sha=EXECUTION_SHA,
            output_directory=tmp_path / "protected-output",
        )
