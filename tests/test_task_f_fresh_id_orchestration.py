import copy
import json
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from oge.evaluation.task_f_fresh import build_fresh_evaluation_plan
from oge.evaluation.task_f_fresh_orchestration import (
    EXPECTED_SPECIFICATION_SHA256,
    HOST_COUNTS,
    LOCAL_RELAY_SCHEMA_VERSION,
    SOURCE_TRAINING_SHA,
    build_task_f_pipeline_manifest,
    collect_task_f_host_summaries,
    index_feature_artifacts,
    publish_task_f_host_result,
    publish_task_f_local_compute_relay,
    stage_task_f_local_compute_relay,
    validate_local_source_gate,
    validate_remote_source_gate_matches_local,
    validate_source_gate_documents,
    validate_task_f_pipeline_manifest,
)
from oge.studies.artifacts import atomic_write_json
from oge.studies.hashing import canonical_sha256
from oge.studies.supervisor import build_artifact_manifest
from oge.studies.staged_pipeline import (
    PIPELINE_SCHEMA_VERSION,
    AtomicStageLedger,
    execute_resource_queues,
    validate_staged_manifest,
)
from oge.training import generate_research_run_matrix


def _run_matrix():
    return generate_research_run_matrix(
        anchor_seeds=(0, 1, 2, 3, 4),
        adam_factorial_seeds=(0, 1, 2),
        sgdm_seeds=(0, 1, 2),
    )


def _placement(run_matrix):
    groups = {
        "curie": {
            "task-f-adam-lr1e-3-seed0",
            "task-f-adam-lr1e-3-seed1",
            "task-f-adam-lr3e-4-seed0",
            "task-f-sgdm-lr0.1-seed0",
        },
        "lise": {
            "task-f-adam-lr1e-3-seed3",
            "task-f-adam-lr1e-3-seed4",
            "task-f-sgdm-lr0.1-seed1",
        },
        "precision_medicine": {
            "task-f-adam-lr1e-3-seed2",
            "task-f-adam-lr3e-4-seed1",
            "task-f-adam-lr3e-4-seed2",
            "task-f-sgdm-lr0.1-seed2",
        },
    }
    concurrency = {"curie": 4, "lise": 2, "precision_medicine": 4}
    by_host = {host: [] for host in groups}
    for run in run_matrix["runs"]:
        host = next(host for host, members in groups.items() if run["sibling_group_id"] in members)
        by_host[host].append(run["run_id"])
    queues = {"execution_sha": SOURCE_TRAINING_SHA, "hosts": {}}
    for host, run_ids in by_host.items():
        queues["hosts"][host] = {
            str(index): {"gpu_uuid": f"GPU-{host}-{index}", "run_ids": []}
            for index in range(concurrency[host])
        }
        for offset, run_id in enumerate(sorted(run_ids)):
            queues["hosts"][host][str(offset % concurrency[host])]["run_ids"].append(run_id)
    assignment = {
        "execution_sha": SOURCE_TRAINING_SHA,
        "host_assignments": {
            host: {
                "expected_run_count": len(by_host[host]),
                "concurrency": concurrency[host],
                "sibling_group_ids": sorted(members),
            }
            for host, members in groups.items()
        },
    }
    location = {}
    for host, host_queues in queues["hosts"].items():
        for index, queue in host_queues.items():
            for run_id in queue["run_ids"]:
                location[run_id] = (host, int(index), queue["gpu_uuid"])
    return assignment, queues, location


def _manifest():
    run_matrix = _run_matrix()
    plan = build_fresh_evaluation_plan(run_matrix)
    assignment, queues, location = _placement(run_matrix)
    observed = []
    for record in plan["records"]:
        if record["dataset_split"] != "id_train" or record["checkpoint_role"] == "best_val":
            continue
        if record["depth_tap"] == "penultimate" and record["checkpoint_epoch"] not in {10, 60, 120, 160, 200}:
            continue
        host, index, uuid = location[record["run_id"]]
        observed.append(
            {
                **record,
                "host_id": host,
                "gpu_index": index,
                "gpu_uuid": uuid,
            }
        )
    assert len(observed) == 310
    manifest = build_task_f_pipeline_manifest(
        evaluation_plan=plan,
        observed_export_jobs=observed,
        host_assignment=assignment,
        gpu_queues=queues,
        evaluation_git_sha="e" * 40,
    )
    return plan, manifest


def test_task_f_manifest_freezes_exact_supplemental_geometry_and_alignment_coverage():
    _, manifest = _manifest()
    assert manifest["identity"] == {
        **manifest["identity"],
        "source_training_sha": SOURCE_TRAINING_SHA,
        "evaluation_git_sha": "e" * 40,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
    }
    assert manifest["counts"]["supplemental_exports"] == 1010
    assert manifest["counts"]["supplemental_by_host"] == {
        "curie": 404,
        "lise": 233,
        "precision_medicine": 373,
    }
    assert manifest["counts"]["geometry_by_host"] == {
        "curie": 264,
        "lise": 156,
        "precision_medicine": 240,
    }
    assert manifest["counts"]["alignments"] == 657
    assert set(manifest["production_sentinel_by_host"]) == {
        "curie",
        "lise",
        "precision_medicine",
    }
    assert all(
        len(record["export_job_ids"]) == 1
        and len(record["bridge_job_ids"]) == 2
        and record["validation"] == "checksum_shape_bridge_geometry_reconstruction"
        for record in manifest["production_sentinel_by_host"].values()
    )
    assert len(manifest["jobs"]) == 1320 + 1320 + 660 + 657
    stages = Counter(job["stage"] for job in manifest["jobs"])
    assert stages == {
        "feature_export": 1320,
        "bridge": 1320,
        "geometry": 660,
        "alignment": 657,
    }
    for job in manifest["jobs"]:
        if job["stage"] == "alignment":
            assert all(
                dependency.startswith(("bridge::", "geometry::"))
                for dependency in job["dependencies"]
            )


def test_task_f_manifest_rejects_gpu_drift_protected_reference_and_count_drift():
    plan, manifest = _manifest()
    broken = copy.deepcopy(manifest)
    broken["counts"]["supplemental_exports"] = 1009
    broken.pop("manifest_sha256")
    with pytest.raises(ValueError, match="counts"):
        validate_task_f_pipeline_manifest(broken)

    run_matrix = _run_matrix()
    assignment, queues, location = _placement(run_matrix)
    observed = []
    for record in plan["records"][:1]:
        host, index, uuid = location[record["run_id"]]
        observed.append({**record, "host_id": host, "gpu_index": index, "gpu_uuid": uuid})
    with pytest.raises(ValueError, match="310"):
        build_task_f_pipeline_manifest(
            evaluation_plan=plan,
            observed_export_jobs=observed,
            host_assignment=assignment,
            gpu_queues=queues,
            evaluation_git_sha="e" * 40,
        )

    protected = copy.deepcopy(manifest)
    protected["jobs"][0]["record"]["dataset_split"] = "id_test"
    protected.pop("manifest_sha256")
    with pytest.raises(ValueError, match="protected"):
        validate_task_f_pipeline_manifest(protected)


def _source_documents():
    documents = {}
    for host, counts in HOST_COUNTS.items():
        documents[host] = {
            "marker": {
                "status": "REMOTE_VERIFIED",
                "host_id": host,
                "execution_sha": SOURCE_TRAINING_SHA,
                "run_count": counts["runs"],
                "export_count": counts["source_exports"],
            },
            "validation": {
                "status": "PASS",
                "host_id": host,
                "execution_sha": SOURCE_TRAINING_SHA,
                "specification_sha256": EXPECTED_SPECIFICATION_SHA256,
                "run_count": counts["runs"],
                "export_count": counts["source_exports"],
            },
        }
    return documents


def test_global_source_gate_requires_all_three_exact_terminal_witnesses():
    result = validate_source_gate_documents(_source_documents())
    assert result["status"] == "PASS"
    assert result["run_count"] == 50
    assert result["export_count"] == 310
    missing = _source_documents()
    missing.pop("lise")
    with pytest.raises(RuntimeError, match="three"):
        validate_source_gate_documents(missing)
    drift = _source_documents()
    drift["curie"]["validation"]["export_count"] -= 1
    with pytest.raises(RuntimeError, match="accounting"):
        validate_source_gate_documents(drift)


def test_feature_index_ignores_only_atomic_temporary_bundles(tmp_path):
    root = tmp_path / "exports"
    temporary = root / ".task-f-fixture.tmp"
    temporary.mkdir(parents=True)
    (temporary / "manifest.json").write_text("{}\n", encoding="utf-8")
    assert index_feature_artifacts([root]) == {}

    visible = root / "visible-incomplete-bundle"
    visible.mkdir()
    (visible / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        index_feature_artifacts([root])


def _local_source_fixture(tmp_path, host, manifest):
    source = tmp_path / host
    (source / "control").mkdir(parents=True)
    (source / "control" / "host_finalizer.status").write_text(
        "UPLOADING\n", encoding="utf-8"
    )
    source_jobs = [
        job
        for job in manifest["jobs"]
        if job["host_id"] == host
        and job["stage"] == "feature_export"
        and job["materialization"] == "source"
    ]
    export_counts = Counter(job["record"]["run_id"] for job in source_jobs)
    rows = []
    for run_id, export_count in sorted(export_counts.items()):
        checkpoint = source / "runs" / run_id / "checkpoints" / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"")
        rows.append(
            {
                "run_id": run_id,
                "last_pt_bytes": 0,
                "export_count": export_count,
            }
        )
    validation = {
        "status": "PASS",
        "host_id": host,
        "execution_sha": SOURCE_TRAINING_SHA,
        "specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "run_count": HOST_COUNTS[host]["runs"],
        "export_count": HOST_COUNTS[host]["source_exports"],
        "runs": rows,
    }
    atomic_write_json(source / "host_validation.json", validation)
    return source, validation


def test_local_source_gate_starts_from_pass_witness_without_remote_marker(tmp_path):
    _, manifest = _manifest()
    source, validation = _local_source_fixture(tmp_path, "curie", manifest)
    result = validate_local_source_gate(
        source_root=source,
        host_id="curie",
        manifest=manifest,
    )
    assert result["status"] == "PASS"
    assert result["run_count"] == 20
    assert result["export_count"] == 124
    assert result["finalizer_status_at_start"] == "UPLOADING"
    assert not (source / "remote_marker" / "REMOTE_COMPLETE.json").exists()
    assert result["validation_sha256"] == canonical_sha256(validation)

    (source / "control" / "host_finalizer.status").write_text(
        "FAILED upload\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not upload-ready"):
        validate_local_source_gate(
            source_root=source,
            host_id="curie",
            manifest=manifest,
        )


def test_overlap_publication_requires_matching_global_remote_source_gate(
    tmp_path, monkeypatch
):
    import oge.evaluation.task_f_fresh_orchestration as orchestration

    remote_gate = validate_source_gate_documents(_source_documents())
    local_gate = {
        "status": "PASS",
        "host_id": "curie",
        "source_training_sha": SOURCE_TRAINING_SHA,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "validation_sha256": remote_gate["hosts"]["curie"]["validation_sha256"],
    }
    validate_remote_source_gate_matches_local(
        local_gate=local_gate,
        remote_gate=remote_gate,
    )
    drift = copy.deepcopy(remote_gate)
    drift["hosts"]["curie"]["validation_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="differs"):
        validate_remote_source_gate_matches_local(
            local_gate=local_gate,
            remote_gate=drift,
        )

    artifact = tmp_path / "artifacts"
    state = tmp_path / "state"
    operational = artifact / "operational_bundle"
    operational.mkdir(parents=True)
    state.mkdir()
    terminal = {
        "schema_version": "task_f_fresh_id_host_terminal_v1",
        "status": "PASS",
        "host_id": "curie",
        "source_training_sha": SOURCE_TRAINING_SHA,
        "evaluation_git_sha": "e" * 40,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "parent_manifest_sha256": "m" * 64,
        "stage_counts": {"geometry": 264},
        "protected_data_access": False,
    }
    atomic_write_json(operational / "HOST_COMPLETE.json", terminal)
    atomic_write_json(state / "HOST_COMPUTE_COMPLETE.json", terminal)
    uploads = []

    def fake_upload(source, *, hf_cli, bucket, destination, control_root):
        uploads.append(destination)
        return {"status": "REMOTE_VERIFIED", "destination": destination}

    monkeypatch.setattr(orchestration, "upload_artifact_tree", fake_upload)
    with pytest.raises(RuntimeError, match="differs"):
        publish_task_f_host_result(
            artifact_root=artifact,
            state_root=state,
            host_id="curie",
            expected_evaluation_git_sha="e" * 40,
            hf_cli="hf",
            remote_source_gate=drift,
            local_source_gate=local_gate,
        )
    assert uploads == []
    assert not (state / "HOST_COMPLETE.json").exists()

    result = publish_task_f_host_result(
        artifact_root=artifact,
        state_root=state,
        host_id="curie",
        expected_evaluation_git_sha="e" * 40,
        hf_cli="hf",
        remote_source_gate=remote_gate,
        local_source_gate=local_gate,
    )
    assert result["remote"]["status"] == "REMOTE_VERIFIED"
    assert len(uploads) == 1
    assert json.loads((state / "HOST_COMPLETE.json").read_text())["remote"][
        "status"
    ] == "REMOTE_VERIFIED"


def test_local_compute_relay_is_atomic_identity_bound_and_not_a_final_terminal(
    tmp_path, monkeypatch
):
    import oge.evaluation.task_f_fresh_orchestration as orchestration

    artifact = tmp_path / "artifacts"
    operational = artifact / "operational_bundle"
    state = tmp_path / "state"
    operational.mkdir(parents=True)
    state.mkdir()
    evaluation_sha = "e" * 40
    collector_sha = "c" * 40
    terminal = {
        "schema_version": "task_f_fresh_id_host_terminal_v1",
        "status": "PASS",
        "host_id": "curie",
        "source_training_sha": SOURCE_TRAINING_SHA,
        "evaluation_git_sha": evaluation_sha,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "stage_counts": {"geometry": HOST_COUNTS["curie"]["geometry"]},
        "protected_data_access": False,
    }
    local_gate = {
        "status": "PASS",
        "host_id": "curie",
        "source_training_sha": SOURCE_TRAINING_SHA,
        "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
        "run_count": HOST_COUNTS["curie"]["runs"],
        "export_count": HOST_COUNTS["curie"]["source_exports"],
    }
    atomic_write_json(operational / "HOST_COMPLETE.json", terminal)
    atomic_write_json(
        operational / "seed_records.json", {"record_count": 1, "records": [{}]}
    )
    atomic_write_json(
        operational / "alignment_inventory.json",
        {"alignment_count": 0, "artifacts": []},
    )
    build_artifact_manifest(operational)
    atomic_write_json(state / "HOST_COMPUTE_COMPLETE.json", terminal)
    atomic_write_json(state / "LOCAL_SOURCE_GATE_PASS.json", local_gate)

    capsule = stage_task_f_local_compute_relay(
        artifact_root=artifact,
        state_root=state,
        host_id="curie",
        expected_evaluation_git_sha=evaluation_sha,
        collector_git_sha=collector_sha,
    )
    relay = json.loads((capsule / "LOCAL_COMPUTE_RELAY.json").read_text())
    assert relay["status"] == "LOCAL_COMPUTE_VERIFIED"
    assert not relay["source_remote_gate_required_for_this_relay"]
    assert not relay["final_research_terminal"]
    assert not relay["protected_data_access"]
    assert stage_task_f_local_compute_relay(
        artifact_root=artifact,
        state_root=state,
        host_id="curie",
        expected_evaluation_git_sha=evaluation_sha,
        collector_git_sha=collector_sha,
    ) == capsule

    uploads = []

    def fake_upload(source, *, hf_cli, bucket, destination, control_root):
        uploads.append((Path(source), destination))
        return {"status": "REMOTE_VERIFIED", "destination": destination}

    monkeypatch.setattr(orchestration, "upload_artifact_tree", fake_upload)
    result = publish_task_f_local_compute_relay(
        artifact_root=artifact,
        state_root=state,
        host_id="curie",
        expected_evaluation_git_sha=evaluation_sha,
        collector_git_sha=collector_sha,
        hf_cli="hf",
    )
    assert result["status"] == "LOCAL_COMPUTE_RELAY_VERIFIED"
    assert not result["source_remote_gate_satisfied"]
    assert not result["final_research_terminal"]
    assert uploads[0][1].endswith(
        f"/{evaluation_sha}/_local_compute/{collector_sha}/curie"
    )


@pytest.mark.parametrize(
    ("overlap", "expected_order"),
    [
        (False, ["remote", "compute", "publish"]),
        (True, ["local", "compute", "remote", "match", "publish"]),
    ],
)
def test_run_host_overlap_changes_only_source_gate_order(
    tmp_path, monkeypatch, overlap, expected_order
):
    import scripts.supervise_task_f_fresh_id as supervisor

    _, manifest = _manifest()
    remote_gate = validate_source_gate_documents(_source_documents())
    local_gate = {
        "status": "PASS",
        "host_id": "curie",
        "validation_sha256": remote_gate["hosts"]["curie"]["validation_sha256"],
    }
    order = []
    state = tmp_path / "state"
    state.mkdir()
    validation_input = tmp_path / "id_validation.npz"
    validation_input.write_bytes(b"fixture")
    args = SimpleNamespace(
        hf_cli=tmp_path / "hf",
        manifest=tmp_path / "manifest.json",
        overlap_source_upload=overlap,
        source_root=tmp_path / "source",
        host_id="curie",
        state_root=state,
        gate_poll_seconds=0.0,
        gate_timeout_hours=1.0,
        hf_command_timeout_seconds=1.0,
        id_validation_input=validation_input,
        dataset_config=tmp_path / "dataset.yaml",
        data_root=tmp_path / "data",
        id_train_input=tmp_path / "id_train.npz",
        expected_evaluation_git_sha="e" * 40,
        artifact_root=tmp_path / "artifacts",
        python=tmp_path / "python",
        batch_size=512,
        blas_threads=4,
        minimum_free_gb=0.0,
    )
    monkeypatch.setattr(supervisor, "_json", lambda path: manifest)
    monkeypatch.setattr(supervisor, "verify_hf_preflight", lambda *a, **k: None)
    monkeypatch.setattr(supervisor, "verify_id_input", lambda *a, **k: None)
    monkeypatch.setattr(supervisor, "build_id_input", lambda *a, **k: None)

    def local(**kwargs):
        order.append("local")
        return local_gate

    def remote(**kwargs):
        order.append("remote")
        return remote_gate

    def compute(**kwargs):
        order.append("compute")
        return {"status": "PASS"}

    def match(**kwargs):
        order.append("match")

    def publish(**kwargs):
        order.append("publish")
        return {"status": "PASS"}

    monkeypatch.setattr(supervisor, "validate_local_source_gate", local)
    monkeypatch.setattr(supervisor, "wait_for_global_source_gate", remote)
    monkeypatch.setattr(supervisor, "execute_task_f_host", compute)
    monkeypatch.setattr(supervisor, "validate_remote_source_gate_matches_local", match)
    monkeypatch.setattr(supervisor, "publish_task_f_host_result", publish)
    assert supervisor._run_host(args) == 0
    assert order == expected_order


def test_generic_pipeline_adapter_exact_resume_dependencies_and_resource_queues(tmp_path):
    manifest = validate_staged_manifest(
        {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "identity": {"adapter": "fixture", "version": 1},
            "jobs": [
                {"job_id": "a", "stage": "source", "resource": "gpu:0", "dependencies": []},
                {"job_id": "b", "stage": "source", "resource": "gpu:1", "dependencies": []},
                {"job_id": "c", "stage": "join", "resource": "cpu:0", "dependencies": ["a", "b"]},
            ],
        }
    )
    ledger = AtomicStageLedger(tmp_path / "ledger.json", manifest)
    active = 0
    peak = 0
    lock = threading.Lock()

    def worker(job):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {"value": job["job_id"]}

    execute_resource_queues(jobs=manifest["jobs"][:2], ledger=ledger, worker=worker)
    assert peak == 2
    execute_resource_queues(jobs=manifest["jobs"][2:], ledger=ledger, worker=worker)
    assert ledger.snapshot()["status"] == "PASS"
    resumed = AtomicStageLedger(tmp_path / "ledger.json", manifest)
    assert execute_resource_queues(jobs=manifest["jobs"], ledger=resumed, worker=worker) == []

    changed = copy.deepcopy(manifest)
    changed["identity"]["version"] = 2
    changed.pop("manifest_sha256")
    with pytest.raises(ValueError, match="identity"):
        AtomicStageLedger(tmp_path / "ledger.json", changed)


def test_generic_pipeline_rejects_unknown_dependency_and_cycle():
    with pytest.raises(ValueError, match="absent"):
        validate_staged_manifest(
            {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "identity": {},
                "jobs": [
                    {"job_id": "a", "stage": "x", "resource": "cpu", "dependencies": ["missing"]}
                ],
            }
        )
    with pytest.raises(ValueError, match="cycle"):
        validate_staged_manifest(
            {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "identity": {},
                "jobs": [
                    {"job_id": "a", "stage": "x", "resource": "cpu", "dependencies": ["b"]},
                    {"job_id": "b", "stage": "x", "resource": "cpu", "dependencies": ["a"]},
                ],
            }
        )


def _collector_roots(tmp_path, *, evaluation_sha, collector_sha=None):
    alignment_counts = {"curie": 270, "lise": 126, "precision_medicine": 261}
    seed_counts = {"curie": 768, "lise": 444, "precision_medicine": 708}
    roots = {}
    for host in HOST_COUNTS:
        root = tmp_path / host
        root.mkdir()
        destination = f"hf://fixture/{host}"
        terminal = {
            "status": "PASS",
            "host_id": host,
            "source_training_sha": SOURCE_TRAINING_SHA,
            "evaluation_git_sha": evaluation_sha,
            "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
            "stage_counts": {
                "geometry": HOST_COUNTS[host]["geometry"],
                "alignment": alignment_counts[host],
            },
            "protected_data_access": False,
        }
        atomic_write_json(root / "HOST_COMPLETE.json", terminal)
        atomic_write_json(
            root / "seed_records.json",
            {
                "record_count": seed_counts[host],
                "records": [{"host": host, "index": index} for index in range(seed_counts[host])],
            },
        )
        atomic_write_json(
            root / "alignment_inventory.json",
            {
                "alignment_count": alignment_counts[host],
                "artifacts": [
                    {
                        "output_identity_sha256": f"{index:064x}",
                        "manifest_sha256": f"{index + 1:064x}",
                    }
                    for index in range(alignment_counts[host])
                ],
            },
        )
        if collector_sha is not None:
            local_gate = {
                "status": "PASS",
                "host_id": host,
                "source_training_sha": SOURCE_TRAINING_SHA,
                "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
                "run_count": HOST_COUNTS[host]["runs"],
                "export_count": HOST_COUNTS[host]["source_exports"],
            }
            atomic_write_json(root / "HOST_COMPUTE_COMPLETE.json", terminal)
            atomic_write_json(root / "LOCAL_SOURCE_GATE_PASS.json", local_gate)
            atomic_write_json(
                root / "LOCAL_COMPUTE_RELAY.json",
                {
                    "schema_version": LOCAL_RELAY_SCHEMA_VERSION,
                    "status": "LOCAL_COMPUTE_VERIFIED",
                    "host_id": host,
                    "source_training_sha": SOURCE_TRAINING_SHA,
                    "evaluation_git_sha": evaluation_sha,
                    "collector_git_sha": collector_sha,
                    "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
                    "host_terminal_sha256": __import__("hashlib").sha256(
                        (root / "HOST_COMPLETE.json").read_bytes()
                    ).hexdigest(),
                    "host_compute_sha256": __import__("hashlib").sha256(
                        (root / "HOST_COMPUTE_COMPLETE.json").read_bytes()
                    ).hexdigest(),
                    "local_source_gate_sha256": __import__("hashlib").sha256(
                        (root / "LOCAL_SOURCE_GATE_PASS.json").read_bytes()
                    ).hexdigest(),
                    "source_remote_gate_required_for_this_relay": False,
                    "final_research_terminal": False,
                    "protected_data_access": False,
                },
            )
        artifact_manifest = build_artifact_manifest(root)
        marker = {
            "status": "REMOTE_VERIFIED",
            "destination": destination,
            "artifact_manifest_sha256": __import__("hashlib").sha256(
                (root / "artifact_manifest.json").read_bytes()
            ).hexdigest(),
        }
        (root / "REMOTE_COMPLETE.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        roots[host] = root
    return roots


def _stub_aggregation(monkeypatch):
    import oge.analysis.task_f_fresh_id as analysis

    monkeypatch.setattr(
        analysis,
        "build_aggregation_contract",
        lambda plan: {"expected_seeds_by_cell": {}, "contexts": []},
    )
    monkeypatch.setattr(
        analysis,
        "aggregate_paired_records",
        lambda **kwargs: {"status": "PASS", "aggregate_sha256": "a" * 64},
    )

    def fake_write(*, payload, output_directory):
        output_directory.mkdir(parents=True)
        return output_directory

    monkeypatch.setattr(analysis, "write_aggregation_artifacts", fake_write)


def test_central_collector_requires_three_complete_host_summaries(tmp_path, monkeypatch):
    evaluation_sha = "e" * 40
    roots = _collector_roots(tmp_path, evaluation_sha=evaluation_sha)
    _stub_aggregation(monkeypatch)
    terminal = collect_task_f_host_summaries(
        host_roots=roots,
        evaluation_plan={"fixture": True},
        output_directory=tmp_path / "central",
        expected_evaluation_git_sha=evaluation_sha,
    )
    assert terminal["status"] == "PASS"
    assert terminal["collection_mode"] == "remote_final"
    assert terminal["source_remote_gate_satisfied"]
    assert terminal["final_research_terminal"]
    assert terminal["seed_record_count"] == 1920
    assert terminal["alignment_count"] == 657
    assert terminal["id_equivalence"] == "PENDING_PROTECTED_ID_TEST"


def test_local_collector_uses_verified_compute_capsules_without_remote_source_gate(
    tmp_path, monkeypatch
):
    evaluation_sha = "e" * 40
    collector_sha = "c" * 40
    roots = _collector_roots(
        tmp_path, evaluation_sha=evaluation_sha, collector_sha=collector_sha
    )
    _stub_aggregation(monkeypatch)
    terminal = collect_task_f_host_summaries(
        host_roots=roots,
        evaluation_plan={"fixture": True},
        output_directory=tmp_path / "central-local",
        expected_evaluation_git_sha=evaluation_sha,
        collection_mode="local_compute_relay",
        expected_collector_git_sha=collector_sha,
    )
    assert terminal["status"] == "PASS"
    assert terminal["collection_mode"] == "local_compute_relay"
    assert not terminal["source_remote_gate_satisfied"]
    assert not terminal["final_research_terminal"]

    roots["curie"].joinpath("LOCAL_SOURCE_GATE_PASS.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum"):
        collect_task_f_host_summaries(
            host_roots=roots,
            evaluation_plan={"fixture": True},
            output_directory=tmp_path / "central-corrupt",
            expected_evaluation_git_sha=evaluation_sha,
            collection_mode="local_compute_relay",
            expected_collector_git_sha=collector_sha,
        )
