import copy
import json
import threading
import time
from collections import Counter

import pytest

from oge.evaluation.task_f_fresh import build_fresh_evaluation_plan
from oge.evaluation.task_f_fresh_orchestration import (
    EXPECTED_SPECIFICATION_SHA256,
    HOST_COUNTS,
    SOURCE_TRAINING_SHA,
    build_task_f_pipeline_manifest,
    collect_task_f_host_summaries,
    validate_source_gate_documents,
    validate_task_f_pipeline_manifest,
)
from oge.studies.artifacts import atomic_write_json
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


def test_central_collector_requires_three_complete_host_summaries(tmp_path, monkeypatch):
    alignment_counts = {"curie": 270, "lise": 126, "precision_medicine": 261}
    seed_counts = {"curie": 768, "lise": 444, "precision_medicine": 708}
    roots = {}
    for host in HOST_COUNTS:
        root = tmp_path / host
        root.mkdir()
        destination = f"hf://fixture/{host}"
        atomic_write_json(
            root / "HOST_COMPLETE.json",
            {
                "status": "PASS",
                "host_id": host,
                "source_training_sha": SOURCE_TRAINING_SHA,
                "task_f_specification_sha256": EXPECTED_SPECIFICATION_SHA256,
                "stage_counts": {
                    "geometry": HOST_COUNTS[host]["geometry"],
                    "alignment": alignment_counts[host],
                },
            },
        )
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
    terminal = collect_task_f_host_summaries(
        host_roots=roots,
        evaluation_plan={"fixture": True},
        output_directory=tmp_path / "central",
    )
    assert terminal["status"] == "PASS"
    assert terminal["seed_record_count"] == 1920
    assert terminal["alignment_count"] == 657
    assert terminal["id_equivalence"] == "PENDING_PROTECTED_ID_TEST"
