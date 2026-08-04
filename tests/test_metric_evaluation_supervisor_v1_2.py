from pathlib import Path

import pytest

from oge.evaluation.supervisor import (
    host_jobs,
    parse_source_roots,
    resolve_checkpoint_path,
)


def _fixture():
    jobs = [
        {
            "job_id": "job-a",
            "assigned_host": "curie",
            "source": {
                "source_execution_id": "seed0",
                "source_stage": "remaining",
                "artifact_relative_path": "trials/a/checkpoints/last.pt",
            },
        },
        {
            "job_id": "job-b",
            "assigned_host": "curie",
            "source": {
                "source_execution_id": "followup",
                "source_stage": "complete",
                "artifact_relative_path": "trials/b/checkpoints/best_val.pt",
            },
        },
    ]
    inventory = {"checkpoint_jobs": jobs}
    execution = {
        "hosts": {"curie": {"checkpoint_job_ids": ["job-a", "job-b"]}}
    }
    return inventory, execution


def test_source_root_mapping_resolves_only_frozen_source_relative_path():
    roots = parse_source_roots(
        ["seed0:remaining=/srv/seed0", "followup:complete=/srv/followup"]
    )
    inventory, execution = _fixture()
    jobs = host_jobs(inventory, execution, host_id="curie")

    assert resolve_checkpoint_path(jobs[0], roots) == Path(
        "/srv/seed0/trials/a/checkpoints/last.pt"
    )
    assert resolve_checkpoint_path(jobs[1], roots) == Path(
        "/srv/followup/trials/b/checkpoints/best_val.pt"
    )


def test_source_root_and_host_assignment_drift_are_rejected():
    inventory, execution = _fixture()
    with pytest.raises(ValueError, match="absolute"):
        parse_source_roots(["seed0:remaining=relative/path"])
    with pytest.raises(ValueError, match="missing"):
        resolve_checkpoint_path(
            inventory["checkpoint_jobs"][0],
            {"followup:complete": Path("/srv/followup")},
        )
    with pytest.raises(ValueError, match="absent"):
        host_jobs(inventory, execution, host_id="lise")
