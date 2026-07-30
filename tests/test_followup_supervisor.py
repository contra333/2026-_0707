import json
import subprocess
from pathlib import Path

import pytest

from oge.studies.artifacts import atomic_write_json, sha256_file
from oge.studies.supervisor import (
    build_artifact_manifest,
    parse_hf_sync_dry_run,
    update_supervisor_state,
    upload_artifact_tree,
)


def test_supervisor_state_is_atomic_and_checksummed(tmp_path):
    path = tmp_path / "state.json"
    state = update_supervisor_state(path, {"host_id": "curie"}, "RUNNING")
    assert state["status"] == "RUNNING"
    assert json.loads(path.read_text())["host_id"] == "curie"
    assert path.with_name("state.json.sha256").is_file()


def test_artifact_manifest_hashes_every_immutable_file(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    (root / "checkpoint.pt").write_bytes(b"checkpoint")
    atomic_write_json(root / "trial.json", {"status": "completed"})
    manifest = build_artifact_manifest(root)
    by_path = {row["path"]: row for row in manifest["files"]}
    assert set(by_path) == {
        "checkpoint.pt",
        "trial.json",
        "trial.json.sha256",
    }
    assert by_path["checkpoint.pt"]["sha256"] == sha256_file(
        root / "checkpoint.pt"
    )
    assert (root / "artifact_manifest.json.sha256").is_file()


def test_hf_dry_run_requires_zero_deletes_and_upload_only_actions():
    valid = "\n".join(
        [
            json.dumps(
                {
                    "type": "header",
                    "summary": {
                        "uploads": 1,
                        "downloads": 0,
                        "deletes": 0,
                        "skips": 0,
                    },
                }
            ),
            json.dumps(
                {
                    "type": "operation",
                    "action": "upload",
                    "path": "trial.json",
                }
            ),
        ]
    )
    assert parse_hf_sync_dry_run(valid)["header"]["summary"]["uploads"] == 1
    deleted = valid.replace('"deletes": 0', '"deletes": 1')
    with pytest.raises(ValueError, match="zero deletes"):
        parse_hf_sync_dry_run(deleted)
    downloaded = valid.replace('"action": "upload"', '"action": "download"')
    with pytest.raises(ValueError, match="non-upload"):
        parse_hf_sync_dry_run(downloaded)


def test_upload_uses_no_delete_plan_apply_listing_and_last_marker(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    atomic_write_json(study / "trial.json", {"status": "completed"})
    build_artifact_manifest(study)
    control = tmp_path / "control"
    destination = (
        "hf://buckets/contra333/ICLR_RUN/servers/curie/"
        "role_pair_followup_20260730"
    )
    prefix = "servers/curie/role_pair_followup_20260730"
    calls = []
    remote = {}

    def fake_run(command, **kwargs):
        command = list(command)
        calls.append(command)
        stdout = ""
        if command[1:3] == ["buckets", "list"]:
            stdout = json.dumps(
                [
                    {"type": "file", "path": path, **metadata}
                    for path, metadata in sorted(remote.items())
                ]
            )
        elif "--dry-run" in command:
            local_files = [path for path in study.rglob("*") if path.is_file()]
            rows = [
                {
                    "type": "header",
                    "summary": {
                        "uploads": len(local_files),
                        "downloads": 0,
                        "deletes": 0,
                        "skips": 0,
                    },
                }
            ]
            rows.extend(
                {
                    "type": "operation",
                    "action": "upload",
                    "path": path.relative_to(study).as_posix(),
                    "size": path.stat().st_size,
                }
                for path in local_files
            )
            stdout = "\n".join(json.dumps(row) for row in rows) + "\n"
        elif "--plan" in command:
            plan_path = Path(command[command.index("--plan") + 1])
            local_files = [path for path in study.rglob("*") if path.is_file()]
            rows = [
                {
                    "type": "header",
                    "summary": {
                        "uploads": len(local_files),
                        "downloads": 0,
                        "deletes": 0,
                        "skips": 0,
                    },
                }
            ]
            rows.extend(
                {
                    "type": "operation",
                    "action": "upload",
                    "path": path.relative_to(study).as_posix(),
                    "size": path.stat().st_size,
                }
                for path in local_files
            )
            plan_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
        elif "--apply" in command:
            for path in study.rglob("*"):
                if path.is_file():
                    remote[f"{prefix}/{path.relative_to(study).as_posix()}"] = {
                        "size": path.stat().st_size,
                        "xet_hash": "fake",
                    }
        elif command[1:3] == ["buckets", "cp"]:
            marker = Path(command[3])
            remote[f"{prefix}/REMOTE_COMPLETE.json"] = {
                "size": marker.stat().st_size,
                "xet_hash": "marker",
            }
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    evidence = upload_artifact_tree(
        study,
        hf_cli="/fake/hf",
        bucket="contra333/ICLR_RUN",
        destination=destination,
        control_root=control,
        run=fake_run,
    )
    assert evidence["status"] == "REMOTE_VERIFIED"
    assert f"{prefix}/REMOTE_COMPLETE.json" in remote
    sync_commands = [
        command for command in calls if command[1:3] == ["buckets", "sync"]
    ]
    assert "--dry-run" in sync_commands[0]
    assert "--no-delete" in sync_commands[0]
    assert "--plan" in sync_commands[1]
    assert "--no-delete" in sync_commands[1]
    assert "--apply" in sync_commands[2]
