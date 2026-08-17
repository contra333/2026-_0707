import json
from pathlib import Path

import pytest

import oge.training.task_f_source_finalizer as finalizer
from oge.studies.supervisor import SupervisorBlockedError
from oge.training.task_f_source_finalizer import (
    EXPORT_GATE_COMPLETE,
    EXPORT_GATE_FAILED,
    EXPORT_GATE_WAITING,
    classify_export_states,
    index_source_export_artifacts,
    parse_hf_json_listing,
    wait_for_export_completion,
)


def _write(path: Path, value: str) -> None:
    path.write_text(value + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ([None, None], EXPORT_GATE_WAITING),
        (["RUNNING 1/31 run-a", None], EXPORT_GATE_WAITING),
        (["COMPLETE jobs=31", "RUNNING 2/31 run-b"], EXPORT_GATE_WAITING),
        (["COMPLETE jobs=31", "COMPLETE jobs=31"], EXPORT_GATE_COMPLETE),
        (["FAILED rc=1", None], EXPORT_GATE_FAILED),
        (["unexpected", "COMPLETE jobs=31"], EXPORT_GATE_FAILED),
    ],
)
def test_export_state_classification(states, expected):
    decision, _ = classify_export_states(states)
    assert decision == expected


def test_waiter_preserves_running_as_wait_then_completes(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    _write(control / "export_gpu0.status", "RUNNING 1/31 run-a")
    sleeps = []
    ticks = iter([0.0, 0.0, 1.0, 2.0])

    def finish_exports(seconds):
        sleeps.append(seconds)
        _write(control / "export_gpu0.status", "COMPLETE jobs=31")
        _write(control / "export_gpu1.status", "COMPLETE jobs=31")

    states = wait_for_export_completion(
        control_root=control,
        concurrency=2,
        poll_seconds=60.0,
        timeout_hours=1.0,
        sleep=finish_exports,
        monotonic=lambda: next(ticks),
    )
    assert states == ["COMPLETE jobs=31", "COMPLETE jobs=31"]
    assert sleeps == [60.0]
    assert "FAILED" not in (control / "host_finalizer.status").read_text()
    assert "decision=WAITING" in (control / "host_finalizer.log").read_text()
    assert "decision=COMPLETE" in (control / "host_finalizer.log").read_text()


def test_waiter_fails_only_on_terminal_failed_state(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    _write(control / "export_gpu0.status", "COMPLETE jobs=31")
    _write(control / "export_gpu1.status", "FAILED rc=9")
    ticks = iter([0.0, 0.0])
    with pytest.raises(SupervisorBlockedError, match="gpu1=FAILED rc=9"):
        wait_for_export_completion(
            control_root=control,
            concurrency=2,
            poll_seconds=0.0,
            timeout_hours=1.0,
            sleep=lambda _: None,
            monotonic=lambda: next(ticks),
        )
    assert (control / "host_finalizer.status").read_text().startswith(
        "FAILED export_state="
    )


def test_empty_export_state_vector_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        classify_export_states([])


def test_flat_output_identity_exports_are_grouped_by_manifest_run(monkeypatch, tmp_path):
    exports = tmp_path / "exports"
    first = exports / ("a" * 64)
    second = exports / ("b" * 64)
    third = exports / ("c" * 64)
    indexed = {
        ("run-a", "last", 10, "penultimate", "id_train"): first,
        ("run-a", "last", 60, "penultimate", "id_train"): second,
        ("run-b", "last", 200, "stage1", "id_train"): third,
    }
    monkeypatch.setattr(finalizer, "index_feature_artifacts", lambda roots: indexed)
    expected = {
        ("run-a", 10, "last", "penultimate", "id_train"),
        ("run-a", 60, "last", "penultimate", "id_train"),
        ("run-b", 200, "last", "stage1", "id_train"),
    }

    grouped = index_source_export_artifacts(
        exports,
        expected_export_keys=expected,
    )

    assert grouped == {"run-a": (first, second), "run-b": (third,)}


@pytest.mark.parametrize(
    "indexed",
    [
        {("run-a", "last", 10, "penultimate", "id_train"): Path("a" * 64)},
        {
            ("run-a", "last", 10, "penultimate", "id_train"): Path("a" * 64),
            ("run-x", "last", 10, "penultimate", "id_train"): Path("b" * 64),
        },
    ],
)
def test_source_export_index_rejects_missing_or_unexpected_coverage(
    monkeypatch, tmp_path, indexed
):
    monkeypatch.setattr(finalizer, "index_feature_artifacts", lambda roots: indexed)
    expected = {
        ("run-a", 10, "last", "penultimate", "id_train"),
        ("run-b", 200, "last", "stage1", "id_train"),
    }

    with pytest.raises(ValueError, match="source export coverage mismatch"):
        index_source_export_artifacts(
            tmp_path / "exports",
            expected_export_keys=expected,
        )


def test_source_export_index_preserves_duplicate_rejection(monkeypatch, tmp_path):
    def reject_duplicates(roots):
        raise ValueError("duplicate Task F feature artifact")

    monkeypatch.setattr(finalizer, "index_feature_artifacts", reject_duplicates)
    with pytest.raises(ValueError, match="duplicate Task F feature artifact"):
        index_source_export_artifacts(
            tmp_path / "exports",
            expected_export_keys=set(),
        )


@pytest.mark.parametrize("stdout", ["", "\n", "  \n\t"])
def test_empty_hf_listing_is_an_empty_prefix(stdout):
    assert parse_hf_json_listing(stdout) == []


def test_nonempty_hf_listing_requires_a_json_list_of_mappings():
    rows = [{"type": "file", "path": "servers/curie/artifact.json"}]
    assert parse_hf_json_listing(json.dumps(rows)) == rows
    with pytest.raises(ValueError, match="list of mappings"):
        parse_hf_json_listing('{"path": "artifact.json"}')
    with pytest.raises(ValueError, match="list of mappings"):
        parse_hf_json_listing('["artifact.json"]')
