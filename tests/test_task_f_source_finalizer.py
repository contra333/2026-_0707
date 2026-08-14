from pathlib import Path

import pytest

from oge.studies.supervisor import SupervisorBlockedError
from oge.training.task_f_source_finalizer import (
    EXPORT_GATE_COMPLETE,
    EXPORT_GATE_FAILED,
    EXPORT_GATE_WAITING,
    classify_export_states,
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
