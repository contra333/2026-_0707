"""Reusable exact-once orchestration for immutable staged jobs.

The module deliberately knows nothing about Task F, datasets, checkpoints, or
accelerators.  Study adapters provide an immutable job graph and worker
callbacks; this layer validates dependencies and preserves atomic execution
state across process restarts.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from oge.studies.artifacts import atomic_write_json
from oge.studies.hashing import canonical_sha256


PIPELINE_SCHEMA_VERSION = "oge_staged_pipeline_v1"
LEDGER_SCHEMA_VERSION = "oge_staged_pipeline_ledger_v1"
TERMINAL_STATUSES = {"PASS", "FAILED"}


def validate_staged_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized immutable staged-job manifest."""

    if value.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise ValueError("unsupported staged pipeline manifest")
    identity = value.get("identity")
    jobs = value.get("jobs")
    if not isinstance(identity, Mapping) or not isinstance(jobs, Sequence):
        raise ValueError("staged pipeline requires identity and jobs")
    normalized_jobs: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for raw in jobs:
        if not isinstance(raw, Mapping):
            raise ValueError("pipeline jobs must be mappings")
        job = dict(raw)
        job_id = str(job.get("job_id", ""))
        stage = str(job.get("stage", ""))
        resource = str(job.get("resource", ""))
        dependencies = [str(item) for item in job.get("dependencies", ())]
        if not job_id or not stage or not resource:
            raise ValueError("pipeline jobs require job_id, stage, and resource")
        if job_id in by_id:
            raise ValueError(f"duplicate pipeline job_id {job_id!r}")
        if len(dependencies) != len(set(dependencies)) or job_id in dependencies:
            raise ValueError(f"invalid dependencies for {job_id!r}")
        job.update(
            {
                "job_id": job_id,
                "stage": stage,
                "resource": resource,
                "dependencies": dependencies,
            }
        )
        by_id[job_id] = job
        normalized_jobs.append(job)
    unknown = sorted(
        {dependency for job in normalized_jobs for dependency in job["dependencies"]}
        - set(by_id)
    )
    if unknown:
        raise ValueError(f"pipeline dependencies are absent: {unknown}")

    indegree = {job_id: 0 for job_id in by_id}
    children: dict[str, list[str]] = defaultdict(list)
    for job in normalized_jobs:
        for dependency in job["dependencies"]:
            indegree[job["job_id"]] += 1
            children[dependency].append(job["job_id"])
    queue = deque(sorted(job_id for job_id, degree in indegree.items() if degree == 0))
    visited: list[str] = []
    while queue:
        job_id = queue.popleft()
        visited.append(job_id)
        for child in sorted(children[job_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(visited) != len(by_id):
        raise ValueError("pipeline dependency graph contains a cycle")

    normalized = dict(value)
    normalized["identity"] = dict(identity)
    normalized["jobs"] = normalized_jobs
    supplied_hash = normalized.pop("manifest_sha256", None)
    manifest_hash = canonical_sha256(normalized)
    if supplied_hash is not None and supplied_hash != manifest_hash:
        raise ValueError("staged pipeline manifest SHA-256 mismatch")
    normalized["manifest_sha256"] = manifest_hash
    return normalized


class AtomicStageLedger:
    """Thread-safe no-reinterpretation state for one immutable job graph."""

    def __init__(self, path: str | Path, manifest: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.manifest = validate_staged_manifest(manifest)
        self._lock = threading.Lock()
        if self.path.exists():
            import json

            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state.get("schema_version") != LEDGER_SCHEMA_VERSION:
                raise ValueError("unsupported staged pipeline ledger")
            if state.get("manifest_sha256") != self.manifest["manifest_sha256"]:
                raise ValueError("preserved pipeline ledger identity mismatch")
            expected = {job["job_id"] for job in self.manifest["jobs"]}
            if set(state.get("jobs", {})) != expected:
                raise ValueError("preserved pipeline ledger job set mismatch")
            self.state = state
        else:
            self.state = {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "manifest_sha256": self.manifest["manifest_sha256"],
                "identity": dict(self.manifest["identity"]),
                "status": "PENDING",
                "jobs": {
                    job["job_id"]: {
                        "status": "PENDING",
                        "attempt_count": 0,
                        "result": None,
                        "failure": None,
                    }
                    for job in self.manifest["jobs"]
                },
            }
            atomic_write_json(self.path, self.state)

    def job_status(self, job_id: str) -> str:
        with self._lock:
            return str(self.state["jobs"][job_id]["status"])

    def dependencies_pass(self, job: Mapping[str, Any]) -> bool:
        with self._lock:
            return all(
                self.state["jobs"][dependency]["status"] == "PASS"
                for dependency in job["dependencies"]
            )

    def start(self, job_id: str, **runtime: Any) -> bool:
        """Mark a nonterminal job running; return False for preserved PASS."""

        with self._lock:
            record = self.state["jobs"][job_id]
            if record["status"] == "PASS":
                return False
            if record["status"] == "FAILED":
                raise RuntimeError(f"job {job_id} has a preserved failure")
            record.update(
                {
                    "status": "RUNNING",
                    "attempt_count": int(record["attempt_count"]) + 1,
                    "runtime": runtime,
                    "failure": None,
                }
            )
            self.state["status"] = "RUNNING"
            atomic_write_json(self.path, self.state)
            return True

    def pass_job(self, job_id: str, result: Mapping[str, Any]) -> None:
        with self._lock:
            record = self.state["jobs"][job_id]
            record.update({"status": "PASS", "result": dict(result), "failure": None})
            self._refresh_status()

    def fail_job(self, job_id: str, error: BaseException | str) -> None:
        with self._lock:
            record = self.state["jobs"][job_id]
            record.update(
                {
                    "status": "FAILED",
                    "failure": {
                        "type": type(error).__name__ if isinstance(error, BaseException) else "error",
                        "detail": str(error),
                    },
                }
            )
            self._refresh_status()

    def _refresh_status(self) -> None:
        statuses = {record["status"] for record in self.state["jobs"].values()}
        self.state["status"] = (
            "FAILED"
            if "FAILED" in statuses
            else "PASS"
            if statuses == {"PASS"}
            else "RUNNING"
            if "RUNNING" in statuses or "PASS" in statuses
            else "PENDING"
        )
        atomic_write_json(self.path, self.state)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self.state,
                "jobs": {
                    key: dict(value) for key, value in self.state["jobs"].items()
                },
            }


def execute_resource_queues(
    *,
    jobs: Sequence[Mapping[str, Any]],
    ledger: AtomicStageLedger,
    worker: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Run one ordered queue per resource while preserving exact job identity."""

    queues: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for job in jobs:
        queues[str(job["resource"])].append(job)
    results: list[dict[str, Any]] = []
    result_lock = threading.Lock()

    def run_queue(resource: str, queue: Sequence[Mapping[str, Any]]) -> None:
        for job in queue:
            job_id = str(job["job_id"])
            if ledger.job_status(job_id) == "PASS":
                continue
            if not ledger.dependencies_pass(job):
                error = RuntimeError(f"job {job_id} dependencies are not PASS")
                ledger.fail_job(job_id, error)
                raise error
            try:
                if not ledger.start(job_id, resource=resource):
                    continue
                result = dict(worker(job))
                ledger.pass_job(job_id, result)
            except BaseException as exc:
                ledger.fail_job(job_id, exc)
                raise
            with result_lock:
                results.append({"job_id": job_id, **result})

    if not queues:
        return results
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures = [
            pool.submit(run_queue, resource, queue)
            for resource, queue in sorted(queues.items())
        ]
        for future in futures:
            future.result()
    return sorted(results, key=lambda row: row["job_id"])
