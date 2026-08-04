import copy
import json
from pathlib import Path

import pytest

from oge.studies.aggregation import validate_followup_aggregate
from oge.studies.hashing import canonical_sha256

ROOT = Path(__file__).parents[1]
AGGREGATE_PATH = (
    ROOT
    / "results/wrn28_10_optimizer_hpo_v1_2/followup_20260730/aggregate.json"
)


def test_checked_in_followup_aggregate_is_complete_and_hash_valid():
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    validate_followup_aggregate(aggregate)
    assert aggregate["counts"] == {
        "aggregate_records": 42,
        "new_followup": 30,
        "reused_seed0": 12,
        "unique_config_seed_identities": 42,
        "unique_configurations": 14,
    }
    assert {item["status"] for item in aggregate["host_completion"].values()} == {
        "REMOTE_VERIFIED"
    }
    assert all(
        item["dry_run_summary"]["deletes"] == 0
        for item in aggregate["host_completion"].values()
    )
    assert len(aggregate["pair_statistics"]) == 4


def test_followup_aggregate_hash_rejects_mutation():
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    aggregate["records"][0]["final_validation"]["accuracy"] = 0.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_followup_aggregate(aggregate)


def test_followup_aggregate_rejects_duplicate_config_seed_identity():
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    duplicate = copy.deepcopy(aggregate["records"][0])
    aggregate["records"][1]["config_hash"] = duplicate["config_hash"]
    aggregate["records"][1]["training_seed"] = duplicate["training_seed"]
    unhashed = dict(aggregate)
    unhashed.pop("aggregate_hash")
    aggregate["aggregate_hash"] = canonical_sha256(unhashed)
    with pytest.raises(ValueError, match="duplicate identities"):
        validate_followup_aggregate(aggregate)
