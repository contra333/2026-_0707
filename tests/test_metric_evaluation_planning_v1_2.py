import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

from oge.evaluation.planning import (
    APPROVED_HOSTS,
    DEFINITION_VERSION,
    RUNTIME_MERGE_GIT_SHA,
    build_checkpoint_inventory,
    build_execution_manifest,
    load_materialized_evaluation_plan,
    materialize_evaluation_plan,
    render_evaluation_plan_files,
    validate_checkpoint_inventory,
    validate_execution_manifest,
    verify_materialized_evaluation_plan,
)
from oge.studies.hashing import canonical_sha256

ROOT = Path(__file__).parents[1]
AGGREGATE = (
    ROOT
    / "results/wrn28_10_optimizer_hpo_v1_2/followup_20260730/aggregate.json"
)
DATASET_CONFIG = ROOT / "configs/datasets/oge_cifar10_holdout_v1.yaml"
PLAN_ROOT = ROOT / "configs/evaluation/wrn28_10_cifar10_metric_v1_2"


def _inventory():
    return build_checkpoint_inventory(
        aggregate_path=AGGREGATE,
        aggregate_sha256_path=AGGREGATE.with_name(f"{AGGREGATE.name}.sha256"),
        dataset_config_path=DATASET_CONFIG,
    )


def _rehash_inventory(inventory):
    inventory = copy.deepcopy(inventory)
    inventory.pop("inventory_hash", None)
    inventory["inventory_hash"] = canonical_sha256(inventory)
    return inventory


def _rehash_execution(execution):
    execution = copy.deepcopy(execution)
    execution.pop("execution_hash", None)
    execution["execution_hash"] = canonical_sha256(execution)
    return execution


def test_checked_in_plan_is_deterministic_and_checksum_valid():
    expected = _inventory()
    inventory, execution = load_materialized_evaluation_plan(PLAN_ROOT)

    assert inventory == expected
    verify_materialized_evaluation_plan(PLAN_ROOT, expected)
    assert execution["inventory"]["inventory_hash"] == inventory["inventory_hash"]
    assert execution["state"] == {
        "execution_started": False,
        "protected_accessed": False,
        "server_sync": "not_started",
        "hf_upload": "not_started",
    }


def test_inventory_freezes_42_sources_30_role_identities_and_60_jobs():
    inventory = _inventory()
    assert inventory["counts"] == {
        "source_training_identities": 42,
        "source_configurations": 14,
        "authorized_training_identities": 30,
        "authorized_configurations": 10,
        "excluded_pair_only_training_identities": 12,
        "excluded_pair_only_checkpoint_references": 24,
        "checkpoint_jobs": 60,
    }
    identities = inventory["training_identities"]
    authorized = [item for item in identities if item["disposition"] == "authorized_role"]
    excluded = [item for item in identities if item["disposition"] == "excluded_pair_only"]
    assert len(authorized) == 30
    assert len(excluded) == 12
    assert all(any(label.startswith("role:") for label in item["labels"]) for item in authorized)
    assert all(not any(label.startswith("role:") for label in item["labels"]) for item in excluded)

    config_seeds = defaultdict(set)
    for item in authorized:
        config_seeds[item["config_hash"]].add(item["training_seed"])
    assert len(config_seeds) == 10
    assert all(seeds == {0, 1, 2} for seeds in config_seeds.values())
    assert len(inventory["checkpoint_jobs"]) == 60
    assert Counter(job["checkpoint_role"] for job in inventory["checkpoint_jobs"]) == {
        "last": 30,
        "best_val": 30,
    }


def test_checkpoint_roles_preserve_aggregate_identity_and_separate_outputs():
    inventory = _inventory()
    aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    aggregate_by_identity = {
        (row["config_hash"], row["training_seed"]): row for row in aggregate["records"]
    }
    outputs = set()
    shas = set()
    for job in inventory["checkpoint_jobs"]:
        source = aggregate_by_identity[(job["config_hash"], job["training_seed"])]
        reference = source["checkpoints"][job["checkpoint_role"]]
        assert job["checkpoint_sha256"] == reference["sha256"]
        assert job["completed_epoch"] == reference["completed_epoch"]
        assert job["reporting_role"] == {
            "last": "confirmatory_primary",
            "best_val": "deployment_control",
        }[job["checkpoint_role"]]
        assert job["output_namespace"] == (
            f"evaluations/{DEFINITION_VERSION}/{job['checkpoint_sha256']}/"
        )
        assert job["checkpoint_sha256"] not in shas
        assert job["output_namespace"] not in outputs
        shas.add(job["checkpoint_sha256"])
        outputs.add(job["output_namespace"])


def test_dataset_policy_is_exact_and_excludes_compatibility_splits():
    policy = _inventory()["dataset_policy"]
    assert {key: item["count"] for key, item in policy["id_splits"].items()} == {
        "id_train": 45000,
        "id_validation": 5000,
        "id_test": 10000,
    }
    assert policy["near_order"] == ["cifar100", "tin"]
    assert policy["far_order"] == ["mnist", "svhn", "texture", "places365"]
    assert set(policy["excluded_splits"]) == {
        "id_test_openood",
        "ood_validation_tin",
    }
    execution_keys = set(policy["id_splits"]) | set(policy["ood_splits"])
    assert "id_test_openood" not in execution_keys
    assert "ood_validation_tin" not in execution_keys


def test_source_local_execution_is_20_each_with_4_2_4_concurrency():
    inventory = _inventory()
    rendered = render_evaluation_plan_files(inventory)
    execution = yaml.safe_load(rendered["execution.yaml"])
    assert tuple(execution["hosts"]) == APPROVED_HOSTS
    assert {host: value["checkpoint_job_count"] for host, value in execution["hosts"].items()} == {
        "curie": 20,
        "lise": 20,
        "precision_medicine": 20,
    }
    assert {host: value["concurrency"] for host, value in execution["hosts"].items()} == {
        "curie": 4,
        "lise": 2,
        "precision_medicine": 4,
    }
    assert execution["hosts"]["curie"]["coordinator"] is True
    assert execution["hosts"]["lise"]["coordinator"] is False
    assert execution["hosts"]["precision_medicine"]["coordinator"] is False
    jobs = {job["job_id"]: job for job in inventory["checkpoint_jobs"]}
    for host_id, host in execution["hosts"].items():
        assert all(jobs[job_id]["assigned_host"] == host_id for job_id in host["checkpoint_job_ids"])


def test_materialization_is_byte_identical_and_matches_checked_in_files(tmp_path):
    inventory = _inventory()
    first = tmp_path / "first"
    second = tmp_path / "second"
    materialize_evaluation_plan(inventory, first)
    materialize_evaluation_plan(inventory, second)
    for name in (
        "checkpoint_inventory.json",
        "checkpoint_inventory.json.sha256",
        "execution.yaml",
        "execution.yaml.sha256",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert (first / name).read_bytes() == (PLAN_ROOT / name).read_bytes()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("inventory_hash", "hash mismatch"),
        ("duplicate_output", "output namespace collision"),
        ("pair_authorized", "authorization counts mismatch"),
        ("wrong_role", "invalid role"),
        ("wrong_dataset", "near-OOD policy mismatch"),
    ],
)
def test_inventory_validator_rejects_identity_and_policy_drift(mutation, message):
    inventory = _inventory()
    if mutation == "inventory_hash":
        inventory["counts"]["checkpoint_jobs"] = 59
    elif mutation == "duplicate_output":
        inventory["checkpoint_jobs"][1]["output_namespace"] = inventory["checkpoint_jobs"][0]["output_namespace"]
        inventory = _rehash_inventory(inventory)
    elif mutation == "pair_authorized":
        excluded = next(item for item in inventory["training_identities"] if item["disposition"] == "excluded_pair_only")
        excluded["disposition"] = "authorized_role"
        excluded["exclusion_reason"] = None
        inventory = _rehash_inventory(inventory)
    elif mutation == "wrong_role":
        inventory["checkpoint_jobs"][0]["checkpoint_role"] = "snapshot"
        inventory = _rehash_inventory(inventory)
    else:
        inventory["dataset_policy"]["near_order"] = ["tin", "cifar100"]
        inventory["dataset_policy_hash"] = canonical_sha256(inventory["dataset_policy"])
        inventory = _rehash_inventory(inventory)
    with pytest.raises(ValueError, match=message):
        validate_checkpoint_inventory(inventory)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("started", "must not record started"),
        ("duplicate", "exactly once"),
        ("cross_host", "not source-local"),
        ("concurrency", "concurrency mismatch"),
        ("delete", "must forbid deletes"),
    ],
)
def test_execution_validator_rejects_launch_and_assignment_drift(mutation, message):
    inventory = _inventory()
    rendered = render_evaluation_plan_files(inventory)
    execution = yaml.safe_load(rendered["execution.yaml"])
    if mutation == "started":
        execution["state"]["execution_started"] = True
    elif mutation == "duplicate":
        execution["hosts"]["lise"]["checkpoint_job_ids"][0] = execution["hosts"]["curie"]["checkpoint_job_ids"][0]
    elif mutation == "cross_host":
        curie_id = execution["hosts"]["curie"]["checkpoint_job_ids"][0]
        lise_id = execution["hosts"]["lise"]["checkpoint_job_ids"][0]
        execution["hosts"]["curie"]["checkpoint_job_ids"][0] = lise_id
        execution["hosts"]["lise"]["checkpoint_job_ids"][0] = curie_id
    elif mutation == "concurrency":
        execution["hosts"]["lise"]["concurrency"] = 4
    else:
        execution["output"]["upload_policy"] = "sync_with_delete"
    execution = _rehash_execution(execution)
    with pytest.raises(ValueError, match=message):
        validate_execution_manifest(execution, inventory)


def test_every_job_binds_runtime_dataset_float64_and_future_authorization():
    inventory = _inventory()
    for job in inventory["checkpoint_jobs"]:
        assert job["bindings"] == {
            "definition_version": DEFINITION_VERSION,
            "runtime_merge_git_sha": RUNTIME_MERGE_GIT_SHA,
            "dataset_policy_hash": inventory["dataset_policy_hash"],
            "membership_manifest_sha256": inventory["dataset_policy"]["membership_manifest"]["sha256"],
            "fit_dtype": "float64",
            "protected_authorization_policy": "separate_active_execution_issue_required",
        }
        assert job["planned_status"] == "not_started"
        assert job["source"]["hf_checkpoint_uri"].endswith(
            f"/checkpoints/{'last.pt' if job['checkpoint_role'] == 'last' else 'best_val.pt'}"
        )
