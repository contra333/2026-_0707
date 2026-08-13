import copy
from pathlib import Path

import numpy as np
import pytest
import torch

import oge.feature_export.task_f as task_f
from oge.feature_export import (
    TASK_F_ARTIFACT_SCHEMA_VERSION,
    TASK_F_CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
    export_task_f_from_files,
    ordered_sample_id_sha256,
    specification_payload,
    validate_task_f_checkpoint_payload,
    validate_task_f_manifest,
    verify_task_f_artifact,
    write_task_f_artifact,
)
from oge.models import WideResNet
from oge.models.wide_resnet import (
    WRN_FEATURE_TAP_CONTRACT_VERSION,
    WRN_FEATURE_TAP_NAMES,
)
from oge.studies.hashing import canonical_json_bytes, canonical_sha256


CARD13_MINIMUM_FIELDS = {
    "run_id",
    "training_seed",
    "branch_policy",
    "total_weight_decay",
    "coupled_ratio",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "depth_tap",
    "dataset_split",
    "ordered_sample_id_sha256",
    "feature_shape",
    "feature_dtype",
    "oge_git_sha",
    "specification_sha256",
    "execution_only",
}


def _sha(character: str) -> str:
    return character * 64


def _model_config() -> dict:
    return {
        "name": "wrn28_10",
        "num_classes": 10,
        "depth": 28,
        "widen_factor": 10,
        "dropout_rate": 0.0,
        "init_policy": "msr_fan_in",
    }


def _sibling_members() -> dict:
    common = {
        "training_seed": 7,
        "initialization_sha256": _sha("a"),
        "data_stream_sha256": _sha("b"),
    }
    return {
        "zero": {
            **common,
            "run_id": "fixture-zero",
            "branch_policy": "zero_decay",
            "total_weight_decay": 0.0,
            "coupled_ratio": None,
        },
        "alpha_0": {
            **common,
            "run_id": "fixture-alpha-0",
            "branch_policy": "adam_coupled_decoupled",
            "total_weight_decay": 1e-4,
            "coupled_ratio": 0.0,
        },
        "alpha_0_5": {
            **common,
            "run_id": "fixture-alpha-0-5",
            "branch_policy": "adam_coupled_decoupled",
            "total_weight_decay": 1e-4,
            "coupled_ratio": 0.5,
        },
        "alpha_1": {
            **common,
            "run_id": "fixture-alpha-1",
            "branch_policy": "adam_coupled_decoupled",
            "total_weight_decay": 1e-4,
            "coupled_ratio": 1.0,
        },
    }


def _provenance(
    *,
    role: str = "alpha_0_5",
    epoch: int = 200,
    checkpoint_role: str = "last",
    execution_only: bool = True,
) -> dict:
    members = _sibling_members()
    current = members[role]
    return {
        "schema_version": TASK_F_CHECKPOINT_PROVENANCE_SCHEMA_VERSION,
        "run_id": current["run_id"],
        "training_seed": current["training_seed"],
        "branch_policy": current["branch_policy"],
        "total_weight_decay": current["total_weight_decay"],
        "coupled_ratio": current["coupled_ratio"],
        "checkpoint_epoch": epoch,
        "checkpoint_role": checkpoint_role,
        "oge_git_sha": "c" * 40,
        "execution_only": execution_only,
        "initialization_sha256": current["initialization_sha256"],
        "data_stream_sha256": current["data_stream_sha256"],
        "sibling_group_id": "fixture-seed-7-anchor",
        "sibling_role": role,
        "sibling_members": members,
        "model_config": _model_config(),
    }


def _runtime() -> dict:
    return {
        "python_version": "fixture-python",
        "python_implementation": "CPython",
        "numpy_version": "fixture-numpy",
        "pytorch_version": "fixture-pytorch",
        "device_type": "cpu",
        "device": "cpu",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "cpu_count": 4,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": None,
            "NUMEXPR_NUM_THREADS": None,
        },
        "numpy_blas": {
            "available": True,
            "blas": {"name": "fixture-blas", "version": "1"},
        },
    }


def _write_fixture_artifact(
    tmp_path: Path,
    *,
    tap: str = "stage1",
    provenance: dict | None = None,
    features: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
) -> Path:
    widths = {"stage1": 160, "stage2": 320, "stage3": 640, "penultimate": 640}
    if features is None:
        features = np.arange(2 * widths[tap], dtype=np.float32).reshape(2, widths[tap])
    if sample_ids is None:
        sample_ids = np.asarray(["fixture:0000", "fixture:0001"])
    return write_task_f_artifact(
        artifact_root=tmp_path,
        features=features,
        sample_ids=sample_ids,
        checkpoint_sha256=_sha("e"),
        checkpoint_provenance=_provenance() if provenance is None else provenance,
        depth_tap=tap,
        dataset_split="synthetic_id_fixture",
        runtime=_runtime(),
    )


def test_wrn_multi_depth_taps_preserve_exact_legacy_logits_and_locations():
    torch.manual_seed(11)
    model = WideResNet(num_classes=10, depth=28, widen_factor=10).eval()
    with torch.no_grad():
        model.bn.weight.zero_()
        model.bn.bias.fill_(1.0)
    images = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        ordinary_logits = model(images)
        feature_logits, penultimate = model(images, return_features=True)
        tap_logits, taps = model(images, return_feature_taps=True)

        value = model.conv1(images)
        value = model.block1(value)
        expected_stage1 = torch.flatten(model.avgpool(value), 1)
        value = model.block2(value)
        expected_stage2 = torch.flatten(model.avgpool(value), 1)
        value = model.block3(value)
        expected_stage3 = torch.flatten(model.avgpool(value), 1)
        expected_penultimate = torch.flatten(model.avgpool(model.relu(model.bn(value))), 1)

    assert model.feature_tap_contract_version == WRN_FEATURE_TAP_CONTRACT_VERSION
    assert WRN_FEATURE_TAP_CONTRACT_VERSION == "wrn_multi_depth_feature_taps_v1"
    assert tuple(taps) == WRN_FEATURE_TAP_NAMES == (
        "stage1",
        "stage2",
        "stage3",
        "penultimate",
    )
    assert model.feature_tap_dims == {
        "stage1": 160,
        "stage2": 320,
        "stage3": 640,
        "penultimate": 640,
    }
    torch.testing.assert_close(ordinary_logits, feature_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(ordinary_logits, tap_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(penultimate, taps["penultimate"], rtol=0.0, atol=0.0)
    for name, expected in {
        "stage1": expected_stage1,
        "stage2": expected_stage2,
        "stage3": expected_stage3,
        "penultimate": expected_penultimate,
    }.items():
        torch.testing.assert_close(taps[name], expected, rtol=0.0, atol=0.0)
        assert taps[name].dtype == torch.float32
        assert torch.isfinite(taps[name]).all()
    assert [tuple(taps[name].shape) for name in WRN_FEATURE_TAP_NAMES] == [
        (1, 160),
        (1, 320),
        (1, 640),
        (1, 640),
    ]
    assert torch.all(taps["penultimate"] == 1.0)
    assert not torch.equal(taps["stage3"], taps["penultimate"])
    assert all(not module._forward_hooks for module in model.modules())


def test_wrn_rejects_ambiguous_dual_feature_return_request():
    model = WideResNet(num_classes=2, depth=10, widen_factor=1).eval()
    with pytest.raises(ValueError, match="mutually exclusive"):
        model(
            torch.zeros(1, 3, 32, 32),
            return_features=True,
            return_feature_taps=True,
        )


def test_task_f_exporter_v1_rejects_gpu_devices_without_accessing_cuda():
    with pytest.raises(ValueError, match="CPU-only"):
        task_f.collect_runtime_provenance("cuda:0")


def test_task_f_artifact_has_deterministic_manifest_spec_and_separate_output_identity(
    tmp_path,
):
    first = _write_fixture_artifact(tmp_path / "first")
    second = _write_fixture_artifact(tmp_path / "second")
    first_verified = verify_task_f_artifact(first)
    second_verified = verify_task_f_artifact(second)
    first_manifest = first_verified["manifest"]
    second_manifest = second_verified["manifest"]

    assert first_manifest == second_manifest
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert first_manifest["schema_version"] == TASK_F_ARTIFACT_SCHEMA_VERSION
    assert first_manifest["specification_sha256"] == canonical_sha256(
        specification_payload()
    )
    assert CARD13_MINIMUM_FIELDS.issubset(first_manifest)
    assert first_manifest["execution_only"] is True
    assert (first / "manifest.json").read_bytes() == canonical_json_bytes(first_manifest) + b"\n"
    assert set(first_verified["verified_files"]) == {
        "features.npy",
        "manifest.json",
        "sample_ids.npy",
    }

    changed = np.arange(320, dtype=np.float32).reshape(2, 160)
    changed[0, 0] = -1.0
    third = _write_fixture_artifact(tmp_path / "third", features=changed)
    third_manifest = verify_task_f_artifact(third)["manifest"]
    assert third_manifest["specification_sha256"] == first_manifest["specification_sha256"]
    assert third_manifest["output_identity_sha256"] != first_manifest["output_identity_sha256"]


def test_task_f_sample_order_digest_and_artifact_checksum_are_verified(tmp_path):
    ordered = np.asarray(["fixture:a", "fixture:b"])
    reversed_ids = ordered[::-1]
    assert ordered_sample_id_sha256(ordered) != ordered_sample_id_sha256(reversed_ids)

    artifact = _write_fixture_artifact(tmp_path, sample_ids=ordered)
    np.save(artifact / "sample_ids.npy", reversed_ids, allow_pickle=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_task_f_artifact(artifact)


def test_task_f_manifest_rejects_every_missing_card13_minimum_field(tmp_path):
    manifest = verify_task_f_artifact(_write_fixture_artifact(tmp_path))["manifest"]
    for field in sorted(CARD13_MINIMUM_FIELDS):
        incomplete = copy.deepcopy(manifest)
        incomplete.pop(field)
        with pytest.raises(ValueError, match="missing fields"):
            validate_task_f_manifest(incomplete)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["sibling_members"].pop("alpha_1"), "complete zero/alpha quartet"),
        (
            lambda value: value["sibling_members"]["alpha_1"].__setitem__(
                "initialization_sha256", _sha("f")
            ),
            "initialization_sha256",
        ),
        (
            lambda value: value["sibling_members"]["alpha_0"].__setitem__(
                "data_stream_sha256", _sha("f")
            ),
            "data_stream_sha256",
        ),
        (
            lambda value: value["sibling_members"]["alpha_0_5"].__setitem__(
                "run_id", "wrong-current-run"
            ),
            "run_id differs",
        ),
        (
            lambda value: value["sibling_members"]["alpha_0"].__setitem__(
                "coupled_ratio", 0.5
            ),
            "coupled_ratio",
        ),
    ],
)
def test_task_f_checkpoint_rejects_missing_or_mismatched_sibling_identities(
    mutation, match
):
    provenance = _provenance()
    mutation(provenance)
    with pytest.raises(ValueError, match=match):
        task_f.validate_task_f_checkpoint_provenance(provenance)


def test_task_f_checkpoint_never_infers_missing_provenance_from_legacy_fields():
    legacy_payload = {
        "checkpoint_type": "last",
        "completed_epoch": 200,
        "model_state": {},
        "oge_git_sha": "c" * 40,
        "run_id": "legacy-run",
        "resolved_config": {"training": {"seed": 7}},
    }
    with pytest.raises(ValueError, match="task_f_provenance"):
        validate_task_f_checkpoint_payload(legacy_payload)


@pytest.mark.parametrize(
    "role,policy",
    [
        ("wd_1e_3_coupled", "adam"),
        ("wd_1e_3_decoupled", "adamw"),
        ("sgdm_coupled", "sgd"),
        ("sgdm_decoupled", "sgdw"),
    ],
)
def test_task_f_checkpoint_accepts_full_matrix_generic_sibling_roles(role, policy):
    provenance = _provenance(role="zero")
    generic_member = {
        "run_id": f"fixture-{role}",
        "training_seed": 7,
        "branch_policy": policy,
        "total_weight_decay": 1e-3,
        "coupled_ratio": None,
        "initialization_sha256": _sha("a"),
        "data_stream_sha256": _sha("b"),
    }
    provenance["sibling_members"] = {
        "zero": provenance["sibling_members"]["zero"],
        role: generic_member,
    }
    provenance["sibling_role"] = role
    for key in (
        "run_id",
        "training_seed",
        "branch_policy",
        "total_weight_decay",
        "coupled_ratio",
        "initialization_sha256",
        "data_stream_sha256",
    ):
        provenance[key] = generic_member[key]
    validated = task_f.validate_task_f_checkpoint_provenance(provenance)
    assert validated["sibling_members"][role]["branch_policy"] == policy


def test_task_f_checkpoint_rejects_generic_role_without_zero_reference():
    provenance = _provenance(role="alpha_0_5")
    provenance["sibling_members"].pop("zero")
    with pytest.raises(ValueError, match="contain zero"):
        task_f.validate_task_f_checkpoint_provenance(provenance)


def test_task_f_schema_and_exporter_reject_protected_references_before_file_access(
    tmp_path,
):
    protected_input = tmp_path / "protected_ood" / "svhn.npz"
    nonexistent_production_checkpoint = Path("/production/checkpoints/last.pt")
    with pytest.raises(ValueError, match="protected"):
        export_task_f_from_files(
            checkpoint_path=nonexistent_production_checkpoint,
            input_npz_path=protected_input,
            artifact_root=tmp_path / "artifacts",
            dataset_split="id_train",
            depth_tap="penultimate",
        )

    manifest = verify_task_f_artifact(_write_fixture_artifact(tmp_path / "valid"))[
        "manifest"
    ]
    protected_split = copy.deepcopy(manifest)
    protected_split["dataset_split"] = "id_test"
    with pytest.raises(ValueError, match="ID-only"):
        validate_task_f_manifest(protected_split)
    protected_role = copy.deepcopy(manifest)
    protected_role["dataset_role"] = "far_ood"
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_task_f_manifest(protected_role)
    protected_model_path = copy.deepcopy(manifest)
    protected_model_path["model_config"]["dataset_path"] = "/data/svhn"
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_task_f_manifest(protected_model_path)
    with pytest.raises(ValueError, match="protected"):
        ordered_sample_id_sha256(["svhn:test:0001"])


def test_task_f_epoch_200_supports_all_taps_and_other_snapshots_penultimate_only(
    tmp_path,
):
    widths = {"stage1": 160, "stage2": 320, "stage3": 640, "penultimate": 640}
    specification_hashes = set()
    output_hashes = set()
    for tap, width in widths.items():
        artifact = _write_fixture_artifact(
            tmp_path / tap,
            tap=tap,
            features=np.zeros((2, width), dtype=np.float32),
        )
        manifest = verify_task_f_artifact(artifact)["manifest"]
        assert manifest["depth_tap"] == tap
        specification_hashes.add(manifest["specification_sha256"])
        output_hashes.add(manifest["output_identity_sha256"])
    assert len(specification_hashes) == 1
    assert len(output_hashes) == 4

    snapshot = _provenance(epoch=120, checkpoint_role="snapshot")
    penultimate = _write_fixture_artifact(
        tmp_path / "snapshot-penultimate",
        tap="penultimate",
        provenance=snapshot,
        features=np.zeros((2, 640), dtype=np.float32),
    )
    assert verify_task_f_artifact(penultimate)["manifest"]["checkpoint_epoch"] == 120
    with pytest.raises(ValueError, match="penultimate only"):
        _write_fixture_artifact(
            tmp_path / "snapshot-stage3",
            tap="stage3",
            provenance=snapshot,
            features=np.zeros((2, 640), dtype=np.float32),
        )


def test_task_f_atomic_failure_leaves_no_partial_artifact(tmp_path, monkeypatch):
    original = task_f._write_npy
    calls = 0

    def fail_second_write(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture write failure")
        original(path, value)

    monkeypatch.setattr(task_f, "_write_npy", fail_second_write)
    with pytest.raises(RuntimeError, match="fixture write failure"):
        _write_fixture_artifact(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_task_f_completed_artifact_is_never_overwritten(tmp_path):
    first = _write_fixture_artifact(tmp_path)
    before = {path.name: path.read_bytes() for path in first.iterdir()}
    with pytest.raises(FileExistsError, match="already exists"):
        _write_fixture_artifact(tmp_path)
    after = {path.name: path.read_bytes() for path in first.iterdir()}
    assert after == before
    assert not any(path.name.startswith(".task-f-") for path in tmp_path.iterdir())


def test_task_f_high_level_export_uses_only_synthetic_cpu_checkpoint_and_id_input(
    tmp_path,
):
    torch.manual_seed(5)
    model = WideResNet(num_classes=10, depth=28, widen_factor=10).eval()
    provenance = _provenance(epoch=120, checkpoint_role="snapshot")
    checkpoint = tmp_path / "synthetic_snapshot.pt"
    torch.save(
        {
            "checkpoint_type": "snapshot",
            "completed_epoch": 120,
            "model_state": model.state_dict(),
            "oge_git_sha": provenance["oge_git_sha"],
            "run_id": provenance["run_id"],
            "task_f_provenance": provenance,
        },
        checkpoint,
    )
    input_npz = tmp_path / "synthetic_id_fixture.npz"
    np.savez(
        input_npz,
        images=np.zeros((1, 3, 32, 32), dtype=np.float32),
        sample_ids=np.asarray(["fixture:synthetic:0000"]),
        is_id=np.ones(1, dtype=np.bool_),
    )

    artifact = export_task_f_from_files(
        checkpoint_path=checkpoint,
        input_npz_path=input_npz,
        artifact_root=tmp_path / "artifacts",
        dataset_split="synthetic_id_fixture",
        depth_tap="penultimate",
        device="cpu",
        batch_size=1,
    )
    verified = verify_task_f_artifact(artifact)
    manifest = verified["manifest"]
    assert manifest["checkpoint_epoch"] == 120
    assert manifest["checkpoint_role"] == "snapshot"
    assert manifest["feature_shape"] == [1, 640]
    assert manifest["feature_dtype"] == "float32"
    assert manifest["runtime"]["device_type"] == "cpu"
    assert {
        "python_version",
        "pytorch_version",
        "numpy_version",
        "numpy_blas",
        "torch_num_threads",
        "torch_num_interop_threads",
        "thread_environment",
    }.issubset(manifest["runtime"])
    assert manifest["execution_only"] is True
    assert np.load(artifact / "features.npy", allow_pickle=False).shape == (1, 640)
