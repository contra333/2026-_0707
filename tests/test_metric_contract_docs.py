import re
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).parents[1]
REFERENCE_DIR = REPOSITORY_ROOT / "docs/reference_cards"
CONTRACT_PATH = REFERENCE_DIR / "11_metric_contract_v1_2.md"


def test_metric_contract_has_frozen_scope_and_decision_complete_markers():
    contract = CONTRACT_PATH.read_text(encoding="utf-8")

    required = (
        "definition_version: metric_contract_v1.2",
        "`last.pt` | optimizer에서 terminal geometry/OOD로 이어지는 confirmatory primary",
        "`best_val.pt` | validation-selected deployment control",
        "`id_test_primary` | `id_test` | 10,000",
        'np.quantile(id_scores, 0.05, method="linear")',
        "GDA-ClassDensity",
        "`DIM=320`",
        "fixed `neco_dim=100`",
        "`geometry/nc1_pinv`",
        "covariance_entropy_rank",
        "covariance_trace_top_rank",
        "covariance_participation_ratio",
        "`success | degenerate | failed`",
        "runtime implementation pending",
    )
    for marker in required:
        assert marker in contract

    for unresolved_marker in ("TBD", "UNSPECIFIED", "id_test_compat"):
        assert unresolved_marker not in contract


def test_metric_contract_local_markdown_links_resolve():
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", contract)
    local_links = [link for link in links if "://" not in link and not link.startswith("#")]
    assert local_links
    for link in local_links:
        assert (CONTRACT_PATH.parent / link).resolve().is_file(), link


def test_metric_contract_source_pins_are_full_commits():
    sources = yaml.safe_load(
        (REPOSITORY_ROOT / "docs/sources.lock.yaml").read_text(encoding="utf-8")
    )["sources"]
    expected = {
        "temperature_scaling_official_repository": "ce1154ec7d5235eee046f28875c3dd8421f11d45",
        "fd_shifts_repository": "c4467aec134e99691359da209f811d91283fc1e3",
        "mahalanobis_pp_official_repository": "53de550be7f88acad959f5fdd8f2430fb9a8b6ea",
        "vim_official_repository": "dabf9e5b242dbd31c15e09ff12af3d11f009f79c",
        "neco_official_repository": "6a55640669f0aad3e23f45ce2f6a8e6400c929ba",
        "dadapy_repository": "c37e52ca3cbe00107696720e5e3efa09f8370bbb",
    }
    for key, commit in expected.items():
        assert sources[key]["commit"] == commit
        assert re.fullmatch(r"[0-9a-f]{40}", sources[key]["commit"])
    assert sources["local_metric_contract_v1_2_reference_card"]["path"] == (
        "docs/reference_cards/11_metric_contract_v1_2.md"
    )


def test_dataset_roles_and_membership_identity_stay_frozen():
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/datasets/oge_cifar10_holdout_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    datasets = config["datasets"]
    assert datasets["id_train"]["expected_count"] == 45_000
    assert datasets["id_validation"]["expected_count"] == 5_000
    assert datasets["id_test"]["expected_count"] == 10_000
    assert datasets["id_test"]["role"] == "metric_contract_primary"
    assert datasets["id_test_openood"]["expected_count"] == 9_000
    assert datasets["id_test_openood"]["role"] == "compatibility_only"
    assert datasets["id_test"]["expected_sha256"] == (
        "4178c1a97951afab348d63734d426d7e3cb0d13ddf3c0aebae331467d723fd73"
    )
    assert datasets["id_test_openood"]["expected_sha256"] == (
        "84559ec5225425e7f7363058d5f9442ed21a39c7c66b63c3f1bb8b1ddb935e13"
    )


def test_affected_reference_cards_point_to_the_v1_2_contract():
    for name in (
        "04_openood_v1_5_protocol.md",
        "06_feature_ood_detectors.md",
        "08_raw_feature_artifact_contract.md",
        "09_core_representation_metrics.md",
    ):
        text = (REFERENCE_DIR / name).read_text(encoding="utf-8")
        assert "11_metric_contract_v1_2.md" in text

    card04 = (REFERENCE_DIR / "04_openood_v1_5_protocol.md").read_text(
        encoding="utf-8"
    )
    assert 'np.quantile(id_scores, 0.05, method="linear")' in card04
    card06 = (REFERENCE_DIR / "06_feature_ood_detectors.md").read_text(
        encoding="utf-8"
    )
    assert "named **`GDA-ClassDensity`**" in card06
    assert "reserved for a future" in card06
    card09 = (REFERENCE_DIR / "09_core_representation_metrics.md").read_text(
        encoding="utf-8"
    )
    assert "NC1 = Tr(Sigma_W Sigma_B^dagger) / K" in card09
    assert "covariance_participation_ratio" in card09
