import re
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).parents[1]
REFERENCE_DIR = REPOSITORY_ROOT / "docs/reference_cards"
CONTRACT_PATH = REFERENCE_DIR / "12_fixed_readout_intervention_protocol_v2.md"


def test_v2_contract_freezes_owner_decisions_and_stage_gates_remaining_numbers():
    contract = CONTRACT_PATH.read_text(encoding="utf-8")

    required = (
        "fixed_readout_training_rule_intervention_v2",
        "WRN-28-10/CIFAR-10",
        "ResNet-18/CIFAR-100",
        "`adam_coupled`",
        "`adamw_decoupled`",
        "`adam_zero`",
        "`sgdm_coupled`",
        "`sgdw_decoupled`",
        "`sgdm_zero`",
        "`detector/mahalanobis_raw`",
        "`detector/mahalanobis_pp`",
        "`detector/knn_raw_k50`",
        "`detector/knn_l2_k50`",
        "`detector/ctm_prototype_cosine`",
        "`detector/vim_residual_author_dim`",
        "`ood_score/energy_t1`",
        "Relative Mahalanobis-Raw",
        "Relative Mahalanobis++",
        "Appendix decomposition: ViM",
        "Excluded: ReAct",
        "fork_from_prefix",
        "Stage 2 -- existing-artifact mechanism gate",
        "Freeze after Stage 2 and before any fresh protected OOD result",
    )
    for marker in required:
        assert marker in contract

    assert "Adam and SGDM prefixes are independent experimental blocks" in contract
    assert "Metric Contract v1.2" in contract
    assert "does not retroactively change v1.2" in contract
    assert "A scalar geometry--AUROC correlation alone is" in contract


def test_v2_contract_local_markdown_links_resolve():
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", contract)
    local_links = [link for link in links if "://" not in link and not link.startswith("#")]
    assert local_links
    for link in local_links:
        assert (CONTRACT_PATH.parent / link).resolve().is_file(), link


def test_v2_source_and_historical_manifest_are_registered_without_tracking_inputs():
    sources = yaml.safe_load(
        (REPOSITORY_ROOT / "docs/sources.lock.yaml").read_text(encoding="utf-8")
    )["sources"]
    assert sources["local_fixed_readout_intervention_v2_reference_card"]["path"] == (
        "docs/reference_cards/12_fixed_readout_intervention_protocol_v2.md"
    )
    for key in (
        "flavors_of_margin_paper",
        "szyc_ood_reliability_paper",
        "training_induced_ood_wacv2026_paper",
        "geometry_based_mahalanobis_ood_paper",
    ):
        assert key in sources

    manifest = (
        REPOSITORY_ROOT / "docs/history/local_research_draft_manifest.md"
    ).read_text(encoding="utf-8")
    assert "untracked and local" in manifest
    for filename in (
        "7월 23일 문제 정의 및 논문 뼈대.md",
        "7월 30일 평가 metric 정의 조사.md",
        "0806_논문뼈대 수정본.md",
        "0807_논문뼈대 v2.md",
        "0806 추가실험 계획 및 논문뼈대.zip",
        "OOD_metric_audit_handoff.zip",
        "metric_contract_v1_2_definitions_and_c1_c4_results.zip",
        "0806_논문뼈대 수정본.md:Zone.Identifier",
    ):
        assert f"`{filename}`" in manifest


def test_v2_does_not_mutate_v1_2_detector_or_resume_boundaries():
    card09 = (REFERENCE_DIR / "09_core_representation_metrics.md").read_text(
        encoding="utf-8"
    )
    card11 = (REFERENCE_DIR / "11_metric_contract_v1_2.md").read_text(
        encoding="utf-8"
    )
    card05 = (REFERENCE_DIR / "05_training_protocol.md").read_text(
        encoding="utf-8"
    )

    assert "only **Metric Contract v1.2** kNN definition" in card09
    assert "`detector/knn_raw_k50`" in card09
    assert "formula, artifact key, reporting tier, split, 수치 결과는 변경하지 않았다" in card11
    assert "optimizer name or optimizer hyperparameter changes" in card05
