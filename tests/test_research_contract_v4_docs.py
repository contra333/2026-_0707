import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
CARD = (
    ROOT
    / "docs/reference_cards/13_paired_trajectory_component_attribution_protocol_v4.md"
)
OUTLINE = ROOT / "docs/paper/intervention_supporting_theory_outline.md"


def test_v4_contract_freezes_paired_trajectory_and_novelty_boundary():
    text = CARD.read_text(encoding="utf-8")
    required = (
        "fixed_readout_paired_trajectory_component_attribution_v4",
        "same model initialization",
        "same initialization and data stream",
        "Total Adam-family budget: **36 runs**",
        "Total SGDM-family budget: **9 runs**",
        "0, 1, 10, 30, 60, 61, 120, 121, 160, 161, 200",
        "Epoch-200 `last.pt` is the primary endpoint",
        "`best_val.pt` selected by ID validation only",
        "stage1",
        "stage2",
        "stage3",
        "penultimate",
        "update dynamics",
        "representation geometry",
        "s_MD(z) = s_RMD(z) + s_Marginal(z)",
        "m_MD(i,o) = m_RMD(i,o) + m_Marginal(i,o)",
        "global-referenced class-relative component",
        "computational hybrids",
        "unique physical or causal mediation decomposition",
        "matched effective shrinkage",
        "Channel-matched confirmation",
        "ResNet-18/CIFAR-10",
        "ResNet-18/CIFAR-100",
        "DenseNet-BC-100",
        "ConvNeXt-Tiny/ImageNet-200",
        "SSB-hard, NINCO",
        "iNaturalist, Textures, OpenImage-O",
        "ImageNet-1K pretrained weights are",
        "forbidden because they expose",
        "optional follow-up",
        "Ordinary resume remains strict",
        "minimum-detectable-effect justification",
        "v3 historical MD--Marginal--RMD analysis remains `PASS`",
        "v2 radial Stage-2 result remains immutable `FAILED`",
    )
    for marker in required:
        assert marker in text
    for excluded_novelty in (
        "MD--Marginal--RMD relation",
        "L2 feature normalization",
        "size--stretch factorization",
    ):
        assert excluded_novelty in text


def test_v4_contract_and_outline_local_links_resolve():
    for path in (CARD, OUTLINE):
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8"))
        for link in links:
            if "://" not in link and not link.startswith("#"):
                assert (path.parent / link).resolve().is_file(), (path, link)


def test_v4_sources_and_architecture_truth_are_registered():
    sources = yaml.safe_load((ROOT / "docs/sources.lock.yaml").read_text())["sources"]
    assert sources["local_paired_trajectory_component_attribution_v4_reference_card"][
        "path"
    ] == "docs/reference_cards/13_paired_trajectory_component_attribution_protocol_v4.md"
    assert sources["relative_mahalanobis_paper"]["stable_source"] == "arXiv:2106.09022"
    geometry = sources["geometry_based_mahalanobis_ood_paper"]
    assert geometry["stable_source"] == "arXiv:2510.15202v3"
    assert geometry["title"] == "A Geometry-Based View of Mahalanobis OOD Detection"
    assert sources["l2_regularization_batch_weight_normalization_paper"][
        "stable_source"
    ] == "arXiv:1706.05350"
    assert sources["three_mechanisms_weight_decay_paper"]["stable_source"] == (
        "arXiv:1810.12281"
    )
    architecture = (ROOT / "docs/reference_cards/02_architectures.md").read_text()
    assert "`densenet_bc_100_k12`" in architecture
    assert "| Planned |" in architecture
    assert "ImageNet-1K pretrained weights are forbidden" in architecture
    assert "v4 multi-depth taps pending" in architecture


def test_historical_v3_pair_manifest_remains_seed_matched_and_bounded():
    pairs = json.loads(
        (
            ROOT
            / "configs/evaluation/fixed_readout_component_attribution_v3/historical_pairs.json"
        ).read_text()
    )
    reuse = json.loads(
        (ROOT / "configs/evaluation/fixed_readout_stage2/reuse_manifest.json").read_text()
    )
    by_id = {row["bundle_id"]: row for row in reuse["bundles"]}
    assert pairs["schema_version"] == "fixed_readout_component_pairs_v3"
    assert len(pairs["pairs"]) == 9
    for pair in pairs["pairs"]:
        assert pair["left_bundle_id"] in by_id
        assert pair["right_bundle_id"] in by_id
        assert by_id[pair["left_bundle_id"]]["training_seed"] == by_id[
            pair["right_bundle_id"]
        ]["training_seed"]


def test_v2_failure_and_v3_discovery_remain_historical_evidence():
    status = (ROOT / "docs/STATUS.md").read_text(encoding="utf-8")
    v2 = (
        ROOT / "docs/reference_cards/12_fixed_readout_intervention_protocol_v2.md"
    ).read_text(encoding="utf-8")
    v1 = (ROOT / "docs/reference_cards/11_metric_contract_v1_2.md").read_text(
        encoding="utf-8"
    )
    assert "Stage 2 mechanism gate result: `FAILED`" in status
    assert "V3 historical discovery result: `PASS`" in status
    assert "fixed_readout_training_rule_intervention_v2" in v2
    assert "metric_contract_v1.2" in v1


def test_active_current_docs_point_to_v4_not_v3_main():
    active_paths = (
        ROOT / "README.md",
        ROOT / "docs/PROJECT_CONTEXT.md",
        ROOT / "docs/STATUS.md",
        ROOT / "docs/reference_cards/02_architectures.md",
        ROOT / "docs/reference_cards/05_training_protocol.md",
        OUTLINE,
    )
    for path in active_paths:
        text = path.read_text(encoding="utf-8")
        assert "13_component_attribution_intervention_protocol_v3.md" not in text
    status = (ROOT / "docs/STATUS.md").read_text(encoding="utf-8")
    assert "Research Contract v4 is the active paper protocol" in status
    assert "no from-scratch v4 paired trajectory" in status
