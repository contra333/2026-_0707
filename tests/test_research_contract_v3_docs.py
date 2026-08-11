import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
CARD = ROOT / "docs/reference_cards/13_component_attribution_intervention_protocol_v3.md"
OUTLINE = ROOT / "docs/paper/intervention_supporting_theory_outline.md"


def test_v3_contract_freezes_intervention_attribution_and_novelty_boundary():
    text = CARD.read_text(encoding="utf-8")
    required = (
        "fixed_readout_component_attribution_intervention_v3",
        "s_MD(z) = s_RMD(z) + s_Marginal(z)",
        "m_MD(i,o) = m_RMD(i,o) + m_Marginal(i,o)",
        "global-referenced class-relative component",
        "shared-prefix coupled/decoupled/zero-decay intervention",
        "five fresh prefix seeds",
        "three fresh prefix seeds",
        "ResNet-18/CIFAR-10",
        "ResNet-18/CIFAR-100",
        "DenseNet-BC-100",
        "ConvNeXt-Tiny/ImageNet-200",
        "SSB-hard, NINCO",
        "iNaturalist, Textures, OpenImage-O",
        "ImageNet-1K pretrained weights are",
        "forbidden because they expose",
        "Ordinary resume remains strict",
        "computational hybrids",
        "not claimed to be the unique",
        "matched effective shrinkage",
        "minimum-detectable-effect justification",
        "historical cache CLI consumes score",
        "not publish size/stretch artifacts",
        "populated Adam `step`, `exp_avg`, and `exp_avg_sq`",
        "The v2 radial failure stays `FAILED`",
    )
    for marker in required:
        assert marker in text
    for excluded_novelty in (
        "MD--Marginal--RMD relation",
        "L2 feature normalization",
        "size--stretch factorization",
    ):
        assert excluded_novelty in text


def test_v3_contract_and_outline_local_links_resolve():
    for path in (CARD, OUTLINE):
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8"))
        for link in links:
            if "://" not in link and not link.startswith("#"):
                assert (path.parent / link).resolve().is_file(), (path, link)


def test_v3_sources_and_architecture_truth_are_registered():
    sources = yaml.safe_load((ROOT / "docs/sources.lock.yaml").read_text())["sources"]
    assert sources["local_component_attribution_intervention_v3_reference_card"][
        "path"
    ] == "docs/reference_cards/13_component_attribution_intervention_protocol_v3.md"
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


def test_historical_pair_manifest_is_seed_matched_and_bounded_to_reuse_population():
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


def test_v2_failure_record_and_v1_2_contract_remain_present():
    status = (ROOT / "docs/STATUS.md").read_text(encoding="utf-8")
    v2 = (
        ROOT / "docs/reference_cards/12_fixed_readout_intervention_protocol_v2.md"
    ).read_text(encoding="utf-8")
    v1 = (ROOT / "docs/reference_cards/11_metric_contract_v1_2.md").read_text(
        encoding="utf-8"
    )
    assert "Stage 2 mechanism gate result: `FAILED`" in status
    assert "fixed_readout_training_rule_intervention_v2" in v2
    assert "metric_contract_v1.2" in v1


def test_v3_status_header_matches_completed_foundation_boundary():
    text = CARD.read_text(encoding="utf-8")
    assert "historical component-attribution implementation and production run: PASS" in text
    assert "shared-prefix fork runtime and CPU fixture validation: PASS" in text
    assert "production run pending" not in text
    assert "CPU tests pending" not in text
