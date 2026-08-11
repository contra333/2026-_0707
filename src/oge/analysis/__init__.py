"""Reproducible analysis utilities for completed experiment artifacts."""

from .fixed_readout_component_attribution import (
    mahalanobis_score_components,
    pair_outcome_summary,
    pair_transition_summary,
    paired_component_attribution,
    quadratic_size_stretch,
    reconstruction_record,
    symmetric_product_change,
    woodbury_low_rank_record,
)


def run_analysis(*args, **kwargs):
    """Lazily import the optional SciPy-backed v1.2 report generator."""

    from .metric_contract_v1_2 import run_analysis as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "mahalanobis_score_components",
    "pair_outcome_summary",
    "pair_transition_summary",
    "paired_component_attribution",
    "quadratic_size_stretch",
    "reconstruction_record",
    "run_analysis",
    "symmetric_product_change",
    "woodbury_low_rank_record",
]
