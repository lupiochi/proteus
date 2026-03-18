"""
PROTEUS: Protein Trajectory Uncertainty Score

Zero-shot detection of protein conformational flexibility from the latent
trajectory of a flow-matching structure predictor.

Core API:
    from proteus import score_embeddings, load_embeddings
    from proteus.stats import partial_spearman, bootstrap_auroc
"""

from .score import (
    l2_delta_max,
    l2_delta_mean,
    l2_delta_p90,
    cos_dist_mean,
    cos_dist_p90,
    ensemble_spread,
    compute_all_features,
)
from .io import load_embeddings, save_embeddings

__version__ = "1.0.0"
__all__ = [
    "l2_delta_max",
    "l2_delta_mean",
    "l2_delta_p90",
    "cos_dist_mean",
    "cos_dist_p90",
    "ensemble_spread",
    "compute_all_features",
    "load_embeddings",
    "save_embeddings",
]
