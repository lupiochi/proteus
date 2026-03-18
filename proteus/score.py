"""
proteus/score.py

PROTEUS conformational ambiguity scoring functions.

All functions are model-agnostic — they operate on numpy arrays and do not
depend on any specific structure predictor. The inputs are:
  seq_emb:     (L, D) sequence-only embedding, extracted with zero coordinate input
  struct_embs: list of K arrays of shape (L, D), each from an independent sampler run

The primary score is l2_delta_max. All other features are reported in the
supplementary formulation search but l2_delta_max is recommended for general use.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Primary score
# ---------------------------------------------------------------------------

def l2_delta_max(
    seq_emb: np.ndarray,
    struct_embs: Sequence[np.ndarray],
) -> float:
    """Maximum per-residue L2 displacement between the sequence-only embedding
    and the mean structural embedding.

    This is the primary PROTEUS conformational ambiguity score. Higher values
    indicate greater conformational flexibility.

    Args:
        seq_emb:     (L, D) sequence-only trunk embedding (flow timestep t=1,
                     zero coordinate input).
        struct_embs: K arrays of shape (L, D), each the trunk embedding from
                     one independent sampler run at flow timestep t=0.

    Returns:
        Scalar score >= 0.
    """
    seq_emb = np.asarray(seq_emb, dtype=np.float32)
    struct_mean = np.mean([np.asarray(e, dtype=np.float32) for e in struct_embs], axis=0)
    delta = np.linalg.norm(struct_mean - seq_emb, axis=-1)   # (L,)
    return float(delta.max())


# ---------------------------------------------------------------------------
# Secondary features (formulation search)
# ---------------------------------------------------------------------------

def l2_delta_mean(
    seq_emb: np.ndarray,
    struct_embs: Sequence[np.ndarray],
) -> float:
    """Mean per-residue L2 displacement (less sensitive to single flexible loops
    than l2_delta_max, but more robust to outlier residues)."""
    seq_emb = np.asarray(seq_emb, dtype=np.float32)
    struct_mean = np.mean([np.asarray(e, dtype=np.float32) for e in struct_embs], axis=0)
    delta = np.linalg.norm(struct_mean - seq_emb, axis=-1)
    return float(delta.mean())


def l2_delta_p90(
    seq_emb: np.ndarray,
    struct_embs: Sequence[np.ndarray],
) -> float:
    """90th-percentile per-residue L2 displacement."""
    seq_emb = np.asarray(seq_emb, dtype=np.float32)
    struct_mean = np.mean([np.asarray(e, dtype=np.float32) for e in struct_embs], axis=0)
    delta = np.linalg.norm(struct_mean - seq_emb, axis=-1)
    return float(np.percentile(delta, 90))


def cos_dist_mean(
    seq_emb: np.ndarray,
    struct_embs: Sequence[np.ndarray],
) -> float:
    """Mean per-residue cosine distance between sequence and structural embeddings.

    Note: this feature is strongly length-confounded in raw form; use partial
    Spearman controlling for length when interpreting as a biological signal.
    """
    seq_emb = np.asarray(seq_emb, dtype=np.float32)
    struct_mean = np.mean([np.asarray(e, dtype=np.float32) for e in struct_embs], axis=0)
    norm_s = np.linalg.norm(seq_emb,    axis=-1)
    norm_t = np.linalg.norm(struct_mean, axis=-1)
    cos_sim = np.sum(seq_emb * struct_mean, axis=-1) / (norm_s * norm_t + 1e-8)
    return float((1.0 - cos_sim).mean())


def cos_dist_p90(
    seq_emb: np.ndarray,
    struct_embs: Sequence[np.ndarray],
) -> float:
    """90th-percentile per-residue cosine distance."""
    seq_emb = np.asarray(seq_emb, dtype=np.float32)
    struct_mean = np.mean([np.asarray(e, dtype=np.float32) for e in struct_embs], axis=0)
    norm_s = np.linalg.norm(seq_emb,    axis=-1)
    norm_t = np.linalg.norm(struct_mean, axis=-1)
    cos_sim = np.sum(seq_emb * struct_mean, axis=-1) / (norm_s * norm_t + 1e-8)
    return float(np.percentile(1.0 - cos_sim, 90))


def ensemble_spread(struct_embs: Sequence[np.ndarray]) -> float:
    """Mean per-residue variance across structural conformations.

    Caution: this feature captures prediction *uncertainty* (correlated with
    pLDDT) rather than trajectory displacement. After controlling for pLDDT it
    becomes non-significant (partial rho ~ 0.03). Provided for completeness.
    """
    stack = np.stack([np.asarray(e, dtype=np.float32) for e in struct_embs], axis=0)
    return float(stack.var(axis=0).mean())


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_features(
    seq_emb: np.ndarray,
    struct_embs: Sequence[np.ndarray],
) -> dict[str, float]:
    """Compute all PROTEUS features and return as a dict.

    Args:
        seq_emb:     (L, D) sequence-only embedding.
        struct_embs: list of K (L, D) structural embeddings.

    Returns:
        Dict with keys: l2_delta_max, l2_delta_mean, l2_delta_p90,
        cos_dist_mean, cos_dist_p90, ensemble_spread, length.
    """
    seq_emb = np.asarray(seq_emb, dtype=np.float32)
    struct_embs = [np.asarray(e, dtype=np.float32) for e in struct_embs]
    struct_mean = np.mean(struct_embs, axis=0)

    diff  = struct_mean - seq_emb
    l2    = np.linalg.norm(diff, axis=-1)
    ns    = np.linalg.norm(seq_emb,    axis=-1)
    nt    = np.linalg.norm(struct_mean, axis=-1)
    cos   = 1.0 - np.sum(seq_emb * struct_mean, axis=-1) / (ns * nt + 1e-8)
    stack = np.stack(struct_embs, axis=0)

    return {
        "l2_delta_max":    float(l2.max()),
        "l2_delta_mean":   float(l2.mean()),
        "l2_delta_p90":    float(np.percentile(l2, 90)),
        "cos_dist_mean":   float(cos.mean()),
        "cos_dist_p90":    float(np.percentile(cos, 90)),
        "ensemble_spread": float(stack.var(axis=0).mean()),
        "length":          int(seq_emb.shape[0]),
    }
