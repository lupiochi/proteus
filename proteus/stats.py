"""
proteus/stats.py

Statistical helpers for PROTEUS analyses.

These utilities are used throughout the analysis scripts for confound control,
significance testing, and AUROC estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from typing import Sequence


# ---------------------------------------------------------------------------
# Partial Spearman correlation
# ---------------------------------------------------------------------------

def rank_residualise(
    values: np.ndarray | pd.Series,
    covariates: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Residualise rank(values) on rank(covariates) via OLS.

    If covariates is empty / None, returns centred ranks (no adjustment).
    """
    v = pd.Series(values).rank().values
    if covariates is None or (hasattr(covariates, "__len__") and len(covariates) == 0):
        return v - v.mean()
    cov = np.asarray(covariates)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    # Rank each covariate column
    cov_ranked = np.column_stack([pd.Series(cov[:, i]).rank().values
                                  for i in range(cov.shape[1])])
    return v - LinearRegression().fit(cov_ranked, v).predict(cov_ranked)


def partial_spearman(
    df: pd.DataFrame,
    feature: str,
    outcome: str,
    covariates: list[str],
) -> tuple[float, float, int]:
    """Partial Spearman rho of feature with outcome, controlling for covariates.

    Uses rank-based OLS residualisation (equivalent to the standard definition
    of partial Spearman correlation and robust to non-linearity in covariates).

    Returns:
        (rho, p_value, n)
    """
    cols = [feature, outcome] + covariates
    ok = df[cols].dropna()
    if len(ok) < 5:
        return np.nan, np.nan, 0

    cov_matrix = ok[covariates].values if covariates else None
    r_feat = rank_residualise(ok[feature].values, cov_matrix)
    r_out  = rank_residualise(ok[outcome].values,  cov_matrix)
    rho, p = spearmanr(r_feat, r_out)
    return float(rho), float(p), len(ok)


# ---------------------------------------------------------------------------
# Bootstrap AUROC
# ---------------------------------------------------------------------------

def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute AUROC with 95% bootstrap confidence interval.

    Returns:
        (auroc, lower_95ci, upper_95ci)
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y_true, y_score)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(base), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Permutation test for AUROC significance.

    Returns:
        (observed_auroc, empirical_p_value)
    """
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y_true, y_score)
    null = []
    for _ in range(n_perm):
        shuffled = rng.permutation(y_true)
        null.append(roc_auc_score(shuffled, y_score))
    p = float(np.mean(np.array(null) >= observed))
    return float(observed), p


# ---------------------------------------------------------------------------
# Mann-Whitney U with effect size
# ---------------------------------------------------------------------------

def mwu_with_effect(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test returning (p_value, rank_biserial_r).

    Rank-biserial r = 1 - 2U / (n_a * n_b).
    r > 0 means values in a tend to be larger than in b.
    """
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    r = float(1 - 2 * u / (len(a) * len(b)))
    return float(p), r


# ---------------------------------------------------------------------------
# Length-matched Wilcoxon
# ---------------------------------------------------------------------------

def length_matched_wilcoxon(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    length_col: str,
    positive_label: int = 1,
    seed: int = 42,
) -> tuple[float, int]:
    """Nearest-neighbour length-matched Wilcoxon signed-rank test.

    Matches each positive example to the nearest-length negative example
    (without replacement), then tests whether paired score differences
    are systematically positive.

    Returns:
        (p_value, n_pairs)
    """
    from scipy.stats import wilcoxon

    pos = df[df[label_col] == positive_label][[score_col, length_col]].dropna()
    neg = df[df[label_col] != positive_label][[score_col, length_col]].dropna()

    rng = np.random.default_rng(seed)
    neg_avail = neg.copy().reset_index(drop=True)
    pairs = []
    for _, row in pos.iterrows():
        if neg_avail.empty:
            break
        dists = (neg_avail[length_col] - row[length_col]).abs()
        best = dists.idxmin()
        pairs.append((row[score_col], neg_avail.loc[best, score_col]))
        neg_avail = neg_avail.drop(best).reset_index(drop=True)

    if len(pairs) < 10:
        return np.nan, len(pairs)

    pos_scores = np.array([p[0] for p in pairs])
    neg_scores = np.array([p[1] for p in pairs])
    stat, p = wilcoxon(pos_scores, neg_scores)
    return float(p), len(pairs)
