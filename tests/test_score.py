"""
tests/test_score.py

Unit tests for proteus.score — all tests use synthetic numpy arrays and require
no GPU, no SimpleFold, and no external data.
"""

import numpy as np
import pytest

from proteus.score import (
    l2_delta_max,
    l2_delta_mean,
    l2_delta_p90,
    cos_dist_mean,
    cos_dist_p90,
    ensemble_spread,
    compute_all_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_embs(rng):
    L, D, K = 50, 512, 10
    seq_emb = rng.standard_normal((L, D)).astype(np.float32)
    struct_embs = [rng.standard_normal((L, D)).astype(np.float32) for _ in range(K)]
    return seq_emb, struct_embs


# ---------------------------------------------------------------------------
# Shape and type checks
# ---------------------------------------------------------------------------

def test_l2_delta_max_is_scalar(synthetic_embs):
    seq, structs = synthetic_embs
    result = l2_delta_max(seq, structs)
    assert isinstance(result, float)


def test_l2_delta_max_nonneg(synthetic_embs):
    seq, structs = synthetic_embs
    assert l2_delta_max(seq, structs) >= 0.0


def test_compute_all_features_keys(synthetic_embs):
    seq, structs = synthetic_embs
    features = compute_all_features(seq, structs)
    expected = {"l2_delta_max", "l2_delta_mean", "l2_delta_p90",
                "cos_dist_mean", "cos_dist_p90", "ensemble_spread", "length"}
    assert set(features.keys()) == expected


def test_compute_all_features_length(synthetic_embs):
    seq, structs = synthetic_embs
    features = compute_all_features(seq, structs)
    assert features["length"] == seq.shape[0]


# ---------------------------------------------------------------------------
# Sanity checks: zero-displacement case
# ---------------------------------------------------------------------------

def test_l2_delta_max_zero_when_equal(rng):
    """If seq_emb == every struct_emb, all displacements are zero."""
    seq = rng.standard_normal((20, 128)).astype(np.float32)
    structs = [seq.copy() for _ in range(5)]
    assert l2_delta_max(seq, structs) == pytest.approx(0.0, abs=1e-5)
    assert l2_delta_mean(seq, structs) == pytest.approx(0.0, abs=1e-5)


def test_cos_dist_zero_when_equal(rng):
    """Cosine distance is zero when seq_emb == struct_emb."""
    seq = rng.standard_normal((20, 128)).astype(np.float32)
    structs = [seq.copy() for _ in range(5)]
    assert cos_dist_mean(seq, structs) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Ordering checks
# ---------------------------------------------------------------------------

def test_l2_max_geq_mean(synthetic_embs):
    seq, structs = synthetic_embs
    assert l2_delta_max(seq, structs) >= l2_delta_mean(seq, structs)


def test_l2_max_geq_p90(synthetic_embs):
    seq, structs = synthetic_embs
    assert l2_delta_max(seq, structs) >= l2_delta_p90(seq, structs)


# ---------------------------------------------------------------------------
# Single conformation edge case
# ---------------------------------------------------------------------------

def test_single_conformation(rng):
    seq = rng.standard_normal((30, 64)).astype(np.float32)
    structs = [rng.standard_normal((30, 64)).astype(np.float32)]
    features = compute_all_features(seq, structs)
    assert features["l2_delta_max"] >= 0.0


# ---------------------------------------------------------------------------
# dtype tolerance
# ---------------------------------------------------------------------------

def test_float16_input(rng):
    """float16 inputs (as stored on disk) should be handled without error."""
    seq = rng.standard_normal((40, 512)).astype(np.float16)
    structs = [rng.standard_normal((40, 512)).astype(np.float16) for _ in range(3)]
    result = l2_delta_max(seq, structs)
    assert np.isfinite(result)
