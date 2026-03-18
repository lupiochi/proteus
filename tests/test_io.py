"""
tests/test_io.py

Unit tests for proteus.io — round-trip save/load and convention auto-detection.
"""

import numpy as np
import pytest

from proteus.io import load_embeddings, save_embeddings, ProteusEmbedding


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path, rng=np.random.default_rng(1)):
    L, D, K = 30, 128, 5
    seq_emb = rng.standard_normal((L, D)).astype(np.float32)
    struct_embs = [rng.standard_normal((L, D)).astype(np.float32) for _ in range(K)]

    path = tmp_path / "test_protein.npz"
    save_embeddings(path, seq_emb, struct_embs)
    loaded = load_embeddings(path)

    assert loaded is not None
    assert loaded.protein_id == "test_protein"
    assert loaded.seq_emb.shape == (L, D)
    assert len(loaded.struct_embs) == K
    assert loaded.struct_embs[0].shape == (L, D)


def test_roundtrip_values_close(tmp_path, rng=np.random.default_rng(2)):
    """Values should survive float16 compression with reasonable tolerance."""
    seq = rng.standard_normal((20, 64)).astype(np.float32)
    structs = [rng.standard_normal((20, 64)).astype(np.float32) for _ in range(3)]

    path = tmp_path / "p.npz"
    save_embeddings(path, seq, structs)
    loaded = load_embeddings(path)

    np.testing.assert_allclose(loaded.seq_emb, seq, atol=1e-2)


# ---------------------------------------------------------------------------
# Missing-key graceful handling
# ---------------------------------------------------------------------------

def test_load_returns_none_for_empty_npz(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(path, some_unrelated_key=np.zeros(5))
    loaded = load_embeddings(path)
    assert loaded is None


def test_load_returns_none_for_no_struct_embs(tmp_path):
    """NPZ with seq key but no conf keys should return None."""
    path = tmp_path / "no_structs.npz"
    np.savez(path, **{"t1.0": np.zeros((10, 64))})
    loaded = load_embeddings(path)
    assert loaded is None


# ---------------------------------------------------------------------------
# Convention auto-detection
# ---------------------------------------------------------------------------

def test_bc_convention_detected(tmp_path):
    """bc-convention: t1.0=seq, t0.0_conf*=struct."""
    L, D = 15, 32
    path = tmp_path / "bc.npz"
    np.savez(path, **{
        "t1.0": np.ones((L, D)),
        "t0.0_conf0": np.zeros((L, D)),
        "t0.0_conf1": np.zeros((L, D)),
    })
    loaded = load_embeddings(path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.seq_emb, np.ones((L, D), dtype=np.float32))
    assert len(loaded.struct_embs) == 2


def test_inverse_convention_detected(tmp_path):
    """Inverse convention: t0.0=seq, t1.0_conf*=struct. Should be auto-remapped."""
    L, D = 15, 32
    path = tmp_path / "inv.npz"
    np.savez(path, **{
        "t0.0": np.ones((L, D)),
        "t1.0_conf0": np.zeros((L, D)),
    })
    loaded = load_embeddings(path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.seq_emb, np.ones((L, D), dtype=np.float32))


def test_struct_emb_ordering(tmp_path):
    """Struct embeddings should be returned in ascending conf index order."""
    L, D = 10, 16
    path = tmp_path / "order.npz"
    np.savez(path, **{
        "t1.0": np.zeros((L, D)),
        "t0.0_conf2": np.full((L, D), 2.0),
        "t0.0_conf0": np.full((L, D), 0.0),
        "t0.0_conf1": np.full((L, D), 1.0),
    })
    loaded = load_embeddings(path)
    assert loaded.struct_embs[0][0, 0] == pytest.approx(0.0, abs=0.01)
    assert loaded.struct_embs[1][0, 0] == pytest.approx(1.0, abs=0.01)
    assert loaded.struct_embs[2][0, 0] == pytest.approx(2.0, abs=0.01)
