"""
proteus/io.py

Loading and saving PROTEUS embedding files (NPZ format).

File format (matches the t-convention used in the paper):
  t0.0          (L, D)  sequence-only embedding   (flow t = 0, zero coords)
  t1.0_conf{k}  (L, D)  structural embedding k    (flow t = 1, k = 0..K-1)

Legacy files written before convention alignment used the inverse keys
(t1.0 for sequence-only, t0.0_conf{k} for structural). load_embeddings
auto-detects and remaps both conventions transparently.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import NamedTuple


class ProteusEmbedding(NamedTuple):
    """Container for PROTEUS embeddings extracted from a single protein."""
    protein_id: str
    seq_emb: np.ndarray              # (L, D) sequence-only embedding
    struct_embs: list[np.ndarray]    # K arrays of shape (L, D)


def load_embeddings(path: str | Path) -> ProteusEmbedding | None:
    """Load a PROTEUS NPZ file.

    Default convention (paper-aligned):
        t0.0          -> sequence-only embedding
        t1.0_conf{k}  -> structural conformation k

    Legacy convention (auto-detected):
        t1.0          -> sequence-only embedding
        t0.0_conf{k}  -> structural conformation k

    Returns None if the file lacks the required keys.
    """
    path = Path(path)
    data = np.load(path)
    keys = list(data.keys())

    has_paper    = ("t0.0" in keys) and any(k.startswith("t1.0_conf") for k in keys)
    has_legacy   = ("t1.0" in keys) and any(k.startswith("t0.0_conf") for k in keys)

    if has_paper:
        seq_key, conf_prefix = "t0.0", "t1.0_conf"
    elif has_legacy:
        seq_key, conf_prefix = "t1.0", "t0.0_conf"
    else:
        return None

    seq_emb = data[seq_key].astype(np.float32)

    conf_keys = sorted(
        [k for k in keys if k.startswith(conf_prefix)],
        key=lambda k: int(k.split(conf_prefix)[1]),
    )
    if not conf_keys:
        return None

    struct_embs = [data[k].astype(np.float32) for k in conf_keys]
    return ProteusEmbedding(
        protein_id=path.stem,
        seq_emb=seq_emb,
        struct_embs=struct_embs,
    )


def save_embeddings(
    path: str | Path,
    seq_emb: np.ndarray,
    struct_embs: list[np.ndarray],
) -> None:
    """Save PROTEUS embeddings using the paper convention.

    Stored as float16 to keep proteome-scale archives compact:
        t0.0          -> sequence-only embedding
        t1.0_conf{k}  -> structural conformation k
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"t0.0": seq_emb.astype(np.float16)}
    for k, emb in enumerate(struct_embs):
        arrays[f"t1.0_conf{k}"] = emb.astype(np.float16)
    np.savez_compressed(path, **arrays)
