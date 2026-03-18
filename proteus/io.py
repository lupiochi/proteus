"""
proteus/io.py

Loading and saving PROTEUS embedding files (NPZ format).

File format (bc-convention):
  t1.0          (L, D)  sequence-only embedding  (flow t = 1, zero coords)
  t0.0_conf{k}  (L, D)  structural embedding k   (flow t = 0, k = 0..K-1)

This convention is produced by the SimpleFold adapter and is expected by all
PROTEUS analysis scripts.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import NamedTuple


class ProteusEmbedding(NamedTuple):
    """Container for PROTEUS embeddings extracted from a single protein."""
    protein_id: str
    seq_emb: np.ndarray          # (L, D) sequence-only embedding
    struct_embs: list[np.ndarray]  # K arrays of shape (L, D)


def load_embeddings(path: str | Path) -> ProteusEmbedding | None:
    """Load a PROTEUS NPZ file.

    Supports both bc-convention (t1.0 / t0.0_conf*) and the inverse convention
    (t0.0 / t1.0_conf*) — the latter is auto-detected and remapped.

    Returns None if the file lacks the required keys.
    """
    path = Path(path)
    data = np.load(path)
    keys = list(data.keys())

    # Detect convention
    has_bc = any(k.startswith("t0.0_conf") for k in keys)
    seq_key    = "t1.0"       if has_bc else "t0.0"
    conf_prefix = "t0.0_conf" if has_bc else "t1.0_conf"

    if seq_key not in keys:
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
    """Save PROTEUS embeddings in bc-convention NPZ format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"t1.0": seq_emb.astype(np.float16)}
    for k, emb in enumerate(struct_embs):
        arrays[f"t0.0_conf{k}"] = emb.astype(np.float16)
    np.savez_compressed(path, **arrays)
