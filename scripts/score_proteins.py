#!/usr/bin/env python3
"""
scripts/score_proteins.py

Compute PROTEUS conformational ambiguity scores from pre-extracted NPZ embeddings.

This script does NOT require SimpleFold or any GPU — it operates entirely on
the NPZ files produced by extract_embeddings.py.

Usage
-----
    python scripts/score_proteins.py \
        --embeddings_dir embeddings/ \
        --output scores.csv

    # Score a single file:
    python scripts/score_proteins.py \
        --embeddings_dir embeddings/ \
        --output scores.csv \
        --pattern "P*.npz"

Output columns
--------------
    protein_id, length, l2_delta_max, l2_delta_mean, l2_delta_p90,
    cos_dist_mean, cos_dist_p90, ensemble_spread
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from proteus.io import load_embeddings
from proteus.score import compute_all_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Score proteins from PROTEUS NPZ embeddings.")
    parser.add_argument("--embeddings_dir", required=True, type=Path,
                        help="Directory containing NPZ files.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output CSV file path.")
    parser.add_argument("--pattern", type=str, default="*.npz",
                        help="Glob pattern for NPZ files (default '*.npz').")
    args = parser.parse_args()

    npz_files = sorted(args.embeddings_dir.glob(args.pattern))
    if not npz_files:
        logger.error("No NPZ files found in %s matching '%s'", args.embeddings_dir, args.pattern)
        return

    logger.info("Scoring %d proteins...", len(npz_files))

    records = []
    for npz_path in npz_files:
        emb = load_embeddings(npz_path)
        if emb is None:
            logger.warning("Skipping %s: missing required keys", npz_path.name)
            continue
        features = compute_all_features(emb.seq_emb, emb.struct_embs)
        records.append({"protein_id": emb.protein_id, **features})

    df = pd.DataFrame(records)
    cols = ["protein_id", "length", "l2_delta_max", "l2_delta_mean", "l2_delta_p90",
            "cos_dist_mean", "cos_dist_p90", "ensemble_spread"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(df), args.output)


if __name__ == "__main__":
    main()
