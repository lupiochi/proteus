#!/usr/bin/env python3
"""
scripts/extract_embeddings.py

Extract PROTEUS embeddings from protein sequences using a flow-matching predictor.

Reads a FASTA file, runs the SimpleFold flow-matching sampler for each sequence
(capturing trunk embeddings inline at the target flow-timestep), and writes one
NPZ file per protein in bc-convention format.

Usage
-----
    python scripts/extract_embeddings.py \
        --fasta proteins.fasta \
        --model_path /path/to/simplefold_weights \
        --output_dir embeddings/ \
        --n_conformations 10 \
        --num_steps 25 \
        --device mps

Output
------
One NPZ file per protein: {output_dir}/{protein_id}.npz
  t1.0            (L, D)  sequence-only embedding
  t0.0_conf{k}    (L, D)  structural conformation k  (k = 0..K-1)

These files are readable by `proteus.io.load_embeddings`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into (protein_id, sequence) pairs."""
    proteins = []
    current_id = None
    seqlines = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    proteins.append((current_id, "".join(seqlines)))
                current_id = line[1:].split()[0]
                seqlines = []
            else:
                seqlines.append(line)
    if current_id is not None:
        proteins.append((current_id, "".join(seqlines)))
    return proteins


def main():
    parser = argparse.ArgumentParser(description="Extract PROTEUS embeddings from FASTA.")
    parser.add_argument("--fasta", required=True, type=Path, help="Input FASTA file.")
    parser.add_argument("--model_path", required=True, type=Path,
                        help="Path to SimpleFold model weights directory.")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory to write NPZ files.")
    parser.add_argument("--n_conformations", type=int, default=10,
                        help="Number of independent structural conformations (default 10).")
    parser.add_argument("--num_steps", type=int, default=25,
                        help="Euler-Maruyama sampler steps (default 25).")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device ('cpu', 'cuda', 'mps'). Auto-detected if omitted.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip proteins whose NPZ file already exists.")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Skip sequences longer than this many residues.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictor
    from proteus.adapters.simplefold import SimpleFoldPredictor
    predictor = SimpleFoldPredictor.from_pretrained(
        args.model_path,
        device=args.device,
        num_steps=args.num_steps,
    )

    proteins = parse_fasta(args.fasta)
    logger.info("Found %d sequences in %s", len(proteins), args.fasta)

    n_ok = 0
    n_skip = 0
    n_err = 0

    for protein_id, sequence in proteins:
        out_path = args.output_dir / f"{protein_id}.npz"

        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue

        if args.max_length and len(sequence) > args.max_length:
            logger.warning("Skipping %s: length %d > max_length %d",
                           protein_id, len(sequence), args.max_length)
            n_skip += 1
            continue

        try:
            output = predictor.embed(
                sequence,
                n_conformations=args.n_conformations,
                protein_id=protein_id,
            )
            from proteus.io import save_embeddings
            save_embeddings(out_path, output.seq_emb, output.struct_embs)
            n_ok += 1
            logger.info("OK  %s  L=%d", protein_id, len(sequence))
        except Exception as exc:
            logger.error("FAIL %s: %s", protein_id, exc)
            n_err += 1

    logger.info("Done. OK=%d  skipped=%d  failed=%d", n_ok, n_skip, n_err)
    if n_err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
