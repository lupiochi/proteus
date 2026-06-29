# PROTEUS

![PROTEUS Logo](logo_PROTEUS.png)

**PRO**tein **T**raj**E**ctory **U**ncertainty **S**core — a zero-shot method
for detecting protein conformational plasticity from the latent trajectory of a
flow-matching structure predictor.

Preprint: https://www.biorxiv.org/content/10.64898/2026.04.27.721098v1

[![Paper](https://img.shields.io/badge/paper-blue)](https://www.biorxiv.org/content/10.64898/2026.04.27.721098v1)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lupiochi/PROTEUS/blob/main/colab/PROTEUS_Colab.ipynb)

## What is PROTEUS?

Flow-matching structure predictors learn to denoise atomic coordinates from a
Gaussian prior (conditioned only on sequence) to a converged protein structure.
PROTEUS extracts the model's internal trunk embeddings at two points along this
trajectory — the **sequence-only** regime at flow timestep `t = 0` (no
coordinate input) and the **structure-converged** regime at `t = 1` — and
computes the maximum per-residue L2 displacement between them.

This single number, **`l2_delta_max`**, is a zero-shot proxy for conformational
plasticity: rigid proteins show little displacement (the sequence tightly
constrains the structure), while flexible or fold-switching proteins show large
displacement (the model's internal representation rotates substantially as
structural information is added).

**Key results from the paper:**
- Monotone ordering of five protein classes spanning the full conformational
  spectrum: *de novo* designed < natural single-domain < monostate <
  fold-switching < DIBS folding-upon-binding
- AUROC = 0.770 [0.719–0.817] on a 386-protein fold-switch benchmark
  (zero-shot; ESM-2 3B and SaProt 1.3B ≈ 0.50)
- Open/closed crystal-pair discrimination: OC23 AUROC = 0.808 and OC85
  AUROC = 0.902 against monostate controls (0.776 / 0.872 against
  MD-validated rigid ATLAS proteins)
- Within-protein correlation with MD-derived RMSF across 1,290 ATLAS
  proteins: median ρ = 0.396; partial ρ = 0.210 after jointly controlling
  for pLDDT and AIUPred disorder
- Identification of proteins harbouring buried phosphorylation sites:
  AUROC = 0.930 vs monostate, 0.914 vs MD-validated rigid (n = 492)
- Proteome-wide application to *Escherichia coli* K-12 (UP000000625):
  3,549 ordered proteins ranked by conformational plasticity

## Installation

### PROTEUS scoring (CPU only, no GPU dependencies)

```bash
git clone https://github.com/lupiochi/PROTEUS.git
cd PROTEUS
pip install -e .          # numpy, scipy, pandas, scikit-learn
pip install -e ".[dev]"   # add pytest / ruff for development
```

This is enough to load and score pre-extracted NPZ embeddings (e.g. from the
Zenodo archive) using `scripts/score_proteins.py` and the `proteus` Python API.

### Embedding extraction (requires SimpleFold + GPU)

To extract embeddings from new sequences you also need
[SimpleFold](https://github.com/apple/ml-simplefold) (Apple Inc., MIT) installed
separately, since it is not on PyPI:

```bash
git clone https://github.com/apple/ml-simplefold
cd ml-simplefold
pip install -e .          # installs the top-level packages used by PROTEUS:
                          #   inference, model, utils, embedding, processor
```

Then install the PROTEUS GPU extras (`torch + einops`):

```bash
pip install -e ".[simplefold]"
```

SimpleFold model weights (e.g. `simplefold_360M`, the variant used in the paper)
are downloaded on first use into `~/.cache/simplefold` by SimpleFold's own
loader; see SimpleFold's README for details.

## Quick start

### Extract embeddings from a FASTA (GPU recommended)

```bash
python scripts/extract_embeddings.py \
    --fasta_path my_proteins.fasta \
    --output_dir simplefold_embeddings/ \
    --simplefold_model simplefold_360M \
    --n_conformations 10 \
    --num_steps 25 \
    --tau 0.3
```

This is the production extraction script that produced every embedding in the
paper. It runs the SimpleFold Euler–Maruyama sampler `K = 10` times per
sequence and writes one NPZ per protein under `simplefold_embeddings/embeddings/`,
with the trunk latent captured inline at the requested flow timesteps (default
`t = 0` and `t = 1` in the paper convention).

### Score from pre-extracted embeddings (CPU only)

```bash
python scripts/score_proteins.py \
    --embeddings_dir simplefold_embeddings/embeddings/ \
    --output scores.csv
```

This reduces every NPZ to a row of PROTEUS features (`l2_delta_max` and
secondary scores) and requires no GPU.

### Programmatic use

```python
from proteus import compute_all_features
from proteus.io import load_embeddings

emb = load_embeddings("simplefold_embeddings/embeddings/P12345.npz")
features = compute_all_features(emb.seq_emb, emb.struct_embs)
print(features["l2_delta_max"])
```

The scoring functions in `proteus.score` operate on plain numpy arrays
(`(L, D)` for the sequence-only embedding and a list of `K` `(L, D)` arrays for
the structural ensemble), so they can be used with any flow-matching predictor
that exposes per-residue trunk embeddings — only the extraction step in
`scripts/extract_embeddings.py` is SimpleFold-specific.

## Repository structure

```
proteus/                  Core scoring package (no GPU deps)
  score.py                l2_delta_max and all PROTEUS features
  io.py                   NPZ embedding loader / writer
  stats.py                Partial Spearman, bootstrap AUROC, confound helpers

scripts/
  extract_embeddings.py   Production SimpleFold extraction CLI (GPU)
                          — paper-correct t-convention (t=0 sequence, t=1 structure)
  score_proteins.py       Score a directory of NPZ embeddings → CSV (CPU only)

LICENSE                   MIT license
pyproject.toml            Build / dependency manifest
CITATION.cff              Machine-readable citation metadata
```

## NPZ file convention

`extract_embeddings.py` writes one NPZ file per protein. Keys follow the same
flow-time convention used in the paper:

| Key             | Shape  | Meaning                                          |
| --------------- | ------ | ------------------------------------------------ |
| `t0.0`          | (L, D) | Sequence-only embedding (zero coords, t = 0)     |
| `t1.0_conf{k}`  | (L, D) | Structural conformation k (k = 0..K−1, t = 1)    |
| `sequence`      | str    | Amino-acid sequence                              |
| `length`        | int    | Sequence length L                                |

If `--aggregate_conformations` is passed, the per-conformation arrays are
replaced by `t1.0_mean` and `t1.0_var`. `proteus.io.load_embeddings` also
auto-detects the legacy inverted convention used in earlier archives.

## Inline latent capture

The extraction script captures the trunk embedding (`trunk_out`) directly from
the Euler–Maruyama sampler loop at the target flow timestep, rather than running
a separate post-hoc forward pass after sampling. This roughly halves the
forward-pass count compared with a naive two-pass extraction and produces
equivalent embeddings. The same approach applies to any Euler–Maruyama-based
flow-matching predictor.

## Citation

```bibtex
@article{piochi2026proteus,
  title   = {Unraveling protein conformational plasticity with PROTEUS},
  author  = {Piochi, Luiz Felipe and Karami, Yasaman and Khakzad, Hamed},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.04.27.721098},
  url     = {https://www.biorxiv.org/content/10.64898/2026.04.27.721098v1}
}
```

## License

MIT — see [LICENSE](LICENSE). The extraction script depends on SimpleFold
(Apple Inc., MIT License); SimpleFold is not bundled with this repository.
