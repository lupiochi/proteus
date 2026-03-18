# PROTEUS

**PRO**tein **T**raj**E**ctory **U**ncertainty **S**core — a zero-shot method for detecting
protein conformational flexibility from the latent trajectory of a flow-matching structure predictor.

[![Paper](https://img.shields.io/badge/paper-Nature_Machine_Intelligence-blue)](TODO)
[![Zenodo](https://img.shields.io/badge/data-Zenodo-orange)](TODO)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## What is PROTEUS?

Flow-matching structure predictors learn to denoise atomic coordinates from a Gaussian prior
(conditioned only on sequence) to a converged protein structure. PROTEUS extracts the model's
internal trunk embeddings at two points along this trajectory — the **sequence-only** regime
(no coordinate input) and the **structure-converged** regime — and computes the maximum
per-residue L2 displacement between them.

This single number, **l2_delta_max**, is a zero-shot proxy for conformational flexibility:
rigid proteins show little displacement (the sequence tightly constrains the structure),
while flexible or fold-switching proteins show large displacement (the model's internal
representation rotates substantially as structural information is added).

**Key results:**
- AUROC = 0.770 [0.719–0.817] on a 368-protein fold-switch benchmark (zero-shot; ESM-2/SaProt ≈ 0.50)
- Independent validation on open/closed crystal structures (OC23, AUROC = 0.808)
- Correlation with MD-derived RMSF across 1,369 ATLAS proteins (partial ρ = +0.313 | pLDDT + length)
- Identification of buried phosphorylation sites in flexible proteins (AUROC = 0.970)
- Proteome-wide application to *Streptococcus pyogenes* M1

## Quick start

### Reproduce paper figures from pre-computed scores (no GPU needed)

All PROTEUS scores used in the paper are archived on Zenodo. Download them and
run any analysis script directly:

```bash
pip install proteus-score
# Download pre-computed scores from Zenodo
wget TODO_ZENODO_URL -O data.tar.gz && tar xf data.tar.gz

# Fold-switch benchmark + confound analysis
python analysis/fold_switch/run_confound_analysis.py \
    --scores data/fold_switch/sf_rawdata.csv \
    --resolution data/fold_switch/resolution.csv \
    --output_dir results/confound/

# Validations
python analysis/validations/run_oc23_analysis.py \
    --scores_dir data/oc23/embeddings/ \
    --output_dir results/oc23/
```

### Score new proteins (GPU required)

PROTEUS requires a flow-matching structure predictor as its backend.
The default adapter targets **SimpleFold** (Apple Inc., MIT License):

```bash
# Install SimpleFold separately (see https://github.com/TODO/simplefold)
pip install proteus-score[simplefold]

# Extract embeddings
python scripts/extract_embeddings.py \
    --fasta my_proteins.fasta \
    --output_dir embeddings/ \
    --model simplefold_360M

# Score
python scripts/score_proteins.py \
    --embeddings_dir embeddings/ \
    --output scores.csv
```

PROTEUS is **model-agnostic** — any flow-matching predictor can be plugged in
by implementing the `FlowMatchingPredictor` interface (see `proteus/predictor.py`).

## Installation

```bash
pip install proteus-score          # scoring + analysis only (no GPU deps)
pip install proteus-score[simplefold]  # with SimpleFold adapter
pip install proteus-score[dev]     # with dev/test dependencies
```

## Repository structure

```
proteus/                  Core package
  score.py                l2_delta_max and all PROTEUS features
  io.py                   NPZ embedding loader
  stats.py                Partial Spearman, bootstrap AUROC, confound helpers
  predictor.py            Abstract FlowMatchingPredictor interface
  adapters/
    simplefold.py         SimpleFold adapter (inline latent-capture optimisation)

scripts/
  extract_embeddings.py   Embedding extraction pipeline (model-agnostic CLI)
  score_proteins.py       Score a directory of NPZ embeddings → CSV
  fetch_pdb_metadata.py   Retrieve resolution + metadata from RCSB GraphQL API
  prepare_fasta.py        FASTA preparation utilities

analysis/
  fold_switch/            Fold-switch benchmark + confound analysis
  validations/            OC23, DIBS, Tsuboyama, ATLAS analyses
  applications/           Phospho sites, GAS proteome
  benchmarks/             Thermostability, ProteinGym comparisons

data/                     Pre-computed scores (Zenodo pointer for embeddings)
notebooks/                Figure reproduction notebooks
tests/                    Unit tests
```

## The inline latent-capture optimisation

A core technical contribution of this work is an optimisation to the SimpleFold
Euler-Maruyama sampler that reduces forward-pass count by ~2.1× (from 521 to 251
passes per protein) while producing equivalent or slightly improved embeddings.
The key insight is that the model already computes the trunk embedding (`latent`)
at every sampler step but discards it — we capture it inline at the step closest
to the desired flow-timestep instead of running a separate post-hoc extraction pass.

This optimisation is implemented in `proteus/adapters/simplefold.py` and is
applicable to any Euler-Maruyama-based flow-matching predictor.

## Citation

```bibtex
@article{TODO,
  title   = {Flow-matching structure predictors implicitly encode protein conformational flexibility},
  author  = {TODO},
  journal = {Nature Machine Intelligence},
  year    = {TODO}
}
```

## License

MIT. The SimpleFold adapter calls SimpleFold (Apple Inc., MIT License) as an external
dependency. SimpleFold is not bundled with this repository.
