# PROTEUS Data

Pre-computed embeddings and benchmark datasets are archived on Zenodo.

## Downloading embeddings

```bash
# Fold-switch benchmark embeddings (368 proteins, K=10 conformations)
zenodo_get TODO_ZENODO_DOI --output-dir data/fold_switch/

# OC23 open/close validation embeddings (23 proteins x 2 states)
zenodo_get TODO_ZENODO_DOI --output-dir data/oc23/

# GAS proteome embeddings (1,489 S. pyogenes M1 proteins)
zenodo_get TODO_ZENODO_DOI --output-dir data/gas_proteome/
```

## Directory structure (after download)

```
data/
  fold_switch/
    embeddings/          # NPZ files, one per protein
    labels.csv           # protein_id, label (0=monostate, 1=fold-switch)
    resolution.csv       # PDB resolution metadata
  oc23/
    embeddings/          # NPZ files, protein_id_open.npz / protein_id_close.npz
    OC23.csv             # metadata with RMSD and state labels
  gas_proteome/
    embeddings/          # NPZ files for all 1,489 GAS proteins
    gas_metadata.csv     # UniProt accessions, VF labels, functional categories
```

## Generating embeddings from scratch

If you want to recompute embeddings using your own SimpleFold installation:

```bash
python scripts/extract_embeddings.py \
    --fasta data/fold_switch/sequences.fasta \
    --model_path /path/to/simplefold_weights \
    --output_dir data/fold_switch/embeddings/ \
    --n_conformations 10 \
    --num_steps 25
```

See `scripts/extract_embeddings.py --help` for all options.
