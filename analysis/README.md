# Analysis Scripts

All analysis scripts used to produce the results in Piochi et al. (2026).
Scripts are organized by result section. Each script is self-contained and
reads pre-computed embeddings from `data/` (see `data/README.md`).

## fold_switch/

Core fold-switch benchmark (n=368 proteins, 197 monostate / 171 fold-switch).

| Script | Figure |
|---|---|
| `run_confound_analysis.py` | Fig 2, Supp |
| `run_formulation_search.py` | Supp (formulation search) |
| `run_trajectory_analysis.py` | Fig 2 (trajectory profile) |
| `run_resolution_confound_combined.py` | Supp (resolution) |

## validations/

Independent validation datasets (no overlap with training data, no supervision).

| Script | Figure | Dataset |
|---|---|---|
| `run_oc23_analysis.py` | Fig 3 | OC23 open/close crystal pairs |
| `run_dibs_analysis.py` | Fig 3 | DIBS disorder-to-order transitions |
| `run_aiupred_comparison.py` | Fig 3 | AIUPred intrinsic disorder |
| `run_atlas_analysis.py` | Fig 3 | ATLAS MD flexibility (RMSF) |
| `run_phospho_analysis.py` | Fig 4 | Buried phosphorylation sites |

## applications/

Proteome-scale case study.

| Script | Figure |
|---|---|
| `run_gas_proteome_analysis.py` | Fig 5 |

## benchmarks/

Baseline comparisons (ESM-2, SaProt, ablations).

| Script | Purpose |
|---|---|
| `run_proteus_evaluation.py` | ESM-2 and SaProt baselines |
| `run_score_function_analysis.py` | Formulation comparison (full) |
| `run_nconf_ablation.py` | Effect of K (number of conformations) |

## Running a script

All scripts follow the same pattern:

```bash
# Example: fold-switch confound analysis
python analysis/fold_switch/run_confound_analysis.py \
    --scores_csv data/fold_switch/scores.csv \
    --labels_csv data/fold_switch/labels.csv \
    --output_dir results/confound/
```

Run any script with `--help` for full usage.
