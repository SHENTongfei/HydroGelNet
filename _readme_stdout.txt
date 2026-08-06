# SIMPLEX

**SIMPLEX: Composition-Space Deep Learning for Hydrogel Adhesion with Out-of-Distribution Extrapolation to Model-Discovered High-Performance Formulations**

SIMPLEX (Simplex composition encoding with Interaction-aware attention,
Multi-modal fusion, Pretraining-ready regularisation, Learnable domain
constraints and EXtrapolation evaluation) is a dual-modality residual deep
network that maps the monomer composition of hydrogels (six molar fractions
on the composition simplex plus their 15 pairwise interactions) to underwater
glass adhesion strength (kPa).

**Key idea — model-guided extrapolation evaluation.** A model trained on 180
low-performance formulations must rank 161 high-performance formulations
discovered by the original study's sequential model-based optimisation (SMBO)
loop — a target-value extrapolation beyond the training range. Under this
protocol SIMPLEX significantly outperforms tree ensembles on ranking
(external Spearman rho 0.50 vs 0.21 for random forest; paired bootstrap
difference +0.18, 95% CI [0.07, 0.31]) and achieves the best top-k screening
precision, while matching the strongest baseline in-distribution.

## Results (5-fold grouped CV × 5 seeds)

| Metric | SIMPLEX | RandomForest | ElasticNet |
|--------|---------|--------------|------------|
| Internal CV R² | 0.709 ± 0.077 | 0.719 (tie) | — |
| External Spearman ρ | **0.501** [0.369, 0.619] | 0.211 | 0.494 |
| External Top-20 precision | **0.25** | 0.10 | 0.05 |

## Data

- `sheng-hu/hydrogels` (MIT licence), accompanying Liao et al., *Nature*
  644, 89–95 (2025), DOI 10.1038/s41586-025-09269-4.
- Internal: `data/raw/df_180.csv` (180 round-1 formulations).
- External: `data/raw/df_341.csv` minus `df_180.csv` (161 SMBO-discovered
  formulations).

## Reproduce

```bash
cd code
C:/Users/TS/.conda/envs/HydroGelNet/python.exe run_all.py --all
```

Stages: download → build → qc → tune → train → baselines → stats → gate →
interpret → figures → tables → paper. PERF-GATE must pass before writing.

## Manuscript

- `paper/frontiers_SIMPLEX.tex` — Frontiers LaTeX (Overleaf project
  6a6a083446657df2cc7a741e).
- `paper/SupplementaryMaterial.md` / `paper/simplex_supplementary.tex`.
- `paper/manuscript.md` — Markdown source.
- `FRONTIERS_SUBMISSION.txt` — title + article type (Original Research).

## Environment

Python 3.11 (conda env `HydroGelNet`, cloned from `py311`), PyTorch 2.14.0
+ cu130, NVIDIA RTX 5080 16 GB. Fixed seeds [42, 2024, 7, 1337, 20260731];
all preprocessing fitted inside training folds only.

