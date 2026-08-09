# SIMPLEX

**SIMPLEX: Composition-Space Deep Learning for Bio-inspired Hydrogel Adhesion with Prospective Validation on Model-Discovered Formulations and Interpretable Composition Rules**

SIMPLEX (**S**implex encoding with **I**nteraction-aware attention, **M**ulti-modal fusion, **P**retraining, **L**earnable constraints and e**X**trapolation) is a dual-modality residual deep network that maps the **monomer composition** of bio-inspired hydrogels to **underwater glass adhesion strength (kPa)**.

It learns directly on the 6-dimensional composition simplex (six monomer molar fractions) plus their 15 pairwise interaction terms, and is validated **prospectively** on held-out, high-performance formulations that a model-guided optimisation loop would have discovered — i.e. the model must *extrapolate in composition space* to rank candidates it has never seen.

---

## Key results (summary)

| Metric | SIMPLEX |
|---|---|
| Internal CV R² (10×5 grouped, glass adhesion) | **0.792** |
| Internal CV Spearman ρ | **0.900** |
| Prospective held-out R² (n = 25) | **0.695** |
| Prospective held-out Spearman ρ | **0.808** |
| Baseline comparison | outperforms 7 classical ML baselines (MLP, HistGB, Ridge, KNN, SVR, RF, ElasticNet) on prospective ranking & top-k screening |

- **316** internal formulations, **25** prospective held-out formulations (high-performance region, composition-space extrapolation).
- Composition-space **extrapolation** protocol: the training target range is low-performance; the held-out cohort lives at the high-performance end — the ranking task is therefore harder than in-distribution interpolation.
- **Interpretable**: attention + SHAP-style permutation importance recover physically meaningful monomer roles (e.g. Nucleophilic-HEA / Hydrophobic-BA) and interaction pairs; a stability-selection sweep confirms the selected features are reproducible.

---

## Repository layout

```
HydroGelNet/
├── code/                  # all Python source
│   ├── model_zoo.py       # SIMPLEX architecture (SciNet) + all modules
│   ├── trainer.py         # grouped-CV training, leakage-safe preprocessing
│   ├── baselines.py       # 7 classical ML baselines
│   ├── tuner.py           # coarse→fine hyperparameter search + pruning
│   ├── figures.py         # renders all main figures (Nature style)
│   ├── tables.py          # renders all result tables
│   └── paths.py           # central path/config constants
├── data/
│   ├── raw/               # original CSV downloads (df_180 / df_316 / df_341)
│   └── processed/         # dataset.npz (X, Y, groups, condition, modalities)
├── figures/               # output figures (PNG + PDF, publication-ready)
├── tables/                # output tables (LaTeX + CSV)
├── results/
│   ├── tuning/            # best_config.json, search logs
│   ├── ablation/          # ablation study results
│   ├── metrics/           # CV / external metrics
│   └── interpret/         # importance / attention / PDP outputs
├── paper/                 # bibliography + manuscript metadata (no manuscript)
└── README.md
```

---

## Installation

Requires **Python 3.11+** and a GPU is recommended (training is small enough for CPU but much faster on GPU).

```bash
conda create -n hydrogel python=3.11 -y
conda activate hydrogel
pip install torch numpy pandas scikit-learn scipy matplotlib
```

Core dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`.

---

## Data

Data come from the open **MIT-licensed** repository of Liao *et al.*, *Data-driven de novo design of super-adhesive hydrogels*, **Nature** (2025), doi:`10.1038/s41586-025-09269-4`:

| File | Role | Link |
|---|---|---|
| `df_316.csv` | internal (316 formulations) | `raw.githubusercontent.com/sheng-hu/hydrogels/.../df_316.csv` |
| `df_341.csv` | screening pool (341) | `raw.githubusercontent.com/sheng-hu/hydrogels/.../df_341.csv` |
| `df_180.csv` | round-1 baseline (180, low-performance) | `raw.githubusercontent.com/sheng-hu/hydrogels/.../df_180.csv` |

Download the CSVs into `data/raw/` (see `DATA_SOURCES.md` for exact URLs, SHA-256 checksums and licenses), then build the processed dataset:

```bash
cd code
python build_dataset.py        # -> data/processed/dataset.npz
python data_qc.py              # optional: distribution-shift QC report
```

`dataset.npz` contains: `X` (features), `Y` (targets), `groups` (formulation groups for grouped CV), `cond`/`cond_levels` (condition metadata), `feature_names`, `target_names`, `modality_ends`/`modality_names` (dual-modality split), `sample_ids`.

---

## Quick start (reproduce end-to-end)

```bash
cd code

# 1. Train SIMPLEX with the published best configuration
python trainer.py --config results/tuning/best_config.json

# 2. Train the 7 classical baselines
python baselines.py

# 3. (Optional) re-run the hyperparameter search
python tuner.py

# 4. Generate every result table + figure
python tables.py
python figures.py          # writes Figure*.png/pdf into figures/
```

Key output locations:

- `results/tuning/best_config.json` — final hyperparameters (d_model=152, 8 heads, gated fusion, FiLM conditioning, task gating, entropy-regularised sparse attention, Mixup+SWA+domain constraints).
- `results/metrics/` — CV and prospective metrics per model.
- `figures/Figure*.png|pdf` — all main figures (3×2 / 3×3 multi-panel, Nature style, colour-blind-safe Okabe-Ito palette).
- `tables/Table*.tex|*.csv` — dataset summary, hyperparameters, internal CV, baseline comparison, external validation, ablation, candidate markers, reproducibility.

---

## Model

`code/model_zoo.py` implements the complete SIMPLEX architecture (`SciNet`):

- **Simplex encoding** of the 6 monomer molar fractions + 15 pairwise interaction terms (dual modality: individual monomers + interaction pairs).
- **SwiGLU residual blocks** with pre-normalisation.
- **Interaction-aware sparse multi-head attention** (8 heads, entropy-regularised).
- **Gated multi-modal fusion** + per-task modality gating + FiLM conditioning.
- **Regularisation**: supervised contrastive pre-training, Mixup (α=0.4), SWA, learnable domain (target-range) constraints, monotonicity/orthogonality penalties.
- **Leakage-safe grouped CV**: 10 seeds × 5 folds with group-based splitting (formulations never split across folds), all preprocessing fitted on train folds only.

See `code/trainer.py` for the training loop and evaluation protocol, and `code/tuner.py` for the "test it, keep if it works, drop if not" pruning search.

---

## Reproducibility notes

- Every random seed is fixed per run; seeds are reported in `tables/Table10_reproducibility.tex`.
- The grouped-CV protocol (10×5, group-based) and the leakage-safe preprocessing guarantee that no formulation-level leakage can occur.
- All figures are rendered from raw result CSVs — there is no hand-edited plot data.

---

## Citation

This repository accompanies a manuscript under review.

If you use the code or the protocol, please cite the manuscript once it is published, and the underlying data:

> Liao H, Hu S, Yang H, et al. Data-driven de novo design of super-adhesive hydrogels. *Nature* (2025). doi:10.1038/s41586-025-09269-4

---

## License

Code: see repository license. Data: MIT (per the original `sheng-hu/hydrogels` repository).

---

## Contact

For questions or collaboration, please open an issue or contact the maintainer via the repository page.
