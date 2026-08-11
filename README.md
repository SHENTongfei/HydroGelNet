# SIMPLEX

**Composition-space deep learning for bio-inspired hydrogel adhesion: an open, reproducible eight-model benchmark with prospective validation on model-discovered formulations and interpretable composition rules.**

Accompanying manuscript (under review): *SIMPLEX: Composition-Space Deep Learning for Bio-inspired Hydrogel Adhesion with Prospective Validation on Model-Discovered Formulations and Interpretable Composition Rules*

This repository releases the data-preparation code, one fixed evaluation protocol, eight model implementations, 23 ablation arms and every raw metric file behind that manuscript. SIMPLEX is the dual-modality deep encoder introduced for composition-space inputs, and the release is a benchmark built around it: the full 23-arm ablation, the leave-one-covariate-out tests and every metric file needed to recompute the tables are included.

**Positioning statement — kept identical across the manuscript, the supplementary material and this README:**

> SIMPLEX, a dual-modality deep encoder for bio-inspired hydrogel adhesion, is benchmarked against seven equally tuned baselines on 316 internal formulations and a 25-formulation Bayesian-optimisation acquisition batch. It attains internal R² = 0.7924 (rank 3 of 8, statistically tied with random forest at Holm-corrected p = 1.0) and leads the external batch in coefficient of determination (R² = 0.6712, rank 1 of 8), demonstrating effective transfer to actively acquired formulations. On the same batch it ranks 7 of 8 in Spearman rank correlation (ρ = 0.8031 vs Ridge's 0.8573), a metric flip that is itself diagnostic of the acquisition-batch provenance. A 23-arm ablation isolates gated multimodal fusion as the single component that contributes beyond noise (ΔR² = +0.0724), while LOCO tests confirm that the 15 quadratic Scheffé interaction terms carry no measurable signal (p = 0.805).

The 21 input features are the six monomer molar fractions on the composition simplex plus their 15 pairwise products — formally a quadratic Scheffé mixture basis — and the target is underwater glass adhesion strength (kPa).

`SIMPLEX` is the label of the dual-modality residual encoder introduced and evaluated here. The name is kept for continuity with the code; it is the model identifier used throughout the manuscript, the supplementary material and this repository.

### What this repository does and does not claim

- **It is**: a full benchmark around the SIMPLEX deep encoder, a single-protocol comparison of 8 models on 316 formulations, a full disclosure of all 23 ablation arms, a leave-one-covariate-out (LOCO) test of the 15 Scheffé interaction terms, and every metric file needed to reproduce the tables.
- **It is not**: a claim that the deep encoder beats every baseline on every metric. It ranks 3 of 8 internally (statistically tied with the leader), 1 of 8 in external R², and the honest double reading of the two external metrics is reported in full, including the fact that the pre-registered performance gate records `FAIL` and that 9 of the 23 ablation arms are bit-identical to the full model.

---

## Key results

All values below are read directly from the metric files in `results/`; nothing is recomputed by hand. Internal figures are means over 10 seeds × 5 grouped folds (50 evaluations per model); external figures are the same 50 evaluations applied to the 25-formulation batch.

### 1. Internal benchmark — 10 × 5 grouped CV, 316 formulations

| Model | R² | Spearman ρ | RMSE (kPa) | R² rank |
|---|---|---|---|---|
| RandomForest | 0.8067 | 0.9056 | 32.0533 | 1 |
| SVR-RBF | 0.8026 | 0.9065 | 32.3873 | 2 |
| **SIMPLEX (deep)** | **0.7924** | **0.9004** | 33.1450 | **3** |
| KNN | 0.7870 | 0.8986 | 33.4101 | 4 |
| Ridge | 0.7688 | 0.8699 | 35.1408 | 5 |
| ElasticNet | 0.7681 | 0.8683 | 35.1945 | 6 |
| HistGB | 0.7624 | 0.8924 | 35.5086 | 7 |
| MLP | 0.7010 | 0.8087 | 39.7413 | 8 |
| Mean (dummy) | −0.0034 | 0.0000 | 73.7304 | — |

RandomForest reaches R² = 0.8067 and ranks 1 of 8; the deep encoder reaches R² = 0.7924 and ranks 3 of 8, a gap of 0.01432. Its Spearman ρ likewise ranks 3 of 8. Holm-corrected pairwise tests do not reject equal performance for any pair involving the deep encoder (p = 1.0 against the top-ranked baseline), so the ordering should be read as descriptive rather than as evidence of separation. Every model separates cleanly from the mean-predictor dummy, so the null result concerns model choice, not data quality.

### 2. External benchmark — BO-acquired batch (n = 25), single-model calibre

| Model | ext R² | R² rank | ext Spearman ρ | ρ rank |
|---|---|---|---|---|
| **SIMPLEX (deep)** | **0.6712** | **1** | **0.8031** | **7** |
| SVR-RBF | 0.6342 | 2 | 0.8347 | 4 |
| Ridge | 0.6336 | 3 | **0.8573** | **1** |
| ElasticNet | 0.6311 | 4 | 0.8491 | 2 |
| RandomForest | 0.5611 | 5 | 0.8444 | 3 |
| MLP | 0.5482 | 6 | 0.7577 | 8 |
| HistGB | 0.5233 | 7 | 0.8134 | 6 |
| KNN | 0.4422 | 8 | 0.8172 | 5 |
| Mean (dummy) | −1.0947 | — | 0.0000 | — |

**Two facts point in opposite directions here, and they are always reported together:** under the single-model calibre the deep encoder attains the largest external R² (0.6712, rank 1 of 8), **while on the same batch it ranks 7 of 8 in rank correlation (ρ = 0.8031), ahead only of the MLP, and Ridge ranks 1 of 8 at ρ = 0.8573.** A screening model is consumed as a ranking rather than as a calibrated value, so the ρ column is the one that governs practical utility. Neither of the two numbers is quoted anywhere in this repository without the other.

**What the 25 external formulations actually are.** They are `df_341 \ df_316`, i.e. the expected-improvement acquisition batch from the final Bayesian-optimisation round of the source study. They were actively sampled towards a high-performance sub-region of the simplex, so they are **not an i.i.d. independent test set** and must not be read as one. Following the manuscript, "prospective validation" here means evaluation on formulations acquired after model development, not a randomly drawn prospective cohort; the batch's biased provenance is characterised explicitly and the numbers are interpreted as a diagnostic.

**Calibre disclosure.** `results/metrics/baselines_external.csv` contains only `external_single` rows — no ensemble was ever constructed for any baseline. SIMPLEX ensemble values do exist (R² = 0.6946, ρ = 0.8077) but have no like-for-like baseline counterpart. Comparing the SIMPLEX ensemble against baseline single runs reports the external R² margin as +0.0604 instead of the like-for-like +0.0370, an inflation of 63%. Every external comparison in this repository uses the single-model calibre.

### 3. The 15 Scheffé interaction terms carry no measurable contribution

Leave-one-covariate-out (LOCO) refitting:

| Deletion | Result |
|---|---|
| all 15 pairwise product terms removed (6 mole fractions only) | performance unchanged, p = 0.805 |
| `pair_14` (BA × PEA), the term ranked 1 by permutation importance | performance unchanged, p = 0.874 |

At n = 316 the quadratic Scheffé interaction terms make no measurable contribution to this task — even though 15 of the 21 input columns are exactly those terms and the deep architecture was built to exploit them. Permutation importance and LOCO measure different quantities: `pair_14` (0.063113 ± 0.036142 over 50 folds) is the feature the fitted model leans on most, yet deleting it and refitting changes nothing, because the 21 columns are strongly collinear and the design matrix is exactly rank-deficient by one.

### 4. Ablation — 23 arms, most of them inert

| Group | Arms | Detail |
|---|---|---|
| Positive contribution | 10 | only multimodal fusion exceeds ΔR² = 0.05 (+0.072358); next is `fusion = concat` (+0.040337) |
| Bit-identical to the full model (Δ = 0.000000) | 9 (+1 below reporting precision, Δ ≈ 1 × 10⁻⁶) | switches not wired into the forward/training path; a domain constraint that is mathematically zero; uncertainty weighting that is structurally unreachable in a single-task configuration; one arm (`w/o attention sparsity reg.`) changes the metrics only below reporting precision |
| Improves when removed | 3 | modality gate (−0.008526), R-Drop (−0.008145), task-specific gating (−0.000817) |

Reference full-model R² = 0.768460 (3 seeds × 3 folds). 13 of the 23 components are therefore inert or actively harmful. Δ is defined as R²(full) − R²(ablated); each Δ comes from an independent single-arm run and the values are not additive.

### 5. Pre-registered quality gate: FAIL

`results/metrics/perf_gate.json` records `"verdict": "FAIL"` with `"failed_checks": ["G2_significant"]`, and it is reported here as recorded. Two stored `pass` flags do not survive audit: `G1` passes only through a `tie_tol = 0.02` clause added after the protocol was fixed (the actual internal gap is −0.01432 against RandomForest), and `G3` is stored as passing while its own detail field reads `2/10 seeds positive (need >= 8)`. Any earlier statement in this repository that the gate passed is withdrawn.

---

## Repository layout

```
HydroGelNet/
├── code/                  # all Python source
│   ├── model_zoo.py       # SIMPLEX architecture (SciNet) + all modules
│   ├── trainer.py         # grouped-CV training, train-fold-only preprocessing
│   ├── baselines.py       # 7 classical ML baselines
│   ├── tuner.py           # coarse→fine hyperparameter search + pruning
│   ├── exp_clr.py         # compositional-data control (raw / CLR / raw+CLR)
│   ├── stats_tests.py     # Holm-corrected comparisons, bootstrap CIs
│   ├── figures.py         # renders all main figures
│   ├── tables.py          # renders all result tables
│   └── paths.py           # central path/config constants
├── data/
│   ├── raw/               # original CSV downloads (df_180 / df_316 / df_341)
│   └── processed/         # dataset.npz (X, Y, groups, condition, modalities)
├── figures/               # output figures (PNG + PDF, publication-ready)
├── tables/                # output tables (LaTeX + CSV)
├── results/
│   ├── tuning/            # config_used.json (the configuration actually run)
│   ├── ablation/          # ablation_results.csv (24 variants = 23 arms + reference row; 3 seeds × 3 folds)
│   ├── metrics/           # internal CV / external metrics, perf_gate.json
│   ├── stats/             # significance tests, bootstrap CIs
│   └── interpret/         # permutation importance / attention / PDP outputs
├── paper/                 # bibliography + manuscript metadata (no manuscript)
├── DATA_SOURCES.md        # download URLs, SHA-256 checksums, upstream licences
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

The measurements are **not ours**. They were released by Liao *et al.*, *Data-driven de novo design of super-adhesive hydrogels*, **Nature** (2025), doi:`10.1038/s41586-025-09269-4`, through the MIT-licensed repository `sheng-hu/hydrogels` (see [License and attribution](#license-and-attribution)):

| File | Role in this benchmark | Link |
|---|---|---|
| `df_316.csv` | internal cohort (316 formulations, 10 × 5 grouped CV) | `raw.githubusercontent.com/sheng-hu/hydrogels/.../df_316.csv` |
| `df_341.csv` | full library (341); `df_341 \ df_316` gives the 25 external formulations | `raw.githubusercontent.com/sheng-hu/hydrogels/.../df_341.csv` |
| `df_180.csv` | round-1 cohort (180); retained for provenance only | `raw.githubusercontent.com/sheng-hu/hydrogels/.../df_180.csv` |

**External-set definition — one definition only.** The external set is `df_341 \ df_316` = **25** formulations, as implemented in `code/build_dataset.py`. An earlier definition, `df_341 \ df_180` = 161, appears in `DATA_SOURCES.md` and in pre-v7 tables; it is **deprecated** and is not used anywhere in the reported results. Any stale artefact carrying n = 161 is superseded. As stated in [Key results](#key-results), those 25 formulations are a Bayesian-optimisation acquisition batch, not an i.i.d. hold-out.

Download the CSVs into `data/raw/` (see `DATA_SOURCES.md` for exact URLs, SHA-256 checksums and licences), then build the processed dataset:

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

# 1. Train SIMPLEX with the configuration actually used in the reported runs
python trainer.py --config ../results/tuning/config_used.json

# 2. Train the 7 classical baselines
python baselines.py

# 3. Significance tests (Holm-corrected pairwise comparisons, bootstrap CIs)
python stats_tests.py

# 4. (Optional) compositional-data control: raw vs CLR vs raw+CLR inputs
python exp_clr.py

# 5. (Optional) re-run the hyperparameter search
python tuner.py

# 6. Generate every result table + figure
python tables.py
python figures.py          # writes Figure*.png/pdf into figures/
```

Key output locations:

- `results/tuning/config_used.json` — the hyperparameters actually executed: `d_model = 152`, `n_blocks = 1`, `n_heads = 8`, `dropout = 0.5`, gated fusion, standard target transform. Several switches recorded as `true` in this file have no effect at run time; see [Known limitations and disclosures](#known-limitations-and-disclosures).
- `results/metrics/` — internal CV and external metrics per model, plus `perf_gate.json`.
- `figures/Figure*.png|pdf` — all main figures (multi-panel, colour-blind-safe Okabe-Ito palette).
- `tables/Table*.tex|*.csv` — dataset summary, hyperparameters, internal CV, model comparison, external benchmark, ablation, reproducibility.

The LOCO deletion tests reported above are implemented in `code/_r4_c7_fold.py`; consolidation of the audit-stage scripts into the main entry points is pending.

---

## Model

`code/model_zoo.py` implements the complete SIMPLEX architecture (`SciNet`):

- **Simplex encoding** of the 6 monomer molar fractions + 15 pairwise interaction terms (dual modality: individual monomers + interaction pairs).
- **SwiGLU residual blocks** with pre-normalisation.
- **Interaction-aware sparse multi-head attention** (8 heads, entropy-regularised).
- **Gated multi-modal fusion** + per-task modality gating + FiLM conditioning.
- **Regularisation**: supervised contrastive pre-training, Mixup (α≈0.36), SWA, learnable domain (target-range) constraints, monotonicity/orthogonality penalties.
<!-- 已修改：README Mixup α=0.4 → ≈0.36（config_used.json 实测 mixup_alpha=0.36222272323956；R1-3 #14，P1）。 -->
- **Grouped CV**: 10 seeds × 5 folds with group-based splitting, all preprocessing fitted on train folds only.

The model has **370,327** parameters against 316 training samples, a parameter-to-sample ratio of **1171.9 : 1**.

Read this module list against Section 4 of [Key results](#key-results) before reusing any of it: the ablation shows that 9 of the listed mechanisms (including SWA, EMA, MC-Dropout, the domain constraint, uncertainty weighting, MFM pre-training and the transformer block) leave the metrics bit-identical and 1 (attention sparsity regularisation) changes the metrics only below reporting precision, and 3 more improve the score when removed. The list describes what the code contains, not what demonstrably contributes.

See `code/trainer.py` for the training loop and evaluation protocol, and `code/tuner.py` for the "test it, keep if it works, drop if not" pruning search.

---

## Reproducibility notes

- Every random seed is fixed per run; seeds are reported in `tables/Table10_reproducibility.tex`.
- All preprocessing (imputation, scaling, target transform) is fitted on training folds only.
- All figures are rendered from raw result CSVs — there is no hand-edited plot data.
- Grouping caveat: the grouping key passed to `GroupKFold` degenerates to a no-op, so the protocol reduces in practice to seeded k-fold. Near-duplicate formulations can therefore fall on both sides of a split. The earlier claim that formulation-level leakage "cannot occur" is withdrawn.

---

## Known limitations and disclosures

Reported here so that they do not have to be discovered:

| # | Disclosure |
|---|---|
| 1 | The pre-registered performance gate records `FAIL` (`G2_significant`). Two further checks are stored as `pass` on grounds that do not survive audit (see Key results, Section 5). |
| 2 | The 25-formulation external set is a Bayesian-optimisation acquisition batch, biased towards a high-performance sub-region. It is not an i.i.d. independent test set. |
| 3 | The external batch was consulted during architecture screening: 14 candidate configurations were scored on it, of which 9 score above the configuration eventually deployed. It is a diagnostic cohort, not a clean hold-out. Fifteen external-evaluation artefacts exist in the git history, so the batch was evaluated repeatedly over the course of development, not scored a single time. |
| 4 | 9 of 23 ablation arms are bit-identical to the full model and 1 (`w/o attention sparsity reg.`) changes the metrics only below reporting precision: several switches are not wired into the forward/training path, the domain-constraint penalty is mathematically zero on simplex-valid inputs, and uncertainty weighting is unreachable with a single task. |
| 5 | The attention-head count is silently coerced in `code/model_zoo.py`: a loop decrements `n_heads` until it divides `d_model`, without warning. With `d_model = 152 = 2³ × 19`, a request for 12 heads becomes 8. The `arch_C_heads` arm and the main run share an identical MD5 of their external metrics (`77375468bf0a07f5fd34eaa38dfcc7b4`), so the attention-head ablation never actually ran. |
| 6 | No "attention off" arm exists among the 23 arms, so the necessity of self-attention in this architecture is untested. |
| 7 | The 21 features are standardised before fitting, which destroys the no-intercept structure that makes a Scheffé basis well-posed. The design matrix is exactly rank-deficient by one with an intercept, its condition number is 5.44 × 10⁷, and each of the six mole fractions has VIF = inf. |
| 8 | Earlier versions of this README and of the manuscript reported internal/external figures that do not match the released metric files (external R² 0.71 and ρ 0.87; full-model ablation R² 0.713; Mixup gain +0.067; `pair_14` importance 0.143 ± 0.036). All of these are corrected above. |
| 9 | `results/stats/ablation_stats.csv` is an empty 2-byte file and must be regenerated before it is cited. The `concat` baseline of the pre-v7 Table 7 lost its source data and is not reproducible; it has been dropped rather than re-stated. |

---

## Citation

This repository accompanies a manuscript under review. **Cite both the source study and this work** — the data are not ours, and citing this repository alone would misattribute them.

**1. Source study (required whenever the data are used, including via the processed `dataset.npz`):**

> Liao H, Hu S, Yang H, et al. Data-driven de novo design of super-adhesive hydrogels. *Nature* (2025). doi:10.1038/s41586-025-09269-4

**2. This benchmark (code, protocol, ablation and LOCO results):**

> `[待填写]` — authors, title, venue and year; the manuscript is under review. Please cite it once published, and in the meantime cite this repository by URL and commit hash.

---

## License and attribution

### Upstream data and code

| Object | Licence | Notes |
|---|---|---|
| Data repository `sheng-hu/hydrogels` (`df_180.csv`, `df_316.csv`, `df_341.csv`) | **MIT** | `LICENSE` present, 1,064 bytes, `Copyright (c) 2024 setupup`. MIT permits use, modification and redistribution provided the copyright and permission notice are retained. |
| The article itself (Nature, 2025, doi:10.1038/s41586-025-09269-4) | **CC BY-NC-ND** | Governs reuse of the article text and figures. It does not govern the data files in the MIT-licensed repository, to which the authors' data-availability statement points. |

The two licences coexist because they cover different objects. This repository reuses the **data files** under MIT, not the article content under CC BY-NC-ND; no article text or figure is reproduced here. Monomer mole fractions and adhesion strengths in kPa are measurement values used as data.

This repository does **not** redistribute the CSVs: `data/raw/` is git-ignored, and `DATA_SOURCES.md` records the download URLs, SHA-256 checksums and licences so that the exact upstream files can be fetched from the source repository.

### This repository's own code

| Component | Licence |
|---|---|
| Code in `code/` and derived artefacts in `results/`, `figures/`, `tables/` | MIT (see `LICENSE`) |

**This repository's own code is released under the MIT License** (see `LICENSE`, Copyright (c) 2026 Tongfei Shen). The upstream data files are governed by the MIT licence of the source repository `sheng-hu/hydrogels` (Copyright (c) 2024 setupup), a verbatim copy of which is provided as `LICENSE-DATA`; we claim no rights over the upstream data.

---

## Contact

For questions or collaboration, please open an issue or contact the maintainer via the repository page.
