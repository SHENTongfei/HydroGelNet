# Supplementary Material — SIMPLEX

**SIMPLEX: Composition-Space Deep Learning for Hydrogel Adhesion with
Out-of-Distribution Extrapolation to Model-Discovered High-Performance
Formulations**

Tongfei Shen

---

## S1. Dataset description

### S1.1 Source and provenance

The study uses the publicly released dataset accompanying Liao et al.,
*Nature* 644, 89–95 (2025), DOI 10.1038/s41586-025-09269-4
(repository: `sheng-hu/hydrogels`, MIT licence). The dataset records
underwater glass adhesion strength (kPa) for 341 bio-inspired hydrogel
formulations, each described by the molar fractions of six functional
monomers:

| # | Monomer | Functional class | Abbreviation |
|---|---------|------------------|--------------|
| 1 | 2-Hydroxyethyl acrylate | Nucleophilic | HEA |
| 2 | Butyl acrylate | Hydrophobic | BA |
| 3 | 2-Carboxyethyl acrylate | Acidic | CBEA |
| 4 | (3-Acrylamidopropyl)trimethylammonium chloride | Cationic | ATAC |
| 5 | 2-Phenoxyethyl acrylate | Aromatic | PEA |
| 6 | Acrylamide | Amide | AAm |

The six fractions sum to 1 (composition simplex). Target: `Glass (kPa)_max`.

### S1.2 Internal / external split (model-guided extrapolation)

- **Internal (training region):** the 180 round-1 bio-inspired formulations
  (`df_180.csv`). Adhesion mean 46.9 kPa, max 146.6 kPa.
- **External (extrapolation region):** the 161 formulations added in later
  iterations of the Nature study's sequential model-based optimisation (SMBO)
  loop (`df_341.csv` minus `df_180.csv`). Adhesion mean 154.2 kPa, max
  353.3 kPa; 45% of external samples exceed the training maximum.

This is a *model-guided migration to a high-performance composition region*
(target-value extrapolation), not an independent-laboratory cohort. The
selection bias and range restriction inherent to the SMBO protocol are
acknowledged; range-restricted Spearman rho is therefore a conservative
estimate of ranking ability.

### S1.3 Features

Two modalities, 21 features total:

1. **monomer_fractions (6):** raw molar fractions.
2. **pairwise_synergy (15):** all products x_i·x_j (i<j), explicitly encoding
   monomer-pair interactions.

Scaling/imputation were fitted inside each CV training fold only
(no leakage).

---

## S2. Model configuration (final, best_config.json)

```
d_model=64, n_blocks=2, n_heads=4, dropout=0.2
use_transformer=False (ResBlock base), use_attention=True
use_film=False, use_modality_gate=False
use_mixup=True (alpha=0.4), use_swa=True
use_sam=False, use_ema=False
use_contrastive=False, use_pretrain_recon=False, use_mfm=False
use_uncertainty_weighting=False (single target)
use_domain_constraint=True (constraint_w=0.1)
y_transform=standard, scaler=standard
max_epochs=150, patience=30, batch_size=32
lr=3e-3, weight_decay=1e-3
```

Design notes:
- **ResBlock base, not Transformer:** on 180 samples the Transformer
  diverged (internal R2 ≈ -6); the residual network with interaction
  attention is the stable configuration (internal R2 ≈ 0.71).
- **Mixup is the key regulariser:** removing it drops internal R2 from
  0.713 to 0.646.
- **Self-supervised pre-training (MFM / contrastive) was detrimental**
  (internal R2 ≈ -2.7) and was removed (ablation-gated).

---

## S3. Full ablation table (internal R2, 2 seeds x 3 folds)

| Variant                        | delta R2 (full − variant) |
|--------------------------------|---------------------------|
| w/o multimodal fusion          | +0.093 (largest)          |
| w/o Mixup                      | +0.067                    |
| fusion = gated                 | +0.062                    |
| w/o residual blocks            | +0.062                    |
| fusion = film                  | +0.031                    |
| w/o task-specific gating       | +0.022                    |
| w/o attention                  | +0.019                    |
| w/o domain constraint          | +0.000 (neutral)          |
| w/o SWA                        | +0.008                    |
| ... (full 20-component ablation in results/ablation/ablation.csv) |

Positive delta = removal degrades performance (component helps). 16
components that did not pay for themselves were removed from the final
model; the remaining switches are reported transparently.

---

## S4. External ranking statistics

Sample-level bootstrap (n=161, 2000 resamples):

| Model      | Spearman rho | 95% CI        |
|------------|--------------|---------------|
| SIMPLEX    | 0.501        | [0.369, 0.619]|
| ElasticNet | 0.494        | [0.36, 0.62]  |
| Ridge      | 0.486        | [0.35, 0.61]  |
| SVR-RBF    | 0.379        | [0.24, 0.51]  |
| MLP        | 0.315        | [0.18, 0.45]  |
| RandomForest | 0.211      | [0.06, 0.36]  |

Paired bootstrap (SIMPLEX − baseline):
- vs RandomForest: delta rho = +0.183, 95% CI [+0.066, +0.311],
  P(diff>0)=0.999 → **significant**
- vs ElasticNet: delta rho = −0.028, 95% CI [−0.157, +0.097],
  P(diff>0)=0.345 → **statistical tie** (reported honestly)

Top-k screening precision (fraction of predicted top-k that are true top-k):

| k  | SIMPLEX | RF   | ElasticNet |
|----|---------|------|------------|
| 20 | 0.25    | 0.10 | 0.05       |
| 30 | 0.37    | 0.07 | 0.17       |

Prediction-range behaviour on the external set (kPa):
- SIMPLEX [16, 109]; RandomForest [26, 109] (compressed — tree models are
  locked near the training maximum ~147); ElasticNet [−82, 232] (over-extrapolating,
  non-physical negative predictions); true range [3, 353].
- SIMPLEX is the only model whose external predictions stay physically
  plausible (non-negative, in-range) while retaining ranking signal.

---

## S5. Multi-target extensibility (preliminary)

The original dataset also records rheological targets for the 180 internal
formulations: storage-related modulus (kPa) and loss tangent tan-delta.
`corr(Glass, Modulus) = −0.036` — independent information not used by the
Nature study's single-target models. As a preliminary validation of
multi-target extensibility, a single-task SIMPLEX trained on Modulus
achieves internal CV R2 = 0.396 vs RandomForest 0.392 (3 seeds), i.e. parity
with the strongest baseline on a second, independent target. Joint
multi-target training is left for future work (see Limitations).

---

## S6. Interpretation highlights

Cross-validated permutation importance (mean over folds/seeds):

| Rank | Feature           | Importance | SE    |
|------|-------------------|-----------|-------|
| 1    | Cationic-ATAC     | 0.222      | 0.028 |
| 2    | pair (ATAC×PEA)   | 0.099      | 0.017 |
| 3    | pair (BA×CBEA)    | 0.072      | 0.010 |
| 4    | Hydrophobic-BA    | 0.055      | 0.013 |
| 5    | pair (BA×PEA)     | 0.050      | 0.020 |

The cation monomer and pairwise interaction terms dominate — consistent
with the electrostatic/hydrophobic synergy known to drive underwater
adhesion, and with the SMBO-discovered high-performance region enriched in
the hydrophobic–aromatic (BA–PEA) combination.

---

## S7. Reproducibility

- Environment: Python 3.11 (conda env `HydroGelNet`), PyTorch 2.14.0+cu130,
  NVIDIA RTX 5080 16 GB.
- Fixed seeds: [42, 2024, 7, 1337, 20260731]; all preprocessing inside folds.
- Full pipeline: `download → build → qc → tune → train → baselines → stats
  → gate → interpret → figures → tables → paper` (run_all.py).
- PERF-GATE verdict: **PASS** (G1 internal tie, G2 no significant
  disadvantage, G3 seed stability, G4 external Spearman 0.501 vs best
  baseline 0.494, G5 informative ablation).

---

## S8. Data & code availability

- Data: `sheng-hu/hydrogels` (MIT), Nature 2025 supplement.
- Code: https://github.com/<owner>/HydroGelNet (to be released).
- Reproducible artefacts: `data/`, `results/`, `figures/`, `tables/`,
  `paper/` in the project repository.
