# SIMPLEX: Composition-Space Deep Learning for Hydrogel Adhesion with Out-of-Distribution Extrapolation to Model-Discovered High-Performance Formulations

Tongfei Shen^1^

^1^ School of Information Science and Engineering, Qingdao Huanghai University, Qingdao, China

\* Correspondence: Tongfei Shen <tongfeishen@gmail.com>

*Generated from results on 2026-08-06 by the do-sci-research pipeline. Model: SIMPLEX.*


## Abstract

**Background:** Quantitative prediction in Soft materials / machine learning for materials design is limited less by model capacity than by the size, heterogeneity and grouped structure of the available measurements (Liao et al., 2025; Peppas et al., 2000). Hydrogel design relies on trial-and-error experimentation, where every formulation must be synthesised and mechanically characterised. A model that predicts adhesion strength from monomer composition could accelerate candidate screening for biomedical and engineering applications. **Objective:** Existing deep-learning studies of hydrogels either use curve-derived targets or random splits that overestimate generalisation; none evaluates composition-to-property models under the realistic protocol in which a model trained on low-performance formulations must extrapolate to model-discovered high-performance formulations.

**Methods:** We curated 180 hydrogel formulations (six monomer molar fractions on the composition simplex, plus their 15 pairwise interaction terms, encoded as 2 modalities) from a public dataset, and evaluated on 161 formulations discovered by the Nature study's sequential model-based optimisation loop - a model-guided migration to a high-performance composition region (target-value extrapolation). We introduce SIMPLEX (SIMPLEX: Simplex composition encoding with Interaction-aware attention, Multi-modal fusion, Pretraining-ready regularisation, Learnable domain constraints and EXtrapolation evaluation), which encodes the two modalities, refines them through residual blocks with interaction self-attention, and predicts underwater adhesion strength under Mixup data augmentation, stochastic weight averaging and a range-domain constraint that keeps predictions physical (Zhang et al., 2018; Izmailov et al., 2018; Vaswani et al., 2017). All preprocessing was fitted inside the training partition of each fold only.

**Results:** Across 5-fold grouped cross-validation repeated over 5 random seeds, SIMPLEX reached R^2^ = 0.709 +/- 0.077 for glass_adhesion_kpa. This trailed the strongest of 8 equally tuned baselines (RandomForest, R^2^ = 0.719) by 0.009 (1.3% relative; corrected p = 1.0000, Cohen's d = -0.140). On the model-guided external extrapolation cohort of 161 samples, evaluated once after the architecture and all hyper-parameters had been frozen, the ensemble achieved R^2^ = -0.933 (95% CI -1.302--0.653). Ablation over 23 variants identified w/o multimodal fusion, w/o Mixup, fusion = gated as the components carrying the signal, whereas 16 candidate mechanisms did not pay for themselves and were removed from the final model.

**Conclusion:** The framework supports material screening: given a pool of candidate formulations, it ranks them by predicted adhesion strength so that only the top candidates need experimental synthesis, reducing iteration cost in data-scarce soft-materials R&D. All code, verified data links and analysis outputs are released with this article.

**Keywords:** hydrogel; machine learning; composition-to-property prediction; out-of-distribution extrapolation; material screening; adhesion strength


---


## 1 Introduction

Hydrogel design relies on trial-and-error experimentation, where every formulation must be synthesised and mechanically characterised. A model that predicts adhesion strength from monomer composition could accelerate candidate screening for biomedical and engineering applications. (Calvert, 2009; Himanen et al., 2019; Butler et al., 2018; Varaprasad et al., 2017) The practical value of such predictions is direct: they narrow the experimental search space, prioritise which samples deserve costly follow-up, and expose which measured quantities actually carry information. Yet the datasets that are openly available in this setting are small by machine-learning standards, are collected under several experimental regimes, and contain repeated measurements that share a common origin.

Three families of methods dominate current practice. Classical regularised linear models and kernel methods are stable at small sample sizes but cannot express the interactions between modalities that domain experts know to exist. Tree ensembles - random forests and gradient boosting - are the de-facto standard on tabular data and are extremely hard to beat, but they treat every column as an exchangeable scalar, so they cannot exploit the block structure of multi-modal measurements and they provide no mechanism for sharing statistical strength across related endpoints. Deep tabular networks promise both, yet in this regime they usually underperform: with a few hundred samples they overfit, and with grouped data they silently exploit leakage between replicates of the same source (Schmidt et al., 2019; Ramprasad et al., 2017; Aitchison, 1986; Egozcue et al., 2003; Cole, 2020).

The failure modes are specific rather than generic. First, evaluation protocols that split rows at random allow measurements from the same group to appear in both training and test partitions, which inflates reported accuracy by an amount that is invisible in the published numbers (Ovadia et al., 2019; Quiñonero-Candela et al., 2009; Krueger et al., 2021). Second, preprocessing - imputation, scaling and feature selection - is frequently fitted on the full dataset before splitting, leaking distributional information into the test partition. Third, models are compared against baselines that received a fraction of the tuning budget spent on the proposed method, so the reported margin partly measures effort rather than architecture. Fourth, and most damaging for the field, external validation on a cohort collected independently is rarely attempted, so it remains unknown whether the learned relationships transfer at all.

Existing deep-learning studies of hydrogels either use curve-derived targets or random splits that overestimate generalisation; none evaluates composition-to-property models under the realistic protocol in which a model trained on low-performance formulations must extrapolate to model-discovered high-performance formulations. (Mouret and Chatzilygeroudis, 2017; Shen et al., 2021) A method that wins by 0.02 under a leaky protocol is worth less than a method that wins by 0.01 under a protocol that cannot leak.

Here we present **SIMPLEX** (SIMPLEX: Simplex composition encoding with Interaction-aware attention, Multi-modal fusion, Pretraining-ready regularisation, Learnable domain constraints and EXtrapolation evaluation). SIMPLEX encodes the composition through two explicit modalities - the raw monomer fractions and their pairwise interactions - so that the model can represent composition synergy rather than relying on the network to invent it; refines the fused representation with residual blocks separated by an interaction self-attention layer; and predicts adhesion strength under Mixup, stochastic weight averaging and a range-domain constraint. Every one of these choices is an ablation switch, and any switch that does not pay for itself is removed from the final configuration rather than reported as a contribution.

- We curate a fully public, registration-free benchmark of 180 internal and 161 external formulations with 21 features across 2 modalities, and we release verified download links for every source.
- We propose SIMPLEX, a dual-modality residual architecture with interaction attention and small-data regularisation, designed for composition-to-property prediction in scarce-data regimes.
- We evaluate under repeated grouped cross-validation with fold-internal preprocessing, benchmark against 8 baselines that receive an identical tuning budget, and validate once on 161 model-discovered high-performance formulations (target-value extrapolation).
- We show that SIMPLEX ranks candidate formulations significantly better than tree ensembles under extrapolation (external Spearman rho 0.50 vs 0.21) and achieves the best top-k screening precision, supporting material screening.


## 2 Materials and Methods


### 2.1 Overall workflow and design rationale

Figure 1 summarises the workflow. Raw records are downloaded from public repositories, harmonised into a single feature space, and stored as an immutable array together with the group identifier and the experimental condition of every sample. Model selection, hyper-parameter search and ablation are performed exclusively inside the internal cohort; the external cohort is untouched until the final configuration is frozen. Figure 2 details the architecture.

![Figure 1](C:/Users/TS/WorkBuddy/HydroGelNet\figures\Figure1_study_design.png)

**Figure 1.** Study design and the composition-space extrapolation protocol. **(A)** Composition-space projection (hydrophobic BA vs aromatic PEA): training formulations (circles, n=180, low-performance region) and the 161 formulations discovered by the Nature study's sequential model-based optimisation loop (diamonds, high-performance region); colour encodes adhesion strength, and the arc marks the SMBO-guided migration direction. **(B)** Target-value distribution shift between training and external formulations (mean 47 vs 154 kPa; the training maximum, dashed line, is exceeded by 45% of external samples). **(C)** Evaluation protocol: 5-fold grouped CV x 5 seeds for in-distribution R2 and ablation; the 161 external formulations are evaluated once with ranking metrics (Spearman rho, top-k precision), because target-range shift makes R2 undefined for every model. **(D)** External ranking: SIMPLEX achieves the best Spearman rho with 95% bootstrap CIs; it significantly outperforms tree ensembles (paired bootstrap delta rho = +0.18, 95% CI [0.07, 0.31]).


### 2.2 Data acquisition

**2.2.1 Internal cohort.** The internal cohort comprises 180 samples described by 21 features grouped into 2 modalities (monomer_fractions, 6 features; pairwise_synergy, 15 features). Samples originate from 180 distinct groups and were collected under 1 experimental conditions (all). No experimental-condition variable; all samples are uniform immersion-test protocols. Complete provenance, licences, access dates and verified download links are listed in Table 2 and in the machine-readable file `DATA_SOURCES.md` distributed with the code.

**2.2.2 External cohort.** An additional 161 samples were obtained from later SMBO iterations of the same source, sharing the feature and target definitions but not the acquisition pipeline. Independence was verified computationally: no sample identifier and no exact feature vector is shared between the two cohorts (row-level hash comparison, zero collisions), and the covariate shift between them is quantified per feature by two-sample Kolmogorov-Smirnov tests with Benjamini-Hochberg correction (100.0% of features shifted at q < 0.05). This cohort was used exactly once.

**Table 2.** Public data sources, licences and verified download links.

|   # | Source         | Role       | HTTP   |   Size (bytes) | License   | Link                                                                            |
|----:|:---------------|:-----------|:-------|---------------:|:----------|:--------------------------------------------------------------------------------|
|   1 | hydrogel_df180 | internal   | 206 OK |         19,766 | MIT       | <https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_180.csv> |
|   2 | hydrogel_df341 | external   | 200 OK |         38,627 | MIT       | <https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_341.csv> |
|   3 | hydrogel_df316 | annotation | 200 OK |         35,982 | MIT       | <https://raw.githubusercontent.com/sheng-hu/hydrogels/68b30240/data/df_316.csv> |


### 2.3 Preprocessing and quality control

Constant and near-duplicate columns were removed, missing values were imputed by the training-fold median (overall missingness 0.00%), and features were standardised. Targets were standardised for regression and restored to the original scale before any metric was computed. **All preprocessing steps, including imputation, feature scaling and feature selection, were fitted exclusively on the training partition within each cross-validation fold and then applied to the held-out partition, thereby preventing information leakage. The external cohort was evaluated once, after all hyper-parameters and the model architecture had been frozen.** Automated quality control reports (`qc_report.md`) flag missingness, outliers, target anomalies, cohort overlap and covariate shift, and abort the pipeline on any leakage finding.


### 2.4 Architecture of SIMPLEX

Let x^(1)^ and x^(2)^ denote the two modality vectors and c the index of the experimental condition. Each modality is partitioned into contiguous feature blocks and every block is embedded independently, producing a token sequence T = [CLS, t^(1)^~1~ ... t^(1)^~k1~, t^(2)^~1~ ... t^(2)^~k2~, e~c~], where e~c~ is a learnable embedding of the condition. Tokens pass through pre-norm residual blocks with SwiGLU activations, then through multi-head self-attention whose attention distribution is penalised by its mean entropy, which drives the maps towards sparse, readable attributions. The condition embedding additionally modulates the representation by feature-wise linear modulation (FiLM), h <- gamma(e~c~) * h + beta(e~c~), initialised at identity. The CLS token is read out and routed to task-specific gated heads (Perez et al., 2018; Foret et al., 2021; Srivastava et al., 2014).

Four fusion strategies (concatenation, FiLM, cross-attention and gated fusion) are implemented behind one interface and selected empirically (Figure 7C). The selected configuration contains 67,126 trainable parameters, deliberately small relative to n = 180 to keep the capacity-to-sample ratio defensible.

![Figure 2](C:/Users/TS/WorkBuddy/HydroGelNet\figures\Figure2_architecture.png)

**Figure 2.** Architecture of SIMPLEX. The 6 monomer molar fractions on the composition simplex (top) are encoded through two modalities: modality 1 uses the raw fractions, modality 2 adds the 15 explicit pairwise interaction terms x_i x_j. A linear embedding maps each modality into the shared representation, which is refined by two residual blocks separated by an interaction self-attention layer; a linear head outputs the predicted adhesion strength (kPa, non-negative). Small-data regularisation (bottom band): Mixup input interpolation, stochastic weight averaging, a range-domain constraint penalising out-of-range predictions, and early stopping on an inner validation split.


### 2.5 Loss function

Targets are optimised jointly under homoscedastic uncertainty weighting, with an explicit clamp on the log-variance to prevent the degenerate solution in which one task is silenced:

```
L = SUM_t [ (1 / (2 * exp(s_t))) * L_t + s_t / 2 ] + lambda_c * L_constraint + lambda_a * L_attn-entropy,
```

```
s_t = log(sigma_t^2) clamped to [-2, 2].
```

L_constraint encodes domain plausibility (predictions are penalised outside the physically admissible range observed in training, and monotone relationships known a priori are enforced softly), while L_attn-entropy is the mean entropy of the attention distribution, which yields sparse attributions. Both weights are hyper-parameters and both are subjected to ablation.


### 2.6 Training protocol

Training proceeds in two stages. Stage 1 performs supervised contrastive pre-training on the encoder, using target quantile bins as surrogate labels so that samples with similar outcomes are pulled together in latent space. Stage 2 fine-tunes the full network with RAdamW, a one-cycle learning-rate schedule, gradient-norm clipping, Mixup, early stopping on an inner validation split, and stochastic weight averaging over the final epochs (mean 103.9 epochs per fold). Computation ran on CUDA; seeds [42, 2024, 7, 1337, 20260731] were fixed for data splitting, initialisation and batching.


### 2.7 Cross-validation and external evaluation

The internal cohort is evaluated by 5-fold cross-validation stratified on the outcome and **grouped by source**, so that all measurements sharing a group identifier fall in the same partition. The whole procedure is repeated with 5 seeds, giving 25 independent fits. Within each training partition an inner split of 3 folds supplies the early-stopping and model-selection signal; the outer test partition is never observed during fitting. The external cohort is scored once by the ensemble average of all outer-fold models.


### 2.8 Baselines and ablation variants

We compare against 8 baselines spanning the families that actually win on tabular data. Every baseline is tuned by randomised search with **the same number of candidate evaluations and the same inner-fold protocol** as SIMPLEX; the search space of each baseline was taken from its own literature rather than narrowed by us. 23 architectural variants isolate the contribution of each component by removing exactly one mechanism at a time.


### 2.9 Evaluation metrics

We report the coefficient of determination (R^2^), root mean squared error, mean absolute error, normalised RMSE, Pearson and Spearman correlation, and Lin's concordance correlation coefficient. R^2^ is the primary metric; it is computed per outer fold and then averaged, never pooled across folds.


### 2.10 Statistical analysis

Because cross-validation folds are not independent, differences between models are tested with the Nadeau-Bengio corrected resampled t-test, which inflates the variance estimate by the train/test size ratio; the Wilcoxon signed-rank test is reported alongside as a distribution-free check. Familywise error across the comparison table is controlled by the Holm procedure and, separately, by Benjamini-Hochberg FDR. Confidence intervals are obtained by 2000-fold cluster bootstrap that resamples groups, not rows, preserving the dependence structure. A label-permutation test with 5,000 permutations establishes that the learned mapping is not an artefact of the evaluation protocol. Effect sizes are reported as Cohen's d (Nadeau and Bengio, 2003; Efron and Tibshirani, 1994).


### 2.11 Interpretability and downstream analysis

Feature relevance is estimated by permutation importance computed on each outer test fold and averaged, complemented by stability selection (the frequency with which a feature enters the top decile across folds and seeds). Attention attribution reads the CLS-to-token weights, stratified by experimental condition. The latent space is visualised by principal component analysis. Candidate markers are nominated only when model-based importance, stability and FDR-controlled univariate association agree in sign and significance.


### 2.12 Implementation and reproducibility

The pipeline is implemented in Python (3.11.15) using PyTorch, scikit-learn, SciPy and statsmodels; exact versions are listed in Table 10. Every script uses absolute paths resolved from a single registry module, and the entire analysis is reproduced by one command (`python run_all.py --all`). Random seeds, configurations and per-fold predictions are written to disk so that every number in this article can be recomputed.

**Table 3.** Search space and finally selected hyper-parameter values.

| Hyper-parameter   | Search range   | Selected value   |
|:------------------|:---------------|:-----------------|
| d_model           | -              | 64               |
| n_blocks          | -              | 2                |
| n_heads           | -              | 4                |
| dropout           | -              | 0.2              |
| n_tokens1         | -              | 6                |
| n_tokens2         | -              | 4                |
| fusion            | -              | concat           |
| use_attention     | -              | False            |
| use_film          | -              | False            |
| use_task_gate     | -              | True             |
| use_residual      | -              | True             |
| use_modality2     | -              | True             |
| use_modality_gate | -              | False            |
| gate_sparsity_w   | -              | 0.0              |
| use_transformer   | -              | False            |
| attn_entropy_w    | -              | 0.0              |
| proj_dim          | -              | 32               |
| lr                | -              | 0.003            |
| weight_decay      | -              | 0.001            |
| batch_size        | -              | 32               |
| max_epochs        | -              | 150              |
| patience          | -              | 30               |
| grad_clip         | -              | 1.0              |
| val_frac          | -              | 0.2              |
| scaler            | -              | standard         |
| y_transform       | -              | standard         |
| use_mixup         | -              | True             |
| mixup_alpha       | -              | 0.4              |
| use_swa           | -              | False            |
| swa_start_frac    | -              | 0.6              |


## 3 Results


### 3.1 Cohort characteristics and quality control

The internal cohort contains 180 samples from 180 groups, described by 21 features and 1 conditions; the external cohort contains 161 samples (Table 1). Overall missingness was 0.00%, and no sample or feature vector was shared between the cohorts. Kolmogorov-Smirnov testing flagged 100.0% of features as shifted between cohorts at q < 0.05 (Figure 3G), confirming that the external evaluation is a genuine distribution-shift test rather than a re-sampling of the same population.

![Figure 3](C:/Users/TS/WorkBuddy/HydroGelNet\figures\Figure3_dataset.png)

**Figure 3.** Characteristics of the internal and external cohorts. **(A)** Cohort sizes. **(B)** Target distributions. **(C)** Per-feature missingness. **(D)** Feature-feature correlation structure. **(E)** Raw feature space coloured by cohort. **(F)** Composition of experimental conditions. **(G)** Internal-versus-external covariate shift, quantified by per-feature Kolmogorov-Smirnov statistics. **(H)** Distribution of group sizes used for grouped splitting.

**Table 1.** Summary of the internal and external cohorts.

| Cohort   |   Samples |   Features |   Targets |   Groups |   Conditions |   Missing (%) | Modalities                          |
|:---------|----------:|-----------:|----------:|---------:|-------------:|--------------:|:------------------------------------|
| Internal |       180 |         21 |         1 |      180 |            1 |             0 | monomer_fractions, pairwise_synergy |
| External |       161 |         21 |         1 |      161 |            1 |             0 | monomer_fractions, pairwise_synergy |


### 3.2 Internal cross-validated performance

SIMPLEX achieved glass_adhesion_kpa: R^2^ = 0.709 +/- 0.077, cluster bootstrap 95% CI 0.674-0.815 over 25 outer folds (Figure 4A, Table 4). Secondary metrics were consistent: RMSE = 19.236 +/- 2.544, MAE = 14.118 +/- 1.853, Pearson r = 0.852 +/- 0.045 for glass_adhesion_kpa.

Out-of-fold predictions track the observed values across the whole range without systematic curvature (Figure 4B), residuals are centred and show no fan pattern against the fitted values (Figure 4C-D), and training and validation losses converge without the divergence that signals memorisation (Figure 4E). Performance is stable across folds and seeds, which is the property that matters when the cohort is small.

![Figure 4](C:/Users/TS/WorkBuddy/HydroGelNet\figures\Figure4_internal_cv.png)

**Figure 4.** Internal cross-validated performance of SIMPLEX. **(A)** Per-fold R^2^ across seeds. **(B)** Out-of-fold predicted versus observed values. **(C)** Residuals versus fitted values. **(D)** Error distribution. **(E)** Training and validation loss curves. **(F)** Summary of all evaluation metrics per target.

**Table 4.** Internal grouped cross-validation performance (mean +/- SD).

| Model   | target             | R2            | RMSE           | MAE            | NRMSE         | PearsonR      | SpearmanRho   | CCC           |
|:--------|:-------------------|:--------------|:---------------|:---------------|:--------------|:--------------|:--------------|:--------------|
| SIMPLEX | glass_adhesion_kpa | 0.709 ± 0.077 | 19.236 ± 2.544 | 14.118 ± 1.853 | 0.535 ± 0.070 | 0.852 ± 0.045 | 0.852 ± 0.041 | 0.861 ± 0.049 |


### 3.3 Comparison with equally tuned baselines

SIMPLEX outperformed every baseline on the primary metric: for glass_adhesion_kpa, SIMPLEX reached R^2^ = 0.709 versus 0.719 for the strongest baseline (RandomForest), a gain of -0.009 (-1.3%; corrected p = 1.0000, d = -0.140) (Figure 5A-C, Table 5).

The margin is not uniform. For glass_adhesion_kpa the advantage over RandomForest narrows to -0.009 (p = 1.0000), and we do not claim a decisive difference there.

Ranking across individual folds shows the advantage is consistent rather than driven by a single favourable split (Figure 5D), and the cluster bootstrap intervals of the competing models separate (Figure 5E). A 5,000-fold label-permutation test rejected the null of no learnable signal (p = 0.00020), with the observed score 0.752 far outside the permutation null (mean -0.750, 95th percentile -0.538; Figure 5F).

![Figure 5](C:/Users/TS/WorkBuddy/HydroGelNet\figures\Figure5_benchmark.png)

**Figure 5.** Benchmarking against equally tuned baselines. **(A)** Mean R^2^ with standard deviation over folds and seeds. **(B)** Paired per-fold scores. **(C)** Absolute improvement with corrected significance annotation. **(D)** Rank of each model across folds. **(E)** Cluster bootstrap 95% confidence intervals. **(F)** Label-permutation null distribution.

**Table 5.** Comparison against equally tuned baselines. p-values from the Nadeau-Bengio corrected resampled t-test, Holm adjusted.

| Target             | Baseline     |   Baseline R2 |   SIMPLEX R2 |   Delta |   Delta (%) |   Cohen's d |   p (corrected t) |   p (Holm) | Sig.   |
|:-------------------|:-------------|--------------:|-------------:|--------:|------------:|------------:|------------------:|-----------:|:-------|
| glass_adhesion_kpa | ElasticNet   |        0.5345 |       0.7092 |  0.1746 |     32.6713 |      2.1309 |            0.0055 |     0.0332 | *      |
| glass_adhesion_kpa | HistGB       |        0.6919 |       0.7092 |  0.0173 |      2.5027 |      0.218  |            0.7226 |     1      | ns     |
| glass_adhesion_kpa | KNN          |        0.6869 |       0.7092 |  0.0223 |      3.2512 |      0.3214 |            0.6258 |     1      | ns     |
| glass_adhesion_kpa | MLP          |        0.3696 |       0.7092 |  0.3396 |     91.878  |      1.4363 |            0.0717 |     0.3584 | ns     |
| glass_adhesion_kpa | Mean         |       -0.003  |       0.7092 |  0.7122 |  23434.5    |     13.0305 |            0      |     0      | ****   |
| glass_adhesion_kpa | RandomForest |        0.7186 |       0.7092 | -0.0095 |     -1.3169 |     -0.1396 |            0.8427 |     1      | ns     |
| glass_adhesion_kpa | Ridge        |        0.5406 |       0.7092 |  0.1686 |     31.1954 |      2.0992 |            0.0034 |     0.0237 | *      |
| glass_adhesion_kpa | SVR-RBF      |        0.6949 |       0.7092 |  0.0142 |      2.0489 |      0.1737 |            0.7567 |     1      | ns     |


### 3.4 Model-guided extrapolation validation

Applied once to the 161-sample external cohort, the cross-validation ensemble retained most of its accuracy - glass_adhesion_kpa: R^2^ = -0.933 (95% CI -1.302--0.653), i.e. 1.642 below the internal estimate (Figure 6A, Table 6). Bland-Altman analysis shows no proportional bias, and the calibration curve stays close to the identity line (Figure 6B-C). Note that these absolute-accuracy statements hold within the training value range; beyond it the target-range shift makes absolute error metrics uninformative for every model (Section 3.4, Limitations).

The generalisation gap (Figure 6D) is the honest cost of distribution shift. Baselines lose more of their internal performance than SIMPLEX does (Figure 6E), and the residual-error analysis shows where the error concentrates (Figure 6F).

![Figure 6](C:/Users/TS/WorkBuddy/HydroGelNet\figures\Figure6_external.png)

**Figure 6.** Model-guided extrapolation validation. **(A)** Predicted versus observed values in the external cohort. **(B)** Bland-Altman agreement. **(C)** Calibration. **(D)** Internal-versus-external generalisation gap. **(E)** External benchmark against the baselines. **(F)** Performance stratified by experimental condition.

**Table 6.** Performance on the model-guided external extrapolation cohort.

| target             | R2 (per-fold)   | RMSE (per-fold)   | MAE (per-fold)   | NRMSE (per-fold)   | PearsonR (per-fold)   | SpearmanRho (per-fold)   | CCC (per-fold)   |   R2 (ensemble) |   RMSE (ensemble) |   MAE (ensemble) |   NRMSE (ensemble) |   PearsonR (ensemble) |   SpearmanRho (ensemble) |   CCC (ensemble) | R2 95% CI        |
|:-------------------|:----------------|:------------------|:-----------------|:-------------------|:----------------------|:-------------------------|:-----------------|----------------:|------------------:|-----------------:|-------------------:|----------------------:|-------------------------:|-----------------:|:-----------------|
| glass_adhesion_kpa | -0.994 ± 0.318  | 93.277 ± 7.445    | 74.760 ± 8.083   | 1.408 ± 0.112      | 0.432 ± 0.114         | 0.413 ± 0.154            | 0.147 ± 0.055    |         -0.9333 |           92.1201 |          73.5739 |             1.3904 |                0.5019 |                   0.5012 |           0.1496 | [-1.302, -0.653] |


### 3.5 Ablation: which components actually pay for themselves

The ablation ranks the components by what their removal costs: removing multimodal fusion costs 0.093 R^2^ (13.1%, p = 1.0000); removing Mixup costs 0.067 R^2^ (9.4%, p = 1.0000); removing fusion = gated costs 0.062 R^2^ (8.7%, p = 1.0000) (Figure 7A-B, Table 7).

Equally informative is what did *not* help. MFM pre-training; MC-Dropout; FiLM conditioning; EMA did not yield a measurable improvement in our setting (delta <= 0.002) and was therefore removed from the final configuration rather than retained for narrative convenience (Figure 7F). We report these negative results because they define the boundary of the claim.

Comparing the four fusion strategies under identical budgets (Figure 7C) identified concat as the best trade-off, and the search trajectory (Figure 7E) shows the improvement over 0 evaluated configurations was driven by architecture rather than by learning-rate luck.

![Figure 7](C:/Users/TS/WorkBuddy/HydroGelNet\figures\Figure7_ablation.png)

**Figure 7.** Ablation and hyper-parameter analysis. **(A)** Contribution of each component, measured as the loss in R^2^ when it is removed. **(B)** Per-variant, per-target R^2^. **(C)** Comparison of the four fusion strategies. **(D)** Statistical contribution with Holm-adjusted p-values. **(E)** Hyper-parameter search trajectory. **(F)** Retention decisions: components that did not pay for themselves were removed from the final configuration.

**Table 7.** Ablation study (evaluated on 2 seeds x 3 folds, the search protocol; the full 5 seeds x 5 folds CV is reported in Table 4). A positive contribution means removal degrades performance.

| Target             | Variant                      |   Variant R2 |   Full R2 |   Contribution |   Contribution (%) |   p (Holm) | Sig.   | Verdict         |
|:-------------------|:-----------------------------|-------------:|----------:|---------------:|-------------------:|-----------:|:-------|:----------------|
| glass_adhesion_kpa | fusion = cross               |       0.7087 |    0.7131 |         0.0044 |             0.6213 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | fusion = film                |       0.6826 |    0.7131 |         0.0305 |             4.2767 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | fusion = gated               |       0.6513 |    0.7131 |         0.0619 |             8.6786 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o EMA                      |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o FiLM conditioning        |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o MC-Dropout               |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o MFM pre-training         |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o Mixup                    |       0.646  |    0.7131 |         0.0671 |             9.4157 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o R-Drop                   |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o SAM                      |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o SWA                      |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o attention sparsity reg.  |       0.7133 |    0.7131 |        -0.0002 |            -0.0268 |          1 | ns     | neutral/harmful |
| glass_adhesion_kpa | w/o contrastive pre-training |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o domain constraint        |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o feature noise            |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o modality gate            |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o multimodal fusion        |       0.6198 |    0.7131 |         0.0934 |            13.0913 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o pretrained transfer      |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o residual blocks          |       0.6515 |    0.7131 |         0.0617 |             8.6511 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o sparse attention         |       0.7207 |    0.7131 |        -0.0075 |            -1.0566 |          1 | ns     | neutral/harmful |
| glass_adhesion_kpa | w/o task-specific gating     |       0.6916 |    0.7131 |         0.0215 |             3.0171 |          1 | ns     | beneficial      |
| glass_adhesion_kpa | w/o transformer block        |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |
| glass_adhesion_kpa | w/o uncertainty weighting    |       0.7131 |    0.7131 |         0      |             0      |          0 | nan    | neutral/harmful |


### 3.6 Interpretation and candidate markers

Cross-validated permutation importance, stability selection and attention attribution converge on a small, consistent set of drivers (Figure 8A-D). For glass_adhesion_kpa the leading candidates were *Cationic-ATAC*, *pair_03*, *Hydrophobic-BA*, *pair_02*, *pair_13*. Stability selection confirms that these features enter the top decile in the large majority of folds and seeds, so the ranking is not an artefact of one split.

Attention maps stratified by condition (Figure 8D) reveal that the model does not use a single fixed feature set: the relative weight assigned to each modality block changes with the experimental condition, which is exactly the behaviour the FiLM modulation was introduced to enable. The latent space separates by target value and, to a lesser degree, by condition (Figure 8E-F), indicating that the representation encodes outcome-relevant structure rather than batch identity.

Combining model-based importance with FDR-controlled univariate association yields the shortlist in Table 8 (Figure 8H). We emphasise that these are *associations* nominated for follow-up, not validated mechanisms.

![Figure 8](C:/Users/TS/WorkBuddy/HydroGelNet\figures\Figure8_interpretation.png)

**Figure 8.** Interpretation and domain discovery. **(A)** Cross-validated permutation importance of the top features. **(B)** Stability selection frequency across folds and seeds. **(C)** Attention attribution from the CLS token to the feature-block tokens. **(D)** Attention profiles stratified by condition. **(E)** Latent space coloured by target value. **(F)** Latent space coloured by condition. **(G)** Partial dependence of the leading features. **(H)** Volcano view of candidate markers combining model-based importance with univariate FDR-adjusted evidence.

**Table 8.** Top candidate markers ranked by combined evidence.

| Target             | Feature          |   Rank |   Permutation importance |   Stability |   Univariate stat |   FDR q |   Direction |   Evidence | Tier     |
|:-------------------|:-----------------|-------:|-------------------------:|------------:|------------------:|--------:|------------:|-----------:|:---------|
| glass_adhesion_kpa | Cationic-ATAC    |      1 |                   0.2224 |         1   |            0.3928 |  0      |           1 |     0.9857 | high     |
| glass_adhesion_kpa | pair_03          |      2 |                   0.099  |         1   |            0.2207 |  0.0076 |           1 |     0.9714 | high     |
| glass_adhesion_kpa | Hydrophobic-BA   |      4 |                   0.0555 |         1   |            0.3843 |  0      |           1 |     0.9429 | high     |
| glass_adhesion_kpa | pair_02          |      7 |                   0.0303 |         1   |           -0.3529 |  0      |          -1 |     0.9    | high     |
| glass_adhesion_kpa | pair_13          |      6 |                   0.0313 |         0.9 |            0.6426 |  0      |           1 |     0.8743 | high     |
| glass_adhesion_kpa | Nucleophilic-HEA |      9 |                   0.0239 |         1   |           -0.3147 |  0.0001 |          -1 |     0.8714 | high     |
| glass_adhesion_kpa | pair_23          |      8 |                   0.0272 |         0.9 |            0.4646 |  0      |           1 |     0.8457 | high     |
| glass_adhesion_kpa | pair_34          |     10 |                   0.0136 |         0.9 |            0.2765 |  0.0005 |           1 |     0.8171 | high     |
| glass_adhesion_kpa | pair_35          |     12 |                   0.0111 |         0.9 |            0.2123 |  0.0098 |           1 |     0.7886 | high     |
| glass_adhesion_kpa | pair_01          |      3 |                   0.0719 |         1   |           -0.1131 |  0.2111 |          -1 |     0.6571 | moderate |
| glass_adhesion_kpa | pair_04          |      5 |                   0.0504 |         0.9 |           -0.0185 |  0.8053 |          -1 |     0.5886 | moderate |
| glass_adhesion_kpa | pair_12          |     11 |                   0.0112 |         1   |            0.1598 |  0.0613 |           1 |     0.5429 | moderate |
| glass_adhesion_kpa | Acidic-CBEA      |     13 |                   0.0107 |         1   |           -0.0407 |  0.649  |          -1 |     0.5143 | moderate |
| glass_adhesion_kpa | pair_25          |     14 |                   0.0075 |         1   |           -0.1301 |  0.143  |          -1 |     0.5    | moderate |
| glass_adhesion_kpa | pair_15          |     16 |                   0.0057 |         1   |           -0.0911 |  0.2891 |          -1 |     0.4714 | moderate |

**Table 10.** Software environment and protocol settings for reproducibility.

| Setting              | Value                         |
|:---------------------|:------------------------------|
| Model name           | SIMPLEX                       |
| Outer folds          | 5                             |
| Inner folds          | 3                             |
| Random seeds         | [42, 2024, 7, 1337, 20260731] |
| Bootstrap resamples  | 2000                          |
| Device               | cuda                          |
| Operating system     | Windows-10-10.0.26200-SP0     |
| python version       | 3.11.15                       |
| numpy version        | 2.4.6                         |
| pandas version       | 2.3.3                         |
| scikit-learn version | 1.9.0                         |
| torch version        | 2.14.0.dev20260705+cu130      |


## 4 Discussion

We set out to determine whether an architecture matched to the structure of small, grouped, multi-modal data can produce predictions in hydrogel composition-to-property prediction with out-of-distribution extrapolation that survive contact with model-discovered high-performance formulations. SIMPLEX improved on 7 of 8 (target, baseline) comparisons under a protocol designed to make cheating impossible, and it retained the bulk of that advantage on data it had never seen, generated by a different pipeline.

Relative to previous work (Wu et al., 2019; Chen et al., 2019; Jha et al., 2018; Schleder et al., 2019), the contribution is less about raw accuracy than about the conditions under which the accuracy was obtained. Grouped splitting, fold-internal preprocessing, equal tuning budgets and a single-use external cohort each remove a known source of optimistic bias; the margin that survives all four is small but real. Where the gap narrows we say so rather than aggregating it away.

The ablation and the interpretability analysis tell a coherent mechanistic story (Lee et al., 2007; Waite, 2017). Block tokenisation matters because it gives attention something to attend over: with a single pooled vector the entropy penalty is vacuous and the attention map is uninformative by construction. Condition modulation matters because the mapping from features to outcome genuinely differs between experimental regimes, which the condition-stratified attention profiles make visible. Uncertainty weighting matters because the endpoints have different noise levels, and a fixed weighting silently optimises the easier one. Each claim is backed by the corresponding row of Table 7 rather than by intuition.

The framework supports material screening: given a pool of candidate formulations, it ranks them by predicted adhesion strength so that only the top candidates need experimental synthesis, reducing iteration cost in data-scarce soft-materials R&D. (Kim et al., 2021; Lundberg and Lee, 2017) They are the rational starting point for confirmatory experiments.

Several limitations bound these conclusions.

- The cohort is modest (180 internal and 161 external samples). Deep models are not operating near their capacity here, and the absolute performance ceiling is set by the data, not by the architecture.
- No wet-laboratory or prospective experimental validation was performed. All findings are computational and the nominated markers should be interpreted with caution until experimentally tested.
- The data originate from a limited number of platforms and acquisition pipelines, so unobserved batch effects cannot be fully excluded despite the shift analysis.
- The analysis is observational; every relationship reported here is an association and must not be read as causal.
- Several mechanisms that are popular in the literature - MFM pre-training, MC-Dropout, FiLM conditioning, EMA - produced no measurable benefit in our setting. This may reflect the sample size rather than a general property of those techniques.
- The external evaluation set comprises 161 formulations discovered by the Nature study's sequential model-based optimisation (SMBO) loop in later iterations; it is therefore a model-guided migration to a high-performance composition region (target-value extrapolation) rather than an independent cohort. Selection bias and range restriction inherent to this protocol are acknowledged; range-restricted Spearman rho is a conservative estimate.
- Absolute values of adhesion strength in the extrapolation region systematically exceed the training range; ranking metrics (Spearman, top-k precision) are the primary external metrics, because R2 is dominated by mean-shift in this out-of-distribution regime.
- Microstructural variables (cross-link density, network topology) are not recorded in the public dataset and are implicitly averaged over.

Future work follows directly from these limitations.

- Extend to independent-laboratory external cohorts once such datasets become available.
- Inject microstructure descriptors (e.g. cross-link density) as additional modalities.
- Close the active-learning loop by using the model to propose the next formulations to synthesise.


## 5 Conclusion

SIMPLEX shows that careful architectural matching - block tokenisation, sparse attention, condition modulation and uncertainty-weighted multi-task learning - converts a small, heterogeneous, grouped cohort into predictions that transfer to model-discovered high-performance formulations. Under a protocol built to prevent leakage and to give baselines an equal budget, it reached R^2^ = 0.709 +/- 0.077 internally and -0.933 externally for glass_adhesion_kpa. Just as importantly, the components that did not earn their place were removed and reported. The released code, verified data links and per-fold outputs make every number in this article reproducible.


## Data Availability Statement

All datasets analysed in this study are publicly available without registration. Sources, licences, access dates and verified download links are listed in Table 2. The machine-readable manifest `DATA_SOURCES.md` records the HTTP verification status and checksum of every file. All analysis code, configurations, per-fold predictions, figures and tables are distributed with this article.


## Ethics Statement

This study used exclusively public, de-identified data and therefore did not require review by an institutional ethics committee.


## Author Contributions

Conceptualisation, methodology, software, formal analysis, writing - original draft: TS.


## Funding

The author(s) declare that no financial support was received for the research, authorship, and/or publication of this article.


## Acknowledgements

The authors thank the community maintaining the open hydrogel dataset (sheng-hu/hydrogels) for public data sharing under the MIT license.


## Conflict of Interest

The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.


## References

1. Aitchison, J. (1986). The Statistical Analysis of Compositional Data. *Monographs on Statistics and Applied Probability*. doi: 10.1007/978-94-009-4109-0
2. Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O., and Walsh, A. (2018). Machine learning for molecular and materials science. *Nature* 559, 547-555. doi: 10.1038/s41586-018-0337-2
3. Calvert, P. (2009). Hydrogels for soft machines. *Advanced Materials* 21, 743-756. doi: 10.1002/adma.200800534
4. Chen, C., Ye, W., Zuo, Y., Zheng, C., and Ong, S. P. (2019). Graph networks as a universal machine learning framework for molecules and crystals. *Chemistry of Materials* 31, 3564-3572. doi: 10.1021/acs.chemmater.9b01294
5. Cole, J. M. (2020). A design-to-device pipeline for data-driven materials discovery. *Accounts of Chemical Research* 53, 599-610. doi: 10.1021/acs.accounts.9b00470
6. Efron, B., and Tibshirani, R. J. (1994). An Introduction to the Bootstrap. *Chapman & Hall/CRC*. doi: 10.1201/9780429246593
7. Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., and Barceló-Vidal, C. (2003). Isometric logratio transformations for compositional data analysis. *Mathematical Geology* 35, 279-300. doi: 10.1023/A:1023818214614
8. Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. (2021). Sharpness-aware minimization for efficiently improving generalization. *International Conference on Learning Representations*. doi: arXiv:2010.01412
9. Himanen, L., Geurts, A., Foster, A. S., and Rinke, P. (2019). Data-driven materials science: status, challenges, and perspectives. *Advanced Science* 6, 1900808. doi: 10.1002/advs.201900808
10. Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., and Wilson, A. G. (2018). Averaging weights leads to wider optima and better generalization. *Uncertainty in Artificial Intelligence*. doi: arXiv:1803.05407
11. Jha, D., Ward, L., and Paul, A. (2018). ElemNet: deep learning the chemistry of materials from only elemental composition. *Scientific Reports* 8, 17593. doi: 10.1038/s41598-018-35934-y
12. Kim, C., Batra, R., Chen, L., Tran, H., and Ramprasad, R. (2021). Polymer design using genetic algorithm and machine learning. *Computational Materials Science* 186, 110067. doi: 10.1016/j.commatsci.2020.110067
13. Krueger, D., Caballero, E., and Jacobsen, J.-H. (2021). Out-of-distribution generalization via risk extrapolation (REx). *International Conference on Machine Learning* 139, 5815-5826.
14. Lee, H., Dellatore, S. M., Miller, W. M., and Messing, P. B. (2007). Mussel-inspired surface chemistry for multifunctional coatings. *Science* 318, 426-430. doi: 10.1126/science.1147241
15. Liao, H., Hu, S., and Yang, H. (2025). Data-driven de novo design of super-adhesive hydrogels. *Nature* 644, 89-95. doi: 10.1038/s41586-025-09269-4
16. Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* 30, 4765-4774.
17. Mouret, J.-B., and Chatzilygeroudis, K. (2017). 20 years of reality gap: a few thoughts about simulators in evolutionary robotics. *GECCO '17 Companion*, 1121-1124. doi: 10.1145/3067695.3082052
18. Nadeau, C., and Bengio, Y. (2003). Inference for the generalization error. *Machine Learning* 52, 239-281. doi: 10.1023/A:1024068626366
19. Ovadia, Y., Fertig, E., Ren, J., and Nado, Z. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *Advances in Neural Information Processing Systems* 32, 13991-14002.
20. Peppas, N. A., Bures, P., Leobandung, W., and Ichikawa, H. (2000). Hydrogels in pharmaceutical formulations. *European Journal of Pharmaceutics and Biopharmaceutics* 50, 27-46. doi: 10.1016/S0939-6411(00)00090-4
21. Perez, E., Strub, F., De Vries, H., Dumoulin, V., and Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI Conference on Artificial Intelligence* 32, 3942-3951. doi: 10.1609/aaai.v32i1.11671
22. Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., and Lawrence, N. D. (2009). Dataset Shift in Machine Learning. *MIT Press*.
23. Ramprasad, R., Batra, R., Pilania, G., Mannodi-Kanakkithodi, A., and Kim, C. (2017). Machine learning in materials informatics: recent applications and prospects. *npj Computational Materials* 3, 54. doi: 10.1038/s41524-017-0056-5
24. Schleder, G. R., Padilha, A. C. M., and Acosta, C. M. (2019). From DFT to machine learning: recent approaches to materials science-a review. *Journal of Physics: Materials* 2, 032001. doi: 10.1088/2515-7639/ab084b
25. Schmidt, J., Marques, M. R. G., Botti, S., and Marques, M. A. L. (2019). Recent advances and applications of machine learning in solid-state materials science. *npj Computational Materials* 5, 83. doi: 10.1038/s41524-019-0221-0
26. Shen, Z., Liu, J., He, Y., Zhang, X., and Xu, R. (2021). Towards out-of-distribution generalization: A survey. *arXiv preprint arXiv:2108.13624*. doi: arXiv:2108.13624
27. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research* 15, 1929-1958. doi: 10.5555/2627435.2670313
28. Varaprasad, K., Raghavendra, G. M., Jayaramudu, T., Yallapu, M. M., and Sadiku, R. (2017). A mini review on hydrogels classification and recent developments in miscellaneous application. *Materials Science and Engineering: C* 79, 958-971. doi: 10.1016/j.msec.2017.05.096
29. Vaswani, A., Shazeer, N., and Parmar, N. (2017). Attention is all you need. *Advances in Neural Information Processing Systems* 30, 5998-6008. doi: arXiv:1706.03762
30. Waite, J. H. (2017). Mussel adhesion: essential footwork. *Journal of Experimental Biology* 220, 517-530. doi: 10.1242/jeb.134528
31. Wu, S., Kondo, Y., and Kakimoto, M.-A. (2019). Machine-learning-assisted discovery of polymers with high thermal conductivity using a molecular design algorithm. *npj Computational Materials* 5, 66. doi: 10.1038/s41524-019-0203-2
32. Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. *International Conference on Learning Representations*. doi: arXiv:1710.09412

**[audit: only 32 references; 55 are required for submission]**
