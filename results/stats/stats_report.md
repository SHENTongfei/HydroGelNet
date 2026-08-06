# Statistical report -- SIMPLEX

Primary metric: **R2**. Outer protocol: 5-fold grouped CV repeated over 5 seed(s). All p-values from the Nadeau-Bengio corrected resampled t-test, adjusted with Holm.

## Proposed model vs baselines

| target             | reference    |   proposed_mean |   reference_mean |   delta |   p_corrected_t |   p_holm | significance   |
|:-------------------|:-------------|----------------:|-----------------:|--------:|----------------:|---------:|:---------------|
| glass_adhesion_kpa | ElasticNet   |          0.7092 |           0.5345 |  0.1746 |          0.0055 |   0.0332 | *              |
| glass_adhesion_kpa | HistGB       |          0.7092 |           0.6919 |  0.0173 |          0.7226 |   1      | ns             |
| glass_adhesion_kpa | KNN          |          0.7092 |           0.6869 |  0.0223 |          0.6258 |   1      | ns             |
| glass_adhesion_kpa | MLP          |          0.7092 |           0.3696 |  0.3396 |          0.0717 |   0.3584 | ns             |
| glass_adhesion_kpa | Mean         |          0.7092 |          -0.003  |  0.7122 |          0      |   0      | ****           |
| glass_adhesion_kpa | RandomForest |          0.7092 |           0.7186 | -0.0095 |          0.8427 |   1      | ns             |
| glass_adhesion_kpa | Ridge        |          0.7092 |           0.5406 |  0.1686 |          0.0034 |   0.0237 | *              |
| glass_adhesion_kpa | SVR-RBF      |          0.7092 |           0.6949 |  0.0142 |          0.7567 |   1      | ns             |

3/8 comparisons remain significant after Holm correction.

## Bootstrap confidence intervals (cluster bootstrap)

| scope        | model   | target             | metric   |   point |      lo |       hi |     se |
|:-------------|:--------|:-------------------|:---------|--------:|--------:|---------:|-------:|
| internal_oof | SIMPLEX | glass_adhesion_kpa | R2       |  0.7521 |  0.6737 |   0.8148 | 0.0362 |
| internal_oof | SIMPLEX | glass_adhesion_kpa | RMSE     | 17.9629 | 15.4737 |  20.3096 | 1.2322 |
| internal_oof | SIMPLEX | glass_adhesion_kpa | MAE      | 13.1354 | 11.3732 |  14.9833 | 0.918  |
| external     | SIMPLEX | glass_adhesion_kpa | R2       | -0.9333 | -1.3024 |  -0.6526 | 0.1637 |
| external     | SIMPLEX | glass_adhesion_kpa | RMSE     | 92.1201 | 82.999  | 101.53   | 4.7421 |
| external     | SIMPLEX | glass_adhesion_kpa | MAE      | 73.5739 | 65.4335 |  82.3773 | 4.4005 |

## Permutation test

| target             | metric   |   observed |   null_mean |   null_p95 |   p_value |   n_perm | significance   |
|:-------------------|:---------|-----------:|------------:|-----------:|----------:|---------:|:---------------|
| glass_adhesion_kpa | R2       |    0.75214 |    -0.75005 |   -0.53817 |    0.0002 |     5000 | ***            |

## Ablation

| target             | variant                      |   full_mean |   variant_mean |   delta |   p_holm | significance   | contribution    |
|:-------------------|:-----------------------------|------------:|---------------:|--------:|---------:|:---------------|:----------------|
| glass_adhesion_kpa | fusion = cross               |      0.7131 |         0.7087 |  0.0044 |        1 | ns             | beneficial      |
| glass_adhesion_kpa | fusion = film                |      0.7131 |         0.6826 |  0.0305 |        1 | ns             | beneficial      |
| glass_adhesion_kpa | fusion = gated               |      0.7131 |         0.6513 |  0.0619 |        1 | ns             | beneficial      |
| glass_adhesion_kpa | w/o EMA                      |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o FiLM conditioning        |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o MC-Dropout               |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o MFM pre-training         |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o Mixup                    |      0.7131 |         0.646  |  0.0671 |        1 | ns             | beneficial      |
| glass_adhesion_kpa | w/o R-Drop                   |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o SAM                      |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o SWA                      |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o attention sparsity reg.  |      0.7131 |         0.7133 | -0.0002 |        1 | ns             | neutral/harmful |
| glass_adhesion_kpa | w/o contrastive pre-training |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o domain constraint        |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o feature noise            |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o modality gate            |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o multimodal fusion        |      0.7131 |         0.6198 |  0.0934 |        1 | ns             | beneficial      |
| glass_adhesion_kpa | w/o pretrained transfer      |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o residual blocks          |      0.7131 |         0.6515 |  0.0617 |        1 | ns             | beneficial      |
| glass_adhesion_kpa | w/o sparse attention         |      0.7131 |         0.7207 | -0.0075 |        1 | ns             | neutral/harmful |
| glass_adhesion_kpa | w/o task-specific gating     |      0.7131 |         0.6916 |  0.0215 |        1 | ns             | beneficial      |
| glass_adhesion_kpa | w/o transformer block        |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |
| glass_adhesion_kpa | w/o uncertainty weighting    |      0.7131 |         0.7131 |  0      |        0 |                | neutral/harmful |

