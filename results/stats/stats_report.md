# Statistical report -- SIMPLEX

Primary metric: **R2**. Outer protocol: 5-fold grouped CV repeated over 5 seed(s). All p-values from the Nadeau-Bengio corrected resampled t-test, adjusted with Holm.

## Proposed model vs baselines

| target             | reference    |   proposed_mean |   reference_mean |   delta |   p_corrected_t |   p_holm | significance   |
|:-------------------|:-------------|----------------:|-----------------:|--------:|----------------:|---------:|:---------------|
| glass_adhesion_kpa | ElasticNet   |          0.7925 |           0.7689 |  0.0236 |          0.2766 |   1      | ns             |
| glass_adhesion_kpa | HistGB       |          0.7925 |           0.7633 |  0.0293 |          0.1255 |   0.7532 | ns             |
| glass_adhesion_kpa | KNN          |          0.7925 |           0.7892 |  0.0033 |          0.8888 |   1      | ns             |
| glass_adhesion_kpa | MLP          |          0.7925 |           0.6991 |  0.0934 |          0.0221 |   0.1548 | ns             |
| glass_adhesion_kpa | Mean         |          0.7925 |          -0.0049 |  0.7974 |          0      |   0      | ****           |
| glass_adhesion_kpa | RandomForest |          0.7925 |           0.8093 | -0.0168 |          0.2286 |   1      | ns             |
| glass_adhesion_kpa | Ridge        |          0.7925 |           0.7691 |  0.0235 |          0.2902 |   1      | ns             |
| glass_adhesion_kpa | SVR-RBF      |          0.7925 |           0.799  | -0.0065 |          0.7451 |   1      | ns             |

1/8 comparisons remain significant after Holm correction.

## Bootstrap confidence intervals (cluster bootstrap)

| scope        | model   | target             | metric   |   point |      lo |      hi |     se |
|:-------------|:--------|:-------------------|:---------|--------:|--------:|--------:|-------:|
| internal_oof | SIMPLEX | glass_adhesion_kpa | R2       |  0.7497 |  0.6647 |  0.8178 | 0.0393 |
| internal_oof | SIMPLEX | glass_adhesion_kpa | RMSE     | 18.0512 | 15.4293 | 20.5099 | 1.3116 |
| internal_oof | SIMPLEX | glass_adhesion_kpa | MAE      | 13.1829 | 11.4308 | 15.0491 | 0.9268 |
| external     | SIMPLEX | glass_adhesion_kpa | R2       |  0.7101 |  0.4582 |  0.8571 | 0.1043 |
| external     | SIMPLEX | glass_adhesion_kpa | RMSE     | 33.657  | 24.7468 | 41.3974 | 4.3314 |
| external     | SIMPLEX | glass_adhesion_kpa | MAE      | 26.878  | 19.5185 | 34.786  | 4.0322 |

## Permutation test

| target             | metric   |   observed |   null_mean |   null_p95 |   p_value |   n_perm | significance   |
|:-------------------|:---------|-----------:|------------:|-----------:|----------:|---------:|:---------------|
| glass_adhesion_kpa | R2       |    0.74969 |    -0.75138 |   -0.53711 |    0.0002 |     5000 | ***            |

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

