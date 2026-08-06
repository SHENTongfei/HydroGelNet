# Statistical report -- SIMPLEX

Primary metric: **R2**. Outer protocol: 5-fold grouped CV repeated over 5 seed(s). All p-values from the Nadeau-Bengio corrected resampled t-test, adjusted with Holm.

## Proposed model vs baselines

| target             | reference    |   proposed_mean |   reference_mean |   delta |   p_corrected_t |   p_holm | significance   |
|:-------------------|:-------------|----------------:|-----------------:|--------:|----------------:|---------:|:---------------|
| glass_adhesion_kpa | ElasticNet   |           0.726 |           0.5345 |  0.1915 |          0.0018 |   0.0107 | *              |
| glass_adhesion_kpa | HistGB       |           0.726 |           0.6919 |  0.0342 |          0.4975 |   1      | ns             |
| glass_adhesion_kpa | KNN          |           0.726 |           0.6869 |  0.0392 |          0.2702 |   1      | ns             |
| glass_adhesion_kpa | MLP          |           0.726 |           0.3696 |  0.3564 |          0.0562 |   0.2812 | ns             |
| glass_adhesion_kpa | Mean         |           0.726 |          -0.003  |  0.7291 |          0      |   0      | ****           |
| glass_adhesion_kpa | RandomForest |           0.726 |           0.7186 |  0.0074 |          0.8579 |   1      | ns             |
| glass_adhesion_kpa | Ridge        |           0.726 |           0.5406 |  0.1855 |          0.0009 |   0.0066 | **             |
| glass_adhesion_kpa | SVR-RBF      |           0.726 |           0.6949 |  0.0311 |          0.4121 |   1      | ns             |

3/8 comparisons remain significant after Holm correction.

## Bootstrap confidence intervals (cluster bootstrap)

| scope        | model   | target             | metric   |   point |      lo |       hi |     se |
|:-------------|:--------|:-------------------|:---------|--------:|--------:|---------:|-------:|
| internal_oof | SIMPLEX | glass_adhesion_kpa | R2       |  0.7497 |  0.6647 |   0.8178 | 0.0393 |
| internal_oof | SIMPLEX | glass_adhesion_kpa | RMSE     | 18.0512 | 15.4293 |  20.5099 | 1.3116 |
| internal_oof | SIMPLEX | glass_adhesion_kpa | MAE      | 13.1829 | 11.4308 |  15.0491 | 0.9268 |
| external     | SIMPLEX | glass_adhesion_kpa | R2       | -0.9108 | -1.2698 |  -0.6346 | 0.1612 |
| external     | SIMPLEX | glass_adhesion_kpa | RMSE     | 91.5836 | 82.3606 | 101.084  | 4.7708 |
| external     | SIMPLEX | glass_adhesion_kpa | MAE      | 72.9064 | 64.7953 |  81.7507 | 4.406  |

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

