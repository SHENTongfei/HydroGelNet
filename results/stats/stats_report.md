# Statistical report -- SIMPLEX

Primary metric: **R2**. Outer protocol: 5-fold grouped CV repeated over 10 seed(s). All p-values from the Nadeau-Bengio corrected resampled t-test, adjusted with Holm.

## Proposed model vs baselines

| target             | reference    |   proposed_mean |   reference_mean |   delta |   p_corrected_t |   p_holm | significance   |
|:-------------------|:-------------|----------------:|-----------------:|--------:|----------------:|---------:|:---------------|
| glass_adhesion_kpa | ElasticNet   |          0.7924 |           0.7681 |  0.0243 |          0.142  |   0.7099 | ns             |
| glass_adhesion_kpa | HistGB       |          0.7924 |           0.7624 |  0.03   |          0.073  |   0.4379 | ns             |
| glass_adhesion_kpa | KNN          |          0.7924 |           0.787  |  0.0054 |          0.7686 |   1      | ns             |
| glass_adhesion_kpa | MLP          |          0.7924 |           0.701  |  0.0914 |          0.008  |   0.0561 | ns             |
| glass_adhesion_kpa | Mean         |          0.7924 |          -0.0034 |  0.7958 |          0      |   0      | ****           |
| glass_adhesion_kpa | RandomForest |          0.7924 |           0.8067 | -0.0143 |          0.348  |   1      | ns             |
| glass_adhesion_kpa | Ridge        |          0.7924 |           0.7688 |  0.0237 |          0.1434 |   0.7099 | ns             |
| glass_adhesion_kpa | SVR-RBF      |          0.7924 |           0.8026 | -0.0102 |          0.5863 |   1      | ns             |

1/8 comparisons remain significant after Holm correction.

## Bootstrap confidence intervals (cluster bootstrap)

| scope        | model   | target             | metric   |   point |      lo |      hi |     se |
|:-------------|:--------|:-------------------|:---------|--------:|--------:|--------:|-------:|
| internal_oof | SIMPLEX | glass_adhesion_kpa | R2       |  0.7497 |  0.6647 |  0.8178 | 0.0393 |
| internal_oof | SIMPLEX | glass_adhesion_kpa | RMSE     | 18.0512 | 15.4293 | 20.5099 | 1.3116 |
| internal_oof | SIMPLEX | glass_adhesion_kpa | MAE      | 13.1829 | 11.4308 | 15.0491 | 0.9268 |
| external     | SIMPLEX | glass_adhesion_kpa | R2       |  0.6946 |  0.4385 |  0.8453 | 0.1021 |
| external     | SIMPLEX | glass_adhesion_kpa | RMSE     | 34.5456 | 24.8587 | 43.4227 | 4.7884 |
| external     | SIMPLEX | glass_adhesion_kpa | MAE      | 26.594  | 18.6536 | 35.2461 | 4.2967 |

## Permutation test

| target             | metric   |   observed |   null_mean |   null_p95 |   p_value |   n_perm | significance   |
|:-------------------|:---------|-----------:|------------:|-----------:|----------:|---------:|:---------------|
| glass_adhesion_kpa | R2       |    0.74969 |    -0.75138 |   -0.53711 |    0.0002 |     5000 | ***            |

