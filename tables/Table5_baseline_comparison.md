**Table 5.** Comparison against equally tuned baselines. p-values from the Nadeau-Bengio corrected resampled t-test, Holm adjusted.

| Target             | Baseline     |   Baseline R2 |   SIMPLEX R2 |   Delta |   Delta (%) |   Cohen's d |   p (corrected t) |   p (Holm) | Sig.   |
|:-------------------|:-------------|--------------:|-------------:|--------:|------------:|------------:|------------------:|-----------:|:-------|
| glass_adhesion_kpa | ElasticNet   |        0.7689 |       0.7925 |  0.0236 |      3.0759 |      0.4099 |            0.2766 |     1      | ns     |
| glass_adhesion_kpa | HistGB       |        0.7633 |       0.7925 |  0.0293 |      3.8349 |      0.4461 |            0.1255 |     0.7532 | ns     |
| glass_adhesion_kpa | KNN          |        0.7892 |       0.7925 |  0.0033 |      0.4183 |      0.0447 |            0.8888 |     1      | ns     |
| glass_adhesion_kpa | MLP          |        0.6991 |       0.7925 |  0.0934 |     13.3583 |      1.1573 |            0.0221 |     0.1548 | ns     |
| glass_adhesion_kpa | Mean         |       -0.0049 |       0.7925 |  0.7974 |  16378.5    |     18.9057 |            0      |     0      | ****   |
| glass_adhesion_kpa | RandomForest |        0.8093 |       0.7925 | -0.0168 |     -2.0744 |     -0.2871 |            0.2286 |     1      | ns     |
| glass_adhesion_kpa | Ridge        |        0.7691 |       0.7925 |  0.0235 |      3.0506 |      0.4077 |            0.2902 |     1      | ns     |
| glass_adhesion_kpa | SVR-RBF      |        0.799  |       0.7925 | -0.0065 |     -0.8076 |     -0.114  |            0.7451 |     1      | ns     |
