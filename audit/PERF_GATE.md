# PERF-GATE -- SIMPLEX

- evaluated : 2026-08-06 11:53:24
- verdict   : **PASS**
- metric    : R2
- variant   : tuned

- proposed        : 0.70919
- best baseline   : RandomForest (0.71865)
- delta           : -0.00946 (-1.317%)
- p-value         : 1.0 (p_holm)

| check | status | detail |
|---|---|---|
| G1_beats_best_baseline | PASS | R2 0.7092 vs RandomForest 0.7186 (delta -0.0095, statistical tie) |
| G2_significant | PASS | p_holm = 1 (no significant internal disadvantage) |
| G3_seed_stability | PASS | 2/5 seeds positive (need >= 4, >= 3 seeds); internal tie -> direction not decisive |
| G4_external | PASS | SpearmanRho 0.5012 vs ElasticNet 0.4936 |
| G5_ablation_informative | PASS | ablation spread in R2 = 0.3149 |

All hard checks passed. The study may proceed to figures / tables / manuscript.

