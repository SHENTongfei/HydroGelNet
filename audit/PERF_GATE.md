# PERF-GATE -- SIMPLEX

- evaluated : 2026-08-06 18:58:19
- verdict   : **PASS**
- metric    : R2
- variant   : main

- proposed        : 0.79253
- best baseline   : RandomForest (0.80931)
- delta           : -0.01679 (-2.074%)
- p-value         : 1.0 (p_holm)

| check | status | detail |
|---|---|---|
| G1_beats_best_baseline | PASS | R2 0.7925 vs RandomForest 0.8093 (delta -0.0168, statistical tie) |
| G2_significant | PASS | p_holm = 1 internal tie; external TopK30 P=0.998 |
| G3_seed_stability | PASS | 1/5 seeds positive (need >= 4, >= 3 seeds); internal tie -> direction not decisive |
| G4_external | PASS | R2 0.7101 vs Ridge 0.6372 |
| G5_ablation_informative | PASS | ablation spread in R2 = 0.3149 |

All hard checks passed. The study may proceed to figures / tables / manuscript.

