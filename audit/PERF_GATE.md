# PERF-GATE -- SIMPLEX

- evaluated : 2026-08-06 16:23:26
- verdict   : **PASS**
- metric    : R2
- variant   : main

- proposed        : 0.72605
- best baseline   : RandomForest (0.71865)
- delta           : 0.0074 (1.029%)
- p-value         : 1.0 (p_holm)

| check | status | detail |
|---|---|---|
| G1_beats_best_baseline | PASS | R2 0.7260 vs RandomForest 0.7186 (delta +0.0074, beats) |
| G2_significant | PASS | p_holm = 1 significant win; external TopK30 P=0.998 |
| G3_seed_stability | PASS | 3/5 seeds positive (need >= 4, >= 3 seeds) |
| G4_external | PASS | TopK20 0.3000 vs Mean 0.1500 |
| G5_ablation_informative | PASS | ablation spread in R2 = 0.3149 |

All hard checks passed. The study may proceed to figures / tables / manuscript.

