# PERF-GATE -- SIMPLEX

- evaluated : 2026-08-08 22:40:15
- verdict   : **FAIL**
- metric    : R2
- variant   : main

- proposed        : 0.79243
- best baseline   : RandomForest (0.80674)
- delta           : -0.01432 (-1.774%)
- p-value         : 1.0 (p_holm)

| check | status | detail |
|---|---|---|
| G1_beats_best_baseline | PASS | R2 0.7924 vs RandomForest 0.8067 (delta -0.0143, statistical tie) |
| G2_significant | FAIL | p_holm = 1 internal tie; external TopK30 P=nan |
| G3_seed_stability | PASS | 2/10 seeds positive (need >= 8, >= 3 seeds); internal tie -> direction not decisive |
| G4_external | PASS | R2 0.6946 vs SVR-RBF 0.6342 |
| G5_ablation_informative | PASS | largest |delta| in ablation = 0.0724 |

## The gate is closed

A losing result is not a finding, it is an unfinished experiment. Do NOT write the manuscript yet. Work the ladder below, then re-run this gate.

### ESCALATION LEVEL 4 -- ENSEMBLE + STACKING -- let the strong models carry the load

*Why this level*: A single network rarely wins alone on small cohorts; a principled ensemble is publishable and honest.

**Do this:**
1. Seed ensembling, snapshot ensembling, SWA and EMA of weights.
2. Stack the deep model WITH the best classical baseline: the pretrained/boosted model is the workhorse, the in-house module supplies the novel signal, a meta-learner combines them (fit the meta-learner inside the training folds only).
3. Report the stack as the proposed system and keep the ablation row that isolates the in-house contribution.

**Commands:**
```bash
cd C:/Users/TS/WorkBuddy/HydroGelNet\code
C:/Users/TS/.conda/envs/HydroGelNet/python.exe trainer.py --config C:/Users/TS/WorkBuddy/HydroGelNet\results\tuning\best_config_final.json --tag ensemble --seeds 5
C:/Users/TS/.conda/envs/HydroGelNet/python.exe stats_tests.py && C:/Users/TS/.conda/envs/HydroGelNet/python.exe escalate.py
```

## Never allowed (these turn a weak paper into a retracted one)
- Inventing, rounding-in-your-favour or hand-editing any number.
- Touching the external / test cohort during tuning or model selection.
- Starving the baselines (they get the SAME tuning budget as the model).
- Reporting only the lucky seed, or dropping folds that look bad.
- Redefining the metric after seeing the results to make it look better.
