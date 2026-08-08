# Run log -- SIMPLEX

- started : 2026-08-08 00:46:00
- python  : C:/Users/TS/.conda/envs/HydroGelNet/python.exe
- root    : C:/Users/TS/WorkBuddy/HydroGelNet
- mode    :
- total   : 120.8 min

| stage | status | seconds | command |
|---|---|---|---|
| train | ok | 5996.8 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\trainer.py --config C:/Users/TS/WorkBuddy/HydroGelNet\results\tuning\best_config_final.json --tag main --seeds 10` |
| baselines | ok | 1213.6 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\baselines.py --seeds 10 --n-iter 40` |
| stats | ok | 32.1 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\stats_tests.py` |
| gate | FAIL 1 | 3.8 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\escalate.py --strict` |
