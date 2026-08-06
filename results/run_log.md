# Run log -- SIMPLEX

- started : 2026-08-06 16:21:59
- python  : C:/Users/TS/.conda/envs/HydroGelNet/python.exe
- root    : C:/Users/TS/WorkBuddy/HydroGelNet
- mode    :
- total   : 7.9 min

| stage | status | seconds | command |
|---|---|---|---|
| train | ok | 121.7 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\trainer.py --config C:/Users/TS/WorkBuddy/HydroGelNet\results\tuning\best_config.json --tag main --seeds 5` |
| baselines | ok | 324.5 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\baselines.py --seeds 5 --n-iter 40` |
| stats | ok | 22.5 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\stats_tests.py` |
| gate | FAIL 1 | 2.4 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\escalate.py --strict` |
