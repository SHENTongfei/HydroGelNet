# Run log -- SIMPLEX

- started : 2026-08-06 19:01:15
- python  : C:/Users/TS/.conda/envs/HydroGelNet/python.exe
- root    : C:/Users/TS/WorkBuddy/HydroGelNet
- mode    :
- total   : 22.8 min

| stage | status | seconds | command |
|---|---|---|---|
| train | ok | 272.6 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\trainer.py --config C:/Users/TS/WorkBuddy/HydroGelNet\results\tuning\best_config.json --tag main --seeds 5` |
| baselines | ok | 867.8 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\baselines.py --seeds 5 --n-iter 40` |
| stats | ok | 48.5 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\stats_tests.py` |
| gate | ok | 2.8 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\escalate.py --strict` |
| interpret | ok | 152.9 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\interpret.py --seeds 2` |
| figures | ok | 5.1 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py` |
| tables | ok | 6.6 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\tables.py` |
| paper | ok | 11.2 | `C:/Users/TS/.conda/envs/HydroGelNet/python.exe C:\Users\TS\WorkBuddy\HydroGelNet\code\paper_pdf.py --topic gate` |
