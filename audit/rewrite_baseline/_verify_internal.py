# -*- coding: utf-8 -*-
import csv, statistics

def load(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))

print('===== cv_outer.csv: SIMPLEX internal per-fold =====')
rows = [r for r in load(r'C:\Users\TS\WorkBuddy\HydroGelNet\results\metrics\cv_outer.csv') if r['model'] == 'SIMPLEX' and r['tag'] == 'main']
print('rows:', len(rows))
seeds = sorted(set(r['seed'] for r in rows))
print('seeds:', seeds, 'count:', len(seeds))
r2 = [float(r['R2']) for r in rows]
rmse = [float(r['RMSE']) for r in rows]
mae = [float(r['MAE']) for r in rows]
print('R2 mean:', round(statistics.mean(r2), 6), 'SD:', round(statistics.stdev(r2), 4))
print('RMSE mean:', round(statistics.mean(rmse), 4), 'SD:', round(statistics.stdev(rmse), 4))
print('MAE mean:', round(statistics.mean(mae), 4), 'SD:', round(statistics.stdev(mae), 4))
# per-fold means
for k in range(5):
    sub = [float(r['R2']) for r in rows if int(r['fold']) == k]
    print(f'  fold {k}: n={len(sub)} mean R2={statistics.mean(sub):.6f}')

print()
print('===== baselines.csv: RF and other baselines (internal) =====')
rows = load(r'C:\Users\TS\WorkBuddy\HydroGelNet\results\metrics\baselines.csv')
print('cols:', list(rows[0].keys()))
print('tag values:', sorted(set(r['tag'] for r in rows)))
print('model values:', sorted(set(r['model'] for r in rows)))
# try to find per-fold rows for RF
for m in ['RandomForest', 'RF', 'random_forest', 'SVR', 'KNN', 'Ridge', 'ElasticNet', 'HistGB', 'MLP']:
    sub = [r for r in rows if r['model'] == m]
    if sub:
        r2s = [float(r['R2']) for r in sub]
        rmses = [float(r['RMSE']) for r in sub]
        maes = [float(r['MAE']) for r in sub]
        print(f'{m}: n={len(sub)} R2={statistics.mean(r2s):.6f} (+-{statistics.stdev(r2s):.4f}) RMSE={statistics.mean(rmses):.4f} MAE={statistics.mean(maes):.4f}')
