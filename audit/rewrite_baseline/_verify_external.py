# -*- coding: utf-8 -*-
import csv, statistics

def load(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))

print('===== baselines.csv SVR-RBF =====')
rows = load(r'C:\Users\TS\WorkBuddy\HydroGelNet\results\metrics\baselines.csv')
sub = [r for r in rows if r['model'] == 'SVR-RBF']
r2s = [float(r['R2']) for r in sub]
rmses = [float(r['RMSE']) for r in sub]
maes = [float(r['MAE']) for r in sub]
print(f"SVR-RBF: n={len(sub)} R2={statistics.mean(r2s):.6f} RMSE={statistics.mean(rmses):.4f} MAE={statistics.mean(maes):.4f}")

print()
print('===== baselines_external.csv: external single + ensemble =====')
rows = load(r'C:\Users\TS\WorkBuddy\HydroGelNet\results\metrics\baselines_external.csv')
print('cols:', list(rows[0].keys()))
print('tag values:', sorted(set(r['tag'] for r in rows)))
print('model values:', sorted(set(r['model'] for r in rows)))
for m in sorted(set(r['model'] for r in rows)):
    ens = [r for r in rows if r['model'] == m and r['tag'] == 'external_ensemble']
    if ens:
        e = ens[0]
        print(f"  {m} ENSEMBLE: R2={float(e['R2']):.4f} Spearman={float(e['SpearmanRho']):.4f} RMSE={float(e['RMSE']):.4f} MAE={float(e['MAE']):.4f}")
    sgl = [r for r in rows if r['model'] == m and r['tag'] == 'external_single']
    if sgl:
        r2s = [float(r['R2']) for r in sgl]
        sps = [float(r['SpearmanRho']) for r in sgl]
        print(f"  {m} SINGLE: n={len(sgl)} R2 mean={statistics.mean(r2s):.4f} (SD {statistics.stdev(r2s):.4f}) Spearman mean={statistics.mean(sps):.4f}")
