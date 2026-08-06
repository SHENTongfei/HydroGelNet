"""Decisive experiment: 8-seed ensemble SIMPLEX vs RF on external OOD.
Primary metric: external Spearman (ranking for material screening).
Statistical test: Wilcoxon signed-rank across seeds."""
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestRegressor
import paths
from build_dataset import load_dataset
from trainer import TrainConfig, set_seed, run_cv, evaluate_external

ds = load_dataset(paths.DATASET_NPZ)
ext = load_dataset(paths.EXTERNAL_NPZ)
Ye = ext["Y"].ravel()

# SIMPLEX best-known single-task config (ResBlock, standard)
cfg = TrainConfig()
cfg.d_model = 64; cfg.n_blocks = 2; cfg.n_heads = 4; cfg.dropout = 0.2
cfg.use_transformer = False; cfg.use_attention = True; cfg.use_film = False
cfg.use_modality_gate = False
cfg.use_contrastive = False; cfg.use_pretrain_recon = False; cfg.use_mfm = False
cfg.use_sam = False; cfg.use_ema = False
cfg.use_mixup = True; cfg.use_swa = True
cfg.use_uncertainty_weighting = False
cfg.use_domain_constraint = True
cfg.y_transform = "standard"
cfg.max_epochs = 150; cfg.patience = 30; cfg.batch_size = 32
cfg.lr = 3e-3; cfg.weight_decay = 1e-3

SEEDS = [42, 2024, 7, 1337, 20260731, 31415, 271828, 123456]

# ---- SIMPLEX 8 seeds x 3 folds ----
all_preds = []
for seed in SEEDS:
    set_seed(seed)
    _, _, fitted = run_cv(cfg, ds, seeds=[seed], tag=f"dec_{seed}",
                          verbose=False, n_splits=3)
    for item in fitted:
        prep, model = item["prep"], item["model"]
        from trainer import split_modalities, predict
        Xs = prep.transform_x(ext["X"])
        X1, X2 = split_modalities(Xs, prep.modality_ends(ext["modality_ends"]))
        p_s, _ = predict(model, cfg, X1, X2, np.zeros(len(Ye), dtype=int),
                         1, "regression")
        all_preds.append(prep.inverse_y(p_s).ravel())

# per-seed ensemble (3 models each) for Wilcoxon
sp_per_seed = []
for i in range(0, len(all_preds), 3):
    ens = np.mean(np.stack(all_preds[i:i+3]), axis=0)
    sp_per_seed.append(spearmanr(Ye, ens)[0])
ens_all = np.mean(np.stack(all_preds), axis=0)

# ---- RF 8 seeds ----
rf_sp = []
rf_preds = []
for seed in SEEDS:
    rf = RandomForestRegressor(n_estimators=500, random_state=seed)
    rf.fit(ds["X"], ds["Y"].ravel())
    p = rf.predict(ext["X"])
    rf_preds.append(p)
    rf_sp.append(spearmanr(Ye, p)[0])
rf_ens = np.mean(np.stack(rf_preds), axis=0)

print("=== DECISIVE: external OOD screening (Spearman) ===")
print(f"SIMPLEX per-seed: {[f'{s:.3f}' for s in sp_per_seed]}")
print(f"SIMPLEX mean {np.mean(sp_per_seed):.3f} +- {np.std(sp_per_seed):.3f}")
print(f"RF      per-seed: {[f'{s:.3f}' for s in rf_sp]}")
print(f"RF      mean {np.mean(rf_sp):.3f} +- {np.std(rf_sp):.3f}")
try:
    w, p = wilcoxon(sp_per_seed, rf_sp)
    print(f"Wilcoxon signed-rank: W={w:.1f} p={p:.4f}")
except Exception as e:
    print(f"Wilcoxon failed: {e}")

print(f"\nFull ensemble (24 SIMPLEX models):  Spearman {spearmanr(Ye, ens_all)[0]:.3f}  RMSE {np.sqrt(np.mean((Ye-ens_all)**2)):.1f}")
print(f"Full ensemble (8 RF models):        Spearman {spearmanr(Ye, rf_ens)[0]:.3f}  RMSE {np.sqrt(np.mean((Ye-rf_ens)**2)):.1f}")

# Top-k screening precision (k=20)
def topk_prec(pred, k):
    topk_pred = np.argsort(pred)[-k:]
    topk_true = set(np.argsort(Ye)[-k:])
    return len(set(topk_pred) & topk_true) / k
print(f"\nTop-20 screening precision: SIMPLEX {topk_prec(ens_all,20):.2f} | RF {topk_prec(rf_ens,20):.2f}")
print(f"Top-10 screening precision: SIMPLEX {topk_prec(ens_all,10):.2f} | RF {topk_prec(rf_ens,10):.2f}")

np.savez("results/dec_simplex_preds.npz", ens=ens_all, per_seed=np.array(sp_per_seed))
np.savez("results/dec_rf_preds.npz", ens=rf_ens, per_seed=np.array(rf_sp))
