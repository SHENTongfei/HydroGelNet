"""External calibration + reframed metrics experiment (dr15_e200_lr3, 5 seeds)."""
import numpy as np
import paths
from build_dataset import load_dataset
from trainer import TrainConfig, set_seed, run_cv, split_modalities, predict
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

ds = load_dataset(paths.DATASET_NPZ)
ext = load_dataset(paths.EXTERNAL_NPZ)
Ye = ext["Y"].ravel()
Xte = ext["X"]
me = ext["modality_ends"]

cfg = TrainConfig()
cfg.d_model = 64; cfg.n_blocks = 2; cfg.n_heads = 4; cfg.dropout = 0.15
cfg.use_transformer = False; cfg.use_attention = True; cfg.use_film = False
cfg.use_modality_gate = False
cfg.use_contrastive = False; cfg.use_pretrain_recon = False; cfg.use_mfm = False
cfg.use_sam = False; cfg.use_ema = False
cfg.use_mixup = True; cfg.mixup_alpha = 0.4; cfg.use_swa = True
cfg.use_uncertainty_weighting = False
cfg.use_domain_constraint = True; cfg.constraint_w = 0.1
cfg.y_transform = "standard"; cfg.max_epochs = 200; cfg.patience = 30
cfg.batch_size = 32; cfg.lr = 3e-3; cfg.weight_decay = 1e-3

preds_e, preds_oof, y_oof = [], [], []
for seed in [42, 2024, 7, 1337, 20260731]:
    set_seed(seed)
    m, oof, fitted = run_cv(cfg, ds, seeds=[seed], tag="CAL", verbose=False, n_splits=5)
    # oof may be a DataFrame; extract prediction and true columns
    if hasattr(oof, "columns"):
        col_p = [c for c in oof.columns if c.startswith("y_pred")]
        col_t = [c for c in oof.columns if c.startswith("y_true")]
        preds_oof.append(oof[col_p[0]].values)
        y_oof.append(oof[col_t[0]].values)
    else:
        preds_oof.append(np.asarray(oof).ravel())
        y_oof.append(np.asarray(ds["Y"]).ravel()[: len(oof)])
    for item in fitted:
        prep, model = item["prep"], item["model"]
        Xs = prep.transform_x(Xte)
        X1, X2 = split_modalities(Xs, prep.modality_ends(me))
        p_s, _ = predict(model, cfg, X1, X2, np.zeros(len(Ye), dtype=int), 1,
                         "regression")
        preds_e.append(prep.inverse_y(p_s).ravel())

ens_e = np.mean(np.stack(preds_e), axis=0)
oof_all = np.concatenate(preds_oof)
y_oof_all = np.concatenate(y_oof)

cal = LinearRegression().fit(oof_all.reshape(-1, 1), y_oof_all)
ens_cal = cal.predict(ens_e.reshape(-1, 1))

print("=== EXTERNAL (5-seed x 5-fold ensemble, dr15_e200_lr3) ===")
print(f"  raw R2         = {r2_score(Ye, ens_e):.3f}")
print(f"  CALIBRATED R2  = {r2_score(Ye, ens_cal):.3f}  (a={cal.coef_[0]:.3f}, b={cal.intercept_:.2f})")
print(f"  log R2         = {r2_score(np.log1p(Ye), np.log1p(ens_e)):.3f}")
print(f"  Spearman       = {spearmanr(Ye, ens_e)[0]:.3f}")
print(f"  RMSE(raw)      = {np.sqrt(np.mean((Ye - ens_e)**2)):.1f}")
print(f"  RMSE(cal)      = {np.sqrt(np.mean((Ye - ens_cal)**2)):.1f}")
print(f"  MAPE% (cal)    = {np.mean(np.abs(Ye - ens_cal) / Ye) * 100:.1f}")
thr = np.median(Ye)
auc = roc_auc_score((Ye >= thr).astype(int), ens_e)
print(f"  ROC-AUC top/bottom half = {auc:.3f}")
# save predictions for downstream use
np.savez("C:/Users/TS/WorkBuddy/HydroGelNet/results/preds/external_ens_v3.npz",
         y=Ye, raw=ens_e, cal=ens_cal)
print("saved external_ens_v3.npz")
