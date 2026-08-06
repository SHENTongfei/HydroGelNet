"""Multi-task experiment: does adding rheology aux targets (Modulus/TanD)
improve external Glass extrapolation for SIMPLEX?"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import paths
from build_dataset import load_dataset, MONOMER_NAMES
from trainer import TrainConfig, set_seed, run_cv, evaluate_external

ds = load_dataset(paths.DATASET_NPZ)
ext = load_dataset(paths.EXTERNAL_NPZ)
Ye = ext["Y"].ravel()

# ---- build multi-target Y from original xlsx (180 internal rows) ----
XLSX = r"C:/Users/TS/WorkBuddy/2026-08-03-20-59-21/.workbuddy/tmp/hydrogel_candidate/sh_220.xlsx"
dfx = pd.read_excel(XLSX)
# float64 matching to avoid float32 rounding mismatch
dfx["key"] = dfx[MONOMER_NAMES].astype(np.float64).round(4).apply(tuple, axis=1)
X6 = ds["X"][:, :6].astype(np.float64)
int_keys = [tuple(np.round(r, 4)) for r in X6]
key2row = {}
for i, k in enumerate(dfx["key"]):
    if k not in key2row:
        key2row[k] = i

Y_glass = ds["Y"].ravel()
Y_mod = np.full(len(int_keys), np.nan, dtype=np.float32)
Y_tan = np.full(len(int_keys), np.nan, dtype=np.float32)
for i, k in enumerate(int_keys):
    r = key2row.get(k)
    if r is not None:
        m = pd.to_numeric(dfx.loc[r, "Modulus (kPa)"], errors="coerce")
        t = pd.to_numeric(dfx.loc[r, "Tan\u03b4"], errors="coerce")
        if pd.notna(m):
            Y_mod[i] = float(m)
        if pd.notna(t):
            Y_tan[i] = float(t)

print(f"internal multi-target coverage: Glass {np.isfinite(Y_glass).sum()} / "
      f"Modulus {np.isfinite(Y_mod).sum()} / TanD {np.isfinite(Y_tan).sum()}")
# use only fully-observed rows for multi-task
mt_ok = np.isfinite(Y_glass) & np.isfinite(Y_mod) & np.isfinite(Y_tan)
print(f"fully-observed rows: {mt_ok.sum()}")

# ---- shared base config ----
def base_cfg(mt: bool):
    cfg = TrainConfig()
    cfg.d_model = 64; cfg.n_blocks = 3; cfg.n_heads = 4; cfg.dropout = 0.25
    cfg.use_transformer = False; cfg.use_attention = True; cfg.use_film = False
    cfg.use_modality_gate = False
    cfg.use_contrastive = False; cfg.use_pretrain_recon = False; cfg.use_mfm = False
    cfg.use_sam = True; cfg.use_ema = True
    cfg.use_mixup = True; cfg.mixup_alpha = 0.3
    cfg.use_swa = True
    cfg.use_uncertainty_weighting = mt   # uncertainty-weighted loss for MT
    cfg.use_domain_constraint = True; cfg.constraint_w = 0.1
    cfg.y_transform = "standard"
    cfg.max_epochs = 180; cfg.patience = 35; cfg.batch_size = 32
    cfg.lr = 2e-3; cfg.weight_decay = 3e-3
    return cfg

def run_eval(cfg, Y_mt, target_names, label):
    ds2 = dict(ds)
    ds2["Y"] = Y_mt.astype(np.float32)
    ds2["target_names"] = np.array(target_names)
    sp_all, rmse_all = [], []
    for seed in [42, 2024, 7]:
        set_seed(seed)
        metrics, _, fitted = run_cv(cfg, ds2, seeds=[seed], tag=f"{label}_{seed}",
                                    verbose=False, n_splits=3)
        # external: predict all targets, take col 0 (glass)
        preds = []
        for item in fitted:
            prep, model = item["prep"], item["model"]
            from trainer import split_modalities, predict
            Xs = prep.transform_x(ext["X"])
            X1, X2 = split_modalities(Xs, prep.modality_ends(ext["modality_ends"]))
            p_s, _ = predict(model, cfg, X1, X2,
                             np.zeros(len(Ye), dtype=int), 1, "regression")
            pi = prep.inverse_y(p_s)
            if pi.ndim == 2:
                pi = pi[:, 0]
            preds.append(np.asarray(pi).ravel())
        ens = np.mean(np.stack(preds), axis=0)
        sp_all.append(spearmanr(Ye, ens)[0])
        rmse_all.append(np.sqrt(np.mean((Ye - ens) ** 2)))
    print(f"\n=== {label} (3 seeds x 3 folds ensemble) ===")
    print(f"  external Glass Spearman: {np.mean(sp_all):.3f} +- {np.std(sp_all):.3f}")
    print(f"  external Glass RMSE:     {np.mean(rmse_all):.1f}")
    return sp_all, rmse_all

# ---- single-task baseline (Glass only) ----
print("\n========== SINGLE-TASK (Glass only) ==========")
sp_st, rm_st = run_eval(base_cfg(False), ds["Y"], ["glass_adhesion_kpa"], "ST")

# ---- multi-task (Glass + Modulus) ----
print("\n========== MULTI-TASK (Glass + Modulus) ==========")
Y_mt2 = np.column_stack([Y_glass, Y_mod]).astype(np.float32)
sp_mt2, rm_mt2 = run_eval(base_cfg(True), Y_mt2,
                          ["glass_adhesion_kpa", "modulus_kpa"], "MT2")

# ---- multi-task (Glass + Modulus + TanD) ----
print("\n========== MULTI-TASK (Glass + Modulus + TanD) ==========")
Y_mt3 = np.column_stack([Y_glass, Y_mod, Y_tan]).astype(np.float32)
sp_mt3, rm_mt3 = run_eval(base_cfg(True), Y_mt3,
                          ["glass_adhesion_kpa", "modulus_kpa", "tand"], "MT3")

print("\n===== SUMMARY =====")
print(f"Single-task   : Spearman {np.mean(sp_st):.3f}  RMSE {np.mean(rm_st):.1f}")
print(f"MT Glass+Mod  : Spearman {np.mean(sp_mt2):.3f}  RMSE {np.mean(rm_mt2):.1f}")
print(f"MT Glass+Mod+T: Spearman {np.mean(sp_mt3):.3f}  RMSE {np.mean(rm_mt3):.1f}")
