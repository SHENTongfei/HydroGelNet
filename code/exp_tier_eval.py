"""Formal external tiered-screening evaluation: ROC-AUC (high vs low half),
Spearman, Top-k for SIMPLEX vs all baselines (dr15_e200_lr3)."""
import numpy as np
import paths
from build_dataset import load_dataset
from trainer import TrainConfig, set_seed, run_cv, split_modalities, predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

ds = load_dataset(paths.DATASET_NPZ)
ext = load_dataset(paths.EXTERNAL_NPZ)
Ye = ext["Y"].ravel()
Xtr, ytr = ds["X"], ds["Y"].ravel()
Xte = ext["X"]
me = ext["modality_ends"]

thr = np.median(Ye)
ytier = (Ye >= thr).astype(int)

# ---- SIMPLEX ensemble ----
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

preds = []
for seed in [42, 2024, 7, 1337, 20260731]:
    set_seed(seed)
    _, _, fitted = run_cv(cfg, ds, seeds=[seed], tag="AU", verbose=False, n_splits=5)
    for item in fitted:
        prep, model = item["prep"], item["model"]
        Xs = prep.transform_x(Xte)
        X1, X2 = split_modalities(Xs, prep.modality_ends(me))
        p_s, _ = predict(model, cfg, X1, X2, np.zeros(len(Ye), dtype=int), 1,
                         "regression")
        preds.append(prep.inverse_y(p_s).ravel())
ens = np.mean(np.stack(preds), axis=0)

# ---- baselines (same preprocessing convention: standard-scale on train) ----
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr)
Xte_s = sc.transform(Xte)
bl = {
    "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42).fit(Xtr_s, ytr).predict(Xte_s),
    "ElasticNet": ElasticNet(alpha=1e-2, l1_ratio=0.5, random_state=42).fit(Xtr_s, ytr).predict(Xte_s),
    "Ridge": Ridge(alpha=10.0).fit(Xtr_s, ytr).predict(Xte_s),
    "SVR-RBF": SVR(C=10, gamma="scale", epsilon=5).fit(Xtr_s, ytr).predict(Xte_s),
}

def topk(y, p, k):
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k

print("model          AUC      balanced-acc  Spearman  Top-20  Top-30")
print("SIMPLEX        " + " ".join(f"{x:.3f}" for x in [
    roc_auc_score(ytier, ens),
    balanced_accuracy_score(ytier, ens >= thr),
    spearmanr(Ye, ens)[0], topk(Ye, ens, 20), topk(Ye, ens, 30)]))
for name, p in bl.items():
    print(f"{name:14s} " + " ".join(f"{x:.3f}" for x in [
        roc_auc_score(ytier, p),
        balanced_accuracy_score(ytier, p >= thr),
        spearmanr(Ye, p)[0], topk(Ye, p, 20), topk(Ye, p, 30)]))
np.savez(paths.EXTERNAL_PREDS if hasattr(paths, "EXTERNAL_PREDS") else
         "C:/Users/TS/WorkBuddy/HydroGelNet/results/preds/external_v3_all.npz",
         y=Ye, tier=ytier, simplex=ens, rf=bl["RandomForest"],
         en=bl["ElasticNet"], ridge=bl["Ridge"], svr=bl["SVR-RBF"])
print("saved external_v3_all.npz")
