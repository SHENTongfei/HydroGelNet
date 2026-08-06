"""New experiment design: train on df_316 (incl. high-y up to 353 kPa),
evaluate on the 25 independent final-SMBO formulations. Top-k + ranking."""
import os
import numpy as np
import pandas as pd
import paths
from build_dataset import load_dataset, MONOMER_NAMES
from trainer import TrainConfig, set_seed, run_cv, split_modalities, predict
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr

RAW = paths.RAW_DIR
d316 = pd.read_csv(os.path.join(RAW, "df_316.csv"))
d341 = pd.read_csv(os.path.join(RAW, "df_341.csv"))
ycol = "Glass (kPa)_max"
cols6 = ["Nucleophilic-HEA", "Hydrophobic-BA", "Acidic-CBEA",
         "Cationic-ATAC", "Aromatic-PEA", "Amide-AAm"]

def feats(df):
    X6 = df[cols6].values.astype(np.float32)
    idx = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    X2 = np.stack([X6[:, i] * X6[:, j] for (i, j) in idx], axis=1).astype(np.float32)
    return np.hstack([X6, X2]).astype(np.float32), X6

# 25 new = in 341 but not in 316 (match by rounded composition)
k316 = set(map(str, d316[cols6].round(6).apply(tuple, axis=1).values))
mask25 = ~d341[cols6].round(6).apply(tuple, axis=1).map(str).isin(k316)
d25 = d341[mask25]
print("external n =", len(d25), "| y: min %.0f max %.0f mean %.0f" % (
    d25[ycol].min(), d25[ycol].max(), d25[ycol].mean()))

Xtr, X6tr = feats(d316)
ytr = d316[ycol].values.astype(np.float32)
Xte, X6te = feats(d25)
Yte = d25[ycol].values.astype(np.float32)

ds = dict(X=Xtr, Y=ytr.reshape(-1, 1), groups=np.arange(len(ytr)),
          cond=np.zeros(len(ytr), dtype=np.int32), cond_levels=["all"],
          feature_names=None, target_names=[ycol],
          modality_ends=np.asarray([6, 21], dtype=np.int32),
          modality_names=["monomer_fractions", "pairwise_synergy"],
          task_type="regression", sample_ids=np.arange(len(ytr)))
me = np.asarray([6, 21], dtype=np.int32)

def mk(yt="standard"):
    cfg = TrainConfig()
    cfg.d_model = 64; cfg.n_blocks = 2; cfg.n_heads = 4; cfg.dropout = 0.15
    cfg.use_transformer = False; cfg.use_attention = True; cfg.use_film = False
    cfg.use_modality_gate = False
    cfg.use_contrastive = False; cfg.use_pretrain_recon = False; cfg.use_mfm = False
    cfg.use_sam = False; cfg.use_ema = False
    cfg.use_mixup = True; cfg.mixup_alpha = 0.4; cfg.use_swa = True
    cfg.use_uncertainty_weighting = False
    cfg.use_domain_constraint = True; cfg.constraint_w = 0.1
    cfg.y_transform = yt; cfg.max_epochs = 250; cfg.patience = 30
    cfg.batch_size = 32; cfg.lr = 2e-3; cfg.weight_decay = 1e-3
    return cfg

def topk(y, p, k):
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k

for yt in ["standard", "log1p"]:
    cfg = mk(yt)
    preds = []
    for seed in [42, 2024, 7, 1337, 20260731]:
        set_seed(seed)
        _, _, fitted = run_cv(cfg, ds, seeds=[seed], tag=f"N{yt}_{seed}",
                              verbose=False, n_splits=5)
        for item in fitted:
            prep, model = item["prep"], item["model"]
            Xs = prep.transform_x(Xte)
            X1, X2 = split_modalities(Xs, prep.modality_ends(me))
            p_s, _ = predict(model, cfg, X1, X2, np.zeros(len(Yte), dtype=int), 1,
                             "regression")
            preds.append(prep.inverse_y(p_s).ravel())
    ens = np.mean(np.stack(preds), axis=0)
    thr = np.median(Yte)
    print(f"[{yt}] n_ext={len(Yte)}: R2={r2_score(Yte, ens):.3f} "
          f"Spearman={spearmanr(Yte, ens)[0]:.3f} AUC={roc_auc_score(Yte > thr, ens):.3f} "
          f"TopK5={topk(Yte, ens, 5):.2f} TopK10={topk(Yte, ens, 10):.2f} "
          f"TopK15={topk(Yte, ens, 15):.2f} TopK20={topk(Yte, ens, min(20, len(Yte))):.2f} "
          f"pred=[{ens.min():.0f},{ens.max():.0f}]")
