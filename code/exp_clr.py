"""Test compositional CLR (log-ratio) feature variants for SIMPLEX.
Variants: A=raw fractions+pairwise (current), B=CLR replaces raw fractions,
C=raw+CLR+pairwise. Internal CV R2 + external ranking/absolute metrics.
"""
import numpy as np
import json
import paths
from build_dataset import load_dataset
from trainer import TrainConfig, set_seed, run_cv, split_modalities, predict
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

ds = load_dataset(paths.DATASET_NPZ)
ext = load_dataset(paths.EXTERNAL_NPZ)
Ye = ext["Y"].ravel()
Xte = ext["X"]
me = ext["modality_ends"]
Xtr = ds["X"]

# ---- CLR transform ----
def clr(X6):
    g = np.exp(np.mean(np.log(np.clip(X6, 1e-8, None)), axis=1, keepdims=True))
    return np.log(np.clip(X6, 1e-8, None) / g).astype(np.float32)

raw6 = Xtr[:, :6]
ext6 = Xte[:, :6]
raw_pair = Xtr[:, 6:]
ext_pair = Xte[:, 6:]
clr_tr = clr(raw6)
clr_te = clr(ext6)

variants = {
    "A_raw+pair": (np.hstack([raw6, raw_pair]), [6, 21]),
    "B_clr+pair": (np.hstack([clr_tr, raw_pair]), [6, 21]),
    "C_raw+clr+pair": (np.hstack([raw6, clr_tr, raw_pair]), [12, 27]),
}
Xte_variants = {
    "A_raw+pair": (np.hstack([ext6, ext_pair]), [6, 21]),
    "B_clr+pair": (np.hstack([clr_te, ext_pair]), [6, 21]),
    "C_raw+clr+pair": (np.hstack([ext6, clr_te, ext_pair]), [12, 27]),
}

def mk():
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
    return cfg

def ev(cfg, X, ends, Xt, ends_t):
    d = dict(ds); d["X"] = X; d["modality_ends"] = np.asarray(ends, dtype=np.int32)
    r2s, preds = [], []
    for seed in [42, 2024, 7]:
        set_seed(seed)
        m, _, fitted = run_cv(cfg, d, seeds=[seed], tag="V", verbose=False, n_splits=3)
        r2s.append(float(m["R2"].mean()))
        for item in fitted:
            prep, model = item["prep"], item["model"]
            Xs = prep.transform_x(Xt)
            X1, X2 = split_modalities(Xs, prep.modality_ends(ends_t))
            p_s, _ = predict(model, cfg, X1, X2, np.zeros(len(Ye), dtype=int), 1,
                             "regression")
            preds.append(prep.inverse_y(p_s).ravel())
    ens = np.mean(np.stack(preds), axis=0)
    def topk(k):
        return len(set(np.argsort(-Ye)[:k]) & set(np.argsort(-ens)[:k])) / k
    return {
        "intR2": float(np.mean(r2s)),
        "extR2": float(r2_score(Ye, ens)),
        "logR2": float(r2_score(np.log1p(Ye), np.log1p(ens))),
        "spear": float(spearmanr(Ye, ens)[0]),
        "top20": float(topk(20)),
        "mape": float(np.mean(np.abs(Ye - ens) / Ye) * 100),
        "rmse": float(np.sqrt(np.mean((Ye - ens) ** 2))),
    }

cfg = mk()
print("variant          intR2  extR2  logR2  spear top20  MAPE%  RMSE")
results = {}
for name in variants:
    X, ends = variants[name]
    Xt, ends_t = Xte_variants[name]
    r = ev(cfg, X, ends, Xt, ends_t)
    results[name] = r
    print(f"{name:14s} {r['intR2']:6.3f} {r['extR2']:6.3f} {r['logR2']:6.3f} "
          f"{r['spear']:5.3f} {r['top20']:5.2f} {r['mape']:6.1f} {r['rmse']:6.1f}")
with open(paths.TUNING_DIR + "/clr_compare.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved -> results/tuning/clr_compare.json")
