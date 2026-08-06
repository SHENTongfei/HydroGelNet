"""kNN composition-neighbour features for Top-k breakthrough.
Add features = y-stats of the 5 nearest training neighbours in composition
space (mean, max, distance-weighted). Train on log1p target; evaluate Top-k.
kNN uses leave-one-out inside training (per-fold), full-train for external.
"""
import numpy as np
import paths
from build_dataset import load_dataset
from trainer import TrainConfig, set_seed, run_cv, split_modalities, predict
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist

ds = load_dataset(paths.DATASET_NPZ)
ext = load_dataset(paths.EXTERNAL_NPZ)
Ye = ext["Y"].ravel()
Xte = ext["X"]
me = ext["modality_ends"]

def knn_feats(X6, ref_X6, ref_y, k=5):
    """Nearest-neighbour y statistics (mean/max/weighted)."""
    D = cdist(X6, ref_X6)
    idx = np.argsort(D, axis=1)[:, :k]
    nbr_y = ref_y[idx]
    w = 1.0 / (D[np.arange(len(D))[:, None], idx] + 1e-6)
    w = w / w.sum(axis=1, keepdims=True)
    mean_k = (nbr_y * w).sum(axis=1)          # distance-weighted mean
    max_k = nbr_y.max(axis=1)
    med_k = np.median(nbr_y, axis=1)
    return np.stack([mean_k, max_k, med_k], axis=1).astype(np.float32)

def add_knn(d):
    X = d["X"]
    X6 = X[:, :6]
    y = d["Y"].ravel()
    # LOO-style: for each sample use the other n-1 as reference
    f = np.zeros((len(X), 3), dtype=np.float32)
    for i in range(len(X)):
        m = np.ones(len(X), dtype=bool); m[i] = False
        f[i] = knn_feats(X6[i:i+1], X6[m], y[m])[0]
    Xn = np.hstack([X, f])
    ends = np.asarray(list(d["modality_ends"]) + [X.shape[1] + 3], dtype=np.int32)
    return Xn, ends

def add_knn_ext():
    X6 = Xte[:, :6]
    ytr = ds["Y"].ravel()
    f = knn_feats(X6, ds["X"][:, :6], ytr)
    return np.hstack([Xte, f])

Xtr_n, ends_n = add_knn(ds)
Xte_n = add_knn_ext()
ends_e = np.asarray([6, 21, 24], dtype=np.int32)

def mk(yt="log1p"):
    cfg = TrainConfig()
    cfg.d_model = 64; cfg.n_blocks = 2; cfg.n_heads = 4; cfg.dropout = 0.15
    cfg.use_transformer = False; cfg.use_attention = True; cfg.use_film = False
    cfg.use_modality_gate = False
    cfg.use_contrastive = False; cfg.use_pretrain_recon = False; cfg.use_mfm = False
    cfg.use_sam = False; cfg.use_ema = False
    cfg.use_mixup = True; cfg.mixup_alpha = 0.4; cfg.use_swa = True
    cfg.use_uncertainty_weighting = False
    cfg.use_domain_constraint = False; cfg.constraint_w = 0.0
    cfg.y_transform = yt; cfg.max_epochs = 200; cfg.patience = 30
    cfg.batch_size = 32; cfg.lr = 3e-3; cfg.weight_decay = 1e-3
    return cfg

def topk(y, p, k):
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k

def ev(cfg):
    d2 = dict(ds); d2["X"] = Xtr_n; d2["modality_ends"] = ends_n
    preds, r2s = [], []
    for seed in [42, 2024, 7]:
        set_seed(seed)
        m, _, fitted = run_cv(cfg, d2, seeds=[seed], tag="KN", verbose=False,
                              n_splits=3)
        r2s.append(float(m["R2"].mean()))
        for item in fitted:
            prep, model = item["prep"], item["model"]
            Xs = prep.transform_x(Xte_n)
            X1, X2 = split_modalities(Xs, prep.modality_ends(ends_e))
            p_s, _ = predict(model, cfg, X1, X2, np.zeros(len(Ye), dtype=int), 1,
                             "regression")
            preds.append(prep.inverse_y(p_s).ravel())
    P = np.stack(preds)
    ens = P.mean(axis=0)
    ranks = np.stack([np.argsort(np.argsort(-P[i])) for i in range(len(P))])
    rank_ens = ranks.mean(axis=0)
    return {"intR2": float(np.mean(r2s)),
            "TopK20": float(topk(Ye, ens, 20)),
            "TopK30": float(topk(Ye, ens, 30)),
            "TopK20_RA": float(topk(Ye, -rank_ens, 20)),
            "TopK30_RA": float(topk(Ye, -rank_ens, 30)),
            "spear": float(spearmanr(Ye, ens)[0]),
            "prange": (float(ens.min()), float(ens.max()))}

print("=== log1p + kNN-neighbour features ===")
for name, yt in [("log1p + knn", "log1p"), ("standard + knn", "standard")]:
    r = ev(mk(yt))
    print(f"{name:16s} intR2={r['intR2']:.3f} TopK20={r['TopK20']:.2f} "
          f"TopK30={r['TopK30']:.2f} TK20-RA={r['TopK20_RA']:.2f} "
          f"TK30-RA={r['TopK30_RA']:.2f} spear={r['spear']:.3f} "
          f"pred=[{r['prange'][0]:.0f},{r['prange'][1]:.0f}]")
