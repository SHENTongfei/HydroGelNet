"""log1p target + pairwise ranking loss for Top-k breakthrough."""
import numpy as np
import paths
from build_dataset import load_dataset
from trainer import TrainConfig, set_seed, run_cv, split_modalities, predict
from scipy.stats import spearmanr

ds = load_dataset(paths.DATASET_NPZ)
ext = load_dataset(paths.EXTERNAL_NPZ)
Ye = ext["Y"].ravel()
Xte = ext["X"]
me = ext["modality_ends"]

def mk(yt="log1p", use_rank=False, margin=0.5, w=0.5):
    cfg = TrainConfig()
    cfg.d_model = 64; cfg.n_blocks = 2; cfg.n_heads = 4; cfg.dropout = 0.15
    cfg.use_transformer = False; cfg.use_attention = True; cfg.use_film = False
    cfg.use_modality_gate = False
    cfg.use_contrastive = False; cfg.use_pretrain_recon = False; cfg.use_mfm = False
    cfg.use_sam = False; cfg.use_ema = False
    cfg.use_mixup = True; cfg.mixup_alpha = 0.4; cfg.use_swa = True
    cfg.use_uncertainty_weighting = False
    cfg.use_domain_constraint = False; cfg.constraint_w = 0.0
    cfg.use_ranking_loss = use_rank
    cfg.rank_margin = margin; cfg.rank_loss_w = w
    cfg.y_transform = yt; cfg.max_epochs = 200; cfg.patience = 30
    cfg.batch_size = 32; cfg.lr = 3e-3; cfg.weight_decay = 1e-3
    return cfg

def topk(y, p, k):
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k

def ev(cfg):
    preds, r2s = [], []
    for seed in [42, 2024, 7]:
        set_seed(seed)
        m, _, fitted = run_cv(cfg, ds, seeds=[seed], tag="RK", verbose=False,
                              n_splits=3)
        r2s.append(float(m["R2"].mean()))
        for item in fitted:
            prep, model = item["prep"], item["model"]
            Xs = prep.transform_x(Xte)
            X1, X2 = split_modalities(Xs, prep.modality_ends(me))
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
            "spear": float(spearmanr(Ye, ens)[0])}

print("=== log1p + ranking loss ===")
for name, cfg in [
    ("log1p no-rank", mk("log1p", False)),
    ("log1p rank m0.5 w0.5", mk("log1p", True, 0.5, 0.5)),
    ("log1p rank m1.0 w1.0", mk("log1p", True, 1.0, 1.0)),
    ("log1p rank m0.3 w2.0", mk("log1p", True, 0.3, 2.0)),
]:
    r = ev(cfg)
    print(f"{name:20s} intR2={r['intR2']:.3f} TopK20={r['TopK20']:.2f} "
          f"TopK30={r['TopK30']:.2f} TK20-RA={r['TopK20_RA']:.2f} "
          f"TK30-RA={r['TopK30_RA']:.2f} spear={r['spear']:.3f}")
