"""TopK20 breakthrough: relax range-domain constraint + rank-average ensemble.
Compare: (A) current (constraint on), (B) constraint off, (C) constraint off +
log1p target, (D) rank-average of B. TopK20/30 + pred range."""
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

def mk(yt="standard", cw=0.1):
    cfg = TrainConfig()
    cfg.d_model = 64; cfg.n_blocks = 2; cfg.n_heads = 4; cfg.dropout = 0.15
    cfg.use_transformer = False; cfg.use_attention = True; cfg.use_film = False
    cfg.use_modality_gate = False
    cfg.use_contrastive = False; cfg.use_pretrain_recon = False; cfg.use_mfm = False
    cfg.use_sam = False; cfg.use_ema = False
    cfg.use_mixup = True; cfg.mixup_alpha = 0.4; cfg.use_swa = True
    cfg.use_uncertainty_weighting = False
    cfg.use_domain_constraint = cw > 0; cfg.constraint_w = cw
    cfg.y_transform = yt; cfg.max_epochs = 200; cfg.patience = 30
    cfg.batch_size = 32; cfg.lr = 3e-3; cfg.weight_decay = 1e-3
    return cfg

def topk(y, p, k):
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k

def ev(cfg):
    preds, r2s = [], []
    for seed in [42, 2024, 7]:
        set_seed(seed)
        m, _, fitted = run_cv(cfg, ds, seeds=[seed], tag="TK", verbose=False,
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
    # rank average across models
    ranks = np.stack([np.argsort(np.argsort(-P[i])) for i in range(len(P))])
    rank_ens = ranks.mean(axis=0)
    return {
        "intR2": float(np.mean(r2s)),
        "TopK20": float(topk(Ye, ens, 20)),
        "TopK30": float(topk(Ye, ens, 30)),
        "TopK20_rankavg": float(topk(Ye, -rank_ens, 20)),
        "TopK30_rankavg": float(topk(Ye, -rank_ens, 30)),
        "pred_range": (float(ens.min()), float(ens.max())),
        "spear": float(spearmanr(Ye, ens)[0]),
    }

print("=== TopK breakthrough sweep (3 seeds x 3 folds) ===")
for name, cfg in [
    ("A constraint=0.1 (cur)", mk("standard", 0.1)),
    ("B constraint=0   ", mk("standard", 0.0)),
    ("C log1p, no const", mk("log1p", 0.0)),
]:
    r = ev(cfg)
    print(f"{name:22s} intR2={r['intR2']:.3f} TopK20={r['TopK20']:.2f} "
          f"TopK30={r['TopK30']:.2f} TopK20-RA={r['TopK20_rankavg']:.2f} "
          f"TopK30-RA={r['TopK30_rankavg']:.2f} pred=[{r['pred_range'][0]:.0f},"
          f"{r['pred_range'][1]:.0f}] spear={r['spear']:.3f}")
