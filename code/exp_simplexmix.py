"""SimplexMix experiment: Mixup with composition-simplex projection.
Standard Mixup interpolates features freely; SimplexMix re-normalises the
6 monomer fractions back to the simplex (sum=1) so augmented compositions
stay physically valid, and can explore compositions beyond the observed
range along simplex edges (extrapolation-oriented augmentation).
Compare internal R2 + external metrics vs standard Mixup.
"""
import numpy as np
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

def simplex_mixup(x, y, alpha=0.4, x2=None):
    """Mixup then project the 6 monomer fractions back onto the simplex."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.shape[0], device=x.device)
    xm = lam * x + (1.0 - lam) * x[idx]
    ym = lam * y + (1.0 - lam) * y[idx]
    x2m = None if x2 is None else lam * x2 + (1.0 - lam) * x2[idx]
    # project modality-1 columns (first 6) back to the simplex
    x1 = xm[:, :6]
    s = x1.sum(dim=1, keepdim=True).clamp(min=1e-6)
    xm = torch.cat([x1 / s, xm[:, 6:]], dim=1)
    return xm, ym, x2m

import torch
from model_zoo import mixup_batch

def ev(cfg, fn):
    r2s, preds = [], []
    import trainer as T
    orig = T.mixup_batch
    if fn is not None:
        T.mixup_batch = fn
    try:
        for seed in [42, 2024, 7]:
            set_seed(seed)
            m, _, fitted = run_cv(cfg, ds, seeds=[seed], tag="SM", verbose=False,
                                  n_splits=3)
            r2s.append(float(m["R2"].mean()))
            for item in fitted:
                prep, model = item["prep"], item["model"]
                Xs = prep.transform_x(Xte)
                X1, X2 = split_modalities(Xs, prep.modality_ends(me))
                p_s, _ = predict(model, cfg, X1, X2,
                                 np.zeros(len(Ye), dtype=int), 1, "regression")
                preds.append(prep.inverse_y(p_s).ravel())
    finally:
        T.mixup_batch = orig
    ens = np.mean(np.stack(preds), axis=0)
    def topk(k):
        return len(set(np.argsort(-Ye)[:k]) & set(np.argsort(-ens)[:k])) / k
    return {
        "intR2": float(np.mean(r2s)),
        "extR2": float(r2_score(Ye, ens)),
        "logR2": float(r2_score(np.log1p(Ye), np.log1p(ens))),
        "spear": float(spearmanr(Ye, ens)[0]),
        "top20": float(topk(20)),
        "rmse": float(np.sqrt(np.mean((Ye - ens) ** 2))),
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

def mixed_mixup(x, y, alpha=0.4, x2=None):
    """50/50: standard convex mixup or simplex-projected mixup."""
    if np.random.rand() < 0.5:
        return mixup_batch(x, y, alpha, x2)
    return simplex_mixup(x, y, alpha, x2)

print("=== Mixup variants ===")
for name, fn in [("standard Mixup", None), ("SimplexMix", simplex_mixup), ("mixed 50/50", mixed_mixup)]:
    r = ev(mk(), fn)
    print(f"{name:14s} intR2={r['intR2']:.3f} extR2={r['extR2']:.3f} "
          f"logR2={r['logR2']:.3f} spear={r['spear']:.3f} "
          f"top20={r['top20']:.2f} rmse={r['rmse']:.1f}")
