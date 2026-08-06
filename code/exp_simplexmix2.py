"""SimplexMix hyper-parameter compensation sweep (overwrite main block)."""
import numpy as np
import paths
from build_dataset import load_dataset
from trainer import TrainConfig, set_seed, run_cv, split_modalities, predict
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import torch
from model_zoo import mixup_batch

ds = load_dataset(paths.DATASET_NPZ)
ext = load_dataset(paths.EXTERNAL_NPZ)
Ye = ext["Y"].ravel()
Xte = ext["X"]
me = ext["modality_ends"]

def simplex_mixup(x, y, alpha=0.4, x2=None):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.shape[0], device=x.device)
    xm = lam * x + (1.0 - lam) * x[idx]
    ym = lam * y + (1.0 - lam) * y[idx]
    x2m = None if x2 is None else lam * x2 + (1.0 - lam) * x2[idx]
    x1 = xm[:, :6]
    s = x1.sum(dim=1, keepdim=True).clamp(min=1e-6)
    xm = torch.cat([x1 / s, xm[:, 6:]], dim=1)
    return xm, ym, x2m

def mk(dr=0.15, ep=200, lr=3e-3):
    cfg = TrainConfig()
    cfg.d_model = 64; cfg.n_blocks = 2; cfg.n_heads = 4; cfg.dropout = dr
    cfg.use_transformer = False; cfg.use_attention = True; cfg.use_film = False
    cfg.use_modality_gate = False
    cfg.use_contrastive = False; cfg.use_pretrain_recon = False; cfg.use_mfm = False
    cfg.use_sam = False; cfg.use_ema = False
    cfg.use_mixup = True; cfg.mixup_alpha = 0.4; cfg.use_swa = True
    cfg.use_uncertainty_weighting = False
    cfg.use_domain_constraint = True; cfg.constraint_w = 0.1
    cfg.y_transform = "standard"; cfg.max_epochs = ep; cfg.patience = 30
    cfg.batch_size = 32; cfg.lr = lr; cfg.weight_decay = 1e-3
    return cfg

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
    return {"intR2": float(np.mean(r2s)),
            "extR2": float(r2_score(Ye, ens)),
            "logR2": float(r2_score(np.log1p(Ye), np.log1p(ens))),
            "spear": float(spearmanr(Ye, ens)[0]),
            "top20": float(topk(20)),
            "rmse": float(np.sqrt(np.mean((Ye - ens) ** 2)))}

print("=== SimplexMix hyper-parameter compensation ===")
for name, cfg, fn in [("std dr15/e200", mk(), None),
                      ("SM dr10/e250/lr2", mk(0.10, 250, 2e-3), simplex_mixup),
                      ("SM dr15/e250/lr2", mk(0.15, 250, 2e-3), simplex_mixup),
                      ("SM dr10/e200/lr3", mk(0.10, 200, 3e-3), simplex_mixup),
                      ("SM dr10/e300/lr2", mk(0.10, 300, 2e-3), simplex_mixup)]:
    r = ev(cfg, fn)
    print(f"{name:18s} intR2={r['intR2']:.3f} extR2={r['extR2']:.3f} "
          f"logR2={r['logR2']:.3f} spear={r['spear']:.3f} "
          f"top20={r['top20']:.2f} rmse={r['rmse']:.1f}")
