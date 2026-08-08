"""Cross-validated training engine.

Everything that could leak information is fitted INSIDE the training fold:
imputation, scaling, target standardisation and feature selection. The outer
test fold is touched exactly once, for scoring.

Every ``use_*`` field of TrainConfig is an ablation switch, so the ablation
study in tuner.py simply flips one field at a time and re-runs this file.

Public API used by the rest of the pipeline
-------------------------------------------
    compute_metrics(y_true, y_pred, task_type)   -> dict
    make_outer_splits(ds, seed, n_splits)        -> list[(tr_idx, te_idx)]
    Preprocessor                                 -> leakage-safe transformer
    run_cv(cfg, ds, seeds, tag)                  -> (metrics_df, preds_df, fitted)
    evaluate_external(fitted, ds_ext, task_type) -> (metrics_df, preds_df)

Usage
-----
    python trainer.py --tag main
    python trainer.py --config <abs path to best_config.json> --tag tuned
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import write_guard  # noqa: F401  (purge-before-write for json artifacts)
import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.preprocessing import QuantileTransformer, StandardScaler

import paths
from build_dataset import load_dataset, split_modalities
from model_zoo import (ConstrainedMultiTaskLoss, ModelConfig, build_model,
                       mask_blocks, mixup_batch, orthogonality_penalty,
                       range_penalty, reconstruction_loss,
                       supervised_contrastive_loss)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Model + optimisation + trick configuration. All fields are tunable."""

    # architecture -------------------------------------------------------- #
    d_model: int = 96
    n_blocks: int = 2
    n_heads: int = 4
    dropout: float = 0.15
    n_tokens1: int = 6                   # feature blocks carved out of mod. 1
    n_tokens2: int = 4                   # feature blocks carved out of mod. 2
    fusion: str = "concat"               # concat | film | cross | gated
    use_attention: bool = True
    use_film: bool = True
    use_task_gate: bool = True
    use_residual: bool = True
    use_modality2: bool = True
    use_modality_gate: bool = True      # learn per-modality relevance gates
    gate_sparsity_w: float = 0.01       # L1 on gate logits (0 = no sparsity)
    use_transformer: bool = True        # stack TransformerBlock instead of ResBlock
    attn_entropy_w: float = 1e-3         # 0 disables attention sparsity
    proj_dim: int = 32

    # optimisation -------------------------------------------------------- #
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 250
    patience: int = 30
    grad_clip: float = 1.0
    val_frac: float = 0.2                # inner split used for early stopping
    scaler: str = "standard"             # standard | quantile
    y_transform: str = "standard"        # standard | log1p (target transform)

    # tricks (each one is an ablation switch) ----------------------------- #
    use_mixup: bool = True
    mixup_alpha: float = 0.3
    use_swa: bool = True
    swa_start_frac: float = 0.6
    swa_lr_frac: float = 0.25
    use_contrastive: bool = True
    contrastive_epochs: int = 40
    contrastive_temp: float = 0.1
    contrastive_bins: int = 4
    use_pretrain_recon: bool = True    # masked-feature reconstruction pretraining
    recon_epochs: int = 30
    recon_mask_frac: float = 0.3
    use_uncertainty_weighting: bool = True
    use_domain_constraint: bool = True
    constraint_w: float = 0.05
    y_low: float = -4.0                  # in standardised target units
    y_high: float = 4.0

    # extra tricks (each one is an ablation switch too) ---------------------- #
    use_sam: bool = True                 # sharpness-aware minimisation
    sam_rho: float = 0.05
    use_ema: bool = True                 # exponential moving average weights
    ema_decay: float = 0.999
    use_rdrop: bool = False              # consistency regularisation (KL)
    rdrop_w: float = 0.5
    feature_noise: float = 0.0           # gaussian noise std on inputs (0=off)
    label_smoothing: float = 0.0         # classification only (0=off)

    # pre-training & transfer learning --------------------------------------- #
    use_mfm: bool = False                # BERT-style masked feature modelling
    mfm_epochs: int = 20
    mfm_mask_frac: float = 0.25
    pretrained_path: str = ""            # seed weights from a previous run
    # representation regularisation ------------------------------------------ #
    use_ortho_reg: bool = False          # decorrelate task heads
    ortho_w: float = 1e-4
    # ranking loss (pairwise) mixed into regression loss. Targets the
    # material-screening use case: the model should rank candidate formulas
    # correctly even when absolute values extrapolate (OOD regression).
    use_ranking_loss: bool = False
    rank_loss_w: float = 0.5
    rank_margin: float = 0.1
    # prediction-time uncertainty --------------------------------------------- #
    mc_samples: int = 0                  # >0 enables MC-Dropout at predict time

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "TrainConfig":
        valid = {f for f in TrainConfig().to_dict()}
        return TrainConfig(**{k: v for k, v in d.items() if k in valid})


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    task_type: str = "regression") -> Dict[str, float]:
    """Metric dictionary for one task. y_pred = value (reg) or P(class=1) (clf)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 3:
        return {}

    if task_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r_p = stats.pearsonr(y_true, y_pred)[0] if np.std(y_pred) > 0 else 0.0
        r_s = stats.spearmanr(y_true, y_pred)[0] if np.std(y_pred) > 0 else 0.0
        denom = np.var(y_true) + np.var(y_pred) + (y_true.mean() - y_pred.mean()) ** 2
        ccc = 2 * np.cov(y_true, y_pred)[0, 1] / denom if denom > 0 else 0.0
        out = {
            "R2": float(r2_score(y_true, y_pred)),
            "RMSE": rmse,
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "NRMSE": float(rmse / (y_true.std() + 1e-12)),
            "PearsonR": float(r_p),
            "SpearmanRho": float(r_s),
            "CCC": float(ccc),
        }
        # Top-k screening precision is meaningful for the full external cohort
        # (screening use-case); compute when the evaluation set is large enough.
        if len(y_true) >= 50:
            n = len(y_true)
            top_true = set(np.argsort(-y_true)[:20])
            top_pred = set(np.argsort(-y_pred)[:20])
            top_true30 = set(np.argsort(-y_true)[:30])
            top_pred30 = set(np.argsort(-y_pred)[:30])
            out["TopK20"] = float(len(top_true & top_pred) / 20)
            out["TopK30"] = float(len(top_true30 & top_pred30) / 30)
        return out

    y_bin = (y_true > 0.5).astype(int)
    hard = (y_pred > 0.5).astype(int)
    out = {
        "Accuracy": float(accuracy_score(y_bin, hard)),
        "BalancedAcc": float(balanced_accuracy_score(y_bin, hard)),
        "F1": float(f1_score(y_bin, hard, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_bin, hard)) if len(set(y_bin)) > 1 else 0.0,
    }
    if len(set(y_bin)) > 1:
        out["AUROC"] = float(roc_auc_score(y_bin, y_pred))
        out["AUPRC"] = float(average_precision_score(y_bin, y_pred))
    return out


PRIMARY_METRIC = {"regression": "R2", "classification": "AUROC"}


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _strata(Y: np.ndarray, task_type: str, n_bins: int = 5) -> np.ndarray:
    if task_type == "classification":
        return (Y[:, 0] > 0.5).astype(int)
    y = Y[:, 0]
    try:
        return pd.qcut(y, q=min(n_bins, len(np.unique(y))),
                       labels=False, duplicates="drop").astype(int)
    except Exception:                                      # noqa: BLE001
        return np.zeros(len(y), dtype=int)


def make_outer_splits(ds: dict, seed: int,
                      n_splits: int = paths.N_OUTER_FOLDS
                      ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Grouped + stratified outer folds. Groups never cross the split line."""
    Y, groups = ds["Y"], np.asarray(ds["groups"]).astype(str)
    strata = _strata(Y, ds["task_type"])
    n_groups = len(np.unique(groups))
    idx = np.arange(len(Y))

    if n_groups >= n_splits * 2:
        try:
            sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed)
            return list(sgk.split(idx, strata, groups))
        except Exception:                                  # noqa: BLE001
            gk = GroupKFold(n_splits=n_splits)
            return list(gk.split(idx, Y[:, 0], groups))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        return list(skf.split(idx, strata))
    except Exception:                                      # noqa: BLE001
        return list(KFold(n_splits=n_splits, shuffle=True,
                          random_state=seed).split(idx))


def inner_val_split(ds: dict, train_idx: np.ndarray, seed: int,
                    val_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """Carve an early-stopping validation set out of the training fold."""
    n_splits = max(2, int(round(1.0 / max(val_frac, 1e-6))))
    sub = {k: (v[train_idx] if isinstance(v, np.ndarray) and v.ndim >= 1
               and len(v) == len(ds["Y"]) else v) for k, v in ds.items()}
    sub["task_type"] = ds["task_type"]
    splits = make_outer_splits(sub, seed=seed, n_splits=n_splits)
    tr_rel, va_rel = splits[0]
    return train_idx[tr_rel], train_idx[va_rel]


# --------------------------------------------------------------------------- #
# Leakage-safe preprocessing
# --------------------------------------------------------------------------- #
class Preprocessor:
    """Impute + scale X, standardise Y. Fitted on training rows only.

    y_transform: "standard" -> z-score on raw Y.
                 "log1p"    -> z-score on log1p(Y); inverse applies expm1.
                 Useful for positive targets with wide dynamic range where the
                 model should learn relative (multiplicative) structure.
    """

    def __init__(self, scaler: str = "standard", task_type: str = "regression",
                 y_transform: str = "standard"):
        self.scaler_kind = scaler
        self.task_type = task_type
        self.y_transform = y_transform
        self.imputer: Optional[SimpleImputer] = None
        self.x_scaler = None
        self.y_mean: Optional[np.ndarray] = None
        self.y_std: Optional[np.ndarray] = None
        self.keep_cols: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Preprocessor":
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        Xi = self.imputer.fit_transform(X)
        self.keep_cols = np.std(Xi, axis=0) > 1e-12
        if self.keep_cols.sum() == 0:
            self.keep_cols = np.ones(Xi.shape[1], dtype=bool)
        Xi = Xi[:, self.keep_cols]

        if self.scaler_kind == "quantile":
            self.x_scaler = QuantileTransformer(
                n_quantiles=min(1000, max(10, Xi.shape[0])),
                output_distribution="normal", random_state=0)
        else:
            self.x_scaler = StandardScaler()
        self.x_scaler.fit(Xi)

        if self.task_type == "regression":
            Yt = np.log1p(Y) if self.y_transform == "log1p" else Y
            self.y_mean = np.nanmean(Yt, axis=0)
            self.y_std = np.nanstd(Yt, axis=0) + 1e-8
        else:
            self.y_mean = np.zeros(Y.shape[1])
            self.y_std = np.ones(Y.shape[1])
        return self

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        Xi = self.imputer.transform(X)[:, self.keep_cols]
        return self.x_scaler.transform(Xi).astype(np.float32)

    def transform_y(self, Y: np.ndarray) -> np.ndarray:
        Yt = np.log1p(Y) if self.y_transform == "log1p" else Y
        return ((Yt - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, Y: np.ndarray) -> np.ndarray:
        Yt = Y * self.y_std + self.y_mean
        if self.y_transform == "log1p":
            # numerical safety: clip the log-space prediction to a sane band
            # (train mean +- 4 sigma in log space) before expm1.
            lo = float(self.y_mean[0] - 4.0 * self.y_std[0])
            hi = float(self.y_mean[0] + 4.0 * self.y_std[0])
            Yt = np.clip(Yt, lo, hi)
            return np.expm1(Yt).astype(np.float32)
        return Yt.astype(np.float32)

    def modality_ends(self, orig_ends) -> List[int]:
        """Re-map modality boundaries after constant-column removal."""
        ends, cursor = [], 0
        keep = self.keep_cols
        for e in [int(v) for v in orig_ends]:
            cursor = int(keep[:e].sum())
            ends.append(cursor)
        return ends


# --------------------------------------------------------------------------- #
# Training one model
# --------------------------------------------------------------------------- #
def _to_tensor(a: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(a, dtype=dtype, device=paths.DEVICE)


def _make_model_cfg(cfg: TrainConfig, d1: int, d2: int, n_cond: int,
                    n_tasks: int, task_type: str) -> ModelConfig:
    return ModelConfig(
        in_dim=d1,
        in_dim2=d2 if (cfg.use_modality2 and d2 > 0) else 0,
        n_cond=n_cond if cfg.use_film else 0,
        n_tasks=n_tasks,
        task_type=task_type,
        n_classes=2,
        d_model=cfg.d_model,
        n_blocks=cfg.n_blocks,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        n_tokens1=cfg.n_tokens1,
        n_tokens2=cfg.n_tokens2,
        use_attention=cfg.use_attention,
        use_film=cfg.use_film,
        use_task_gate=cfg.use_task_gate,
        use_residual=cfg.use_residual,
        fusion=cfg.fusion,
        use_modality_gate=cfg.use_modality_gate,
        gate_sparsity_w=cfg.gate_sparsity_w,
        use_transformer=cfg.use_transformer,
        use_mfm=cfg.use_mfm,
        mfm_mask_frac=cfg.mfm_mask_frac,
        pretrained_path=cfg.pretrained_path,
        use_ortho_reg=cfg.use_ortho_reg,
        ortho_w=cfg.ortho_w,
        attn_entropy_w=cfg.attn_entropy_w,
        proj_dim=cfg.proj_dim,
    )


def _batches(n: int, bs: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def _forward_loss(model, mtl, cfg: TrainConfig, xb, x2b, cb, yb, task_type):
    outs, entropy, _ = model(xb, x2b, cb)
    losses = []
    for t, out in enumerate(outs):
        if task_type == "regression":
            losses.append(F.smooth_l1_loss(out.squeeze(-1), yb[:, t]))
        else:
            losses.append(F.cross_entropy(
                out, yb[:, t].long(),
                label_smoothing=cfg.label_smoothing
                if cfg.label_smoothing > 0 else 0.0))
    total = mtl(losses) if mtl is not None else sum(losses) / len(losses)
    if cfg.attn_entropy_w > 0:
        total = total + cfg.attn_entropy_w * entropy
    if cfg.use_domain_constraint and task_type == "regression":
        pen = sum(range_penalty(o.squeeze(-1), cfg.y_low, cfg.y_high)
                  for o in outs) / len(outs)
        total = total + cfg.constraint_w * pen
    if cfg.use_ranking_loss and task_type == "regression":
        # pairwise ranking loss on the FIRST task (glass adhesion):
        # encourage correct ordering of formulas within each batch.
        p0 = outs[0].squeeze(-1)
        y0 = yb[:, 0]
        dy = y0.unsqueeze(1) - y0.unsqueeze(0)      # (B, B)
        dp = p0.unsqueeze(1) - p0.unsqueeze(0)
        mask = dy.abs() > 1e-6
        sign = torch.sign(dy)
        rloss = torch.relu(cfg.rank_margin - sign * dp) * mask
        n = mask.sum().clamp(min=1)
        total = total + cfg.rank_loss_w * (rloss.sum() / n)
    if cfg.use_ortho_reg and len(outs) > 1:
        # decorrelate the per-task head inputs (shared pooled -> gated reps)
        reps = [outs[i] for i in range(len(outs))]
        total = total + cfg.ortho_w * orthogonality_penalty(reps)
    return total, [float(l.detach()) for l in losses], outs


def _rdrop_kl(outs_a: List[torch.Tensor], outs_b: List[torch.Tensor],
              task_type: str) -> torch.Tensor:
    """Symmetrised consistency term for R-Drop (same input, two passes)."""
    terms = []
    for a, b in zip(outs_a, outs_b):
        if task_type == "regression":
            terms.append(F.mse_loss(a, b))
        else:
            pa, pb = F.log_softmax(a, dim=-1), F.softmax(b, dim=-1)
            terms.append((F.kl_div(pa, pb, reduction="batchmean")
                          + F.kl_div(pb.log(), F.softmax(a, dim=-1),
                                     reduction="batchmean")) / 2.0)
    return torch.stack(terms).mean()


def _sam_perturb(model, rho: float) -> None:
    """One sharpness-aware ascent step on the weights (SAM first step)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            norm = p.grad.norm().clamp_min(1e-12)
            p.add_(rho * p.grad / norm)


def _ema_update(ema_params, model_params, decay: float) -> None:
    """In-place EMA update: ema = decay * ema + (1 - decay) * current."""
    with torch.no_grad():
        for ema, cur in zip(ema_params, model_params):
            ema.mul_(decay).add_(cur, alpha=1.0 - decay)


def train_single(cfg: TrainConfig, Xtr, X2tr, Ctr, Ytr, Xva, X2va, Cva, Yva,
                 n_cond: int, task_type: str, seed: int, verbose: bool = False):
    """Train one network. Returns (model, history dict)."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    n_tasks = Ytr.shape[1]
    d1, d2 = Xtr.shape[1], X2tr.shape[1]

    mcfg = _make_model_cfg(cfg, d1, d2, n_cond, n_tasks, task_type)
    model = build_model(mcfg).to(paths.DEVICE)
    mtl = (ConstrainedMultiTaskLoss(n_tasks).to(paths.DEVICE)
           if (cfg.use_uncertainty_weighting and n_tasks > 1) else None)

    xt, x2t = _to_tensor(Xtr), _to_tensor(X2tr)
    ct = _to_tensor(Ctr, torch.long)
    yt = _to_tensor(Ytr)
    xv, x2v = _to_tensor(Xva), _to_tensor(X2va)
    cv_ = _to_tensor(Cva, torch.long)
    yv = _to_tensor(Yva)
    use_x2 = d2 > 0 and cfg.use_modality2
    use_c = n_cond > 0 and cfg.use_film

    def fwd_args(x, x2, c):
        return (x, x2 if use_x2 else None, c if use_c else None)

    history = {"train": [], "val": [], "contrastive": [], "mfm": []}

    # ------------------- stage 1: contrastive pre-training ------------------ #
    if cfg.use_contrastive and cfg.contrastive_epochs > 0 and len(Xtr) >= 32:
        if task_type == "classification":
            pseudo = Ytr[:, 0].astype(int)
        else:
            try:
                pseudo = pd.qcut(Ytr[:, 0], q=cfg.contrastive_bins,
                                 labels=False, duplicates="drop").astype(int)
            except Exception:                              # noqa: BLE001
                pseudo = np.zeros(len(Ytr), dtype=int)
        pt = _to_tensor(pseudo, torch.long)
        opt_c = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
        model.train()
        for _ in range(cfg.contrastive_epochs):
            ep = []
            for bi in _batches(len(Xtr), cfg.batch_size, True, rng):
                if len(bi) < 8:
                    continue
                idx = torch.as_tensor(bi, device=paths.DEVICE)
                z = model.project(*fwd_args(xt[idx], x2t[idx], ct[idx]))
                loss = supervised_contrastive_loss(z, pt[idx],
                                                   cfg.contrastive_temp)
                if float(loss) == 0.0:
                    continue
                opt_c.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt_c.step()
                ep.append(float(loss.detach()))
            history["contrastive"].append(float(np.mean(ep)) if ep else 0.0)

    # ---------------- stage 1b: masked-feature modelling (MFM) --------------- #
    # BERT-style self-supervised pre-training on the raw features themselves:
    # mask a random subset of feature columns, encode, reconstruct the hidden
    # ones. This teaches the encoder the *feature structure* without needing
    # any label -- a genuine "pre-trained representation" trick that works on
    # tabular/multi-modal data where off-the-shelf foundation models do not
    # exist. Turn it on via use_mfm=True (ablation: w/o MFM pre-training).
    if (cfg.use_mfm and cfg.mfm_epochs > 0 and len(Xtr) >= 32
            and hasattr(model, "mfm_forward")):
        opt_m = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
        mask_frac = float(cfg.mfm_mask_frac)
        model.train()
        for _ in range(cfg.mfm_epochs):
            ep = []
            for bi in _batches(len(Xtr), cfg.batch_size, True, rng):
                if len(bi) < 8:
                    continue
                idx = torch.as_tensor(bi, device=paths.DEVICE)
                xb, x2b = xt[idx], x2t[idx]
                if mask_frac > 0:
                    mask = torch.rand(xb.shape, device=xb.device) < mask_frac
                    xb_m = xb.masked_fill(mask, 0.0)
                    if use_x2:
                        mask2 = (torch.rand(x2b.shape, device=x2b.device)
                                 < mask_frac)
                        x2b_m = x2b.masked_fill(mask2, 0.0)
                        recon, target = model.mfm_forward(xb_m, x2b_m)
                        loss = reconstruction_loss(
                            recon,
                            torch.cat([xb, x2b], dim=-1),
                            torch.cat([mask, mask2], dim=-1))
                    else:
                        recon, target = model.mfm_forward(xb_m, None)
                        loss = reconstruction_loss(recon, xb, mask)
                else:
                    recon, target = model.mfm_forward(xb, x2b if use_x2 else None)
                    loss = F.mse_loss(recon, target)
                if float(loss.detach()) == 0.0:
                    continue
                opt_m.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt_m.step()
                ep.append(float(loss.detach()))
            history["mfm"].append(float(np.mean(ep)) if ep else 0.0)
        print(f"      MFM pre-training: {cfg.mfm_epochs} epochs done")

    # ------------------------- stage 2: supervised -------------------------- #
    params = list(model.parameters()) + (list(mtl.parameters()) if mtl else [])
    try:
        opt = torch.optim.RAdam(params, lr=cfg.lr,
                                weight_decay=cfg.weight_decay,
                                decoupled_weight_decay=True)
    except TypeError:
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, int(np.ceil(len(Xtr) / cfg.batch_size)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.max_epochs * steps_per_epoch,
        pct_start=0.25, div_factor=10.0, final_div_factor=100.0)

    swa_model = None
    swa_start = int(cfg.max_epochs * cfg.swa_start_frac)
    if cfg.use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)

    best_val, best_state, bad = np.inf, copy.deepcopy(model.state_dict()), 0
    ema_model = None
    if cfg.use_ema:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
    for epoch in range(cfg.max_epochs):
        model.train()
        ep_loss = []
        for bi in _batches(len(Xtr), cfg.batch_size, True, rng):
            if len(bi) < 2:
                continue
            idx = torch.as_tensor(bi, device=paths.DEVICE)
            xb, x2b, cb, yb = xt[idx], x2t[idx], ct[idx], yt[idx]
            if cfg.feature_noise > 0:
                xb = xb + torch.randn_like(xb) * cfg.feature_noise
                if use_x2:
                    x2b = x2b + torch.randn_like(x2b) * cfg.feature_noise
            if cfg.use_mixup and task_type == "regression":
                xb, yb, x2b = mixup_batch(xb, yb, cfg.mixup_alpha,
                                          x2b if use_x2 else None)
                if not use_x2:
                    x2b = x2t[idx]
            total, _, outs = _forward_loss(model, mtl, cfg,
                                           *fwd_args(xb, x2b, cb), yb,
                                           task_type)
            if cfg.use_rdrop:
                # second forward pass on the same batch for consistency
                outs2 = model(*fwd_args(xb, x2b, cb))[0]
                total = total + cfg.rdrop_w * _rdrop_kl(outs, outs2,
                                                        task_type)
            opt.zero_grad()
            total.backward()
            if cfg.use_sam:
                _sam_perturb(model, cfg.sam_rho)
                total2, _, outs2 = _forward_loss(model, mtl, cfg,
                                                 *fwd_args(xb, x2b, cb),
                                                 yb, task_type)
                if cfg.use_rdrop:
                    outs3 = model(*fwd_args(xb, x2b, cb))[0]
                    total2 = total2 + cfg.rdrop_w * _rdrop_kl(outs2, outs3,
                                                              task_type)
                opt.zero_grad()
                total2.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            sched.step()
            if ema_model is not None:
                _ema_update(ema_model.parameters(), model.parameters(),
                            cfg.ema_decay)
            ep_loss.append(float(total.detach()))

        if cfg.use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)

        model.eval()
        with torch.no_grad():
            vl, _, _ = _forward_loss(model, mtl, cfg,
                                     *fwd_args(xv, x2v, cv_), yv, task_type)
            vl = float(vl)
        history["train"].append(float(np.mean(ep_loss)) if ep_loss else np.nan)
        history["val"].append(vl)

        if vl < best_val - 1e-5:
            best_val, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience and epoch >= swa_start * (1 if cfg.use_swa else 0):
                break
        if verbose and epoch % 25 == 0:
            print(f"      ep{epoch:3d} train={history['train'][-1]:.4f} "
                  f"val={vl:.4f}")

    final = model
    # candidate pools: best checkpoint, SWA average, EMA average
    candidates = [("best", best_state)]
    if cfg.use_swa and swa_model is not None and epoch >= swa_start:
        candidates.append(("swa", swa_model.module.state_dict()))
    if ema_model is not None:
        candidates.append(("ema", ema_model.state_dict()))

    best_key, best_v = "best", best_val
    for key, state in candidates:
        if key == "best":
            continue
        trial = copy.deepcopy(model)
        trial.load_state_dict(state)
        trial.eval()
        with torch.no_grad():
            vl_k, _, _ = _forward_loss(trial, mtl, cfg,
                                       *fwd_args(xv, x2v, cv_), yv,
                                       task_type)
        if float(vl_k) < best_v:
            best_v, best_key = float(vl_k), key
    final.load_state_dict(dict(next(s for k, s in candidates
                                    if k == best_key)))
    history["model_selected"] = best_key
    history["swa_used"] = best_key == "swa"

    history["best_val"] = float(best_v)
    history["epochs_run"] = int(epoch + 1)
    history["mtl_weights"] = (mtl.weights().tolist() if mtl else None)
    final.eval()
    return final, history


@torch.no_grad()
def predict(model, cfg: TrainConfig, X, X2, C, n_cond: int, task_type: str):
    """Return (pred_matrix, latent_matrix). pred = value or P(class=1).

    When ``cfg.mc_samples > 0`` the model is kept in train mode and evaluated
    ``mc_samples`` times, averaging the stochastic (MC-Dropout) passes --
    a cheap uncertainty estimate that also usually improves the point
    prediction slightly.
    """
    use_x2 = X2.shape[1] > 0 and cfg.use_modality2
    use_c = n_cond > 0 and cfg.use_film
    xb, x2b = _to_tensor(X), _to_tensor(X2)
    cb = _to_tensor(C, torch.long)
    n_mc = max(1, int(getattr(cfg, "mc_samples", 0)))
    if n_mc > 1:
        model.train()                      # stochastic dropout stays on
    else:
        model.eval()
    acc, latent_last = None, None
    for _ in range(n_mc):
        outs, _, latent = model(xb, x2b if use_x2 else None,
                                cb if use_c else None)
        if task_type == "regression":
            pred = torch.cat([o for o in outs], dim=1)
        else:
            pred = torch.cat([F.softmax(o, dim=1)[:, 1:2] for o in outs],
                             dim=1)
        acc = pred if acc is None else acc + pred
        latent_last = latent
    model.eval()
    return (acc / n_mc).cpu().numpy(), latent_last.cpu().numpy()


# --------------------------------------------------------------------------- #
# Cross-validation driver
# --------------------------------------------------------------------------- #
def run_cv(cfg: TrainConfig, ds: dict, seeds: List[int], tag: str = "main",
           verbose: bool = True, n_splits: int = paths.N_OUTER_FOLDS):
    """Repeated grouped CV. Returns (metrics_df, preds_df, fitted_models)."""
    X, Y = ds["X"], ds["Y"]
    C = np.asarray(ds["cond"]).astype(int)
    n_cond = int(len(ds["cond_levels"]))
    task_type = ds["task_type"]
    tnames = [str(t) for t in ds["target_names"]]
    ends = ds["modality_ends"]

    rows, pred_rows, fitted = [], [], []
    t0 = time.time()
    for seed in seeds:
        splits = make_outer_splits(ds, seed=seed, n_splits=n_splits)
        for fold, (tr_idx, te_idx) in enumerate(splits):
            tr2, va = inner_val_split(ds, tr_idx, seed=seed,
                                      val_frac=cfg.val_frac)
            prep = Preprocessor(cfg.scaler, task_type,
                                cfg.y_transform).fit(X[tr2], Y[tr2])
            new_ends = prep.modality_ends(ends)

            def prep_x(rows_idx):
                Xs = prep.transform_x(X[rows_idx])
                return split_modalities(Xs, new_ends)

            X1tr, X2tr = prep_x(tr2)
            X1va, X2va = prep_x(va)
            X1te, X2te = prep_x(te_idx)
            Ytr = prep.transform_y(Y[tr2])
            Yva = prep.transform_y(Y[va])

            model, hist = train_single(
                cfg, X1tr, X2tr, C[tr2], Ytr, X1va, X2va, C[va], Yva,
                n_cond, task_type, seed=seed + fold, verbose=False)

            pred_s, _ = predict(model, cfg, X1te, X2te, C[te_idx],
                                n_cond, task_type)
            pred = (prep.inverse_y(pred_s) if task_type == "regression"
                    else pred_s)

            for t, tname in enumerate(tnames):
                m = compute_metrics(Y[te_idx, t], pred[:, t], task_type)
                rows.append({"tag": tag, "model": paths.MODEL_NAME,
                             "seed": seed, "fold": fold, "target": tname,
                             "n_test": len(te_idx),
                             "epochs": hist["epochs_run"],
                             "swa_used": hist["swa_used"], **m})
            for j, gi in enumerate(te_idx):
                row = {"tag": tag, "seed": seed, "fold": fold,
                       "sample_id": str(ds["sample_ids"][gi]),
                       "group": str(ds["groups"][gi]),
                       "cond": int(C[gi])}
                for t, tname in enumerate(tnames):
                    row[f"y_true_{tname}"] = float(Y[gi, t])
                    row[f"y_pred_{tname}"] = float(pred[j, t])
                pred_rows.append(row)

            fitted.append({"seed": seed, "fold": fold, "model": model,
                           "prep": prep, "ends": new_ends, "hist": hist})
            if verbose:
                pm = PRIMARY_METRIC[task_type]
                sc = np.mean([r[pm] for r in rows[-len(tnames):] if pm in r])
                print(f"    seed={seed} fold={fold}  {pm}={sc:.4f}  "
                      f"ep={hist['epochs_run']}  swa={hist['swa_used']}")

    metrics_df = pd.DataFrame(rows)
    preds_df = pd.DataFrame(pred_rows)
    if verbose:
        pm = PRIMARY_METRIC[task_type]
        print(f"  [{tag}] mean {pm} = {metrics_df[pm].mean():.4f} "
              f"+/- {metrics_df[pm].std():.4f}   "
              f"({time.time() - t0:.1f}s)")
    return metrics_df, preds_df, fitted


def evaluate_external(fitted: List[dict], cfg: TrainConfig, ds_ext: dict,
                      tag: str = "external"):
    """Ensemble of all CV models applied ONCE to the held-out cohort."""
    X, Y = ds_ext["X"], ds_ext["Y"]
    C = np.asarray(ds_ext["cond"]).astype(int)
    n_cond = int(len(ds_ext["cond_levels"]))
    task_type = ds_ext["task_type"]
    tnames = [str(t) for t in ds_ext["target_names"]]
    ends = ds_ext["modality_ends"]

    per_model, rows = [], []
    for item in fitted:
        prep, model = item["prep"], item["model"]
        Xs = prep.transform_x(X)
        X1, X2 = split_modalities(Xs, prep.modality_ends(ends))
        p_s, _ = predict(model, cfg, X1, X2, C, n_cond, task_type)
        p = prep.inverse_y(p_s) if task_type == "regression" else p_s
        per_model.append(p)
        for t, tname in enumerate(tnames):
            m = compute_metrics(Y[:, t], p[:, t], task_type)
            rows.append({"tag": f"{tag}_single", "model": paths.MODEL_NAME,
                         "seed": item["seed"], "fold": item["fold"],
                         "target": tname, "n_test": len(Y), **m})

    ens = np.mean(np.stack(per_model, axis=0), axis=0)
    for t, tname in enumerate(tnames):
        m = compute_metrics(Y[:, t], ens[:, t], task_type)
        rows.append({"tag": f"{tag}_ensemble", "model": paths.MODEL_NAME,
                     "seed": -1, "fold": -1, "target": tname,
                     "n_test": len(Y), **m})

    pred_rows = []
    for i in range(len(Y)):
        row = {"tag": f"{tag}_ensemble", "seed": -1, "fold": -1,
               "sample_id": str(ds_ext["sample_ids"][i]),
               "group": str(ds_ext["groups"][i]), "cond": int(C[i])}
        for t, tname in enumerate(tnames):
            row[f"y_true_{tname}"] = float(Y[i, t])
            row[f"y_pred_{tname}"] = float(ens[i, t])
        pred_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="absolute path to a JSON config")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seeds", type=int, default=len(paths.SEEDS))
    ap.add_argument("--quick", action="store_true",
                    help="1 seed and few epochs, for smoke tests only")
    args = ap.parse_args()

    paths.ensure_dirs()
    paths.banner("STEP 5/9  CROSS-VALIDATED TRAINING")

    cfg = TrainConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = TrainConfig.from_dict(json.load(fh))
        print(f"  loaded config: {args.config}")
    seeds = paths.SEEDS[:max(1, args.seeds)]
    if args.quick:
        cfg.max_epochs, cfg.patience, cfg.contrastive_epochs = 30, 8, 5
        seeds = paths.SEEDS[:1]
        print("  QUICK MODE: results are not publishable.")

    ds = load_dataset(paths.DATASET_NPZ)
    ds_ext = load_dataset(paths.EXTERNAL_NPZ)
    print(f"  internal n={len(ds['Y'])}  external n={len(ds_ext['Y'])}  "
          f"task={ds['task_type']}  seeds={seeds}")

    metrics, preds, fitted = run_cv(cfg, ds, seeds, tag=args.tag)
    metrics.to_csv(paths.CV_OUTER_CSV, index=False)
    preds.to_csv(os.path.join(paths.PREDS_DIR, f"preds_cv_{args.tag}.csv"),
                 index=False)

    ext_metrics, ext_preds = evaluate_external(fitted, cfg, ds_ext)
    ext_metrics.to_csv(paths.EXTERNAL_CSV, index=False)
    ext_preds.to_csv(os.path.join(paths.PREDS_DIR, "preds_external.csv"),
                     index=False)

    hist_rows = []
    for item in fitted:
        for e, (tr, va) in enumerate(zip(item["hist"]["train"],
                                         item["hist"]["val"])):
            hist_rows.append({"seed": item["seed"], "fold": item["fold"],
                              "epoch": e, "train_loss": tr, "val_loss": va})
    pd.DataFrame(hist_rows).to_csv(
        os.path.join(paths.METRICS_DIR, "training_history.csv"), index=False)

    write_guard.write_json(
        os.path.join(paths.TUNING_DIR, "config_used.json"), cfg.to_dict())

    pm = PRIMARY_METRIC[ds["task_type"]]
    print(f"\n  internal CV {pm}: {metrics[pm].mean():.4f}")
    ens = ext_metrics[ext_metrics['tag'].str.endswith('ensemble')]
    print(f"  external    {pm}: {ens[pm].mean():.4f}")
    print(f"\nWrote: {paths.CV_OUTER_CSV}\n       {paths.EXTERNAL_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
