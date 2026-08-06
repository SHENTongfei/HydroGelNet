"""Model zoo: every architectural and training technique used in the project.

All modules are switchable through :class:`ModelConfig`, so the ablation study
can disable exactly one technique at a time.

Design note on attention
------------------------
The encoder tokenises each modality into contiguous feature blocks, so the
self-attention layer attends over a real sequence of tokens (feature blocks +
modality tokens + an optional condition token + a CLS token). This matters:
attention over a length-1 sequence would make the entropy-sparsity penalty a
no-op and the resulting "attention analysis" figure meaningless. With block
tokens the attention map is interpretable -- each weight points at a named
group of input features.

Implemented techniques
----------------------
* SwiGLU gated activation
* Pre-norm residual blocks (LayerNorm -> SwiGLU -> Dropout -> Proj -> add)
* Token-level multi-head self-attention with entropy sparsity regularisation
* FiLM conditioning driven by a learnable condition embedding
* Task-specific sigmoid gating of the shared representation
* Four multimodal fusion strategies: concat / film / cross-attention / gated
* Uncertainty-weighted multi-task loss with clamped log-variance
* Supervised contrastive pre-training objective
* Mixup augmentation
* Domain-consistency (monotonicity / range) penalty hooks
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import _runtime_guard  # noqa: F401  (must be first)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Full model configuration. Every ``use_*`` flag is an ablation switch."""

    in_dim: int = 64
    in_dim2: int = 0                 # second modality width (0 disables it)
    n_cond: int = 0                  # number of condition categories (0 = off)
    n_tasks: int = 1
    task_type: str = "regression"    # "regression" or "classification"
    n_classes: int = 2

    d_model: int = 96
    n_blocks: int = 2
    n_heads: int = 4
    dropout: float = 0.15
    cond_emb_dim: int = 16

    n_tokens1: int = 6               # feature blocks carved out of modality 1
    n_tokens2: int = 4               # feature blocks carved out of modality 2

    use_attention: bool = True
    use_film: bool = True
    use_task_gate: bool = True
    use_residual: bool = True
    fusion: str = "concat"           # concat | film | cross | gated

    use_modality_gate: bool = False  # learn per-modality relevance gates
    gate_sparsity_w: float = 0.0     # L1 on gate logits (0 disables sparsity)
    use_transformer: bool = False    # stack TransformerBlock instead of ResBlock

    # BERT-style masked-feature-modelling pre-training (stage 0)
    use_mfm: bool = False
    mfm_mask_frac: float = 0.25      # fraction of feature blocks masked
    # optional transfer learning: seed weights from a previous checkpoint
    pretrained_path: str = ""        # empty = train from scratch

    # extra regularisation
    use_ortho_reg: bool = False      # decorrelate task-head representations
    ortho_w: float = 1e-4

    attn_entropy_w: float = 1e-3     # 0 disables attention sparsity
    proj_dim: int = 32

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    """Gated linear unit variant: out((W1 x) * SiLU(W2 x))."""

    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or dim * 2
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.w1(x) * F.silu(self.w2(x)))


class ResBlock(nn.Module):
    """Pre-norm residual block. Works on (B, D) and (B, T, D) alike."""

    def __init__(self, dim: int, dropout: float = 0.15, residual: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = SwiGLU(dim)
        self.drop = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.ff(self.norm(x)))
        return x + h if self.residual else h


class TransformerBlock(nn.Module):
    """True transformer layer: pre-norm MHA -> SwiGLU FFN, both residual.

    Used when ``use_transformer`` is on; otherwise the encoder stacks ResBlocks.
    The MHA here reuses SparseMultiHeadAttention so the entropy-sparsity
    regularisation and the CLS->token attention map survive the upgrade.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.15,
                 residual: bool = True):
        super().__init__()
        self.residual = residual
        self.attn = SparseMultiHeadAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # attn already returns (x + proj(out), entropy, attn_map) with internal
        # residual; we only add the FFN on top.
        h, entropy, attn_map = self.attn(x)
        h = h + self.drop(self.ffn(self.norm2(h))) if self.residual else \
            self.drop(self.ffn(self.norm2(h)))
        return h, entropy, attn_map


class FiLM(nn.Module):
    """Feature-wise linear modulation: y = gamma(c) * x + beta(c).

    Initialised as the identity so that enabling FiLM never destabilises the
    first optimisation steps.
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta = nn.Linear(cond_dim, feat_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        if x.dim() == 3 and gamma.dim() == 2:
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
        return gamma * x + beta


class SparseMultiHeadAttention(nn.Module):
    """Self-attention over tokens, returning the mean attention entropy.

    Low entropy == the model concentrates on few tokens. Adding
    ``attn_entropy_w * entropy`` to the loss therefore *encourages sparse,
    readable* attention maps instead of a uniform blur.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        n_heads = max(1, n_heads)
        while dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        entropy = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(-1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(b, t, d)
        return x + self.proj(out), entropy.mean(), attn.mean(dim=1)


class CrossAttentionFusion(nn.Module):
    """Tokens of modality A query tokens of modality B."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        n_heads = max(1, n_heads)
        while dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(self.norm_a(a), self.norm_b(b), self.norm_b(b),
                           need_weights=False)
        return a + out


class GatedFusion(nn.Module):
    """Learned convex gate between two pooled modality embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([a, b], dim=-1))
        return g * a + (1.0 - g) * b


class ModalityGate(nn.Module):
    """Per-modality relevance gate: decides *whether* a modality matters.

    For each modality a scalar logit is learned from its pooled embedding; a
    sigmoid turns it into a gate in (0, 1) that scales that modality's token
    sequence. An optional L1 penalty on the logits (``gate_sparsity_w``)
    pushes gates toward 0/1, i.e. the model actively *selects* modalities
    instead of trusting every input block. The final gate values are
    registered on ``self.last_gates`` so the interpretation stage can report
    "modality importance" directly.
    """

    def __init__(self, dim: int, n_modalities: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.SiLU(), nn.Linear(dim // 2, 1))
        self.n_modalities = n_modalities
        self.last_gates: Optional[np.ndarray] = None

    def forward(self, pools: List[torch.Tensor]) -> torch.Tensor:
        """pools: list of (B, D) pooled embeddings, one per modality.

        Returns per-modality gate logits of shape (B, n_modalities).
        """
        logits = torch.stack([self.score(p).squeeze(-1) for p in pools],
                             dim=-1)
        gates = torch.sigmoid(logits)
        self.last_gates = gates.detach().cpu().numpy()
        return logits


class BlockTokenizer(nn.Module):
    """Split a flat feature vector into contiguous blocks and embed each one."""

    def __init__(self, in_dim: int, n_tokens: int, d_model: int,
                 dropout: float = 0.1):
        super().__init__()
        n_tokens = max(1, min(n_tokens, in_dim))
        base, rem = divmod(in_dim, n_tokens)
        self.sizes = [base + (1 if i < rem else 0) for i in range(n_tokens)]
        self.bounds: List[Tuple[int, int]] = []
        cursor = 0
        for s in self.sizes:
            self.bounds.append((cursor, cursor + s))
            cursor += s
        self.projs = nn.ModuleList([nn.Linear(s, d_model) for s in self.sizes])
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.n_tokens = n_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        toks = [proj(x[:, lo:hi]) for proj, (lo, hi)
                in zip(self.projs, self.bounds)]
        h = torch.stack(toks, dim=1) + self.pos
        return self.drop(self.norm(h))


class PredictionHead(nn.Module):
    """Tapered MLP head."""

    def __init__(self, dim: int, out_dim: int = 1, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Main network
# --------------------------------------------------------------------------- #
class SciNet(nn.Module):
    """Token-based multimodal multi-task network used throughout the study."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.tok1 = BlockTokenizer(cfg.in_dim, cfg.n_tokens1, d, cfg.dropout)
        self.has_mod2 = cfg.in_dim2 > 0
        self.tok2 = (BlockTokenizer(cfg.in_dim2, cfg.n_tokens2, d, cfg.dropout)
                     if self.has_mod2 else None)

        if self.has_mod2:
            if cfg.fusion == "cross":
                self.cross = CrossAttentionFusion(d, cfg.n_heads, cfg.dropout)
            elif cfg.fusion == "gated":
                self.gated = GatedFusion(d)
            elif cfg.fusion == "film":
                self.fuse_film = FiLM(d, d)
        if self.has_mod2 and cfg.use_modality_gate:
            self.mod_gate = ModalityGate(d, 2)

        self.has_cond = cfg.n_cond > 0 and cfg.use_film
        if self.has_cond:
            self.cond_emb = nn.Embedding(cfg.n_cond, cfg.cond_emb_dim)
            self.film = FiLM(cfg.cond_emb_dim, d)
            self.cond_to_token = nn.Linear(cfg.cond_emb_dim, d)

        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)

        if cfg.use_transformer:
            # stack true transformer layers (attention entropy is aggregated
            # over layers inside encode below)
            self.blocks = nn.ModuleList([
                TransformerBlock(d, cfg.n_heads, cfg.dropout,
                                 residual=cfg.use_residual)
                for _ in range(cfg.n_blocks)
            ])
            self.attn = None      # MHA lives inside each TransformerBlock
        else:
            self.blocks = nn.ModuleList([
                ResBlock(d, cfg.dropout, residual=cfg.use_residual)
                for _ in range(cfg.n_blocks)
            ])
            self.attn = (SparseMultiHeadAttention(d, cfg.n_heads, cfg.dropout)
                         if cfg.use_attention else None)
        self.final_norm = nn.LayerNorm(d)

        out_dim = 1 if cfg.task_type == "regression" else cfg.n_classes
        self.gates = (nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
            for _ in range(cfg.n_tasks)]) if cfg.use_task_gate else None)
        self.heads = nn.ModuleList([
            PredictionHead(d, out_dim, cfg.dropout) for _ in range(cfg.n_tasks)
        ])
        self.projector = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, cfg.proj_dim))

        # MFM reconstruction heads: one linear map per feature block, turning
        # the d_model token embedding back into the original block width
        self.mfm_heads1 = nn.ModuleList([
            nn.Linear(d, s) for s in self.tok1.sizes])
        self.mfm_heads2 = (nn.ModuleList([
            nn.Linear(d, s) for s in self.tok2.sizes])
            if self.has_mod2 else nn.ModuleList())

    # ------------------------------------------------------------------ #
    def token_names(self) -> List[str]:
        """Human-readable label of every token, in sequence order.

        The order matches the second axis of :meth:`attention_map`.
        """
        names = ["CLS"]
        if self.has_mod2 and self.cfg.fusion == "gated":
            names += ["fused_mod1_mod2"]
        else:
            names += [f"mod1_block{i + 1}" for i in range(self.tok1.n_tokens)]
            if self.has_mod2:
                names += [f"mod2_block{i + 1}"
                          for i in range(self.tok2.n_tokens)]
        if self.has_cond:
            names += ["condition"]
        return names

    def token_bounds(self) -> Dict[str, Tuple[int, int]]:
        """Map each token name to its (start, end) column range in X."""
        out: Dict[str, Tuple[int, int]] = {}
        for i, (lo, hi) in enumerate(self.tok1.bounds):
            out[f"mod1_block{i + 1}"] = (lo, hi)
        if self.has_mod2:
            off = self.cfg.in_dim
            for i, (lo, hi) in enumerate(self.tok2.bounds):
                out[f"mod2_block{i + 1}"] = (off + lo, off + hi)
        return out

    def encode(self, x: torch.Tensor, x2: Optional[torch.Tensor] = None,
               cond: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return (pooled latent, attention entropy, attention map)."""
        h = self.tok1(x)                                    # (B, T1, D)
        gate_penalty = torch.zeros((), device=h.device)

        if self.has_mod2 and x2 is not None and x2.shape[1] > 0:
            h2 = self.tok2(x2)                              # (B, T2, D)
            if self.cfg.use_modality_gate:
                # learn *whether* each modality matters; scale its token
                # sequence by the gate before fusion
                g = torch.sigmoid(self.mod_gate([h.mean(1), h2.mean(1)]))
                h = h * g[:, 0:1].unsqueeze(1)
                h2 = h2 * g[:, 1:2].unsqueeze(1)
                if self.cfg.gate_sparsity_w > 0:
                    # push gates toward 0/1 (hard selection)
                    gate_penalty = (self.cfg.gate_sparsity_w
                                    * g.mean())
            if self.cfg.fusion == "concat":
                h = torch.cat([h, h2], dim=1)
            elif self.cfg.fusion == "cross":
                h = torch.cat([self.cross(h, h2), h2], dim=1)
            elif self.cfg.fusion == "gated":
                h = self.gated(h.mean(1), h2.mean(1)).unsqueeze(1)
            else:                                           # "film"
                h = torch.cat([self.fuse_film(h, h2.mean(1)), h2], dim=1)

        if self.has_cond and cond is not None:
            c = self.cond_emb(cond)
            h = self.film(h, c)
            h = torch.cat([h, self.cond_to_token(c).unsqueeze(1)], dim=1)

        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)

        entropy = gate_penalty
        attn_map = None
        if self.cfg.use_transformer:
            # every TransformerBlock returns its own attention entropy/map;
            # average the maps across layers, accumulate the entropy
            maps = []
            for block in self.blocks:
                h, e, am = block(h)
                entropy = entropy + e
                if am is not None:
                    maps.append(am)
            if maps:
                attn_map = torch.stack(maps).mean(dim=0)[:, 0, :]
            pooled = h.mean(dim=1) if self.attn is None else None
        else:
            for block in self.blocks:
                h = block(h)
            if self.attn is not None:
                h, e, attn_map = self.attn(h)
                entropy = entropy + e
                attn_map = attn_map[:, 0, :]                # CLS -> tokens
        h = self.final_norm(h)
        pooled = (h[:, 0] if (self.attn is not None
                              or self.cfg.use_transformer)
                  else h.mean(dim=1))
        return pooled, entropy, attn_map

    def forward(self, x: torch.Tensor, x2: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None
                ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        pooled, entropy, _ = self.encode(x, x2, cond)
        outs = []
        for i, head in enumerate(self.heads):
            hi = pooled * self.gates[i](pooled) if self.gates is not None else pooled
            outs.append(head(hi))
        return outs, entropy, pooled

    def mfm_forward(self, x: torch.Tensor, x2: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masked-feature-modelling pass: encode with masked blocks, then
        linearly reconstruct the ORIGINAL feature vector.

        Returns (reconstruction, original x_cat) where x_cat is the
        concatenated (x, x2) used as the reconstruction target.
        """
        h = self.tok1(x)
        if self.has_mod2 and x2 is not None and x2.shape[1] > 0:
            h2 = self.tok2(x2)
            if self.cfg.fusion == "concat":
                h = torch.cat([h, h2], dim=1)
            elif self.cfg.fusion == "cross":
                h = torch.cat([self.cross(h, h2), h2], dim=1)
            elif self.cfg.fusion == "gated":
                h = self.gated(h.mean(1), h2.mean(1)).unsqueeze(1)
            else:
                h = torch.cat([self.fuse_film(h, h2.mean(1)), h2], dim=1)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
        if self.cfg.use_transformer:
            for block in self.blocks:
                h, _, _ = block(h)
        else:
            for block in self.blocks:
                h = block(h)
        h = self.final_norm(h)
        # reconstruct every token back to its input dimension via the shared
        # tokenizer projections (they were trained to embed, so we invert).
        # NOTE: with `gated` fusion the token sequence is collapsed to a
        # single fused token, so index i+1 would be out of bounds; replicate
        # the fused representation so every reconstruction head still works.
        n_need = 1 + self.tok1.n_tokens + (
            self.tok2.n_tokens if self.has_mod2 else 0)
        if h.shape[1] < n_need:
            fused = h[:, 1:2].expand(-1, n_need - 1, -1)
            h = torch.cat([h[:, :1], fused], dim=1)
        recon_parts = []
        for i, (lo, hi) in enumerate(self.tok1.bounds):
            # token i+1 is CLS-shifted by 1
            t_emb = h[:, i + 1]
            proj = self.tok1.projs[i]
            # pseudo-inverse is too heavy; use a linear map from d_model to the
            # block width, learned as a head
            recon_parts.append(self.mfm_heads1[i](t_emb))
        x_cat = x
        if self.has_mod2 and x2 is not None and x2.shape[1] > 0:
            for i, (lo, hi) in enumerate(self.tok2.bounds):
                t_emb = h[:, 1 + self.tok1.n_tokens + i]
                recon_parts.append(self.mfm_heads2[i](t_emb))
            x_cat = torch.cat([x, x2], dim=-1)
        recon = torch.cat(recon_parts, dim=-1)
        return recon, x_cat

    @torch.no_grad()
    def attention_map(self, x: torch.Tensor, x2: Optional[torch.Tensor] = None,
                      cond: Optional[torch.Tensor] = None
                      ) -> Optional[torch.Tensor]:
        """(B, T) attention weight from the CLS token to every input token."""
        return self.encode(x, x2, cond)[2]

    def project(self, x: torch.Tensor, x2: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled, _, _ = self.encode(x, x2, cond)
        return F.normalize(self.projector(pooled), dim=-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class ConstrainedMultiTaskLoss(nn.Module):
    """Homoscedastic-uncertainty weighting with a clamped log-variance.

    L = sum_t [ exp(-s_t) * L_t + 0.5 * s_t ],  s_t = log(sigma_t^2)
    The clamp keeps the automatic weights inside a numerically safe band and
    prevents one task from silently switching itself off.
    """

    def __init__(self, n_tasks: int, clamp_min: float = -2.0,
                 clamp_max: float = 2.0):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_tasks))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        s = torch.clamp(self.log_var, self.clamp_min, self.clamp_max)
        total = 0.0
        for i, loss_i in enumerate(losses):
            total = total + torch.exp(-s[i]) * loss_i + 0.5 * s[i]
        return total

    def weights(self) -> np.ndarray:
        s = torch.clamp(self.log_var, self.clamp_min, self.clamp_max)
        return torch.exp(-s).detach().cpu().numpy()


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.1) -> torch.Tensor:
    """SupCon loss. Samples sharing a label are positives for each other."""
    device = z.device
    n = z.shape[0]
    if n < 4:
        return torch.zeros((), device=device)
    sim = torch.matmul(z, z.t()) / temperature
    eye = torch.eye(n, dtype=torch.bool, device=device)
    sim = sim.masked_fill(eye, -1e9)

    labels = labels.reshape(-1, 1)
    positive = (labels == labels.t()) & (~eye)
    if positive.sum() == 0:
        return torch.zeros((), device=device)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = positive.sum(dim=1).clamp(min=1)
    mean_log_prob = (log_prob * positive).sum(dim=1) / pos_count
    valid = positive.sum(dim=1) > 0
    return -mean_log_prob[valid].mean()


def monotonicity_penalty(pred_curve: torch.Tensor) -> torch.Tensor:
    """Penalise negative first differences along the last axis.

    Use when the domain dictates that a predicted profile must increase
    (stress-strain loading branch, cumulative incidence, dose-response, ...).
    """
    diffs = pred_curve[..., 1:] - pred_curve[..., :-1]
    return torch.relu(-diffs).mean()


def range_penalty(pred: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Penalise predictions leaving a physically admissible interval."""
    return (torch.relu(low - pred) + torch.relu(pred - high)).mean()


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.3,
                x2: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convex combination of a batch with a shuffled copy of itself."""
    if alpha <= 0:
        return x, y, x2
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.shape[0], device=x.device)
    xm = lam * x + (1.0 - lam) * x[idx]
    ym = lam * y + (1.0 - lam) * y[idx]
    x2m = None if x2 is None else lam * x2 + (1.0 - lam) * x2[idx]
    return xm, ym, x2m


def mask_features(x: torch.Tensor, x2: Optional[torch.Tensor],
                  mask_frac: float, rng: np.random.Generator
                  ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                             torch.Tensor, Optional[torch.Tensor]]:
    """Masked-feature pre-training (MAE-style, VIME/TabMAE lineage).

    Randomly zero a fraction of the *feature columns* of the batch (a mask of
    zeros, not noise). The encoder must then reconstruct the missing values
    from the remaining columns -- a purely self-supervised pretext task that
    transfers to ANY downstream task and needs NO external pretrained weights,
    which makes it the universal pre-training trick for tabular/multimodal
    data. Returns (x_masked, x2_masked, mask1, mask2) where mask == 1 marks
    columns that were hidden and must be reconstructed.
    """
    n = x.shape[1]
    mask1 = torch.zeros((x.shape[0], n), dtype=torch.bool, device=x.device)
    for i in range(x.shape[0]):
        k = max(1, int(round(mask_frac * n)))
        cols = rng.choice(n, size=k, replace=False)
        mask1[i, cols] = True
    xm = x.clone()
    xm[mask1] = 0.0

    mask2 = None
    x2m = x2
    if x2 is not None and x2.shape[1] > 0:
        n2 = x2.shape[1]
        mask2 = torch.zeros((x2.shape[0], n2), dtype=torch.bool,
                            device=x2.device)
        for i in range(x2.shape[0]):
            k2 = max(1, int(round(mask_frac * n2)))
            cols2 = rng.choice(n2, size=k2, replace=False)
            mask2[i, cols2] = True
        x2m = x2.clone()
        x2m[mask2] = 0.0
    return xm, x2m, mask1, mask2


class ReconstructionHead(nn.Module):
    """Temporary head for masked-feature pre-training: pooled -> raw features.

    It is built and trained during the pre-training stage only and discarded
    afterwards; the encoder weights it has warmed up are what carry over.
    """

    def __init__(self, d_model: int, in_dim: int, in_dim2: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, in_dim + in_dim2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def reconstruction_loss(pred: torch.Tensor, x: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """MSE restricted to the masked (hidden) columns."""
    if mask is None or not mask.any():
        return torch.zeros((), device=pred.device)
    err = (pred - x) ** 2
    return err[mask].mean()


# --------------------------------------------------------------------------- #
# Masked feature modelling (BERT-style pre-training on tabular blocks)
# --------------------------------------------------------------------------- #
def mask_blocks(tokens: torch.Tensor, bounds: List[Tuple[int, int]],
                frac: float = 0.25, rng=None) -> torch.Tensor:
    """Randomly zero-out whole feature-block columns of a token tensor.

    ``tokens`` has shape (B, T, D) where token index 0 is CLS (never masked).
    Returns the same tensor with ``frac`` of the non-CLS blocks zeroed.
    """
    t = tokens.shape[1]
    # non-CLS positions 1..t-1 are all real blocks once CLS is prepended
    mask_idx = list(range(1, t))
    if not mask_idx:
        return tokens
    k = max(1, int(round(len(mask_idx) * frac)))
    chooser = rng if rng is not None else np.random.default_rng()
    chosen = chooser.choice(mask_idx, size=min(k, len(mask_idx)),
                            replace=False)
    masked = tokens.clone()
    masked[:, chosen, :] = 0.0
    return masked


def orthogonality_penalty(representations: List[torch.Tensor]) -> torch.Tensor:
    """Penalise pairwise cosine similarity between task representations."""
    if len(representations) < 2:
        return torch.zeros((), device=representations[0].device)
    sims = []
    for i in range(len(representations)):
        for j in range(i + 1, len(representations)):
            a = F.normalize(representations[i], dim=-1)
            b = F.normalize(representations[j], dim=-1)
            sims.append((a * b).sum(dim=-1).abs().mean())
    return torch.stack(sims).mean() if sims else \
        torch.zeros((), device=representations[0].device)


def load_pretrained(model: nn.Module, path: str,
                    strict: bool = False) -> int:
    """Load weights from a previous checkpoint (transfer learning).

    ``strict=False`` keeps the encoder weights when the heads do not match
    (different number of targets). Returns how many parameter tensors were
    copied.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cur = model.state_dict()
    copied = 0
    for k, v in state.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            copied += 1
    model.load_state_dict(cur)
    return copied


def build_model(cfg: ModelConfig) -> SciNet:
    """Factory kept separate so the tuner can build models from dicts."""
    net = SciNet(cfg)
    if cfg.pretrained_path and os.path.exists(cfg.pretrained_path):
        n = load_pretrained(net, cfg.pretrained_path)
        print(f"  [model_zoo] loaded {n} tensors from pretrained "
              f"checkpoint: {cfg.pretrained_path}")
    return net


if __name__ == "__main__":
    torch.manual_seed(0)
    for fusion in ("concat", "film", "cross", "gated"):
        cfg = ModelConfig(in_dim=24, in_dim2=16, n_cond=3, n_tasks=2,
                          task_type="regression", d_model=64, n_blocks=2,
                          fusion=fusion)
        net = build_model(cfg)
        xb, x2b = torch.randn(16, 24), torch.randn(16, 16)
        cb = torch.randint(0, 3, (16,))
        outs, ent, latent = net(xb, x2b, cb)
        amap = net.attention_map(xb, x2b, cb)
        print(f"[model_zoo] fusion={fusion:<7s} params={net.n_parameters():>7d} "
              f"out={[tuple(o.shape) for o in outs]} "
              f"entropy={float(ent.detach()):.4f} "
              f"latent={tuple(latent.shape)} "
              f"attn={tuple(amap.shape)} tokens={len(net.token_names())}")

    # new: modality gate + transformer stack
    cfg = ModelConfig(in_dim=24, in_dim2=16, n_cond=3, n_tasks=2,
                      task_type="regression", d_model=64, n_blocks=2,
                      use_modality_gate=True, gate_sparsity_w=0.01,
                      use_transformer=True, n_heads=4)
    net = build_model(cfg)
    xb, x2b = torch.randn(16, 24), torch.randn(16, 16)
    cb = torch.randint(0, 3, (16,))
    outs, ent, latent = net(xb, x2b, cb)
    amap = net.attention_map(xb, x2b, cb)
    print(f"[model_zoo] gate+transformer params={net.n_parameters():>7d} "
          f"out={[tuple(o.shape) for o in outs]} "
          f"entropy={float(ent.detach()):.4f} "
          f"attn={tuple(amap.shape)}")
    if net.mod_gate.last_gates is not None:
        print(f"[model_zoo] modality gate means = "
              f"{net.mod_gate.last_gates.mean(0).round(3)}")

    mtl = ConstrainedMultiTaskLoss(2)
    loss = mtl([torch.tensor(0.5, requires_grad=True),
                torch.tensor(1.0, requires_grad=True)])
    print(f"[model_zoo] mtl loss   : {float(loss.detach()):.4f} "
          f"weights={mtl.weights()}")
    z = F.normalize(torch.randn(16, 32), dim=-1)
    print(f"[model_zoo] supcon     : "
          f"{float(supervised_contrastive_loss(z, torch.randint(0, 4, (16,)))):.4f}")

    # masked-feature pre-training pieces
    xb = torch.randn(16, 24)
    x2b = torch.randn(16, 16)
    rng = np.random.default_rng(0)
    xm, x2m, m1, m2 = mask_features(xb, x2b, 0.3, rng)
    print(f"[model_zoo] mask       : x_masked={tuple(xm.shape)} "
          f"mask1_frac={float(m1.float().mean()):.3f} "
          f"mask2_frac={float(m2.float().mean()):.3f}")
    head = ReconstructionHead(64, 24, 16)
    pred = head(torch.randn(16, 64))
    rl = reconstruction_loss(pred, torch.cat([xm, x2m], dim=1),
                             torch.cat([m1, m2], dim=1))
    print(f"[model_zoo] recon head : pred={tuple(pred.shape)} "
          f"recon_loss={float(rl.detach()):.4f}")
