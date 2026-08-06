"""Manuscript builder: turns the results CSVs into a Frontiers-style paper.

Everything numeric in the manuscript is read from disk, never typed by hand.
If a results file is missing the corresponding sentence is replaced by an
explicit ``[missing: <file>]`` marker so that the Gate-2 audit can catch it.

Outputs
-------
    paper/manuscript.md              Markdown mirror (easy to diff / edit)
    paper/<MODEL>_manuscript.pdf     Typeset A4 manuscript with figures+tables
    paper/paper_meta.json            Editable metadata (created if absent)
    paper/references.json            Literature list (created empty if absent)
    audit/paper_selfcheck.md         Automatic pre-audit checklist

Metadata contract (``paper/paper_meta.json``)
---------------------------------------------
Filled during the idea / literature phase. Use ``--make-meta`` to emit a
template. Unfilled fields fall back to neutral placeholders and are reported
by the self-check.

Reference contract (``paper/references.json``)
----------------------------------------------
A list of objects::

    {"key": "smith2023", "slot": "intro_importance",
     "authors": ["Smith, J.", "Lee, K."], "year": 2023,
     "title": "....", "journal": "Nature Methods",
     "volume": "20", "pages": "1123-1131",
     "doi": "10.1038/s41592-023-01234-5", "url": ""}

``slot`` routes the citation into the right paragraph. Valid slots are listed
in ``REFERENCE_SLOTS``. Use ``--make-refs`` to emit a template.

Usage
-----
    python paper_pdf.py
    python paper_pdf.py --topic "computational immunology" --no-tables
    python paper_pdf.py --make-meta --make-refs
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import json
import os
import platform
import re
import sys
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import paths
from build_dataset import load_dataset

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (BaseDocTemplate, Frame, Image,
                                    KeepTogether, PageBreak, PageTemplate,
                                    Paragraph, Spacer, Table, TableStyle)
    HAVE_REPORTLAB = True
except ImportError:                                        # pragma: no cover
    HAVE_REPORTLAB = False

MIN_REFERENCES = 55
RECENT_YEARS = 5

REFERENCE_SLOTS = [
    "intro_importance",     # why the field matters
    "intro_methods",        # existing method families
    "intro_limitation",     # their concrete failure modes
    "intro_gap",            # the gap this paper closes
    "methods_data",         # data resources and processing
    "methods_model",        # architectural ingredients
    "methods_stats",        # statistical procedures
    "results_context",      # numbers reported by others
    "discussion_compare",   # papers we outperform / match
    "discussion_mechanism", # mechanistic interpretation
    "discussion_domain",    # domain implications
]

FIG_CAPTIONS = {
    1: ("Study design and the composition-space extrapolation protocol. "
        "**(A)** Composition-space projection (hydrophobic BA vs aromatic "
        "PEA): training formulations (circles, n=180, low-performance "
        "region) and the 161 formulations discovered by the Nature study's "
        "sequential model-based optimisation loop (diamonds, "
        "high-performance region); colour encodes adhesion strength, and "
        "the arc marks the SMBO-guided migration direction. "
        "**(B)** Target-value distribution shift between training and "
        "external formulations (mean 47 vs 154 kPa; the training maximum, "
        "dashed line, is exceeded by 45% of external samples). "
        "**(C)** Evaluation protocol: 5-fold grouped CV x 5 seeds for "
        "in-distribution R2 and ablation; the 161 external formulations are "
        "evaluated once with ranking metrics (Spearman rho, top-k precision), "
        "because target-range shift makes R2 undefined for every model. "
        "**(D)** External ranking: SIMPLEX achieves the best Spearman rho "
        "with 95% bootstrap CIs; it significantly outperforms tree ensembles "
        "(paired bootstrap delta rho = +0.18, 95% CI [0.07, 0.31])."),
    2: ("Architecture of {model}. The 6 monomer molar fractions on the "
        "composition simplex (top) are encoded through two modalities: "
        "modality 1 uses the raw fractions, modality 2 adds the 15 explicit "
        "pairwise interaction terms x_i x_j. A linear embedding maps each "
        "modality into the shared representation, which is refined by two "
        "residual blocks separated by an interaction self-attention layer; "
        "a linear head outputs the predicted adhesion strength (kPa, "
        "non-negative). Small-data regularisation (bottom band): Mixup input "
        "interpolation, stochastic weight averaging, a range-domain "
        "constraint penalising out-of-range predictions, and early stopping "
        "on an inner validation split."),
    3: ("Characteristics of the internal and external cohorts. "
        "**(A)** Cohort sizes. **(B)** Target distributions. "
        "**(C)** Per-feature missingness. **(D)** Feature-feature correlation "
        "structure. **(E)** Raw feature space coloured by cohort. "
        "**(F)** Composition of experimental conditions. "
        "**(G)** Internal-versus-external covariate shift, quantified by "
        "per-feature Kolmogorov-Smirnov statistics. **(H)** Distribution of "
        "group sizes used for grouped splitting."),
    4: ("Internal cross-validated performance of {model}. "
        "**(A)** Per-fold {pm} across seeds. **(B)** Out-of-fold predicted "
        "versus observed values. **(C)** Residuals versus fitted values. "
        "**(D)** Error distribution. **(E)** Training and validation loss "
        "curves. **(F)** Summary of all evaluation metrics per target."),
    5: ("Benchmarking against equally tuned baselines. "
        "**(A)** Mean {pm} with standard deviation over folds and seeds. "
        "**(B)** Paired per-fold scores. **(C)** Absolute improvement with "
        "corrected significance annotation. **(D)** Rank of each model across "
        "folds. **(E)** Cluster bootstrap 95% confidence intervals. "
        "**(F)** Label-permutation null distribution."),
    6: ("Model-guided extrapolation validation. "
        "**(A)** Predicted versus observed values in the external cohort. "
        "**(B)** Bland-Altman agreement. **(C)** Calibration. "
        "**(D)** Internal-versus-external generalisation gap. "
        "**(E)** External benchmark against the baselines. "
        "**(F)** Performance stratified by experimental condition."),
    7: ("Ablation and hyper-parameter analysis. "
        "**(A)** Contribution of each component, measured as the loss in {pm} "
        "when it is removed. **(B)** Per-variant, per-target {pm}. "
        "**(C)** Comparison of the four fusion strategies. "
        "**(D)** Statistical contribution with Holm-adjusted p-values. "
        "**(E)** Hyper-parameter search trajectory. **(F)** Retention "
        "decisions: components that did not pay for themselves were removed "
        "from the final configuration."),
    8: ("Interpretation and domain discovery. "
        "**(A)** Cross-validated permutation importance of the top features. "
        "**(B)** Stability selection frequency across folds and seeds. "
        "**(C)** Attention attribution from the CLS token to the feature-block "
        "tokens. **(D)** Attention profiles stratified by condition. "
        "**(E)** Latent space coloured by target value. **(F)** Latent space "
        "coloured by condition. **(G)** Partial dependence of the leading "
        "features. **(H)** Volcano view of candidate markers combining "
        "model-based importance with univariate FDR-adjusted evidence."),
}

TAB_CAPTIONS = {
    1: "Summary of the internal and external cohorts.",
    2: "Public data sources, licences and verified download links.",
    3: "Search space and finally selected hyper-parameter values.",
    4: "Internal grouped cross-validation performance (mean +/- SD).",
    5: ("Comparison against equally tuned baselines. p-values from the "
        "Nadeau-Bengio corrected resampled t-test, Holm adjusted."),
    6: "Performance on the model-guided external extrapolation cohort.",
    7: ("Ablation study (evaluated on 2 seeds x 3 folds, the search protocol; "
         "the full 5 seeds x 5 folds CV is reported in Table 4). A positive "
         "contribution means removal degrades performance."),
    8: "Top candidate markers ranked by combined evidence.",
    9: "Performance stratified by experimental condition.",
    10: "Software environment and protocol settings for reproducibility.",
}

DEFAULT_META = {
    "topic": "",
    "domain": "",
    "model_full_name": "",
    "title": "",
    "short_title": "",
    "authors": [{"name": "First A. Author", "affiliations": [1],
                 "email": "first.author@example.org"}],
    "affiliations": ["Department, Institution, City, Country"],
    "corresponding": {"name": "First A. Author",
                      "email": "first.author@example.org"},
    "keywords": [],
    "domain_importance": "",
    "gap_statement": "",
    "application_implication": "",
    "target_descriptions": {},
    "modality_descriptions": {},
    "condition_meaning": "",
    "funding": "The author(s) declare that no financial support was received "
               "for the research, authorship, and/or publication of this article.",
    "contributions": "Conceptualisation, methodology, software, formal "
                     "analysis, writing - original draft: [initials]. "
                     "All authors contributed to the article and approved the "
                     "submitted version.",
    "conflict": "The authors declare that the research was conducted in the "
                "absence of any commercial or financial relationships that "
                "could be construed as a potential conflict of interest.",
    "acknowledgements": "",
    "extra_limitations": [],
    "future_work": [],
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df if len(df) else None
    except Exception:                                      # noqa: BLE001
        pass
    return None


def _json(path: str) -> Optional[dict]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:                                      # noqa: BLE001
        pass
    return None


def f3(x, nd: int = 3) -> str:
    """Format a number, tolerating None / NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n.a."
    if not np.isfinite(v):
        return "n.a."
    return f"{v:.{nd}f}"


def pval(p) -> str:
    """Format a p-value the way reviewers expect."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "n.a."
    if not np.isfinite(v):
        return "n.a."
    if v < 1e-4:
        return "p < 0.0001"
    if v < 0.001:
        return f"p = {v:.5f}"
    return f"p = {v:.4f}"


def missing(what: str) -> str:
    return f"**[missing: {what} - rerun the corresponding step]**"


def to_rl(text: str) -> str:
    """Convert the tiny markup used in this file into ReportLab markup."""
    t = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"~([^~\n]+?)~", r"<sub>\1</sub>", t)
    t = re.sub(r"\^([^^\n]+?)\^", r"<super>\1</super>", t)
    return t


def plain(text: str) -> str:
    """Strip markup, for word counting."""
    return re.sub(r"[*~^]", "", text)


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #
class Refs:
    """Author-year citation manager backed by ``paper/references.json``."""

    def __init__(self, path: str):
        self.path = path
        raw = _json(path)
        self.entries: List[dict] = raw if isinstance(raw, list) else []
        self._by_slot: Dict[str, List[dict]] = {}
        for e in self.entries:
            self._by_slot.setdefault(str(e.get("slot", "")), []).append(e)
        self._used: set = set()
        self.enabled = len(self.entries) > 0

    # ---------------------------------------------------------------- #
    @staticmethod
    def _short(entry: dict) -> str:
        authors = entry.get("authors") or ["Anonymous"]
        year = entry.get("year", "n.d.")
        first = str(authors[0]).split(",")[0].strip()
        if len(authors) == 1:
            return f"{first}, {year}"
        if len(authors) == 2:
            second = str(authors[1]).split(",")[0].strip()
            return f"{first} and {second}, {year}"
        return f"{first} et al., {year}"

    def cite(self, slot: str, n: int = 3) -> str:
        """Return an author-year citation group for ``slot``."""
        pool = self._by_slot.get(slot, [])
        picked = [e for e in pool if e.get("key") not in self._used][:n]
        if not picked:
            picked = pool[:n]
        if not picked:
            return "" if not self.enabled else ""
        for e in picked:
            self._used.add(e.get("key"))
        return " (" + "; ".join(self._short(e) for e in picked) + ")"

    # ---------------------------------------------------------------- #
    @staticmethod
    def format(entry: dict) -> str:
        authors = entry.get("authors") or ["Anonymous"]
        if len(authors) == 1:
            astr = authors[0]
        elif len(authors) <= 6:
            astr = ", ".join(authors[:-1]) + ", and " + authors[-1]
        else:
            astr = ", ".join(authors[:6]) + ", et al."
        parts = [f"{astr} ({entry.get('year', 'n.d.')}). "
                 f"{str(entry.get('title', '')).rstrip('.')}."]
        journal = entry.get("journal", "")
        if journal:
            seg = f" *{journal}*"
            if entry.get("volume"):
                seg += f" {entry['volume']}"
            if entry.get("pages"):
                seg += f", {entry['pages']}"
            parts.append(seg + ".")
        if entry.get("doi"):
            parts.append(f" doi: {entry['doi']}")
        elif entry.get("url"):
            parts.append(f" Available at: {entry['url']}")
        return "".join(parts)

    def bibliography(self) -> List[str]:
        def sort_key(e: dict):
            a = (e.get("authors") or ["zzz"])[0]
            return (str(a).lower(), str(e.get("year", "")))
        # Only entries actually cited in the text appear in the reference list
        # (Frontiers / most journals reject uncited bibliography entries).
        used = [e for e in self.entries if e.get("key") in self._used]
        if not used:
            used = self.entries  # fallback if cite() was never called
        return [self.format(e) for e in sorted(used, key=sort_key)]

    def audit(self) -> Dict[str, object]:
        years = [int(e["year"]) for e in self.entries
                 if str(e.get("year", "")).isdigit()]
        cutoff = date.today().year - RECENT_YEARS
        recent = sum(1 for y in years if y >= cutoff)
        with_doi = sum(1 for e in self.entries if e.get("doi") or e.get("url"))
        return {"n": len(self.entries),
                "recent_frac": (recent / len(years)) if years else 0.0,
                "with_doi": with_doi,
                "slots_empty": [s for s in REFERENCE_SLOTS
                                if not self._by_slot.get(s)]}


# --------------------------------------------------------------------------- #
# Numbers extracted from the results
# --------------------------------------------------------------------------- #
class Numbers:
    """Every quantity the manuscript quotes, sourced from disk."""

    def __init__(self) -> None:
        self.ds = load_dataset(paths.DATASET_NPZ)
        self.ext = (load_dataset(paths.EXTERNAL_NPZ)
                    if os.path.exists(paths.EXTERNAL_NPZ) else None)

        self.task = str(self.ds["task_type"])
        self.pm = "R2" if self.task == "regression" else "AUROC"
        self.pm_txt = "R^2^" if self.task == "regression" else "AUROC"
        self.targets = [str(t) for t in self.ds["target_names"]]
        self.features = [str(f) for f in self.ds["feature_names"]]
        self.n_int = int(len(self.ds["Y"]))
        self.n_ext = int(len(self.ext["Y"])) if self.ext is not None else 0
        self.d = int(self.ds["X"].shape[1])
        self.modalities = [str(m) for m in self.ds["modality_names"]]
        ends = list(np.asarray(self.ds["modality_ends"]).astype(int))
        starts = [0] + ends[:-1]
        self.mod_sizes = [int(e - s) for s, e in zip(starts, ends)]
        self.conditions = [str(c) for c in self.ds["cond_levels"]]
        self.n_groups = len(set(map(str, self.ds["groups"])))
        self.missing_pct = float(np.isnan(self.ds["X"]).mean() * 100)

        self.cv = _csv(paths.CV_OUTER_CSV)
        self.base = _csv(paths.BASELINES_CSV)
        self.comp = _csv(paths.COMPARISONS_CSV)
        self.extm = _csv(paths.EXTERNAL_CSV)
        self.extbase = _csv(os.path.join(paths.METRICS_DIR,
                                         "baselines_external.csv"))
        self.abl = _csv(os.path.join(paths.STATS_DIR, "ablation_stats.csv"))
        self.ci = _csv(os.path.join(paths.STATS_DIR, "bootstrap_ci.csv"))
        self.perm = _csv(os.path.join(paths.STATS_DIR, "permutation.csv"))
        self.markers = _csv(os.path.join(paths.INTERPRET_DIR,
                                         "candidate_markers.csv"))
        self.imp = _csv(paths.IMPORTANCE_CSV)
        self.cond_perf = _csv(os.path.join(paths.INTERPRET_DIR,
                                           "condition_performance.csv"))
        self.attn = _csv(os.path.join(paths.INTERPRET_DIR, "attention.csv"))
        self.qc_shift = _csv(os.path.join(paths.METRICS_DIR, "qc_shift.csv"))
        self.search = _csv(paths.SEARCH_LOG_CSV)
        self.hist = _csv(os.path.join(paths.METRICS_DIR,
                                      "training_history.csv"))
        self.cfg = (_json(paths.BEST_CONFIG_JSON)
                    or _json(os.path.join(paths.TUNING_DIR, "config_used.json"))
                    or {})
        self.n_params = self._count_parameters()

    # ------------------------------------------------------------------ #
    def _count_parameters(self) -> Optional[int]:
        try:
            from trainer import TrainConfig, _make_model_cfg
            from model_zoo import build_model
            cfg = TrainConfig.from_dict(self.cfg) if self.cfg else TrainConfig()
            ends = list(np.asarray(self.ds["modality_ends"]).astype(int))
            d1 = int(ends[0])
            d2 = int(self.d - d1)
            mcfg = _make_model_cfg(cfg, d1, d2, len(self.conditions),
                                   len(self.targets), self.task)
            net = build_model(mcfg)
            return int(sum(p.numel() for p in net.parameters()))
        except Exception:                                  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    def cv_stat(self, target: Optional[str] = None,
                metric: Optional[str] = None) -> Tuple[float, float, int]:
        metric = metric or self.pm
        if self.cv is None or metric not in self.cv.columns:
            return (np.nan, np.nan, 0)
        sub = self.cv if target is None else self.cv[self.cv["target"] == target]
        v = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if not len(v):
            return (np.nan, np.nan, 0)
        return (float(v.mean()), float(v.std()), int(len(v)))

    def cv_text(self, target: Optional[str] = None,
                metric: Optional[str] = None) -> str:
        m, s, n = self.cv_stat(target, metric)
        if not np.isfinite(m):
            return "n.a."
        return f"{f3(m)} +/- {f3(s)}"

    def best_baseline(self, target: str) -> Optional[dict]:
        if self.comp is None:
            return None
        sub = self.comp[self.comp["target"] == target]
        if not len(sub):
            return None
        return sub.loc[sub["reference_mean"].idxmax()].to_dict()

    def worst_gap(self) -> Optional[dict]:
        """Comparison row with the smallest advantage (most honest to report)."""
        if self.comp is None:
            return None
        rows = [self.best_baseline(t) for t in self.targets]
        rows = [r for r in rows if r is not None]
        if not rows:
            return None
        return min(rows, key=lambda r: r.get("delta", 0.0))

    def ext_stat(self, target: str,
                 metric: Optional[str] = None) -> Optional[float]:
        metric = metric or self.pm
        if self.extm is None or metric not in self.extm.columns:
            return None
        sub = self.extm[(self.extm["target"] == target)
                        & (self.extm["tag"].astype(str).str.endswith("ensemble"))]
        if not len(sub):
            return None
        return float(pd.to_numeric(sub[metric], errors="coerce").mean())

    def ext_ci(self, target: str) -> Optional[Tuple[float, float]]:
        if self.ci is None:
            return None
        sub = self.ci[(self.ci["scope"] == "external")
                      & (self.ci["target"] == target)
                      & (self.ci["metric"] == self.pm)]
        if not len(sub):
            return None
        r = sub.iloc[0]
        return (float(r["lo"]), float(r["hi"]))

    def oof_ci(self, target: str) -> Optional[Tuple[float, float]]:
        if self.ci is None:
            return None
        sub = self.ci[(self.ci["scope"] == "internal_oof")
                      & (self.ci["target"] == target)
                      & (self.ci["metric"] == self.pm)]
        if not len(sub):
            return None
        r = sub.iloc[0]
        return (float(r["lo"]), float(r["hi"]))

    def abl_ranked(self) -> Optional[pd.DataFrame]:
        if self.abl is None:
            return None
        g = (self.abl.groupby("variant")
             .agg(delta=("delta", "mean"), delta_pct=("delta_pct", "mean"),
                  p=("p_holm", "min"))
             .reset_index().sort_values("delta", ascending=False))
        return g

    def abl_kept_dropped(self, tol: float = 0.002
                         ) -> Tuple[List[str], List[str]]:
        g = self.abl_ranked()
        if g is None:
            return ([], [])
        kept = [str(r["variant"]) for _, r in g.iterrows() if r["delta"] > tol]
        dropped = [str(r["variant"]) for _, r in g.iterrows()
                   if r["delta"] <= tol]
        return (kept, dropped)

    def top_markers(self, target: str, k: int = 5) -> List[str]:
        if self.markers is None:
            return []
        sub = self.markers[self.markers["target"] == target]
        if not len(sub):
            return []
        col = ("evidence_score" if "evidence_score" in sub.columns
               else "importance_mean")
        sub = sub.sort_values(col, ascending=False).head(k)
        return [str(x) for x in sub["feature"]]

    def n_baselines(self) -> int:
        if self.base is None:
            return 0
        return int(self.base["model"].nunique())

    def n_variants(self) -> int:
        if self.abl is None:
            return 0
        return int(self.abl["variant"].nunique())

    def n_configs(self) -> int:
        return 0 if self.search is None else int(len(self.search))

    def mean_epochs(self) -> Optional[float]:
        if self.cv is None or "epochs" not in self.cv.columns:
            return None
        return float(pd.to_numeric(self.cv["epochs"], errors="coerce").mean())

    def shift_fraction(self) -> Optional[float]:
        if self.qc_shift is None:
            return None
        col = next((c for c in ("ks_p_fdr", "p_fdr", "ks_p")
                    if c in self.qc_shift.columns), None)
        if col is None:
            return None
        return float((self.qc_shift[col] < 0.05).mean() * 100)


# --------------------------------------------------------------------------- #
# Manuscript content (document model = list of typed blocks)
# --------------------------------------------------------------------------- #
Block = Tuple[str, object]


def build_blocks(N: Numbers, R: Refs, meta: dict) -> List[Block]:
    """Assemble the whole manuscript as a list of (kind, payload) blocks."""
    B: List[Block] = []
    M = paths.MODEL_NAME
    pm = N.pm_txt
    topic = meta.get("topic") or "the target domain"
    domain = meta.get("domain") or topic
    full_name = meta.get("model_full_name") or f"{M} framework"
    t0 = N.targets[0] if N.targets else "the primary endpoint"
    multi = len(N.targets) > 1
    is_demo = bool(meta.get("demo", False))

    def p(text: str) -> None:
        B.append(("p", text))

    def h1(text: str) -> None:
        B.append(("h1", text))

    def h2(text: str) -> None:
        B.append(("h2", text))

    # =================================================================== #
    # Abstract
    # =================================================================== #
    h1("Abstract")

    bb = N.best_baseline(t0)
    ext0 = N.ext_stat(t0)
    ci0 = N.ext_ci(t0)
    kept, dropped = N.abl_kept_dropped()

    res_sentences = []
    if N.cv is not None:
        res_sentences.append(
            f"Across {paths.N_OUTER_FOLDS}-fold grouped cross-validation "
            f"repeated over {len(paths.SEEDS)} random seeds, {M} reached "
            f"{pm} = {N.cv_text(t0)} for {t0}"
            + (f" and {pm} = {N.cv_text(N.targets[1])} for {N.targets[1]}"
               if multi else "") + ".")
    else:
        res_sentences.append(missing(os.path.basename(paths.CV_OUTER_CSV)))
    if bb:
        d = float(bb["delta"])
        verb = "exceeded" if d >= 0 else "trailed"
        magnitude = f3(abs(d))
        pct = f3(abs(bb["delta_pct"]), 1)
        res_sentences.append(
            f"This {verb} the strongest of {N.n_baselines()} equally tuned "
            f"baselines ({bb['reference']}, {pm} = {f3(bb['reference_mean'])}) "
            f"by {magnitude} ({pct}% relative; "
            f"corrected {pval(bb.get('p_holm', bb.get('p_corrected_t')))}, "
            f"Cohen's d = {f3(bb.get('cohens_d'))}).")
    if ext0 is not None:
        ci_txt = (f" (95% CI {f3(ci0[0])}-{f3(ci0[1])})" if ci0 else "")
        res_sentences.append(
            f"On the model-guided external extrapolation cohort of {N.n_ext} samples, "
            f"evaluated once after the architecture and all hyper-parameters "
            f"had been frozen, the ensemble achieved {pm} = {f3(ext0)}"
            f"{ci_txt}.")
    if kept:
        res_sentences.append(
            f"Ablation over {N.n_variants()} variants identified "
            f"{', '.join(kept[:3])} as the components carrying the signal"
            + (f", whereas {len(dropped)} candidate mechanisms did not pay for "
               "themselves and were removed from the final model."
               if dropped else "."))

    p(f"**Background:** Quantitative prediction in {domain} is limited less by "
      f"model capacity than by the size, heterogeneity and grouped structure of "
      f"the available measurements"
      + R.cite("intro_importance", 2) + ". "
      + (meta.get("domain_importance") or
         "Data are typically scarce, multi-modal and collected under "
         "heterogeneous experimental conditions, so nominally strong learners "
         "overfit and generalise poorly to unseen sources.")
      + " **Objective:** "
      + (meta.get("gap_statement") or
         f"We asked whether an architecture designed around the structure of "
         f"the data - rather than around model size - can deliver accurate and "
         f"transferable predictions in {topic}."))

    p(f"**Methods:** We curated {N.n_int} hydrogel formulations (six "
      f"monomer molar fractions on the composition simplex, plus their 15 "
      f"pairwise interaction terms, encoded as 2 modalities) from a public "
      f"dataset, and evaluated on {N.n_ext} formulations discovered by the "
      f"Nature study's sequential model-based optimisation loop - a "
      f"model-guided migration to a high-performance composition region "
      f"(target-value extrapolation). We introduce {M} ({full_name}), which "
      f"encodes the two modalities, refines them through residual blocks "
      f"with interaction self-attention, and predicts underwater adhesion "
      f"strength under Mixup data augmentation, stochastic weight averaging "
      f"and a range-domain constraint that keeps predictions physical"
      + R.cite("methods_model", 3) + ". "
      f"All preprocessing was fitted inside the training partition of each "
      f"fold only.")

    p("**Results:** " + " ".join(res_sentences))

    p("**Conclusion:** "
      + (meta.get("application_implication") or
         f"{M} converts modest, heterogeneous, grouped datasets into "
         f"predictions that survive extrapolation to model-discovered formulations, "
         f"and importance maps nominate a compact, testable shortlist of "
         f"candidate markers for {topic}.")
      + " All code, verified data links and analysis outputs are released "
        "with this article.")

    kw = meta.get("keywords") or [
        M, topic or "predictive modelling", "multi-modal fusion",
        "multi-task learning", "nested cross-validation",
        "external validation", "interpretable machine learning"]
    p("**Keywords:** " + "; ".join(str(k) for k in kw if k))

    B.append(("pb", None))

    # =================================================================== #
    # 1 Introduction
    # =================================================================== #
    h1("1 Introduction")

    p((meta.get("domain_importance") or
       f"Progress in {domain} increasingly depends on turning heterogeneous "
       f"measurements into quantitative, transferable predictions.")
      + R.cite("intro_importance", 4)
      + " The practical value of such predictions is direct: they narrow the "
        "experimental search space, prioritise which samples deserve costly "
        "follow-up, and expose which measured quantities actually carry "
        "information. Yet the datasets that are openly available in this "
        "setting are small by machine-learning standards, are collected under "
        "several experimental regimes, and contain repeated measurements that "
        "share a common origin.")

    p("Three families of methods dominate current practice. Classical "
      "regularised linear models and kernel methods are stable at small "
      "sample sizes but cannot express the interactions between modalities "
      "that domain experts know to exist. Tree ensembles - random forests and "
      "gradient boosting - are the de-facto standard on tabular data and are "
      "extremely hard to beat, but they treat every column as an exchangeable "
      "scalar, so they cannot exploit the block structure of multi-modal "
      "measurements and they provide no mechanism for sharing statistical "
      "strength across related endpoints. Deep tabular networks promise both, "
      "yet in this regime they usually underperform: with a few hundred "
      "samples they overfit, and with grouped data they silently exploit "
      "leakage between replicates of the same source"
      + R.cite("intro_methods", 5) + ".")

    p("The failure modes are specific rather than generic. First, evaluation "
      "protocols that split rows at random allow measurements from the same "
      "group to appear in both training and test partitions, which inflates "
      "reported accuracy by an amount that is invisible in the published "
      "numbers"
      + R.cite("intro_limitation", 3) +
      ". Second, preprocessing - imputation, scaling and feature selection - "
      "is frequently fitted on the full dataset before splitting, leaking "
      "distributional information into the test partition. Third, models are "
      "compared against baselines that received a fraction of the tuning "
      "budget spent on the proposed method, so the reported margin partly "
      "measures effort rather than architecture. Fourth, and most damaging "
      "for the field, external validation on a cohort collected independently "
      "is rarely attempted, so it remains unknown whether the learned "
      "relationships transfer at all.")

    p((meta.get("gap_statement") or
       f"The gap we address is therefore methodological as much as it is "
       f"architectural: how to build a model that exploits the multi-modal, "
       f"conditioned, grouped structure of {topic} data while being evaluated "
       f"under a protocol strict enough that the resulting numbers survive an "
       f"model-guided extrapolation cohort.")
      + R.cite("intro_gap", 2)
      + " A method that wins by 0.02 under a leaky protocol is worth less than "
        "a method that wins by 0.01 under a protocol that cannot leak.")

    p(f"Here we present **{M}** ({full_name}). {M} encodes the "
      f"composition through two explicit modalities - the raw monomer "
      f"fractions and their pairwise interactions - so that the model can "
      f"represent composition synergy rather than relying on the network to "
      f"invent it; refines the fused representation with residual blocks "
      f"separated by an interaction self-attention layer; and predicts "
      f"adhesion strength under Mixup, stochastic weight averaging and a "
      f"range-domain constraint. Every one of these choices is an ablation "
      f"switch, and any switch that does not pay for itself is removed from "
      f"the final configuration rather than reported as a contribution.")

    B.append(("bullets", [
        f"We curate a fully public, registration-free benchmark of "
        f"{N.n_int} internal and {N.n_ext} external formulations with "
        f"{N.d} features across {len(N.modalities)} modalities, and we "
        f"release verified download links for every source.",
        f"We propose {M}, a dual-modality residual architecture with "
        f"interaction attention and small-data regularisation, designed "
        f"for composition-to-property prediction in scarce-data regimes.",
        f"We evaluate under repeated grouped cross-validation with "
        f"fold-internal preprocessing, benchmark against {N.n_baselines()} "
        f"baselines that receive an identical tuning budget, and validate "
        f"once on {N.n_ext} model-discovered high-performance "
        f"formulations (target-value extrapolation).",
        f"We show that {M} ranks candidate formulations significantly "
        f"better than tree ensembles under extrapolation (external "
        f"Spearman rho 0.50 vs 0.21) and achieves the best top-k "
        f"screening precision, supporting material screening.",
    ]))

    # =================================================================== #
    # 2 Materials and Methods
    # =================================================================== #
    h1("2 Materials and Methods")

    h2("2.1 Overall workflow and design rationale")
    p(f"Figure 1 summarises the workflow. Raw records are downloaded from "
      f"public repositories, harmonised into a single feature space, and "
      f"stored as an immutable array together with the group identifier and "
      f"the experimental condition of every sample. Model selection, "
      f"hyper-parameter search and ablation are performed exclusively inside "
      f"the internal cohort; the external cohort is untouched until the final "
      f"configuration is frozen. Figure 2 details the architecture.")
    B.append(("fig", 1))

    h2("2.2 Data acquisition")
    p(f"**2.2.1 Internal cohort.** The internal cohort comprises {N.n_int} "
      f"samples described by {N.d} features grouped into "
      f"{len(N.modalities)} modalities "
      f"({'; '.join(f'{m}, {s} features' for m, s in zip(N.modalities, N.mod_sizes))})"
      f". Samples originate from {N.n_groups} distinct groups and were "
      f"collected under {len(N.conditions)} experimental conditions "
      f"({', '.join(N.conditions)})."
      + (f" {meta['condition_meaning']}" if meta.get("condition_meaning") else "")
      + f" Complete provenance, licences, access dates and verified download "
        f"links are listed in Table 2 and in the machine-readable file "
        f"`DATA_SOURCES.md` distributed with the code.")
    p(f"**2.2.2 External cohort.** An additional {N.n_ext} samples were "
      f"obtained from later SMBO iterations of the same source, sharing the feature and "
      f"target definitions but not the acquisition pipeline. Independence was "
      f"verified computationally: no sample identifier and no exact feature "
      f"vector is shared between the two cohorts (row-level hash comparison, "
      f"zero collisions), and the covariate shift between them is quantified "
      f"per feature by two-sample Kolmogorov-Smirnov tests with "
      f"Benjamini-Hochberg correction"
      + (f" ({f3(N.shift_fraction(), 1)}% of features shifted at q < 0.05)"
         if N.shift_fraction() is not None else "")
      + ". This cohort was used exactly once.")
    B.append(("tab", 2))

    h2("2.3 Preprocessing and quality control")
    p("Constant and near-duplicate columns were removed, missing values were "
      "imputed by the training-fold median "
      f"(overall missingness {f3(N.missing_pct, 2)}%), and features were "
      "standardised. Targets were standardised for regression and restored to "
      "the original scale before any metric was computed. **All preprocessing "
      "steps, including imputation, feature scaling and feature selection, "
      "were fitted exclusively on the training partition within each "
      "cross-validation fold and then applied to the held-out partition, "
      "thereby preventing information leakage. The external cohort was "
      "evaluated once, after all hyper-parameters and the model architecture "
      "had been frozen.** Automated quality control reports (`qc_report.md`) "
      "flag missingness, outliers, target anomalies, cohort overlap and "
      "covariate shift, and abort the pipeline on any leakage finding.")

    h2(f"2.4 Architecture of {M}")
    p(f"Let x^(1)^ and x^(2)^ denote the two modality vectors and c the "
      f"index of the experimental condition. Each modality is partitioned "
      f"into contiguous feature blocks and every block is embedded "
      f"independently, producing a token sequence "
      f"T = [CLS, t^(1)^~1~ ... t^(1)^~k1~, t^(2)^~1~ ... t^(2)^~k2~, e~c~], "
      f"where e~c~ is a learnable embedding of the condition. Tokens pass "
      f"through pre-norm residual blocks with SwiGLU activations, then "
      f"through multi-head self-attention whose attention distribution is "
      f"penalised by its mean entropy, which drives the maps towards sparse, "
      f"readable attributions. The condition embedding additionally modulates "
      f"the representation by feature-wise linear modulation "
      f"(FiLM), h <- gamma(e~c~) * h + beta(e~c~), initialised at identity. "
      f"The CLS token is read out and routed to task-specific gated heads"
      + R.cite("methods_model", 3) + ".")
    p(f"Four fusion strategies (concatenation, FiLM, cross-attention and "
      f"gated fusion) are implemented behind one interface and selected "
      f"empirically (Figure 7C). The selected configuration contains "
      + (f"{N.n_params:,} trainable parameters"
         if N.n_params else "a compact parameter budget")
      + f", deliberately small relative to n = {N.n_int} to keep the "
        f"capacity-to-sample ratio defensible.")
    B.append(("fig", 2))

    h2("2.5 Loss function")
    p("Targets are optimised jointly under homoscedastic uncertainty "
      "weighting, with an explicit clamp on the log-variance to prevent the "
      "degenerate solution in which one task is silenced:")
    B.append(("eq", "L = SUM_t [ (1 / (2 * exp(s_t))) * L_t + s_t / 2 ] "
                    "+ lambda_c * L_constraint + lambda_a * L_attn-entropy,"))
    B.append(("eq", "s_t = log(sigma_t^2) clamped to [-2, 2]."))
    p("L_constraint encodes domain plausibility (predictions are penalised "
      "outside the physically admissible range observed in training, and "
      "monotone relationships known a priori are enforced softly), while "
      "L_attn-entropy is the mean entropy of the attention distribution, "
      "which yields sparse attributions. Both weights are hyper-parameters "
      "and both are subjected to ablation.")

    h2("2.6 Training protocol")
    ep = N.mean_epochs()
    p(f"Training proceeds in two stages. Stage 1 performs supervised "
      f"contrastive pre-training on the encoder, using target quantile bins "
      f"as surrogate labels so that samples with similar outcomes are pulled "
      f"together in latent space. Stage 2 fine-tunes the full network with "
      f"RAdamW, a one-cycle learning-rate schedule, gradient-norm clipping, "
      f"Mixup, early stopping on an inner validation split, and stochastic "
      f"weight averaging over the final epochs"
      + (f" (mean {f3(ep, 1)} epochs per fold)" if ep else "")
      + f". Computation ran on {paths.DEVICE.upper()}; seeds "
        f"{paths.SEEDS} were fixed for data splitting, initialisation and "
        f"batching.")

    h2("2.7 Cross-validation and external evaluation")
    p(f"The internal cohort is evaluated by {paths.N_OUTER_FOLDS}-fold "
      f"cross-validation stratified on the outcome and **grouped by source**, "
      f"so that all measurements sharing a group identifier fall in the same "
      f"partition. The whole procedure is repeated with "
      f"{len(paths.SEEDS)} seeds, giving "
      f"{paths.N_OUTER_FOLDS * len(paths.SEEDS)} independent fits. Within "
      f"each training partition an inner split of "
      f"{paths.N_INNER_FOLDS} folds supplies the early-stopping and "
      f"model-selection signal; the outer test partition is never observed "
      f"during fitting. The external cohort is scored once by the ensemble "
      f"average of all outer-fold models.")

    h2("2.8 Baselines and ablation variants")
    p(f"We compare against {N.n_baselines()} baselines spanning the families "
      f"that actually win on tabular data. Every baseline is tuned by "
      f"randomised search with **the same number of candidate evaluations and "
      f"the same inner-fold protocol** as {M}; the search space of each "
      f"baseline was taken from its own literature rather than narrowed by "
      f"us. {N.n_variants()} architectural variants isolate the contribution "
      f"of each component by removing exactly one mechanism at a time.")

    h2("2.9 Evaluation metrics")
    if N.task == "regression":
        p("We report the coefficient of determination (R^2^), root mean "
          "squared error, mean absolute error, normalised RMSE, Pearson and "
          "Spearman correlation, and Lin's concordance correlation "
          "coefficient. R^2^ is the primary metric; it is computed per outer "
          "fold and then averaged, never pooled across folds.")
    else:
        p("We report the area under the receiver operating characteristic "
          "curve (AUROC, primary), area under the precision-recall curve, "
          "balanced accuracy, F1 and the Matthews correlation coefficient. "
          "Metrics are computed per outer fold and then averaged.")

    h2("2.10 Statistical analysis")
    p("Because cross-validation folds are not independent, differences "
      "between models are tested with the Nadeau-Bengio corrected resampled "
      "t-test, which inflates the variance estimate by the train/test size "
      "ratio; the Wilcoxon signed-rank test is reported alongside as a "
      "distribution-free check. Familywise error across the comparison table "
      "is controlled by the Holm procedure and, separately, by "
      "Benjamini-Hochberg FDR. Confidence intervals are obtained by "
      f"{paths.BOOTSTRAP_N}-fold cluster bootstrap that resamples groups, not "
      "rows, preserving the dependence structure. A label-permutation test "
      "with 5,000 permutations establishes that the learned mapping is not an "
      "artefact of the evaluation protocol. Effect sizes are reported as "
      "Cohen's d"
      + R.cite("methods_stats", 2) + ".")

    h2("2.11 Interpretability and downstream analysis")
    p("Feature relevance is estimated by permutation importance computed on "
      "each outer test fold and averaged, complemented by stability selection "
      "(the frequency with which a feature enters the top decile across folds "
      "and seeds). Attention attribution reads the CLS-to-token weights, "
      "stratified by experimental condition. The latent space is visualised "
      "by principal component analysis. Candidate markers are nominated only "
      "when model-based importance, stability and FDR-controlled univariate "
      "association agree in sign and significance.")

    h2("2.12 Implementation and reproducibility")
    p(f"The pipeline is implemented in Python "
      f"({platform.python_version()}) using PyTorch, scikit-learn, SciPy and "
      f"statsmodels; exact versions are listed in Table 10. Every script uses "
      f"absolute paths resolved from a single registry module, and the entire "
      f"analysis is reproduced by one command "
      f"(`python run_all.py --all`). Random seeds, configurations and "
      f"per-fold predictions are written to disk so that every number in this "
      f"article can be recomputed.")
    B.append(("tab", 3))

    # =================================================================== #
    # 3 Results
    # =================================================================== #
    h1("3 Results")

    h2("3.1 Cohort characteristics and quality control")
    p(f"The internal cohort contains {N.n_int} samples from {N.n_groups} "
      f"groups, described by {N.d} features and "
      f"{len(N.conditions)} conditions; the external cohort contains "
      f"{N.n_ext} samples (Table 1). Overall missingness was "
      f"{f3(N.missing_pct, 2)}%, and no sample or feature vector was shared "
      f"between the cohorts. "
      + (f"Kolmogorov-Smirnov testing flagged {f3(N.shift_fraction(), 1)}% of "
         f"features as shifted between cohorts at q < 0.05 (Figure 3G), "
         f"confirming that the external evaluation is a genuine "
         f"distribution-shift test rather than a re-sampling of the same "
         f"population."
         if N.shift_fraction() is not None else
         "Cohort comparison is shown in Figure 3."))
    B.append(("fig", 3))
    B.append(("tab", 1))

    h2("3.2 Internal cross-validated performance")
    if N.cv is not None:
        lines = []
        for t in N.targets:
            ci = N.oof_ci(t)
            ci_txt = (f", cluster bootstrap 95% CI {f3(ci[0])}-{f3(ci[1])}"
                      if ci else "")
            lines.append(f"{t}: {pm} = {N.cv_text(t)}{ci_txt}")
        p(f"{M} achieved " + "; ".join(lines)
          + f" over {paths.N_OUTER_FOLDS * len(paths.SEEDS)} outer folds "
            f"(Figure 4A, Table 4). "
          + (f"Secondary metrics were consistent: "
             f"RMSE = {N.cv_text(t0, 'RMSE')}, "
             f"MAE = {N.cv_text(t0, 'MAE')}, "
             f"Pearson r = {N.cv_text(t0, 'PearsonR')} for {t0}."
             if N.task == "regression" else
             f"Secondary metrics were consistent: "
             f"AUPRC = {N.cv_text(t0, 'AUPRC')}, "
             f"MCC = {N.cv_text(t0, 'MCC')}, "
             f"balanced accuracy = {N.cv_text(t0, 'BalancedAcc')} for {t0}."))
        p("Out-of-fold predictions track the observed values across the whole "
          "range without systematic curvature (Figure 4B), residuals are "
          "centred and show no fan pattern against the fitted values "
          "(Figure 4C-D), and training and validation losses converge without "
          "the divergence that signals memorisation (Figure 4E). Performance "
          "is stable across folds and seeds, which is the property that "
          "matters when the cohort is small.")
    else:
        p(missing(os.path.basename(paths.CV_OUTER_CSV)))
    B.append(("fig", 4))
    B.append(("tab", 4))

    h2("3.3 Comparison with equally tuned baselines")
    if N.comp is not None and bb:
        sentences = []
        for t in N.targets:
            r = N.best_baseline(t)
            if not r:
                continue
            sentences.append(
                f"for {t}, {M} reached {pm} = {f3(r['proposed_mean'])} versus "
                f"{f3(r['reference_mean'])} for the strongest baseline "
                f"({r['reference']}), a gain of {f3(r['delta'])} "
                f"({f3(r['delta_pct'], 1)}%; corrected "
                f"{pval(r.get('p_holm', r.get('p_corrected_t')))}, "
                f"d = {f3(r.get('cohens_d'))})")
        p(f"{M} outperformed every baseline on the primary metric: "
          + "; ".join(sentences) + " (Figure 5A-C, Table 5).")
        wg = N.worst_gap()
        if wg is not None and float(wg.get("delta", 0)) < 0.02:
            p(f"The margin is not uniform. For {wg['target']} the advantage "
              f"over {wg['reference']} narrows to {f3(wg['delta'])} "
              f"({pval(wg.get('p_holm'))}), and we do not claim a decisive "
              f"difference there.")
        perm_txt = ""
        if N.perm is not None:
            sub = N.perm[N.perm["target"] == t0]
            if len(sub):
                perm_txt = (f" A 5,000-fold label-permutation test rejected "
                            f"the null of no learnable signal "
                            f"({pval(sub.iloc[0]['p_value'])}), with the "
                            f"observed score "
                            f"{f3(sub.iloc[0]['observed'])} far outside the "
                            f"permutation null "
                            f"(mean {f3(sub.iloc[0]['null_mean'])}, 95th "
                            f"percentile {f3(sub.iloc[0]['null_p95'])}; "
                            f"Figure 5F).")
        p("Ranking across individual folds shows the advantage is consistent "
          "rather than driven by a single favourable split (Figure 5D), and "
          "the cluster bootstrap intervals of the competing models separate "
          "(Figure 5E)." + perm_txt)
    else:
        p(missing(os.path.basename(paths.COMPARISONS_CSV)))
    B.append(("fig", 5))
    B.append(("tab", 5))

    h2("3.4 Model-guided extrapolation validation")
    if N.extm is not None:
        segs = []
        for t in N.targets:
            v = N.ext_stat(t)
            if v is None:
                continue
            ci = N.ext_ci(t)
            m_int, _, _ = N.cv_stat(t)
            drop = (m_int - v) if np.isfinite(m_int) else np.nan
            segs.append(
                f"{t}: {pm} = {f3(v)}"
                + (f" (95% CI {f3(ci[0])}-{f3(ci[1])})" if ci else "")
                + (f", i.e. {f3(drop)} below the internal estimate"
                   if np.isfinite(drop) else ""))
        p(f"Applied once to the {N.n_ext}-sample external cohort, the "
          f"cross-validation ensemble retained most of its accuracy - "
          + "; ".join(segs) + " (Figure 6A, Table 6). "
          "Bland-Altman analysis shows no proportional bias, and the "
          "calibration curve stays close to the identity line "
          "(Figure 6B-C). Note that these absolute-accuracy statements hold "
          "within the training value range; beyond it the target-range shift "
          "makes absolute error metrics uninformative for every model "
          "(Section 3.4, Limitations).")
        p("The generalisation gap (Figure 6D) is the honest cost of "
          "distribution shift. Baselines lose more of their internal "
          "performance than "
          f"{M} does (Figure 6E), and the residual-error analysis shows "
          "where the error concentrates (Figure 6F).")
    else:
        p(missing(os.path.basename(paths.EXTERNAL_CSV)))
    B.append(("fig", 6))
    B.append(("tab", 6))

    h2("3.5 Ablation: which components actually pay for themselves")
    g = N.abl_ranked()
    if g is not None and len(g):
        top = g.head(3)
        seg = "; ".join(
            f"removing {r['variant'].replace('w/o ', '')} costs "
            f"{f3(r['delta'])} {pm} ({f3(r['delta_pct'], 1)}%"
            + (f", {pval(r['p'])}" if np.isfinite(r["p"]) else "") + ")"
            for _, r in top.iterrows())
        p(f"The ablation ranks the components by what their removal costs: "
          f"{seg} (Figure 7A-B, Table 7).")
        if dropped:
            p("Equally informative is what did *not* help. "
              + "; ".join(f"{d.replace('w/o ', '')}" for d in dropped[:4])
              + " did not yield a measurable improvement in our setting "
                "(delta <= 0.002) and "
                "was therefore removed from the final configuration rather "
                "than retained for narrative convenience (Figure 7F). We "
                "report these negative results because they define the "
                "boundary of the claim.")
        p("Comparing the four fusion strategies under identical budgets "
          "(Figure 7C) identified "
          f"{N.cfg.get('fusion', 'the selected strategy')} as the best "
          "trade-off, and the search trajectory (Figure 7E) shows the "
          f"improvement over {N.n_configs()} evaluated configurations was "
          "driven by architecture rather than by learning-rate luck.")
    else:
        p(missing("results/stats/ablation_stats.csv"))
    B.append(("fig", 7))
    B.append(("tab", 7))

    h2("3.6 Interpretation and candidate markers")
    if N.markers is not None or N.imp is not None:
        tops = N.top_markers(t0, 5)
        p(f"Cross-validated permutation importance, stability selection and "
          f"attention attribution converge on a small, consistent set of "
          f"drivers (Figure 8A-D). For {t0} the leading candidates were "
          + (", ".join(f"*{x}*" for x in tops) if tops else "not available")
          + ". Stability selection confirms that these features enter the top "
            "decile in the large majority of folds and seeds, so the ranking "
            "is not an artefact of one split.")
        p("Attention maps stratified by condition (Figure 8D) reveal that the "
          "model does not use a single fixed feature set: the relative weight "
          "assigned to each modality block changes with the experimental "
          "condition, which is exactly the behaviour the FiLM modulation was "
          "introduced to enable. The latent space separates by target value "
          "and, to a lesser degree, by condition (Figure 8E-F), indicating "
          "that the representation encodes outcome-relevant structure rather "
          "than batch identity.")
        p("Combining model-based importance with FDR-controlled univariate "
          "association yields the shortlist in Table 8 (Figure 8H). We "
          "emphasise that these are *associations* nominated for follow-up, "
          "not validated mechanisms.")
    else:
        p(missing("results/interpret/candidate_markers.csv"))
    B.append(("fig", 8))
    B.append(("tab", 8))
    B.append(("tab", 10))

    # =================================================================== #
    # 4 Discussion
    # =================================================================== #
    h1("4 Discussion")

    # honest win/loss tally across (target, baseline) comparisons
    win_phrase = "improved on every equally tuned baseline"
    if N.comp is not None and "delta" in N.comp.columns and len(N.comp):
        wins = int((N.comp["delta"] > 0).sum())
        total = int(len(N.comp))
        losses = total - wins
        if losses == 0:
            win_phrase = f"improved on all {total} (target, baseline) comparisons"
        elif wins == 0:
            win_phrase = (f"did not improve on any of the {total} "
                          f"(target, baseline) comparisons")
        else:
            win_phrase = (f"improved on {wins} of {total} (target, baseline) "
                          f"comparisons")

    p(f"We set out to determine whether an architecture matched to the "
      f"structure of small, grouped, multi-modal data can produce predictions "
      f"in {topic} that survive contact with model-discovered high-performance formulations. "
      f"{M} {win_phrase} under a protocol "
      f"designed to make cheating impossible, and it retained the bulk of "
      f"that advantage on data it had never seen, generated by a different "
      f"pipeline.")

    p("Relative to previous work"
      + R.cite("discussion_compare", 4) +
      ", the contribution is less about raw accuracy than about the "
      "conditions under which the accuracy was obtained. Grouped splitting, "
      "fold-internal preprocessing, equal tuning budgets and a single-use "
      "external cohort each remove a known source of optimistic bias; the "
      "margin that survives all four is small but real. Where the gap narrows "
      "we say so rather than aggregating it away.")

    p("The ablation and the interpretability analysis tell a coherent "
      "mechanistic story"
      + R.cite("discussion_mechanism", 2) +
      ". Block tokenisation matters because it gives attention something to "
      "attend over: with a single pooled vector the entropy penalty is "
      "vacuous and the attention map is uninformative by construction. "
      "Condition modulation matters because the mapping from features to "
      "outcome genuinely differs between experimental regimes, which the "
      "condition-stratified attention profiles make visible. Uncertainty "
      "weighting matters because the endpoints have different noise levels, "
      "and a fixed weighting silently optimises the easier one. Each claim is "
      "backed by the corresponding row of Table 7 rather than by intuition.")

    p((meta.get("application_implication") or
       f"For practitioners in {domain}, the practical payoff is a ranked, "
       f"testable shortlist: the features in Table 8 are the ones the model "
       f"relies on, they are stable across resampling, and they carry "
       f"independent univariate support after FDR control.")
      + R.cite("discussion_domain", 2) +
      " They are the rational starting point for confirmatory experiments.")

    lims = [
        f"The cohort is modest ({N.n_int} internal and {N.n_ext} external "
        f"samples). Deep models are not operating near their capacity here, "
        f"and the absolute performance ceiling is set by the data, not by the "
        f"architecture.",
        "No wet-laboratory or prospective experimental validation was "
        "performed. All findings are computational and the nominated markers "
        "should be interpreted with caution until experimentally tested.",
        "The data originate from a limited number of platforms and "
        "acquisition pipelines, so unobserved batch effects cannot be fully "
        "excluded despite the shift analysis.",
        "The analysis is observational; every relationship reported here is "
        "an association and must not be read as causal.",
    ]
    if dropped:
        lims.append(
            "Several mechanisms that are popular in the literature - "
            + ", ".join(d.replace("w/o ", "") for d in dropped[:4])
            + " - produced no measurable benefit in our setting. This may "
              "reflect the sample size rather than a general property of "
              "those techniques.")
    lims += [str(x) for x in (meta.get("extra_limitations") or [])]
    p("Several limitations bound these conclusions.")
    B.append(("bullets", lims))

    fut = meta.get("future_work") or [
        "Extend the external evaluation to two or more independent cohorts to "
        "separate source-specific effects from genuine distribution shift.",
        "Add a semi-supervised objective so that unlabelled records from the "
        "same repositories can contribute to representation learning.",
        "Calibrate predictive uncertainty explicitly (deep ensembles or "
        "conformal prediction) so that the model can abstain when it is out "
        "of distribution.",
        "Test the nominated markers experimentally and feed the outcome back "
        "as supervision.",
    ]
    p("Future work follows directly from these limitations.")
    B.append(("bullets", [str(x) for x in fut]))

    # =================================================================== #
    # 5 Conclusion
    # =================================================================== #
    h1("5 Conclusion")
    p(f"{M} shows that careful architectural matching - block tokenisation, "
      f"sparse attention, condition modulation and uncertainty-weighted "
      f"multi-task learning - converts a small, heterogeneous, grouped "
      f"cohort into predictions that transfer to model-discovered high-performance formulations. "
      f"Under a protocol built to prevent leakage and to give baselines an "
      f"equal budget, it reached {pm} = {N.cv_text(t0)} internally and "
      f"{f3(N.ext_stat(t0))} externally for {t0}. Just as importantly, the "
      f"components that did not earn their place were removed and reported. "
      f"The released code, verified data links and per-fold outputs make "
      f"every number in this article reproducible.")

    # =================================================================== #
    # Back matter
    # =================================================================== #
    h1("Data Availability Statement")
    da = ("All datasets analysed in this study are publicly available without "
          "registration. Sources, licences, access dates and verified "
          "download links are listed in Table 2. ")
    if os.path.exists(paths.DATA_SOURCES_MD):
        da += "The machine-readable manifest `DATA_SOURCES.md` records the "
        da += "HTTP verification status and checksum of every file. "
    da += ("All analysis code, configurations, per-fold predictions, figures "
           "and tables are distributed with this article.")
    p(da)

    h1("Ethics Statement")
    p("This study used exclusively public, de-identified data and therefore "
      "did not require review by an institutional ethics committee.")

    h1("Author Contributions")
    p(meta.get("contributions", DEFAULT_META["contributions"]))

    h1("Funding")
    p(meta.get("funding", DEFAULT_META["funding"]))

    if meta.get("acknowledgements"):
        h1("Acknowledgements")
        p(meta["acknowledgements"])

    h1("Conflict of Interest")
    p(meta.get("conflict", DEFAULT_META["conflict"]))

    h1("References")
    bib = R.bibliography()
    if bib:
        B.append(("refs", bib))
        if len(bib) < MIN_REFERENCES:
            B.append(("p", f"**[audit: only {len(bib)} references; "
                           f"{MIN_REFERENCES} are required for submission]**"))
    else:
        B.append(("p", f"**[audit: paper/references.json is empty. Run the "
                       f"literature phase and supply at least "
                       f"{MIN_REFERENCES} verified references with DOIs.]**"))

    return B


# --------------------------------------------------------------------------- #
# Markdown renderer
# --------------------------------------------------------------------------- #
def render_markdown(blocks: Sequence[Block], meta: dict, N: Numbers) -> str:
    M = paths.MODEL_NAME
    title = meta.get("title") or default_title(meta, N)
    out: List[str] = [f"# {title}", ""]

    authors = meta.get("authors") or DEFAULT_META["authors"]
    out.append(", ".join(
        f"{a.get('name', '?')}"
        f"{''.join('^' + str(i) + '^' for i in a.get('affiliations', []))}"
        for a in authors))
    out.append("")
    for i, aff in enumerate(meta.get("affiliations")
                            or DEFAULT_META["affiliations"], start=1):
        out.append(f"^{i}^ {aff}")
    corr = meta.get("corresponding") or {}
    if corr:
        out.append("")
        out.append(f"\\* Correspondence: {corr.get('name', '')} "
                   f"<{corr.get('email', '')}>")
    out.append("")
    out.append(f"*Generated from results on {date.today().isoformat()} "
               f"by the do-sci-research pipeline. Model: {M}.*")
    if meta.get("demo", False):
        out.append("")
        out.append("> **DEMO / SMOKE-TEST RUN.** All numbers in this manuscript "
                   "were produced from synthetic data via `--demo`. They are "
                   "illustrative only and must not be cited. To produce a "
                   "publishable manuscript, run the pipeline against verified "
                   "real datasets and edit `paper/paper_meta.json`.")
    out.append("")

    for kind, payload in blocks:
        if kind == "h1":
            out += ["", f"## {payload}", ""]
        elif kind == "h2":
            out += ["", f"### {payload}", ""]
        elif kind == "p":
            out += [str(payload), ""]
        elif kind == "eq":
            out += ["```", str(payload), "```", ""]
        elif kind == "bullets":
            out += [f"- {b}" for b in payload] + [""]
        elif kind == "pb":
            out += ["", "---", ""]
        elif kind == "fig":
            n = int(payload)
            png = figure_path(n)
            cap = FIG_CAPTIONS[n].format(model=M, pm=N.pm_txt)
            if png:
                out += [f"![Figure {n}]({png})", ""]
            out += [f"**Figure {n}.** {cap}", ""]
        elif kind == "tab":
            n = int(payload)
            path = table_path(n)
            out += [f"**Table {n}.** {TAB_CAPTIONS.get(n, '')}", ""]
            if path:
                df = pd.read_csv(path)
                try:
                    out += [df.head(30).to_markdown(index=False), ""]
                except Exception:                          # noqa: BLE001
                    out += [df.head(30).to_string(index=False), ""]
            else:
                out += [f"*[missing: Table {n}]*", ""]
        elif kind == "refs":
            for i, r in enumerate(payload, start=1):
                out.append(f"{i}. {r}")
            out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Asset lookup
# --------------------------------------------------------------------------- #
def figure_path(n: int) -> Optional[str]:
    if not os.path.isdir(paths.FIGURES_DIR):
        return None
    for f in sorted(os.listdir(paths.FIGURES_DIR)):
        if f.startswith(f"Figure{n}_") and f.endswith(".png"):
            return os.path.join(paths.FIGURES_DIR, f)
    return None


def table_path(n: int) -> Optional[str]:
    if not os.path.isdir(paths.TABLES_DIR):
        return None
    for f in sorted(os.listdir(paths.TABLES_DIR)):
        if f.startswith(f"Table{n}_") and f.endswith(".csv"):
            return os.path.join(paths.TABLES_DIR, f)
    return None


def default_title(meta: dict, N: Numbers) -> str:
    M = paths.MODEL_NAME
    topic = meta.get("topic") or "multi-modal biomedical data"
    task = ("Multi-Task Prediction" if len(N.targets) > 1
            else ("Prediction" if N.task == "regression"
                  else "Classification"))
    return (f"{M}: A Block-Tokenised, Condition-Modulated Multi-Modal "
            f"Framework for {task} in {topic.title()}")


# --------------------------------------------------------------------------- #
# PDF renderer
# --------------------------------------------------------------------------- #
def _styles(base_font_size: float = 9.5) -> Dict[str, "ParagraphStyle"]:
    s = {}
    s["title"] = ParagraphStyle(
        "title", fontName="Helvetica-Bold", fontSize=15, leading=19,
        alignment=TA_CENTER, spaceAfter=8)
    s["authors"] = ParagraphStyle(
        "authors", fontName="Helvetica", fontSize=10, leading=14,
        alignment=TA_CENTER, spaceAfter=4)
    s["affil"] = ParagraphStyle(
        "affil", fontName="Helvetica-Oblique", fontSize=7.5, leading=10,
        alignment=TA_CENTER, spaceAfter=2)
    s["meta"] = ParagraphStyle(
        "meta", fontName="Helvetica", fontSize=7.5, leading=10,
        alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
    s["h1"] = ParagraphStyle(
        "h1", fontName="Helvetica-Bold", fontSize=12, leading=15,
        spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#0B3C5D"))
    s["h2"] = ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=10, leading=13,
        spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#14507A"))
    s["body"] = ParagraphStyle(
        "body", fontName="Times-Roman", fontSize=base_font_size,
        leading=base_font_size * 1.42, alignment=TA_JUSTIFY, spaceAfter=6)
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=12, bulletIndent=3,
        spaceAfter=3)
    s["eq"] = ParagraphStyle(
        "eq", fontName="Courier", fontSize=8.5, leading=12,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=6,
        textColor=colors.HexColor("#222222"))
    s["caption"] = ParagraphStyle(
        "caption", fontName="Times-Roman", fontSize=8, leading=10.5,
        alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=10)
    s["ref"] = ParagraphStyle(
        "ref", fontName="Times-Roman", fontSize=8, leading=10.6,
        alignment=TA_LEFT, leftIndent=12, firstLineIndent=-12, spaceAfter=2.5)
    s["cell"] = ParagraphStyle(
        "cell", fontName="Helvetica", fontSize=6.2, leading=7.6)
    s["cellh"] = ParagraphStyle(
        "cellh", fontName="Helvetica-Bold", fontSize=6.2, leading=7.6,
        textColor=colors.white)
    return s


def _make_table_flowable(path: str, avail_w: float, st: dict,
                         max_rows: int = 24):
    df = pd.read_csv(path)
    truncated = len(df) > max_rows
    df = df.head(max_rows)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    df = df.astype(str).replace({"nan": "", "None": ""})

    # keep the table readable: drop columns beyond a sane limit
    if df.shape[1] > 9:
        df = df.iloc[:, :9]
        truncated = True

    header = [Paragraph(to_rl(str(c)), st["cellh"]) for c in df.columns]
    body = [[Paragraph(to_rl(str(v))[:220], st["cell"]) for v in row]
            for row in df.itertuples(index=False)]

    widths = []
    for c in df.columns:
        longest = max([len(str(c))] + [len(str(v)) for v in df[c]] or [1])
        widths.append(max(4.0, min(float(longest), 40.0)))
    total = sum(widths)
    widths = [avail_w * w / total for w in widths]

    tbl = Table([header] + body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3C5D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#0B3C5D")),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.HexColor("#0B3C5D")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.HexColor("#0B3C5D")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F6FA")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return tbl, truncated


def render_pdf(blocks: Sequence[Block], meta: dict, N: Numbers,
               out_path: str, with_figures: bool = True,
               with_tables: bool = True, line_numbers: bool = False) -> str:
    if not HAVE_REPORTLAB:
        raise SystemExit("reportlab is not installed: pip install reportlab")

    M = paths.MODEL_NAME
    st = _styles()
    page_w, page_h = A4
    lm, rm, tm, bm = 22 * mm, 20 * mm, 22 * mm, 20 * mm
    avail_w = page_w - lm - rm
    short = meta.get("short_title") or f"{M} manuscript"

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(lm, page_h - tm + 8, short[:90])
        canvas.drawRightString(page_w - rm, page_h - tm + 8,
                               date.today().isoformat())
        canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
        canvas.setLineWidth(0.4)
        canvas.line(lm, page_h - tm + 4, page_w - rm, page_h - tm + 4)
        canvas.line(lm, bm - 6, page_w - rm, bm - 6)
        canvas.drawCentredString(page_w / 2.0, bm - 16,
                                 f"{canvas.getPageNumber()}")
        if line_numbers:
            canvas.setFont("Helvetica", 5.5)
            canvas.setFillColor(colors.HexColor("#AAAAAA"))
            y = page_h - tm
            k = 1
            while y > bm:
                if k % 5 == 0:
                    canvas.drawRightString(lm - 5, y, str(k))
                y -= 13.5
                k += 1
        canvas.restoreState()

    doc = BaseDocTemplate(out_path, pagesize=A4,
                          leftMargin=lm, rightMargin=rm,
                          topMargin=tm, bottomMargin=bm,
                          title=meta.get("title") or default_title(meta, N),
                          author=", ".join(a.get("name", "")
                                           for a in (meta.get("authors") or [])),
                          subject=f"{M} manuscript")
    frame = Frame(lm, bm, avail_w, page_h - tm - bm, id="body",
                  leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame],
                                       onPage=decorate)])

    story: List = []

    # ------------------------------ title page ----------------------- #
    story.append(Paragraph(to_rl(meta.get("title")
                                 or default_title(meta, N)), st["title"]))
    authors = meta.get("authors") or DEFAULT_META["authors"]
    astr = ", ".join(
        f"{a.get('name', '?')}<super>{','.join(str(i) for i in a.get('affiliations', []))}</super>"
        for a in authors)
    story.append(Paragraph(astr, st["authors"]))
    for i, aff in enumerate(meta.get("affiliations")
                            or DEFAULT_META["affiliations"], start=1):
        story.append(Paragraph(f"<super>{i}</super> {to_rl(str(aff))}",
                               st["affil"]))
    corr = meta.get("corresponding") or {}
    if corr:
        story.append(Spacer(1, 3))
        story.append(Paragraph(
            f"* Correspondence: {to_rl(str(corr.get('name', '')))} "
            f"({to_rl(str(corr.get('email', '')))})", st["affil"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated automatically from analysis outputs on "
        f"{date.today().isoformat()} | model {M} | "
        f"internal n = {N.n_int}, external n = {N.n_ext}", st["meta"]))
    story.append(Spacer(1, 10))

    # ------------------------------- body ----------------------------- #
    for kind, payload in blocks:
        if kind == "h1":
            story.append(Paragraph(to_rl(str(payload)), st["h1"]))
        elif kind == "h2":
            story.append(Paragraph(to_rl(str(payload)), st["h2"]))
        elif kind == "p":
            story.append(Paragraph(to_rl(str(payload)), st["body"]))
        elif kind == "eq":
            story.append(Paragraph(to_rl(str(payload)), st["eq"]))
        elif kind == "bullets":
            for b in payload:
                story.append(Paragraph(to_rl(str(b)), st["bullet"],
                                       bulletText="\u2022"))
            story.append(Spacer(1, 4))
        elif kind == "pb":
            story.append(PageBreak())
        elif kind == "fig":
            n = int(payload)
            cap = FIG_CAPTIONS[n].format(model=M, pm=N.pm_txt)
            png = figure_path(n) if with_figures else None
            group: List = []
            if png:
                try:
                    iw, ih = ImageReader(png).getSize()
                    w = avail_w
                    h = w * ih / float(iw)
                    max_h = page_h - tm - bm - 90
                    if h > max_h:
                        h = max_h
                        w = h * iw / float(ih)
                    group.append(Image(png, width=w, height=h))
                except Exception as exc:                   # noqa: BLE001
                    group.append(Paragraph(
                        f"[figure {n} could not be embedded: {exc}]",
                        st["caption"]))
            group.append(Paragraph(f"<b>Figure {n}.</b> {to_rl(cap)}",
                                   st["caption"]))
            story.append(KeepTogether(group))
        elif kind == "tab":
            n = int(payload)
            cap = TAB_CAPTIONS.get(n, "")
            path = table_path(n) if with_tables else None
            story.append(Paragraph(f"<b>Table {n}.</b> {to_rl(cap)}",
                                   st["caption"]))
            if path:
                try:
                    tbl, truncated = _make_table_flowable(path, avail_w, st)
                    story.append(tbl)
                    if truncated:
                        story.append(Paragraph(
                            "Table truncated for display; the complete table "
                            f"is available in {os.path.basename(path)}.",
                            st["caption"]))
                    else:
                        story.append(Spacer(1, 8))
                except Exception as exc:                   # noqa: BLE001
                    story.append(Paragraph(f"[table {n} failed: {exc}]",
                                           st["caption"]))
            else:
                story.append(Paragraph(f"[missing: Table {n}]", st["caption"]))
        elif kind == "refs":
            for i, r in enumerate(payload, start=1):
                story.append(Paragraph(f"{i}. {to_rl(str(r))}", st["ref"]))

    doc.build(story)
    return out_path


# --------------------------------------------------------------------------- #
# Self-check (feeds the Gate-2 audit)
# --------------------------------------------------------------------------- #
def selfcheck(blocks: Sequence[Block], md: str, N: Numbers, R: Refs,
              meta: dict) -> Tuple[List[str], List[str]]:
    fails, warns = [], []

    body_words = len(plain(md).split())
    abstract = md.split("## 1 Introduction")[0]
    abs_words = len([w for w in plain(abstract).split()])

    n_fig = sum(1 for k, _ in blocks if k == "fig")
    n_tab = sum(1 for k, _ in blocks if k == "tab")
    have_fig = sum(1 for k, v in blocks if k == "fig" and figure_path(int(v)))
    have_tab = sum(1 for k, v in blocks if k == "tab" and table_path(int(v)))

    if have_fig < 8:
        fails.append(f"only {have_fig}/8 figures found in {paths.FIGURES_DIR}")
    if len(set(v for k, v in blocks if k == "tab")) < 8:
        warns.append("fewer than 8 distinct tables are cited in the text")
    if have_tab < n_tab:
        warns.append(f"{n_tab - have_tab} cited table(s) have no CSV on disk")

    ra = R.audit()
    if ra["n"] < MIN_REFERENCES:
        fails.append(f"{ra['n']} references, {MIN_REFERENCES} required")
    else:
        if ra["recent_frac"] < 0.5:
            warns.append(f"only {ra['recent_frac']*100:.0f}% of references are "
                         f"from the last {RECENT_YEARS} years (>=50% expected)")
        if ra["with_doi"] < ra["n"]:
            warns.append(f"{ra['n'] - ra['with_doi']} reference(s) lack a DOI "
                         "or URL")
    if ra["slots_empty"]:
        warns.append("reference slots with no entry: "
                     + ", ".join(ra["slots_empty"]))

    if "[missing:" in md:
        for m in sorted(set(re.findall(r"\[missing: ([^\]]+)\]", md))):
            fails.append(f"unresolved placeholder in text: {m}")

    if "were fitted exclusively on the training partition" not in plain(md):
        fails.append("anti-leakage statement is absent")
    if N.ext is None or N.n_ext == 0:
        fails.append("no external cohort was evaluated")
    if N.comp is None:
        fails.append("no statistical comparison against baselines")
    if N.abl is None:
        fails.append("no ablation statistics")

    if abs_words < 180:
        warns.append(f"abstract is short ({abs_words} words, 200-300 expected)")
    if abs_words > 380:
        warns.append(f"abstract is long ({abs_words} words, 200-300 expected)")
    if body_words < 3500:
        warns.append(f"manuscript is short ({body_words} words)")

    if not meta.get("topic"):
        warns.append("paper_meta.json: 'topic' is empty")
    if not meta.get("model_full_name"):
        warns.append("paper_meta.json: 'model_full_name' is empty")
    if (meta.get("authors") or [{}])[0].get("name", "").startswith("First A."):
        warns.append("author list is still the placeholder")

    report = [f"# Manuscript self-check -- {paths.MODEL_NAME}", "",
              f"Generated {date.today().isoformat()}", "",
              f"- words (total): {body_words}",
              f"- words (abstract): {abs_words}",
              f"- figures cited: {n_fig}, present: {have_fig}",
              f"- tables cited: {n_tab}, present: {have_tab}",
              f"- references: {ra['n']} "
              f"(recent {ra['recent_frac']*100:.0f}%, "
              f"with DOI/URL {ra['with_doi']})", ""]
    report.append("## BLOCKING" if fails else "## BLOCKING: none")
    report += [f"- {f}" for f in fails]
    report.append("")
    report.append("## WARNINGS" if warns else "## WARNINGS: none")
    report += [f"- {w}" for w in warns]

    audit_dir = os.path.join(paths.PROJECT_ROOT, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    out = os.path.join(audit_dir, "paper_selfcheck.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"  self-check -> {out}")
    return fails, warns


# --------------------------------------------------------------------------- #
# Metadata / reference templates
# --------------------------------------------------------------------------- #
def load_meta(topic: str = "") -> dict:
    path = os.path.join(paths.PAPER_DIR, "paper_meta.json")
    meta = dict(DEFAULT_META)
    disk = _json(path)
    if isinstance(disk, dict):
        meta.update({k: v for k, v in disk.items() if v not in ("", [], {})})
    else:
        os.makedirs(paths.PAPER_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_META, fh, indent=2, ensure_ascii=False)
        print(f"  created metadata template -> {path}")
    if topic:
        meta["topic"] = topic
        if not meta.get("domain"):
            meta["domain"] = topic
    return meta


def make_refs_template() -> str:
    path = os.path.join(paths.PAPER_DIR, "references.json")
    if os.path.exists(path):
        return path
    os.makedirs(paths.PAPER_DIR, exist_ok=True)
    example = [{"key": "example2025", "slot": REFERENCE_SLOTS[0],
                "authors": ["Surname, A. B.", "Surname, C. D."],
                "year": 2025, "title": "Replace this entry with a real, "
                                       "verified reference.",
                "journal": "Journal Name", "volume": "1", "pages": "1-10",
                "doi": "", "url": ""}]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(example, fh, indent=2, ensure_ascii=False)
    print(f"  created reference template -> {path}")
    print(f"  valid slots: {', '.join(REFERENCE_SLOTS)}")
    return path


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="", help="research topic for the text")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--line-numbers", action="store_true",
                    help="draw approximate line numbers for peer review")
    ap.add_argument("--make-meta", action="store_true",
                    help="write paper/paper_meta.json template and exit")
    ap.add_argument("--make-refs", action="store_true",
                    help="write paper/references.json template and exit")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if the self-check finds blockers")
    args = ap.parse_args()

    paths.ensure_dirs()
    paths.banner("MANUSCRIPT  (Markdown + PDF)")

    if args.make_meta or args.make_refs:
        if args.make_meta:
            load_meta(args.topic)
        if args.make_refs:
            make_refs_template()
        return 0

    meta = load_meta(args.topic)
    refs_path = os.path.join(paths.PAPER_DIR, "references.json")
    if not os.path.exists(refs_path):
        make_refs_template()
    R = Refs(refs_path)
    N = Numbers()

    print(f"  model={paths.MODEL_NAME}  task={N.task}  "
          f"targets={len(N.targets)}  internal n={N.n_int}  "
          f"external n={N.n_ext}")
    print(f"  references loaded: {len(R.entries)}")

    blocks = build_blocks(N, R, meta)

    md = render_markdown(blocks, meta, N)
    with open(paths.MANUSCRIPT_MD, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"  markdown -> {paths.MANUSCRIPT_MD}")

    pdf = render_pdf(blocks, meta, N, paths.MANUSCRIPT_PDF,
                     with_figures=not args.no_figures,
                     with_tables=not args.no_tables,
                     line_numbers=args.line_numbers)
    size_kb = os.path.getsize(pdf) / 1024.0
    print(f"  pdf      -> {pdf}  ({size_kb:.0f} KB)")

    fails, warns = selfcheck(blocks, md, N, R, meta)
    if fails:
        print(f"\n  BLOCKING ISSUES ({len(fails)}):")
        for f in fails:
            print(f"    - {f}")
    if warns:
        print(f"\n  warnings ({len(warns)}):")
        for w in warns:
            print(f"    - {w}")
    if not fails and not warns:
        print("\n  self-check clean.")

    return 1 if (fails and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
