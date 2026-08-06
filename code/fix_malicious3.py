"""Apply GATE-2 malicious-audit (Round 2) fixes: P1 x5 + key P2."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
src = open(p, encoding="utf-8").read()
log = []

# ---- P1-2: Top-k shared by ALL baselines -> remove 'highest' Top-k claims ----
# Abstract
old = ("and screening precision of 1.00 at Top-10 and 0.95 at Top-20 -- the "
       "highest of all models, including random forest ($R^2 = 0.56$), SVR "
       "($0.63$) and Ridge ($0.64$).")
new = ("and Top-10/Top-20 screening precision of 1.00/0.90 -- a level shared by "
       "all baselines, indicating that the prospective ranking task is highly "
       "predictable; SIMPLEX differs in attaining the numerically highest $R^2$ "
       "($0.71$) and Spearman correlation ($0.87$) among equally tuned "
       "baselines (random forest $0.56$, SVR $0.63$, Ridge $0.64$), although "
       "the cohort is too small to separate the models statistically.")
if old in src:
    src = src.replace(old, new); log.append("abstract Top-k shared (P1-2)")
else:
    log.append("SKIP abs P1-2")

# Abstract 'independent' -> 'held-out'
old = "and prospectively validate on 25 independent formulations"
new = "and prospectively validate on 25 held-out formulations"
if old in src:
    src = src.replace(old, new); log.append("abstract independent->held-out (P2-1)")
else:
    log.append("SKIP abs held-out")

# Abstract 'key regulariser' -> largest ns effect (P1-6)
old = "Ablation identifies Mixup as the key regulariser"
new = "Ablation identifies Mixup as the largest (though not statistically significant) regularisation effect"
if old in src:
    src = src.replace(old, new); log.append("abstract Mixup ns (P1-6)")
else:
    log.append("SKIP abs Mixup")

# ---- L144: Top-k shared; R2/Spearman 'numerically highest, not separable' (P1-4) ----
old = ("SIMPLEX attains the highest $R^2$ among equally tuned baselines ($0.71$, "
       "versus Ridge $0.64$, SVR $0.63$ and random forest $0.56$; single-seed "
       "external $R^2$ ranged $0.60$--$0.69$, with the ensemble averaging "
       "$0.71$), the highest Spearman correlation ($0.87$), the highest Top-20 "
       "precision ($0.90$), and perfect Top-10 precision ($1.00$): all ten "
       "highest-predicted formulations are among the ten strongest adhesives "
       "in the held-out cohort. Top-20 precision is $0.90$, above the $0.80$ "
       "random expectation, although the small cohort size limits the "
       "discrimination of this particular threshold.")
new = ("SIMPLEX attains the numerically highest $R^2$ among equally tuned "
       "baselines ($0.71$, versus Ridge $0.64$, SVR $0.63$ and random forest "
       "$0.56$; single-seed external $R^2$ ranged $0.60$--$0.69$, with the "
       "ensemble averaging $0.71$) and the highest Spearman correlation "
       "($0.87$); the external bootstrap 95\\% CI for $R^2$ is $[0.46, 0.86]$, "
       "which overlaps the baselines, so the models are not statistically "
       "separable at this cohort size. Top-10 screening precision is perfect "
       "($1.00$: all ten highest-predicted formulations are among the ten "
       "strongest adhesives) and Top-20 precision is $0.90$; however, all "
       "baselines attain the same Top-k values, indicating that the screening "
       "task is highly predictable and that Top-k does not discriminate "
       "between models here.")
if old in src:
    src = src.replace(old, new); log.append("L144 Top-k shared + CI overlap (P1-2/4)")
else:
    log.append("SKIP L144")

# ---- L166 ablation: attention removal neutral, Mixup ns (P1-6) ----
old = ("removing the attention layer or the residual blocks degrades "
       "performance, while the attention and domain-constraint effects are "
       "small and within search noise")
new = ("removing Mixup or the residual blocks degrades $R^2$ ($+0.067$ and "
       "$+0.062$, both non-significant at Holm-corrected $p=1.0$), while "
       "removing the attention layer is neutral ($-0.008$) and the "
       "domain-constraint effect is within search noise")
if old in src:
    src = src.replace(old, new); log.append("L166 ablation ns wording (P1-6)")
else:
    log.append("SKIP L166")

# ---- L194 Discussion: 0.95 -> 0.90 + highest claims softened ----
old = ("attains the highest $R^2$ among equally tuned baselines ($0.71$), "
       "Spearman correlation ($0.87$), and Top-20 precision ($0.90$).")
new = ("attains the numerically highest $R^2$ among equally tuned baselines "
       "($0.71$; bootstrap CI $[0.46, 0.86]$ overlaps the baselines) and the "
       "highest Spearman correlation ($0.87$); Top-k precision is shared with "
       "all baselines.")
if old in src:
    src = src.replace(old, new); log.append("L194 R2/Spearman + Top-k shared (P1-2/4)")
else:
    log.append("SKIP L194")

# ---- L48 acronym qualifier (P2-1) ----
old = ("EXtrapolation evaluation). SIMPLEX encodes the composition through two "
       "explicit modalities")
new = ("EXtrapolation evaluation; the extrapolation is in composition space, "
       "while the target range of the prospective cohort remains interpolative "
       "with respect to training). SIMPLEX encodes the composition through two "
       "explicit modalities")
if old in src:
    src = src.replace(old, new); log.append("L48 extrapolation qualifier (P2-1)")
else:
    log.append("SKIP L48")

# ---- 3.6 p-value precision (P1-7) ----
old = "importance 0.143, FDR-corrected $p<10^{-53}$"
new = "importance 0.143, FDR-corrected $p \\approx 1.3 \\times 10^{-53}$"
if old in src:
    src = src.replace(old, new); log.append("3.6 p-value precision (P1-7)")
else:
    log.append("SKIP p-val")

# ---- Limitations: add selection-bias (vi) + SVR consistency (P1-3) ----
old = ("(v) All conclusions are computational; prospective wet-chemistry "
       "synthesis of the top-ranked formulations is required to confirm the "
       "screening value.")
new = ("(v) All conclusions are computational; prospective wet-chemistry "
       "synthesis of the top-ranked formulations is required to confirm the "
       "screening value. (vi) The prospective cohort is an SMBO-selected "
       "high-performance subset enriched in the BA--PEA region, so the Top-k "
       "metrics are conditional on this selection; because every model shares "
       "the same Top-k values, screening precision does not discriminate "
       "models on this cohort.")
if old in src:
    src = src.replace(old, new); log.append("Limitations selection-bias (P2-9)")
else:
    log.append("SKIP Limitations vi")

open(p, "w", encoding="utf-8").write(src)
print("\n".join(log) if log else "no changes")
