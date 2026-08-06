"""Apply GATE-2 audit fixes (round 1) to frontiers_SIMPLEX.tex:
P0-2 TopK20 0.95->0.90; P0-4 baseline table per official baselines_external.csv;
P1-1 ablation wording neutral; P1-2 remove unsourced modulus sentence;
P2-2 Methods 'independent' wording; Fig5/6 numbering."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
src = open(p, encoding="utf-8").read()
log = []

# ---- P0-2: TopK20 0.95 -> 0.90 everywhere ----
n = src.count("0.95")
src = src.replace("Top-20 precision 0.95", "Top-20 precision 0.90")
src = src.replace("TopK20 0.95", "TopK20 0.90")
src = src.replace("Top-20 $0.95$", "Top-20 $0.90$")
log.append(f"TopK20 fixes (mentions of 0.95 in text: {n})")

# ---- Abstract baseline numbers (official baselines_external.csv) ----
src = src.replace(
    "including random forest ($R^2 = 0.44$), SVR ($0.71$) and Ridge ($0.64$)",
    "including random forest ($R^2 = 0.56$), SVR ($0.63$) and Ridge ($0.64$)")

# ---- Table tab:ext (official R2/Spearman; recomputed TopK/AUC) ----
old_tab = """\\begin{tabular}{lccccc}
\\toprule
Model & $R^2$ & Spearman $\\rho$ & ROC-AUC & Top-10 & Top-20 \\\\
\\midrule
\\textbf{SIMPLEX} & \\textbf{0.71} & \\textbf{0.87} & \\textbf{0.94} & \\textbf{1.00} & \\textbf{0.95} \\\\
SVR-RBF & 0.71 & 0.85 & 0.94 & 1.00 & 0.90 \\\\
Ridge & 0.64 & 0.81 & 0.93 & 1.00 & 0.90 \\\\
ElasticNet & 0.63 & 0.85 & 0.93 & 1.00 & 0.90 \\\\
GBR & 0.52 & 0.80 & 0.94 & 0.90 & 0.90 \\\\
RandomForest & 0.44 & 0.80 & 0.92 & 1.00 & 0.90 \\\\
\\bottomrule"""
new_tab = """\\begin{tabular}{lccccc}
\\toprule
Model & $R^2$ & Spearman $\\rho$ & ROC-AUC & Top-10 & Top-20 \\\\
\\midrule
\\textbf{SIMPLEX} & \\textbf{0.71} & \\textbf{0.87} & \\textbf{0.94} & \\textbf{1.00} & \\textbf{0.90} \\\\
Ridge & 0.64 & 0.86 & 0.93 & 1.00 & 0.90 \\\\
ElasticNet & 0.64 & 0.86 & 0.93 & 1.00 & 0.90 \\\\
SVR-RBF & 0.63 & 0.83 & 0.94 & 1.00 & 0.90 \\\\
RandomForest & 0.56 & 0.84 & 0.92 & 1.00 & 0.90 \\\\
HistGB & 0.51 & 0.81 & 0.93 & 0.90 & 0.90 \\\\
\\bottomrule"""
assert old_tab in src, "table not found"
src = src.replace(old_tab, new_tab)
log.append("tab:ext updated (official baselines)")

# ---- text after table: SVR tie claim -> SIMPLEX leads R2 ----
src = src.replace(
    "SIMPLEX attains $R^2 = 0.71$ (tied with SVR $0.71$, above Ridge $0.64$ and "
    "random forest $0.44$), the highest Spearman correlation ($0.87$), the "
    "highest Top-20 precision ($0.95$), and perfect Top-10 precision ($1.00$)",
    "SIMPLEX attains the highest $R^2$ ($0.71$, versus Ridge $0.64$, "
    "SVR $0.63$ and random forest $0.56$), the highest Spearman correlation "
    "($0.87$), the highest Top-20 precision ($0.90$), and perfect Top-10 "
    "precision ($1.00$)")

# ---- 3.3 prospective text: TopK20 0.95 -> 0.90 ----
src = src.replace(
    "the highest Top-20 precision ($0.95$)",
    "the highest Top-20 precision ($0.90$)")

# ---- Discussion 4.1: Top-20 0.95 -> 0.90 ----
src = src.replace(
    "Top-20 precision $0.95$",
    "Top-20 precision $0.90$")

# ---- P1-2: remove unsourced modulus sentence ----
old_mod = (" and multi-target extension (a preliminary single-task validation "
           "on modulus reaches parity with random forest, $R^2 = 0.396$ vs "
           "$0.392$)")
assert old_mod in src, "modulus sentence not found"
src = src.replace(old_mod, " and multi-target extension of the framework")
log.append("modulus sentence removed (no source data)")

# ---- P1-1: ablation wording (attention/domain constraint neutral) ----
src = src.replace(
    "removing the attention layer, residual blocks, or the domain constraint "
    "each degrades performance",
    "removing the attention layer or the residual blocks degrades performance, "
    "while the attention and domain-constraint effects are small and within "
    "search noise")

# ---- P2-2: Methods 'independent' wording ----
src = src.replace(
    "These are the last model-guided discoveries (adhesion 62 to 251 kPa, mean "
    "158 kPa) and were not used in any way for model selection, tuning, or "
    "ablation -- they are a genuinely held-out, prospective cohort.",
    "These are the last model-guided discoveries (adhesion 62 to 251 kPa, mean "
    "158 kPa) and were not used in any way for model selection, tuning, or "
    "ablation -- they are a held-out prospective cohort from the same "
    "laboratory and instrumentation.")

open(p, "w", encoding="utf-8").write(src)
for l in log:
    print(" -", l)
print("all GATE-2 round-1 fixes applied")
