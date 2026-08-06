"""Fix residual inconsistencies from malicious-audit verification."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
src = open(p, encoding="utf-8").read()
log = []

old = ("random forest, despite matching SIMPLEX internally, drops to $R^2 = 0.44$ "
       "on the prospective cohort because its nearest-neighbour-style "
       "extrapolation fails outside the density of the training composition "
       "space, whereas SIMPLEX's continuous composition encoding transfers.")
new = ("random forest, despite matching SIMPLEX internally, attains $R^2 = 0.56$ "
       "on the prospective cohort, whereas SIMPLEX attains $0.71$, "
       "demonstrating stronger transfer to the held-out, model-discovered "
       "formulations.")
if old in src:
    src = src.replace(old, new); log.append("l144 RF 0.44->0.56 + wording")
else:
    log.append("SKIP l144")

old = ("attains the highest $R^2$ ($0.71$), Spearman correlation ($0.87$), "
       "and Top-20 precision ($0.95$).")
new = ("attains the highest $R^2$ among equally tuned baselines ($0.71$), "
       "Spearman correlation ($0.87$), and Top-20 precision ($0.90$).")
if old in src:
    src = src.replace(old, new); log.append("l194 TopK20 0.90")
else:
    log.append("SKIP l194a")

old = ("The mechanism is visible in the generalisation gap: tree models "
       "extrapolate by nearest-neighbour interpolation and degrade sharply "
       "outside the density of the training composition space (random forest "
       "$R^2$ drops from $0.81$ internally to $0.44$ prospectively), while "
       "SIMPLEX's continuous, interaction-aware composition encoding transfers "
       "with a small gap ($0.79 \\rightarrow 0.71$)")
new = ("The mechanism is visible in the held-out generalisation gap: tree "
       "models degrade more on the prospective cohort (random forest $R^2$ "
       "falls from $0.81$ internally to $0.56$ prospectively), while "
       "SIMPLEX's continuous, interaction-aware composition encoding retains a "
       "smaller gap ($0.79 \\rightarrow 0.71$), even though all prospective "
       "formulations lie within the training adhesion range")
if old in src:
    src = src.replace(old, new); log.append("l194 gap wording")
else:
    log.append("SKIP l194b")

old = ("(iv) The internal difference from random forest is not statistically "
       "significant, and SVR matches SIMPLEX's prospective $R^2$ --- both "
       "facts are reported transparently")
new = ("(iv) The internal difference from random forest is not statistically "
       "significant, and the exact prospective $R^2$ ranking depends on "
       "baseline tuning (SVR attains up to $0.71$ when tuned aggressively) "
       "--- both facts are reported transparently")
if old in src:
    src = src.replace(old, new); log.append("l198 SVR claim")
else:
    log.append("SKIP l198")

open(p, "w", encoding="utf-8").write(src)
print("\n".join(log))
