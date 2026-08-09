# -*- coding: utf-8 -*-
"""Move all MID figure/table refs to sentence end."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex'
src = open(P, encoding='utf-8').read()

reps = [
    # L108
    ("The target-range overlap is substantial (\\Fref{fig:dataset}{I}), so any prospective advantage cannot be attributed to out-of-range extrapolation in the target variable.",
     "The target-range overlap is substantial, so any prospective advantage cannot be attributed to out-of-range extrapolation in the target variable (\\Fref{fig:dataset}{I})."),
    # L112 - groups
    ("Measurements are organised into experimental groups (\\Fref{fig:dataset}{H}), which motivates grouped cross-validation to prevent within-batch leakage into training.",
     "Measurements are organised into experimental groups, which motivates grouped cross-validation to prevent within-batch leakage into training (\\Fref{fig:dataset}{H})."),
    # L112 - shift + condition space
    ("The prospective cohort exhibits measurable per-feature covariate shift relative to the internal cohort (\\Fref{fig:dataset}{G}) and occupies a distinct region of condition space (\\Fref{fig:dataset}{F}); together these panels establish that the prospective cohort is compositionally novel and that transfer, not memorisation, is required.",
     "The prospective cohort exhibits measurable per-feature covariate shift relative to the internal cohort and occupies a distinct region of condition space; together these panels establish that the prospective cohort is compositionally novel and that transfer, not memorisation, is required (\\Fref{fig:dataset}{G}, \\Fref{fig:dataset}{F})."),
    # L112 - pca
    ("The raw feature space, projected onto the leading two principal components, separates adhesion values continuously (\\Fref{fig:dataset}{E}), confirming that composition carries a learnable target signal.",
     "The raw feature space, projected onto the leading two principal components, separates adhesion values continuously, confirming that composition carries a learnable target signal (\\Fref{fig:dataset}{E})."),
    # L125 - oof
    ("Out-of-fold predictions track observations closely without systematic bias (\\Fref{fig:cv}{B}), residuals are centred at zero with no evident curvature in the fitted-versus-residual plot (\\Fref{fig:cv}{D}), and the error distribution is concentrated at low magnitudes (\\Fref{fig:cv}{E}).",
     "Out-of-fold predictions track observations closely without systematic bias, residuals are centred at zero with no evident curvature in the fitted-versus-residual plot, and the error distribution is concentrated at low magnitudes (\\Fref{fig:cv}{B}, \\Fref{fig:cv}{D}, \\Fref{fig:cv}{E})."),
    # L125 - slopes
    ("Per-target $R^2$ slopes (\\Fref{fig:cv}{C}) indicate that SIMPLEX and random forest degrade similarly toward the extremes of the adhesion range, so the tie holds across target levels rather than on an easy subregion.",
     "Per-target $R^2$ slopes indicate that SIMPLEX and random forest degrade similarly toward the extremes of the adhesion range, so the tie holds across target levels rather than on an easy subregion (\\Fref{fig:cv}{C})."),
    # L127 - learning curves
    ("Learning curves (\\Fref{fig:cv}{F}) show both models approaching their plateau by roughly 250 training formulations, and the per-seed-fold heatmap (\\Fref{fig:cv}{G}) together with the seed-to-seed stability boxplot (\\Fref{fig:cv}{I}) confirm that fold-level differences are within seed noise.",
     "Learning curves show both models approaching their plateau by roughly 250 training formulations, and the per-seed-fold heatmap together with the seed-to-seed stability boxplot confirm that fold-level differences are within seed noise (\\Fref{fig:cv}{F}, \\Fref{fig:cv}{G}, \\Fref{fig:cv}{I})."),
    # L127 - hexbin
    ("The density hexbin (\\Fref{fig:cv}{H}) corroborates the absence of systematic miscalibration.",
     "The density hexbin corroborates the absence of systematic miscalibration (\\Fref{fig:cv}{H})."),
    # L158 - paired
    ("Paired per-fold scores (\\Fref{fig:bench}{B}) show no fold on which any baseline significantly exceeds SIMPLEX, and the Holm-adjusted $\\Delta R^2$ panel (\\Fref{fig:bench}{D}) reports no significant internal difference.",
     "Paired per-fold scores show no fold on which any baseline significantly exceeds SIMPLEX, and the Holm-adjusted $\\Delta R^2$ panel reports no significant internal difference (\\Fref{fig:bench}{B}, \\Fref{fig:bench}{D})."),
    # L160 - rank
    ("Cross-fold ranking (\\Fref{fig:bench}{E}) places SIMPLEX and random forest in the same rank group, and the critical-difference diagram (\\Fref{fig:bench}{I}) shows no statistically separated clusters.",
     "Cross-fold ranking places SIMPLEX and random forest in the same rank group, and the critical-difference diagram shows no statistically separated clusters (\\Fref{fig:bench}{E}, \\Fref{fig:bench}{I})."),
    # L160 - CI + perm
    ("Cluster-bootstrap confidence intervals (\\Fref{fig:bench}{G}) and a permutation test (\\Fref{fig:bench}{H}) agree that internal differences are within noise.",
     "Cluster-bootstrap confidence intervals and a permutation test agree that internal differences are within noise (\\Fref{fig:bench}{G}, \\Fref{fig:bench}{H})."),
    # L160 - quality map
    ("On the model-quality map (\\Fref{fig:bench}{F}), SIMPLEX pairs a competitive internal $R^2$ with the highest prospective Spearman correlation, and Top-20 screening precision is at the cohort ceiling for every model (\\Fref{fig:bench}{C}).",
     "On the model-quality map, SIMPLEX pairs a competitive internal $R^2$ with the highest prospective Spearman correlation, and Top-20 screening precision is at the cohort ceiling for every model (\\Fref{fig:bench}{F}, \\Fref{fig:bench}{C})."),
    # L171 - track
    ("Predicted values track observations with errors concentrated at the highest-adhesion formulations (\\Fref{fig:ext}{A}), and the Bland-Altman plot shows no systematic bias (\\Fref{fig:ext}{B}).",
     "Predicted values track observations with errors concentrated at the highest-adhesion formulations, and the Bland-Altman plot shows no systematic bias (\\Fref{fig:ext}{A}, \\Fref{fig:ext}{B})."),
    # L175 - residuals
    ("Residuals on the prospective cohort are centred near zero (\\Fref{fig:ext}{H}), errors grow modestly with predicted rank in the highest quartile (\\Fref{fig:ext}{F}), and calibration remains acceptable across the range (\\Fref{fig:ext}{D}).",
     "Residuals on the prospective cohort are centred near zero, errors grow modestly with predicted rank in the highest quartile, and calibration remains acceptable across the range (\\Fref{fig:ext}{H}, \\Fref{fig:ext}{F}, \\Fref{fig:ext}{D})."),
    # L224 - violin
    ("Violin overlays of the top variants show that the full-model mean sits at the upper edge of the distribution (\\Fref{fig:abl}{E}), confirming that no ablated variant is superior.",
     "Violin overlays of the top variants show that the full-model mean sits at the upper edge of the distribution, confirming that no ablated variant is superior (\\Fref{fig:abl}{E})."),
    # L226 - variant order
    ("Variant $R^2$ values are ordered in (\\Fref{fig:abl}{B}), per-variant effects are summarised in (\\Fref{fig:abl}{F}), and the pruning log records each retention decision (\\Fref{fig:abl}{G}).",
     "Variant $R^2$ values are ordered, per-variant effects are summarised, and the pruning log records each retention decision (\\Fref{fig:abl}{B}, \\Fref{fig:abl}{F}, \\Fref{fig:abl}{G})."),
    # L226 - cumulative
    ("Cumulative importance separates marginal from interaction contributions (\\Fref{fig:abl}{H}), consistent with the interaction-dominated marker set reported next.",
     "Cumulative importance separates marginal from interaction contributions, consistent with the interaction-dominated marker set reported next (\\Fref{fig:abl}{H})."),
    # L256 - volcano + sign map
    ("The marker volcano plot (\\Fref{fig:interp}{H}) and the composition-rule sign map (\\Fref{fig:interp}{I}) summarise the direction and magnitude of every candidate marker.",
     "The marker volcano plot and the composition-rule sign map summarise the direction and magnitude of every candidate marker (\\Fref{fig:interp}{H}, \\Fref{fig:interp}{I})."),
    # L271 - advantage
    ("Because the prospective cohort here was produced by the original study's own optimisation loop and overlaps the internal adhesion range (\\Fref{fig:dataset}{I}), the measured advantage is a statement about composition transfer, not about extrapolation to a never-seen target scale.",
     "Because the prospective cohort here was produced by the original study's own optimisation loop and overlaps the internal adhesion range, the measured advantage is a statement about composition transfer, not about extrapolation to a never-seen target scale (\\Fref{fig:dataset}{I})."),
]

n_ok = 0
n_miss = 0
for old, new in reps:
    if old in src:
        src = src.replace(old, new, 1)
        n_ok += 1
    else:
        n_miss += 1
        # locate approximate position
        key = old[:60]
        idx = src.find(old[:40])
        print(f'MISS: {key[:70]}...  at {idx}')

open(P, 'w', encoding='utf-8').write(src)
print(f'\napplied {n_ok}/{len(reps)} (missed {n_miss})')
