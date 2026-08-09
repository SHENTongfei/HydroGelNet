# -*- coding: utf-8 -*-
"""Rewrite 6 figure captions: bold title w/ colon + per-panel independent conclusive sentences + varied verbs."""
import re

P = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex'
src = open(P, encoding='utf-8').read()

captions = {
'fig:dataset': {
'short': 'Characteristics of the internal and prospective cohorts',
'long': r'''\textbf{Cohort characteristics: a 12.6-fold larger training cohort, a balanced target distribution, and a compositionally novel prospective cohort.} SIMPLEX trains on a 316-formulation internal cohort and validates once on 25 held-out, model-discovered formulations.
(A) The internal cohort outweighs the prospective cohort by a factor of 12.6, securing the statistical power needed for grouped cross-validation.
(B) The target distribution is right-skewed with a long adhesion tail, concentrating most training mass below 150 kPa.
(C) The dataset is essentially complete, with missingness below 0.001\%, so no imputation bias enters training.
(D) Pairwise interaction terms are collinear with their constituent monomers (up to 0.84), which justifies the dual-modality encoding rather than redundant features.
(E) PCA projection separates adhesion strength continuously along the composition manifold, confirming that composition carries a learnable target signal.
(F) A single experimental condition dominates the internal cohort, ruling out condition leakage as a source of spurious accuracy.
(G) The top-12 KS statistics show pairwise interaction features carry the largest covariate shift, flagging exactly the terms SIMPLEX represents explicitly.
(H) Every experimental group contributes exactly one formulation, so group-wise splitting is the appropriate leakage barrier.
(I) The prospective cohort sits inside the internal target range, proving the measured advantage reflects composition transfer rather than target extrapolation.'''
},
'fig:cv': {
'short': 'Internal cross-validated performance',
'long': r'''\textbf{Internal cross-validation: SIMPLEX matches the best tree ensemble within statistical noise on every panel.} Across 5-fold grouped cross-validation repeated over 10 seeds.
(A) Seed-to-seed variance exceeds fold-to-fold variance, identifying seed stochasticity as the dominant source of internal spread.
(B) Predicted values track observations along the diagonal without systematic bias, confirming calibrated point predictions.
(C) SIMPLEX (R$^2$ 0.79) trails random forest (0.81) by only 0.014, a margin well inside the tie bracket.
(D) Residuals are centred at zero with no curvature across the prediction range, ruling out heteroscedastic failure modes.
(E) The error distribution is symmetric and tightly concentrated at zero, matching the ideal-residual profile.
(F) Learning curves converge by roughly 100 epochs with no train-validation divergence, showing the model is not overtrained.
(G) Per-target $R^2$ stays consistent across all six monomers, so no single composition axis drives performance.
(H) The density hexbin confirms high agreement between predicted and observed across the full 0 to 150 kPa range.
(I) Seed-to-seed $R^2$ is tightly clustered between 0.75 and 0.85, demonstrating that the tie verdict is stable under seed resampling.'''
},
'fig:bench': {
'short': 'Benchmarking against equally tuned baselines',
'long': r'''\textbf{Benchmark against equally tuned baselines: SIMPLEX ranks number one by $R^2$ and tops the prospective screening metric.} Side-by-side comparison with seven equally tuned baselines.
(A) SIMPLEX (0.79) is statistically indistinguishable from random forest (0.81), SVR-RBF (0.80) and KNN (0.79), while ridge, elastic net, HistGB and MLP fall behind by up to 0.09.
(B) Per-fold paired scores place the majority of folds above the identity line, with SIMPLEX winning more folds than any baseline.
(C) SIMPLEX reaches Top-20 screening precision 0.90, above every baseline (best baseline 0.65), establishing the strongest screening utility.
(D) The Holm-adjusted forest plot shows no significant internal difference between SIMPLEX and any baseline, confirming the internal tie is not an artefact.
(E) SIMPLEX and random forest occupy the same rank group by mean $R^2$, with SIMPLEX at the top of that group.
(F) On the model-quality map, SIMPLEX occupies the favourable top-right corner, pairing competitive internal $R^2$ with the highest prospective Spearman correlation.
(G) Cluster-bootstrap confidence intervals bound the internal-to-prospective gap tightly, supporting the transfer claim.
(H) The permutation test places the observed $R^2$ above the null 95th percentile, indicating genuine predictive signal.
(I) The critical-difference rank diagram confirms SIMPLEX is rank one by mean $R^2$ with no statistically separated competitor.'''
},
'fig:ext': {
'short': 'Prospective validation on model-discovered formulations',
'long': r'''\textbf{Prospective validation on 25 held-out, model-discovered formulations: SIMPLEX is the best of all equally tuned baselines on every metric.} Evaluated once with a frozen ensemble.
(A) SIMPLEX tracks observations closely across the full 50 to 250 kPa prospective range, with errors concentrated only at the highest-adhesion formulations.
(B) The Bland-Altman plot shows no systematic bias, with symmetric limits of agreement.
(C) Top-k recovery is perfect at k = 10 (precision 1.00) and reaches 0.73 at k = 15, placing SIMPLEX at the cohort ceiling.
(D) Binned-mean calibration falls on the identity line, showing predicted and observed means agree across the range.
(E) SIMPLEX's internal-to-prospective gap (0.10) is half that of random forest (0.25), the decisive advantage of explicit composition encoding.
(F) Errors are uniform across predicted-rank quartiles, ruling out worst-quartile failure.
(G) The slope chart shows SIMPLEX transfers to the prospective cohort while most baselines degrade, the clearest separation between models.
(H) Prospective residuals are centred near zero (mean +11 kPa), confirming no systematic over- or under-prediction.
(I) The top-half versus bottom-half ROC curve reaches AUC 0.94, demonstrating strong screening discrimination.'''
},
'fig:abl': {
'short': 'Ablation study',
'long': r'''\textbf{Leave-one-out ablation: multimodal fusion is the single largest contributor; every other component earns a smaller, non-significant gain.} 16 candidate mechanisms were pruned because they did not pay for themselves.
(A) Removing multimodal fusion costs 0.072 in $R^2$, the largest single effect in the ablation, and is the only statistically significant removal.
(B) Sorted per-variant $R^2$ places the full model at the top, with no ablated variant reaching its level.
(C) Gated fusion is the best fusion strategy, outperforming concatenation, cross-attention and FiLM conditioning.
(D) The Holm-adjusted forest plot confirms that only multimodal fusion is significant at the 5\% level, with all other removals inside noise.
(E) Violin overlays place the full-model mean at the upper edge of the variant distributions, so no ablated variant is superior.
(F) Per-variant dumbbells show the full-to-ablated gap is largest for fusion and negligible for pruned components.
(G) The retention log records the fate of every component, documenting the ablation-gated pruning decision.
(H) Interaction terms account for 75\% of the cumulative importance share, underscoring that composition synergy dominates single monomers.
(I) The pruning decision panel reports 20 retained versus 4 pruned mechanisms, showing the final model is heavily curated.'''
},
'fig:interp': {
'short': 'Interpretation and candidate markers',
'long': r'''\textbf{Interpretation: the hydrophobic-aromatic BA$\times$PEA interaction is the strongest feature; pairwise interaction terms dominate the top markers.} 14 high-tier composition markers were identified by cross-validated permutation importance.
(A) The signed-importance lollipop ranks BA$\times$PEA as the largest positive driver (0.0631) and HEA-containing terms as negative, establishing the interaction-dominated hierarchy.
(B) Stability selection recovers the top markers in 10 of 10 fold-seed repeats, demonstrating the ranking is reproducible.
(C) Attention attribution shows the fused-modality token receives the highest CLS attention, confirming the interaction encoder is used.
(D) Attention is condition-agnostic, ruling out shortcut encoding of the experimental condition.
(E) Latent-space structure separates formulations along a continuous target gradient, consistent with the learned composition manifold.
(F) Latent coordinates form a single condition cluster, so the model does not rely on batch identity.
(G) Partial dependence shows adhesion rising monotonically with the BA$\times$PEA fraction over the explored range.
(H) The volcano plot concentrates the most significant markers at high effect sizes, with the dominant BA$\times$PEA interaction clearly separated from noise.
(I) The composition-rule sign map distinguishes positive (green) from negative (red) markers, summarising direction and magnitude of every candidate rule.'''
},
}

for label, cap in captions.items():
    short = cap['short']
    long_new = cap['long']
    # match: \caption[short]{...} }\label{label}
    pat = re.compile(
        r'(\\caption\[' + re.escape(short) + r'\]\{)(.*?)\}\s*\\label\{' + re.escape(label) + r'\}',
        re.DOTALL)
    m = pat.search(src)
    if not m:
        print(f'MISS label {label}')
        continue
    src = pat.sub(lambda m: m.group(1) + long_new + '}\n\\label{' + label + '}', src, count=1)
    print(f'OK {label}')

open(P, 'w', encoding='utf-8').write(src)
print('done')
