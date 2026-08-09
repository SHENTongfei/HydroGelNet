# -*- coding: utf-8 -*-
"""Shorten SIMPLEX full name (both abstract L38 + body L52)."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex'
src = open(P, encoding='utf-8').read()

# --- abstract L38 (short version, no qualifier) ---
old_abs = ("Here we introduce SIMPLEX (Simplex composition encoding with "
           "Interaction-aware attention, Multi-modal fusion, Pretraining-ready "
           "regularisation, Learnable domain constraints and EXtrapolation "
           "evaluation), a dual-modality residual network")
new_abs = ("Here we introduce SIMPLEX (Simplex encoding with Interaction-aware "
           "attention, Multi-modal fusion, Pretraining, Learnable constraints "
           "and eXtrapolation), a dual-modality residual network")
assert old_abs in src, "abstract full name not found"
src = src.replace(old_abs, new_abs, 1)

# --- body L52 (with composition-space qualifier kept) ---
old_body = (r"Here we present \textbf{SIMPLEX} (Simplex composition encoding with Interaction-aware attention, Multi-modal fusion, Pretraining-ready regularisation, Learnable domain constraints and EXtrapolation evaluation; the extrapolation is in composition space, while the target range of the prospective cohort remains interpolative with respect to training), a dual-modality residual network with interaction-aware attention and small-data regularisation.")
new_body = (r"Here we present \textbf{SIMPLEX} (Simplex encoding with Interaction-aware attention, Multi-modal fusion, Pretraining, Learnable constraints and eXtrapolation; the extrapolation is in composition space, while the target range of the prospective cohort remains interpolative with respect to training), a dual-modality residual network with interaction-aware attention and small-data regularisation.")
assert old_body in src, "body full name not found"
src = src.replace(old_body, new_body, 1)

open(P, 'w', encoding='utf-8').write(src)
print("SIMPLEX full name shortened (abstract + body)")
