"""Update FIG_CAPTIONS[1]/[2] in paper_pdf.py to match new pipeline/architecture."""
import ast

src = open("paper_pdf.py", encoding="utf-8").read()
old_start = src.index("FIG_CAPTIONS = {")
old_end = src.index('    3: ("Characteristics')

new_cap = """FIG_CAPTIONS = {
    1: ("Schematic workflow of the SIMPLEX framework. "
        "(1) A public dataset of 341 hydrogel formulations (Nature 2025, MIT "
        "licence; six functional monomers on the composition simplex) is split "
        "into a training region of 180 low-performance formulations and an "
        "external region of 161 formulations discovered by the original "
        "study's sequential model-based optimisation loop. (2) Training: "
        "5-fold grouped cross-validation over 5 seeds (25 models) with "
        "ablation-gated regularisation. (3) SIMPLEX, a dual-modality encoder "
        "with interaction attention. (4) Extrapolation: the 161 "
        "model-discovered high-performance formulations are evaluated once, "
        "after all hyper-parameters are frozen (target-value shift, mean 47 "
        "to 154 kPa). (5) Screening and insight: external ranking (Spearman "
        "rho 0.50 vs 0.21 for random forest), top-k precision, and "
        "permutation-importance analysis of composition synergy."),
    2: ("Architecture of {model}. The six monomer molar fractions "
        "(composition simplex) are encoded through two modalities: modality 1 "
        "uses the raw fractions, modality 2 adds the 15 explicit pairwise "
        "interaction terms. A linear embedding (d=64) maps both modalities "
        "into the shared representation, which is refined by two residual "
        "blocks separated by an interaction self-attention layer (4 heads); a "
        "pooling + output head predicts the non-negative adhesion strength "
        "(kPa). Small-data regularisation (bottom band): Mixup input "
        "interpolation, stochastic weight averaging, a range-domain "
        "constraint and early stopping on an inner validation split."),
"""
src = src[:old_start] + new_cap + src[old_end:]
open("paper_pdf.py", "w", encoding="utf-8").write(src)
try:
    ast.parse(src)
    print("captions updated, syntax OK")
except SyntaxError as e:
    print(f"line {e.lineno}: {e.msg}")
