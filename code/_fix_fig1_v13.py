# -*- coding: utf-8 -*-
"""Fix fig1: add helper aliases, panel letters A-E, correct v13 numbers."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\code\figures_v2_backup.py'
src = open(P, encoding='utf-8').read()

# 1) add aliases before fig1
old_alias = """# Figure 1 -- SIMPLEX pipeline (schematic)
# =========================================================================== #
def fig1(ctx: Ctx) -> None:"""
new_alias = """# Figure 1 -- SIMPLEX pipeline (schematic)
# =========================================================================== #
# aliases: fig1 body uses short names, definitions carry a _p_ prefix
_pbox = _p_pbox
_parrow = _p_parrow
_stage_label = __stage_label
_psave = _p_psave


def fig1(ctx: Ctx) -> None:"""
assert old_alias in src, "alias anchor not found"
src = src.replace(old_alias, new_alias, 1)

# 2) stage labels 1..5 -> A..E
for num, letter, name in [
    ('1 · Data', 'A · Data', None),
    ('2 · Training region', 'B · Training region', None),
    ('3 · SIMPLEX', 'C · SIMPLEX', None),
    ('4 · Extrapolation', 'D · Prospective validation', None),
    ('5 · Screening', 'E · Screening & insight', None),
]:
    src = src.replace(f'"{num}"', f'"{letter}"', 1)

# 3) correct numbers (old -> new)
reps = [
    # training region
    ('"Training set", "green", fs=7.5, bold=True,\n        sub="n = 180 · low-performance"',
     '"Training set", "green", fs=7.5, bold=True,\n        sub="n = 316 · internal cohort"'),
    ('"5 seeds · 25 models"', '"10 seeds · 50 models"'),
    # extrapolation
    ('"External cohort", "red", fs=7.5, bold=True,\n        sub="n = 161 · SMBO-discovered"',
     '"Prospective cohort", "red", fs=7.5, bold=True,\n        sub="n = 25 · model-discovered"'),
    ('"High-performance\\ncomposition region"', '"Held-out during\\nall tuning"'),
    ('"Target-value shift\\n(mean 47 → 154 kPa)"', '"High-adhesion region\\n(62–251 kPa)"'),
    # screening
    ('"Spearman ρ = 0.50\\nvs RF 0.21"', '"Spearman ρ = 0.87\\n(prospective)"'),
    ('"Top-20 precision 0.25\\nvs RF 0.10"', '"Top-20 precision 0.90\\n(prospective)"'),
]
n_ok = 0
for old, new in reps:
    if old in src:
        src = src.replace(old, new, 1)
        n_ok += 1
    else:
        print('MISS:', old[:50].replace('\n', '\\n'))

open(P, 'w', encoding='utf-8').write(src)
print(f'fig1 fixed: aliases + A-E letters + {n_ok}/{len(reps)} number updates')
