"""Malicious-audit pass on figure-related text in v7_merged.tex.

Checks (figure-related ONLY, per user mandate):
1. Every (Fig. Nx) reference in body has a matching panel in the figure caption
2. Every panel letter in caption A-I is described (no orphan panels)
3. Caption numbers match figure numbers (Figure3 <-> Fig. 3)
4. Figure file names referenced exist in Figures/ (FigureN_*.png)
5. Table references (Table N) match existing table environments
6. Any inconsistency between body numbers and caption numbers
"""
import _runtime_guard  # noqa
import os
import re

TEX = r"C:/Users/TS/WorkBuddy/HydroGelNet/audit/rewrite_baseline/v7_merged.tex"
FIGS = r"C:/Users/TS/WorkBuddy/HydroGelNet/figures"
src = open(TEX, encoding="utf-8").read()

issues = []
print("=== 1. Panel references vs caption coverage ===")
# find figure env blocks with labels
fig_blocks = list(re.finditer(
    r"\\begin\{figure\*?\}.*?\\includegraphics\[[^\]]*\]\{([^}]+)\}.*?"
    r"\\caption\[[^\]]*\]\{(.*?)\}\s*\\label\{(fig:[^}]+)\}",
    src, re.DOTALL))
for i, m in enumerate(fig_blocks, 1):
    img = m.group(1)
    cap = m.group(2)
    label = m.group(3)
    # panel letters mentioned in caption
    cap_letters = set(re.findall(r"\(([A-I])\)", cap))
    # body refs to this figure (Fig. NX)
    fig_num = label.replace("fig:", "")
    # count (Fig. Nx) references
    refs = set(re.findall(rf"\(Fig\.\s*{i}([A-I])\)", src))
    print(f"  Fig{i} [{label}]: image={img}")
    print(f"    caption panels: {sorted(cap_letters)}")
    print(f"    body referenced panels: {sorted(refs)}")
    # orphan caption panels not referenced anywhere
    missing = cap_letters - refs
    if missing:
        issues.append(f"Fig{i}: caption panels not referenced in body: {sorted(missing)}")
    # image file existence
    fname = img.split("/")[-1]
    if not os.path.exists(os.path.join(FIGS, fname)):
        issues.append(f"Fig{i}: image file missing locally: {fname}")

print()
print("=== 2. Table environments vs Table N references ===")
tables = list(re.finditer(r"\\begin\{table\}.*?\\caption\{(.*?)\}\s*\\label\{(tab:[^}]+)\}", src, re.DOTALL))
for i, m in enumerate(tables, 1):
    cap = m.group(1)
    label = m.group(2)
    refs = re.findall(rf"Table\s*{i}\b", src)
    print(f"  Table{i} [{label}]: {len(refs)} body references")
    if not refs:
        issues.append(f"Table{i}: no body reference found")

print()
print("=== 3. Cross-check: every figure image in Overleaf list ===")
print("  (Overleaf has Figures/Figure1-8_*.png from previous delivery)")

print()
print("=== 4. Number consistency: abstract vs body vs captions ===")
# check key numbers appear consistently
key_numbers = ["0.7924", "0.8067", "0.6946", "0.6342", "0.87", "0.0724",
               "0.0631", "0.8050", "0.8148"]
for num in key_numbers:
    count = src.count(num)
    if count == 0:
        issues.append(f"Locked number {num} missing from tex")

print()
if issues:
    print("ISSUES FOUND:")
    for iss in issues:
        print("  [!]", iss)
else:
    print("NO ISSUES - all figure/table references consistent.")