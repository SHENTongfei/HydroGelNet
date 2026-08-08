import _runtime_guard  # noqa
import re

p = r"C:\Users\TS\WorkBuddy\HydroGelNet\paper\frontiers_SupplementaryMaterial.tex"
src = open(p, encoding="utf-8").read()

# Update prospective/Top-K numbers to v7 (locked delivery values)
replacements = [
    ("0.710", "0.6946"),       # any remaining 0.710 -> 0.6946
    ("0.71", "0.69"),          # remaining 0.71 -> 0.69  (broad replace; most safe)
]
# But "0.713" (ablation protocol mean) and "0.95" (Top-20 col) must NOT be touched.
# Strategy: replace only in known-good contexts.

# Find every 0.71 occurrence and inspect context
lines = src.split("\n")
out = []
for ln in lines:
    new_ln = ln
    # SIMPLEX row in Table 3 of supple (line "0.71 & 0.87 & 0.94 & 1.00 & 0.95") -> 0.69 & 0.87 & 0.94 & 1.00 & 0.90
    if "\\textbf{SIMPLEX} & \\textbf{0.71}" in ln:
        new_ln = ln.replace("\\textbf{0.71}", "\\textbf{0.69}").replace("\\textbf{0.95}", "\\textbf{0.90}")
    # SVR-RBF row: was 0.71, actual 0.63
    if "SVR-RBF & 0.71" in ln:
        new_ln = ln.replace("SVR-RBF & 0.71", "SVR-RBF & 0.63")
    # generalisation gap text
    new_ln = new_ln.replace("0.79 \\rightarrow 0.71", "0.79 \\rightarrow 0.69")
    # confidence CI row
    new_ln = re.sub(r"R\^2 & 0\.710 & ", "R^2 & 0.6946 & ", new_ln)
    # G4 prospective statement
    new_ln = new_ln.replace(
        "G4 prospective $R^2 = 0.710$",
        "G4 prospective $R^2 = 0.6946$")
    # final cross-check: any stray 0.71 that is NOT 0.713/0.714/etc.
    out.append(new_ln)

new_src = "\n".join(out)
open(p, "w", encoding="utf-8").write(new_src)

# verify
remaining = []
for ln in new_src.split("\n"):
    if "0.71" in ln and "0.713" not in ln and "0.714" not in ln and "0.715" not in ln:
        remaining.append(ln.strip()[:120])
print(f"Remaining 0.71 lines (should only be 0.713 ablation protocol):")
for r in remaining:
    print(" ", r)