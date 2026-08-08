"""Audit v7_merged.tex: extract every "(Fig. Nx[,Ny]...)" reference and
verify all panel letters A-I for Figures 3-8 are mentioned.
"""
import _runtime_guard  # noqa
import re

P = r"C:/Users/TS/WorkBuddy/HydroGelNet/audit/rewrite_baseline/v7_merged.tex"
src = open(P, encoding="utf-8").read()
body = re.sub(r"\\caption\[[^\]]*\]\{[^}]*\}", "", src)
body = re.sub(r"\\label\{[^}]*\}", "", body)

# Find every (Fig. Nx, Ny, ...) reference; each "Ni" is N then i
pat = re.compile(r"\(Fig\.\s*((?:\d+[A-I][,\s]*)+)\)")
mentions = {}
for m in pat.finditer(body):
    # each token like "7A" or "7A,7D" or "7A 7B" -> split
    tokens = re.findall(r"(\d+)([A-I])", m.group(1))
    for n_str, letter in tokens:
        n = int(n_str)
        if 3 <= n <= 8:
            mentions.setdefault(n, set()).add(letter)

# Also accept "Fig. Nx" without parens
pat2 = re.compile(r"\bFig\.\s*(\d+)([A-I])\b")
for m in pat2.finditer(body):
    n = int(m.group(1))
    if 3 <= n <= 8:
        mentions.setdefault(n, set()).add(m.group(2))

print("=== Per-figure panel-letter coverage ===")
missing_total = 0
for n in range(3, 9):
    got = sorted(mentions.get(n, set()))
    expected = sorted(set("ABCDEFGHI"))
    missing = sorted(set(expected) - set(got))
    if missing:
        print(f"  Figure{n}: present={got}  MISSING={missing}")
        missing_total += len(missing)
    else:
        print(f"  Figure{n}: all panels A-I referenced ({len(got)}/{len(expected)})")

print()
print("=== Per-table coverage ===")
for n in range(1, 5):
    pat = re.compile(rf"\bTable\s*{n}\b")
    n_hits = len(pat.findall(body))
    print(f"  Table {n}: {n_hits} mentions")

print(f"\nTOTAL missing sub-panel references: {missing_total}")