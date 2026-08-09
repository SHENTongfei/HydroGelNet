import _runtime_guard  # noqa
import re

tex = open(r"C:/Users/TS/WorkBuddy/HydroGelNet/audit/rewrite_baseline/v7_merged.tex",
           encoding="utf-8").read()
body = re.sub(r"\\caption\[[^\]]*\]\{[^}]*\}", "", tex)

# Collect all referenced panels: parse "(Fig. 7A, 7D)" / "(Fig. 5A, 5F)" chains
mentioned = {}
pat = re.compile(r"\(Fig\.\s*((?:\d+[A-I][,\s]*)+)\)")
for m in pat.finditer(body):
    for n_str, L in re.findall(r"(\d+)([A-I])", m.group(1)):
        n = int(n_str)
        if 3 <= n <= 8:
            mentioned.setdefault(n, set()).add(L)

missing = []
for n in range(3, 9):
    got = mentioned.get(n, set())
    for L in "ABCDEFGHI":
        if L not in got:
            missing.append(f"Fig{n}{L}")
print("Mentioned panels:", {k: sorted(v) for k, v in sorted(mentioned.items())})
print("MISSING panel refs:", missing if missing
      else "NONE - all 54 panels (Figures 3-8 x A-I) referenced")