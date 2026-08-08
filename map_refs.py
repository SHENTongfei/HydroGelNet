"""Map tex figure refs and citation completeness for v7 rewrite."""
import re
import os

ROOT = r"C:\Users\TS\WorkBuddy\HydroGelNet"
tex = open(os.path.join(ROOT, "paper", "frontiers_SIMPLEX.tex"),
           encoding="utf-8").read()

refs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
print("Tex references", len(refs), "figure files:")
for r in refs:
    print(" -", r)

cites = re.findall(r"\\cite\{([^}]+)\}", tex)
all_keys = set()
for c in cites:
    for k in c.split(","):
        k = k.strip()
        if k:
            all_keys.add(k)
print(f"\nCited keys: {len(all_keys)}")

bib = open(os.path.join(ROOT, "paper", "reference_final.bib"),
           encoding="utf-8").read()
missing = [k for k in all_keys if k not in bib]
print("Keys missing from bib:", missing if missing else "NONE (all present)")

# dashes / colons in whole tex (H29 target)
print("\nWhole-tex dash count:", tex.count("--"))
print("Whole-tex colon count:", tex.count(":"))
