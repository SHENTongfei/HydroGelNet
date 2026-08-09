import _runtime_guard  # noqa
import re

cloud = open(r"C:/Users/TS/WorkBuddy/HydroGelNet/results/overleaf_tex_check.tex",
             encoding="utf-8").read()
cites = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", cloud):
    for k in m.group(1).split(","):
        cites.add(k.strip())
print(f"Cloud tex cited keys: {len(cites)}")
recent = sorted(k for k in cites if k.startswith("recent"))
print(f"New 'recent' keys cited: {len(recent)}")
print(recent)