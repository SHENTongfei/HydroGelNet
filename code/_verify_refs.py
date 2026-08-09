import _runtime_guard  # noqa
import re

tex = open(r"C:/Users/TS/WorkBuddy/HydroGelNet/audit/rewrite_baseline/v7_merged.tex",
           encoding="utf-8").read()
bib = open(r"C:/Users/TS/WorkBuddy/HydroGelNet/paper/reference_final.bib",
           encoding="utf-8").read()
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cites = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        cites.add(k.strip())
dangling = cites - bib_keys
print(f"cited: {len(cites)}, bib: {len(bib_keys)}, dangling: {sorted(dangling)}")
print("all cited keys resolve:", len(dangling) == 0)
print("dashes:", tex.count("\u2014") + tex.count("\u2013") + tex.count("---"))
print("begin/end:", tex.count("begin{document}"), tex.count("end{document}"))
print("captions:", tex.count("caption["))