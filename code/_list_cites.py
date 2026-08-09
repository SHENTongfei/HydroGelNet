import _runtime_guard  # noqa
import re

tex = open(r"C:/Users/TS/WorkBuddy/HydroGelNet/audit/rewrite_baseline/v7_merged.tex",
           encoding="utf-8").read()
for i, m in enumerate(re.finditer(r"\\cite\{([^}]+)\}", tex)):
    line = tex[:m.start()].count("\n") + 1
    print(f"L{line}: cite{{{m.group(1)}}}")