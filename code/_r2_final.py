# -*- coding: utf-8 -*-
"""R2 final: caption bold titles (user-mandated colon) + URL exemptions."""
import re

TEX = r"C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex"
tex = open(TEX, encoding="utf-8").read()
lines = tex.split("\n")
code_lines = [re.sub(r"(?<!\\)%.*$", "", ln).rstrip() for ln in lines]
body = "\n".join(code_lines)
masked = re.sub(r"\$[^$]*\$", "MATH", body)
masked = re.sub(r"\\\[.*?\\\]", "MATH", masked, flags=re.DOTALL)
allowed = re.compile(
    r"\\title\[[^\]]*\]\{[^}]*\}"
    r"|\\ref\{[^}]*\}"
    r"|\\label\{(sec|eq|fig|tab):[^}]*\}"
    r"|\\cite\{[^}]*\}"
    r"|\\Fref\{[^}]*\}\{[^}]*\}"
    r"|\\Tref\{[^}]*\}"
    r"|https?://[^\s)}]+"
    r"|\\subsection\{[^}]*\}"
    r"|\\textbf\{[^}]*:[^}]*\}"
)
masked = allowed.sub("ALLOWED", masked)
hits = []
for i, ln in enumerate(masked.split("\n"), 1):
    for m in re.finditer(":", ln):
        ctx = ln[max(0, m.start() - 30):m.end() + 25]
        hits.append((i, ctx.strip()))
print("prose colons after full exemption:", len(hits))
for h in hits:
    print("  ", h)
print("RESULT:", "PASS" if len(hits) == 0 else "FAIL")
