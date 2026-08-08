"""Map current tex structure for the v7 rewrite."""
import re
import os

tex = open(r"C:\Users\TS\WorkBuddy\HydroGelNet\paper\frontiers_SIMPLEX.tex",
           encoding="utf-8").read()

print("=== sections ===")
for m in re.finditer(r"\\(?:sub)*section\*?\{(.*?)\}", tex):
    print(" ", m.group(1))

print("\n=== tables (captions) ===")
for m in re.finditer(r"\\caption\{(.*?)\}", tex, re.DOTALL):
    cap = " ".join(m.group(1).split())[:110]
    print(" -", cap)

print("\n=== figures (labels) ===")
for m in re.finditer(r"\\label\{(fig:[a-z_]+)\}", tex):
    print(" -", m.group(1))

i = tex.find("\\section{Results}")
j = tex.find("\\section{Discussion}", i)
res = tex[i:j] if i >= 0 and j > i else ""
print(f"\n=== Results section: {len(res.split())} words ===")
print("cites in Results:", len(re.findall(r"\\cite\{", res)))
print("dashes '--' in Results:", res.count("--"))
print("colons ':' in Results:", res.count(":"))
print("Fig refs (Figure~\\ref):", len(re.findall(r"Figure~\w*ref", res)))
