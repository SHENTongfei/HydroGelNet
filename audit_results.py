"""Audit the current Results section."""
import re

with open(r"C:/Users/TS/WorkBuddy/HydroGelNet/paper/frontiers_SIMPLEX.tex", "r", encoding="utf-8") as f:
    tex = f.read()

i = tex.find(r"\section{Results}")
j = tex.find(r"\section{Discussion}", i)
results = tex[i:j]
print(f"Results section: {len(results.split())} words, {len(results)} chars")
fig_refs = re.findall(r"Fig\.~?\\ref", results)
print(f"Figure refs (Fig.N or Figure N): {len(fig_refs)}")
cite_count = len(re.findall(r"\\cite", results))
print(f"Cites in Results: {cite_count} (should be 0)")
tables = re.findall(r"\\begin\{table", results)
print(f"Tables: {len(tables)} (should be 3-5)")
subfig_refs = re.findall(r"\(Fig\.\s*[A-I]\)", results)
print(f"(Fig.A)-(Fig.I) subfigure refs: {len(subfig_refs)}")
print(f"  -- dashes: {results.count('--')}")
print(f"  : colons: {results.count(':')}")
print(f"  ; semicolons: {results.count(';')}")
# Count how many (sd) style stat reports
sd_reports = len(re.findall(r"\(\s*\d+\.\d+", results))
print(f"Stats in parens (NN): {sd_reports}")
