"""Generate SIMPLEX Frontiers LaTeX deliverables:
1. reference_final.bib  (from paper/references.json, 64 verified refs)
2. copies figures to Fig1..Fig8.png naming for the tex
"""
import json, os, shutil, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # HydroGelNet
PAPER = os.path.join(ROOT, "paper")
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "_frontiers")
os.makedirs(OUT, exist_ok=True)

# ---------- 1) references.json -> .bib ----------
with open(os.path.join(PAPER, "references.json"), encoding="utf-8") as f:
    refs = json.load(f)

def bib_key(r):
    return r["key"]

def clean(s):
    return re.sub(r"[{}]", "", str(s))

def bib_entry(r):
    key = bib_key(r)
    authors = " and ".join(a.split(", ")[0] + ", " + " ".join(a.split(", ")[1:])
                          for a in r.get("authors", []))
    fields = []
    fields.append(f"  author = {{{authors}}}")
    fields.append(f"  title = {{{clean(r['title'])}}}")
    fields.append(f"  year = {{{r['year']}}}")
    if r.get("journal"):
        fields.append(f"  journal = {{{clean(r['journal'])}}}")
    elif r.get("url") or "arXiv" in str(r.get("title", "")):
        fields.append(f"  journal = {{{clean(r.get('journal') or 'arXiv preprint')}}}")
    if r.get("volume"):
        fields.append(f"  volume = {{{r['volume']}}}")
    if r.get("pages"):
        fields.append(f"  pages = {{{r['pages']}}}")
    if r.get("doi"):
        fields.append(f"  doi = {{{r['doi']}}}")
    if r.get("url"):
        fields.append(f"  url = {{{r['url']}}}")
    return f"@{'article' if r.get('journal') else 'misc'}{{{key},\n" + ",\n".join(fields) + "\n}\n"

bib = "\n".join(bib_entry(r) for r in refs)
with open(os.path.join(OUT, "reference_final.bib"), "w", encoding="utf-8") as f:
    f.write(bib)
print(f"bib written: {len(refs)} entries -> {OUT}/reference_final.bib")

# ---------- 2) figures -> Fig1..Fig8.png ----------
map_fig = {
    "Figure1_pipeline.png": "Fig1_pipeline.png",
    "Figure2_architecture.png": "Fig2_architecture.png",
    "Figure3_dataset.png": "Fig3_dataset.png",
    "Figure4_internal_cv.png": "Fig4_internal_cv.png",
    "Figure5_benchmark.png": "Fig5_benchmark.png",
    "Figure6_external.png": "Fig6_external.png",
    "Figure7_ablation.png": "Fig7_ablation.png",
    "Figure8_interpretation.png": "Fig8_interpretation.png",
}
figdir = os.path.join(OUT, "Figures")
os.makedirs(figdir, exist_ok=True)
for src_name, dst_name in map_fig.items():
    s = os.path.join(FIG, src_name)
    d = os.path.join(figdir, dst_name)
    if os.path.exists(s):
        shutil.copy2(s, d)
        print(f"  copied {src_name} -> Figures/{dst_name}")
    else:
        print(f"  MISSING {src_name}")

print("deliverable figures staged in", figdir)
