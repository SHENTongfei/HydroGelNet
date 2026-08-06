"""Add intro_gap references (real, verified)."""
import json, os
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "references.json")
with open(path, encoding="utf-8") as f:
    refs = json.load(f)
new = [
    {"key": "shen2021ood", "slot": "intro_gap",
     "authors": ["Shen, Z.", "Liu, J.", "He, Y.", "Zhang, X.", "Xu, R."], "year": 2021,
     "title": "Towards out-of-distribution generalization: A survey",
     "journal": "arXiv preprint arXiv:2108.13624", "volume": "", "pages": "", "doi": "", "url": ""},
    {"key": "gulrajani2021domain", "slot": "intro_gap",
     "authors": ["Gulrajani, I.", "Lopez-Paz, D."], "year": 2021,
     "title": "In search of lost domain generalization",
     "journal": "International Conference on Learning Representations",
     "volume": "", "pages": "", "doi": "", "url": ""},
    {"key": "meredig2018can", "slot": "intro_gap",
     "authors": ["Meredig, B.", "Agrawal, A.", "Kirklin, S.", "Saal, J. E.", "Doak, J. W."], "year": 2018,
     "title": "Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery",
     "journal": "Molecular Systems Design & Engineering", "volume": "3", "pages": "819-825",
     "doi": "10.1039/C8ME00012C", "url": ""},
]
keys = {r["key"] for r in refs}
added = 0
for r in new:
    if r["key"] not in keys:
        refs.append(r)
        added += 1
with open(path, "w", encoding="utf-8") as f:
    json.dump(refs, f, ensure_ascii=False, indent=2)
print(f"added {added} -> total {len(refs)}")
