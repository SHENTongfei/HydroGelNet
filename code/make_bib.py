"""Convert paper/references.json -> paper/simplex.bib (BibTeX)."""
import json
import os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                    "references.json")
bib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "paper", "simplex.bib")

with open(path, encoding="utf-8") as f:
    refs = json.load(f)

def tex_escape(s: str) -> str:
    return (s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("$", "\\$"))

entries = []
for r in refs:
    key = r["key"]
    authors = " and ".join(r.get("authors") or [])
    title = tex_escape(r.get("title", ""))
    journal = tex_escape(r.get("journal", ""))
    year = r.get("year", "")
    vol = r.get("volume", "")
    pages = r.get("pages", "")
    doi = r.get("doi", "")
    url = r.get("url", "")
    # detect @article vs @misc (arXiv/ICLR/MLR without volume -> misc)
    if journal and (vol or pages or doi or year):
        entry = f"@article{{{key},\n  author = {{{authors}}},\n  title = {{{title}}},\n  journal = {{{journal}}},\n  year = {{{year}}},\n"
        if vol:
            entry += f"  volume = {{{vol}}},\n"
        if pages:
            entry += f"  pages = {{{pages}}},\n"
    else:
        entry = f"@misc{{{key},\n  author = {{{authors}}},\n  title = {{{title}}},\n  year = {{{year}}},\n"
        if journal:
            entry += f"  note = {{{journal}}},\n"
    if doi:
        entry += f"  doi = {{{doi}}},\n"
    if url:
        entry += f"  url = {{{url}}},\n"
    entry += "}\n"
    entries.append(entry)

with open(bib_path, "w", encoding="utf-8") as f:
    f.write("\n".join(entries))
print(f"simplex.bib written: {len(entries)} entries")
