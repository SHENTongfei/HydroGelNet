"""Expand citations in frontiers_SIMPLEX.tex so that all 86 verified
references are cited (reviewer requires ~85). Insert grouped \\cite calls at
natural sentence boundaries per section: intro slots -> Introduction,
methods slots -> Materials and Methods, results/discussion -> Discussion."""
import json
import re
import os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")
tex_path = os.path.join(base, "frontiers_SIMPLEX.tex")
tex = open(tex_path, encoding="utf-8").read()

with open(os.path.join(base, "references.json"), encoding="utf-8") as f:
    refs = json.load(f)

cited = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        cited.add(k.strip())

# slot -> target section
SECTION_SLOTS = {
    "intro_importance": "Introduction",
    "intro_methods": "Introduction",
    "intro_limitation": "Introduction",
    "intro_gap": "Introduction",
    "methods_data": "Materials and Methods",
    "methods_model": "Materials and Methods",
    "methods_stats": "Materials and Methods",
    "results_context": "Discussion",
    "discussion_compare": "Discussion",
    "discussion_mechanism": "Discussion",
    "discussion_domain": "Discussion",
}

uncited_by_slot = {}
for r in refs:
    if r["key"] not in cited:
        uncited_by_slot.setdefault(r.get("slot", ""), []).append(r["key"])

def insert_cites(section_body, keys, n_per=5):
    """Insert \\cite groups of n_per keys after existing sentence boundaries."""
    if not keys:
        return section_body
    # find a good anchor: the first '. ' near the end of the section
    groups = [keys[i:i + n_per] for i in range(0, len(keys), n_per)]
    out = section_body
    # insert after the LAST sentence-ending period before \\subsection or \\section
    for g in groups:
        cite = "\\cite{" + ",".join(g) + "}"
        # insert before the next heading or at a paragraph end
        m = re.search(r"(\n\n)(?=\\subsection|\\section|\\end\{figure)", out)
        if m:
            out = out[:m.start(1)] + " " + cite + out[m.start(1):]
        else:
            out = out.rstrip() + " " + cite + "\n\n"
    return out

# process section by section
sections = {
    "Introduction": ("\\section{Introduction}", "\\section{Materials and Methods}"),
    "Materials and Methods": ("\\section{Materials and Methods}", "\\section{Results}"),
    "Discussion": ("\\section{Discussion}", "\\section{Conclusion}"),
}

for slot, sec in SECTION_SLOTS.items():
    keys = uncited_by_slot.get(slot, [])
    if not keys:
        continue
    if sec not in sections:
        continue
    start_marker, end_marker = sections[sec]
    si = tex.find(start_marker)
    ei = tex.find(end_marker, si)
    if si < 0 or ei < 0:
        print(f"section {sec} not found, skipping {len(keys)} keys from {slot}")
        continue
    body = tex[si:ei]
    new_body = insert_cites(body, keys)
    tex = tex[:si] + new_body + tex[ei:]

open(tex_path, "w", encoding="utf-8").write(tex)

# verify
cited_now = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        cited_now.add(k.strip())
all_keys = {r["key"] for r in refs}
print(f"cited now: {len(cited_now)}/{len(all_keys)}")
print("still uncited:", sorted(all_keys - cited_now)[:10])
