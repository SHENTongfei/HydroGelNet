"""Convert SIMPLEX manuscript.md -> Frontiers LaTeX (frontiers_new.tex).
Keeps the TransMICRO LaTeX skeleton (authors, contributions, funding,
conflict, data-availability, Frontiers-Harvard bibliography).
"""
import json
import re
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")

# ---------------- reference mapping: author-year -> bib key ----------------
with open(os.path.join(BASE, "references.json"), encoding="utf-8") as f:
    REFS = json.load(f)
AUTH_YEAR = {}
for r in REFS:
    first = (r.get("authors") or ["?"])[0].split(",")[0].strip()
    key = f"{first}, {r['year']}"
    AUTH_YEAR[key] = r["key"]
    AUTH_YEAR[f"{first} et al., {r['year']}"] = r["key"]
    if len(r.get("authors", [])) == 2:
        second = r["authors"][1].split(",")[0].strip()
        AUTH_YEAR[f"{first} and {second}, {r['year']}"] = r["key"]

def md2latex(text: str) -> str:
    """Basic markdown -> LaTeX."""
    # figure blocks: ![Figure N](path) -> keep placeholder, insert later
    text = re.sub(r"!\[(Figure \d+)\]\([^)]*\)", r"[[FIG:\1]]", text)
    # citations: (Author, Year; Author, Year) -> \cite{key1,key2}
    def cite_repl(m):
        inner = m.group(1)
        keys = []
        for part in inner.split(";"):
            part = part.strip().strip(".")
            for k, v in AUTH_YEAR.items():
                if part.startswith(k) or part == k:
                    keys.append(v)
                    break
        if not keys:
            return f"({inner})"
        return "\\cite{" + ",".join(dict.fromkeys(keys)) + "}"
    text = re.sub(r"\(([^()]*?\d{4}[^()]*?)\)", cite_repl, text)
    # bold / italic
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\textit{\1}", text)
    # R^2 -> R$^2$
    text = text.replace("R^2^", "R$^2$").replace("R²", "R$^2$")
    # unicode arrows etc
    text = text.replace("→", "$\\rightarrow$")
    # bullets
    lines = []
    for ln in text.split("\n"):
        m = re.match(r"^(\s*)- (.+)$", ln)
        if m:
            lines.append("\\begin{itemize}\n\\item " + m.group(2) + "\n\\end{itemize}" if False else "\\item " + m.group(2))
            continue
        if ln.strip() == "":
            lines.append("")
            continue
        lines.append(ln)
    # join: separate paragraphs by blank lines handled by caller
    return "\n".join(lines)

def section_split(text: str):
    """Yield (level, title, body) for headings."""
    parts = re.split(r"^(#{2,4}) (.+)$", text, flags=re.M)
    return parts

with open(os.path.join(BASE, "manuscript.md"), encoding="utf-8") as f:
    MD = f.read()

# extract body from '## 1 Introduction' to '## Data Availability Statement'
body_start = MD.index("## 1 Introduction")
body_end = MD.index("## Data Availability Statement")
body = MD[body_start:body_end]

# extract abstract
abs_start = MD.index("## Abstract") + len("## Abstract")
abs_text = MD[abs_start:body_start].strip()
abs_text = re.sub(r"\*\*Background:\*\*\s*", "", abs_text)
abs_text = re.sub(r"\*\*Objective:\*\*\s*", "", abs_text)
abs_text = re.sub(r"\*\*Methods:\*\*\s*", "", abs_text)
abs_text = re.sub(r"\*\*Results:\*\*\s*", "", abs_text)
abs_text = re.sub(r"\*\*Conclusion:\*\*\s*", "", abs_text)
abs_text = re.sub(r"\*\*Keywords:\*\*\s*", "", abs_text)
abs_text = re.sub(r"\s+", " ", abs_text).strip()

# tail blocks from manuscript
def tail_block(md, title):
    m = re.search(rf"## {title}\n(.*?)(?=\n## |\Z)", md, re.S)
    return m.group(1).strip() if m else ""

data_avail = tail_block(MD, "Data Availability Statement")
author_contrib = tail_block(MD, "Author Contributions")
funding = tail_block(MD, "Funding")
ack = tail_block(MD, "Acknowledgements")
conflict = tail_block(MD, "Conflict of Interest")

# ---- assemble ----
lines = []
lines.append("""\\documentclass[utf8]{FrontiersinHarvard}
\\usepackage{url,hyperref,lineno,microtype,subcaption}
\\usepackage[onehalfspacing]{setspace}
\\usepackage{booktabs}
\\usepackage{multirow}
\\linenumbers

\\def\\keyFont{\\fontsize{8}{11}\\helveticabold }
\\def\\firstAuthorLast{Shen, Lei, Hung, Li, Pan, Huo and Fu {et~al.}}
\\def\\Authors{Tongfei Shen\\,$^{1,2\\dagger}$, Xueqin Lei\\,$^{3\\dagger}$, Hung Ka Lok\\,$^{4\\dagger}$, Huaicheng Li\\,$^{5}$, Zhongze Pan\\,$^{6}$, Miaozhe Huo\\,$^{2,*}$ and Xuepeng Fu\\,$^{7,*,§}$}
\\def\\Address{$^{1}$School of Information Science and Engineering, Qingdao Huanghai University, Qingdao, China \\\\
$^{2}$Department of Computer Science, City University of Hong Kong, Kowloon, China\\\\
$^{3}$College of Physics and Electronic Information Engineering, Zhejiang Normal University, Jinhua 321004, China\\\\
$^{4}$The Centre for Innovation and Entrepreneurship, The Hang Seng University of Hong Kong, Hong Kong, China\\\\
$^{5}$Ability R\\&D Energy Research Centre, School of Energy and Environment, City University of Hong Kong, Kowloon, Hong Kong, China\\\\
$^{6}$School of Engineering, Nanfang College Guangzhou, Guangzhou, China\\\\
$^{7}$Department of Life Science and Agroforestry, Qiqihar University, Qiqihar 161000, China}
\\def\\corrAuthor{Miaozhe Huo and Xuepeng Fu}
\\def\\corrAddress{Department of Computer Science, City University of Hong Kong, Kowloon, China; Department of Life Science and Agroforestry, Qiqihar University, Qiqihar 161000, China}
\\def\\corrEmail{miaozhhuo2-c@my.cityu.edu.hk; 02383@qqhru.edu.cn}

\\begin{document}
\\onecolumn
\\firstpage{1}

\\title[SIMPLEX: composition-space deep learning for hydrogel adhesion]{SIMPLEX: Composition-Space Deep Learning for Hydrogel Adhesion with Out-of-Distribution Extrapolation to Model-Discovered High-Performance Formulations}

\\author[\\firstAuthorLast ]{\\Authors}
\\address{}
\\correspondance{}
\\extraAuth{}
\\maketitle

\\begin{abstract}
\\section{}
""" + abs_text + """

\\tiny
 \\keyFont{ \\section{Keywords} Hydrogel, Machine learning, Composition-to-property prediction, Out-of-distribution extrapolation, Material screening, Adhesion strength}
\\end{abstract}
""")

# body: convert headings and paragraphs
paras = re.split(r"\n\n+", body)
para_lines = []
for p in paras:
    p = p.strip()
    if not p:
        continue
    m = re.match(r"^(#{2,4}) (.+)$", p, re.S)
    if m:
        level, title = m.group(1), m.group(2).strip()
        cmd = {2: "\\section", 3: "\\subsection", 4: "\\subsubsection"}[len(level)]
        para_lines.append(f"\n{cmd}{{{title}}}")
        continue
    # figure placeholder
    figm = re.match(r"^\[\[FIG:(Figure \d+)\]\]$", p)
    if figm:
        n = figm.group(1).split()[-1]
        para_lines.append(f"\\begin{{figure}}[ht]\n\\centering\n\\includegraphics[width=1\\textwidth]{{figures/Figure{n}}}\n\\caption{{Figure {n} caption.}}\n\\label{{fig:{n}}}\n\\end{{figure}}")
        continue
    conv = md2latex(p)
    para_lines.append(conv)
lines.append("\n\n".join(para_lines))

# tail blocks (LaTeX-ify headings)
lines.append("""
\\section*{Conflict of Interest Statement}
""" + md2latex(conflict) + """

\\section*{Author Contributions}
""" + md2latex(author_contrib) + """

\\section*{Funding}
""" + md2latex(funding) + """

\\section*{Acknowledgments}
""" + md2latex(ack) + """

\\section*{Data Availability Statement}
The raw experimental dataset analysed in this study is publicly available from the repository accompanying Liao et al. (Nature 644, 89--95, 2025; DOI 10.1038/s41586-025-09269-4, MIT licence). The training code, the processed tensors, the per-formulation predictions underlying every reported table and figure, and the regeneration scripts are all available from the GitHub repository \\url{https://github.com/SHENTongfei/HydroGelNet}. No additional materials were generated.

\\phantomsection
\\bibliographystyle{Frontiers-Harvard}
\\bibliography{simplex}

\\end{document}
""")

out = "\n".join(lines)
with open(os.path.join(BASE, "simplex_frontiers.tex"), "w", encoding="utf-8") as f:
    f.write(out)
print(f"simplex_frontiers.tex written ({len(out)} chars)")
print("citations found:", len(re.findall(r"\\\\cite\{", out)))
