"""Convert paper/SupplementaryMaterial.md -> paper/simplex_supplementary.tex
(Frontiers suppmat class, mirroring the TransMICRO supplement skeleton)."""
import re
import os

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")
with open(os.path.join(base, "SupplementaryMaterial.md"), encoding="utf-8") as f:
    md = f.read()

# strip the title line (first # line)
md = re.sub(r"^# .*?\n\n", "", md)

def md2tex(text):
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    text = re.sub(r"ρ", r"$\\rho$", text)
    text = re.sub(r"±", r"$\\pm$", text)
    text = re.sub(r"Δ", r"$\\Delta$", text)
    text = re.sub(r"→", r"$\\rightarrow$", text)
    text = text.replace("×", "$\\times$")
    return text

def blocks(text):
    """Convert markdown headings/tables/lists to LaTeX."""
    out = []
    lines = text.split("\n")
    in_item, in_tab = False, False
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^(#{1,4}) (.+)$", ln)
        if m:
            if in_item: out.append("\\end{itemize}"); in_item = False
            if in_tab: out.append("\\end{tabular}"); in_tab = False
            lvl = len(m.group(1))
            title = md2tex(m.group(2).strip())
            if lvl == 1:
                out.append(f"\\section{{{title}}}")
            elif lvl == 2:
                out.append(f"\\subsection{{{title}}}")
            else:
                out.append(f"\\subsubsection{{{title}}}")
            i += 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and lines[i+1].startswith("|---"):
            # table header
            if in_tab: out.append("\\end{tabular}"); in_tab = False
            header = [c.strip() for c in ln.strip("|").split("|")]
            out.append("\\begin{tabular}{" + "l" * len(header) + "}")
            out.append("\\toprule")
            out.append(" & ".join(md2tex(h) for h in header) + " \\\\")
            out.append("\\midrule")
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append(" & ".join(md2tex(c) for c in cells) + " \\\\")
                i += 1
            out.append("\\bottomrule")
            out.append("\\end{tabular}")
            out.append("")
            in_tab = False
            continue
        m = re.match(r"^\s*- (.+)$", ln)
        if m:
            if not in_item:
                out.append("\\begin{itemize}"); in_item = True
            out.append("\\item " + md2tex(m.group(1)))
            i += 1
            continue
        if in_item:
            out.append("\\end{itemize}"); in_item = False
        if ln.strip() == "":
            out.append("")
        else:
            out.append(md2tex(ln))
        i += 1
    if in_item: out.append("\\end{itemize}")
    if in_tab: out.append("\\end{tabular}")
    return "\n".join(out)

body = blocks(md)

tex = """\\documentclass[utf8]{frontiers_suppmat}
\\usepackage{url,hyperref,lineno,microtype}
\\usepackage[onehalfspacing]{setspace}
\\usepackage{booktabs}

\\begin{document}
\\onecolumn
\\firstpage{1}

\\title[Supplementary Material]{{\\helveticaitalic{Supplementary Material: Full Analysis of the SIMPLEX Case Study}}}

\\maketitle

""" + body + """

\\end{document}
"""

with open(os.path.join(base, "simplex_supplementary.tex"), "w", encoding="utf-8") as f:
    f.write(tex)
print(f"simplex_supplementary.tex written ({len(tex)} chars)")
