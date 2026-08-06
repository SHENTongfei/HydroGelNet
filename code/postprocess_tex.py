"""Post-process simplex_frontiers.tex: strip section numbering, map figure
placeholders to real filenames, drop duplicated caption paragraphs."""
import re
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "simplex_frontiers.tex")
t = open(p, encoding="utf-8").read()

# 1) strip numbering: \section{2 Materials and Methods} -> \section{Materials and Methods}
t = re.sub(r"\\(section|subsection|subsubsection)\{(\d+(?:\.\d+)*) ([^}]*)\}",
           r"\\\1{\3}", t)

# 2) figure placeholders -> includegraphics with real filenames
FIGMAP = {1: "Figure1_pipeline", 2: "Figure2_architecture", 3: "Figure3_dataset",
          4: "Figure4_internal_cv", 5: "Figure5_benchmark", 6: "Figure6_external",
          7: "Figure7_ablation", 8: "Figure8_interpretation"}
def fig_repl(m):
    n = int(m.group(1))
    name = FIGMAP.get(n, f"Figure{n}")
    return (f"\\begin{{figure*}}[ht]\n\\centering\n"
            f"\\includegraphics[width=1\\textwidth]{{figures/{name}}}\n"
            f"\\label{{fig:{n}}}\n\\end{{figure*}}")
t = re.sub(r"\[\[FIG:Figure (\d+)\]\]", fig_repl, t)

# 3) drop standalone caption paragraphs ("**Figure N.** ...") right after a figure
def drop_caption(m):
    head = m.group(0).split("\\end{figure*}")[0] + "\\end{figure*}"
    return head
t = re.sub(r"\\begin\{figure\*\}\[ht\].*?\\end\{figure\*\}"
           r"(?:\n\n\\textbf\{Figure \d+\.\}.*?)(?=\n\n|\\section|\\subsection|\\end\{document\})",
           drop_caption, t, flags=re.S)

open(p, "w", encoding="utf-8").write(t)
print(f"post-processed: {len(t)} chars")
print("includegraphics:", len(re.findall(r"includegraphics", t)))
print("numbered sections left:", len(re.findall(r"\\(section|subsection)\{\\d", t)))
print("stray FIG placeholders:", len(re.findall(r"\[\[FIG", t)))
