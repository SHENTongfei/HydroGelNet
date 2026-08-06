"""Fix citation blocks glued before \\subsection titles."""
import re
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
t = open(p, encoding="utf-8").read()

fixes = [
    ("}}\\subsection{Overall workflow and design rationale}",
     "}\n\n\\subsection{Overall workflow and design rationale}"),
    ("}}\\subsection{What does the deep model contribute?}",
     "}\n\n\\subsection{What does the deep model contribute?}"),
]
for old, new in fixes:
    if old in t:
        t = t.replace(old, new)
        print("fixed:", old[:50])
    else:
        print("NOT FOUND:", old[:50])

open(p, "w", encoding="utf-8").write(t)
print("cite count:", len(re.findall(r"\\cite\{", t)))
