"""Move \\cite groups sitting right after \\section{...} titles into the
following paragraph (professional citation placement)."""
import re
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
t = open(p, encoding="utf-8").read()

# pattern: \section{Title} \cite{...} \cite{...} (possibly with trailing text)
pat = re.compile(r"(\\section\{[^}]+\})((?:\s*\\cite\{[^}]+\})+)([^\n]*)")
def repl(m):
    cites = re.findall(r"\\cite\{[^}]+\}", m.group(2))
    rest = re.sub(r"\\cite\{[^}]+\}", "", m.group(2)).strip()
    return m.group(1) + (f" {rest}" if rest else "") + " __CITES__" + " ".join(cites)

t2, n = pat.subn(repl, t)
# move __CITES__ tokens into the following paragraph
t3 = re.sub(r"(__CITES__(?: \\cite\{[^}]+\})*)", lambda m: "", t2)  # strip tokens first
# re-attach: find section markers followed by paragraph, append cites
# simpler: rebuild by replacing '\\section{X}\n\nFirst sentence' with cites appended
tokens = re.findall(r"__CITES__((?: \\cite\{[^}]+\})*)", t2)
body_wo = t3
# append the cite groups to the first paragraph of each affected section
out = body_wo
# find each affected section and append its cites to the paragraph after it
idx = 0
def append_after_section(m):
    nonlocal idx
    cite_block = ""
    if idx < len(tokens):
        cite_block = tokens[idx]
        idx += 1
    return m.group(1) + "\n\n" + m.group(2) + cite_block

# affected sections are Materials and Methods + Discussion (the ones with title cites)
res = re.sub(r"(\\section\{(Materials and Methods|Discussion)\})\n\n([^\n]*?\n)", append_after_section, body_wo)
open(p, "w", encoding="utf-8").write(res)
print("fixed title-cites:", n, "| tokens:", len(tokens))
bad = re.findall(r"\\section\{[^}]+\} \\\\cite", res)
print("remaining title-cites:", len(bad))
