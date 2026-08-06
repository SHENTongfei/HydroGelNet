"""Move \\cite groups directly after \\section{...} titles into the first
paragraph of that section (professional placement)."""
import re
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
t = open(p, encoding="utf-8").read()

pattern = re.compile(r"(\\section\{[^}]+\})\s+((?:\\cite\{[^}]+\}\s*)+)")
def do(m):
    return m.group(1) + " __TITLECITES__{" + m.group(2).strip() + "}"
t2 = pattern.sub(do, t)

out = t2
while "__TITLECITES__" in out:
    i = out.find("__TITLECITES__")
    j = out.find("{", i)
    k = out.find("}", j)
    cites = out[j + 1:k]
    head = out[:i] + "\n"
    rest = out[k + 1:]
    m2 = re.search(r"\n\n([^\n]*?\n)", rest)
    if m2:
        rest = (rest[:m2.start(1)] + "\n\n" + m2.group(1).rstrip("\n")
                + " " + cites + "\n" + rest[m2.end(1):])
    else:
        rest = cites + "\n" + rest
    out = head + rest

open(p, "w", encoding="utf-8").write(out)
left = len(re.findall(r"\\section\{[^}]+\}\s+\\cite", out))
print("title-cites remaining:", left)
