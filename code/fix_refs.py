"""Replace NOT_FOUND citations with verified ones; strengthen finding 2."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
src = open(p, encoding="utf-8").read()

# 1) zhang2021adhesive (NOT FOUND) -> cai2021bioadhesives (verified, J Polym Sci 2021)
n1 = src.count("zhang2021adhesive")
src = src.replace("zhang2021adhesive", "cai2021bioadhesives")

# 2) zhou2015hydrogel (NOT FOUND) -> maier2015saltdisplacement (verified, Science 2015)
n2 = src.count("zhou2015hydrogel")
src = src.replace("zhou2015hydrogel", "maier2015saltdisplacement")

# 3) strengthen finding 2 (cationic electrostatics) with zhao2016polyelectrolyte
#    in the 3.6 citation group that contains lee2007mussel
old3 = ("\\cite{zhang2020catechol,cai2021bioadhesives,maier2015saltdisplacement,"
        "lee2007mussel,waite2017mussel,")
if old3 in src:
    src = src.replace(old3, "\\cite{zhang2020catechol,cai2021bioadhesives,"
                            "maier2015saltdisplacement,zhao2016polyelectrolyte,"
                            "lee2007mussel,waite2017mussel,")
    n3 = 1
else:
    # fallback: the discussion 4.2 group
    old3b = ("\\cite{zhang2020catechol,cai2021bioadhesives,maier2015saltdisplacement,"
             "lee2007mussel,waite2017mussel,gong2010doublenetwork")
    n3 = 0
    if old3b in src:
        src = src.replace(old3b, "\\cite{zhang2020catechol,cai2021bioadhesives,"
                                 "maier2015saltdisplacement,zhao2016polyelectrolyte,"
                                 "lee2007mussel,waite2017mussel,gong2010doublenetwork")
        n3 = 1

open(p, "w", encoding="utf-8").write(src)
print(f"replaced zhang2021adhesive x{n1}, zhou2015hydrogel x{n2}, zhao added x{n3}")
