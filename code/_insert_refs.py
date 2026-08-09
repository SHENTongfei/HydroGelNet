"""Insert real new references into matching cite groups in v7_merged.tex.
Semantic mapping: each new bib key -> the cite-group (identified by an anchor
existing key) where it belongs. This makes the bibliography actually contain
100+ entries in the compiled PDF.
"""
import _runtime_guard  # noqa
import re

TEX = r"C:/Users/TS/WorkBuddy/HydroGelNet/audit/rewrite_baseline/v7_merged.tex"
src = open(TEX, encoding="utf-8").read()

# Map: anchor existing key in a cite group -> list of new keys to append there
INSERT = {
    # Introduction: hydrogels / biohybrid interfaces
    "peppas2000hydrogels": [
        "recent2021pei",          # polymer hydrogel bioadhesives review
        "recent2021herrmann",     # hydrogels in biosensing
        "recent2024li",           # hydrogel flexible strain sensors
        "recent2025segneanu",     # advancements in hydrogels review
    ],
    # Introduction: wet adhesion / bio-inspired
    "cui2021wetadhesion": [
        "recent2021song",         # wet soft bio-adhesion insect-inspired
        "recent2021xue",          # hydrogel tapes strong wet adhesion
        "recent2023lee",          # bioinspired materials underwater adhesion
        "recent2022cheng",        # ultrastrong underwater adhesion non-canonical
        "recent2021zhang",        # sandcastle worm-inspired wet hydrogels
        "recent2023linghu",       # adhesion paradox rough surfaces
    ],
    # Introduction: materials ML / datadriven
    "ramprasad2017machine": [
        "recent2022zhong",        # explainable ML materials science
        "recent2021tao",          # ML perovskite materials
        "recent2023damewood",     # representations of materials for ML
        "recent2022choudhary",    # deep learning methods in materials
        "recent2023heid",         # Chemprop ML package
    ],
    # Introduction: deep tabular small data
    "zhang2018mixup": [
        "recent2025hollmann",     # tabular foundation model small data
        "recent2023dou",          # ML methods small data molecular science
        "recent2023xu",           # small data ML in materials science
    ],
    # Methods: physics-informed / SciML
    "shahriari2016bayesian": [
        "recent2021karniadakis",  # physics-informed ML (Karniadakis)
        "recent2022cuomo",        # SciML PINNs review
    ],
    # Discussion: underwater adhesion chemistry
    "cai2021bioadhesives": [
        "recent2022tan",          # soft self-adhesive conductive polymer
        "recent2021zhu",          # mussel-inspired wet-adhesion hydrogel
        "recent2024panb",         # silk fibroin hydrogel adhesive
        "recent2022yangb",        # biomimetic wet-tissue adhesive
        "recent2021ma",           # ultra-strong bio-glue polypeptide
    ],
    # Discussion: small-data ML materials
    "kim2023smallML": [
        "recent2022rao",          # ML high-entropy alloy discovery Science
        "recent2021mishin",       # ML interatomic potentials
        "recent2025li",           # ML mechanical properties hydrogels
        "recent2022shokrollahi",  # FEM-ML mechanical prediction
        "recent2023kibrete",      # AI predicting mechanical properties composites
    ],
    # Discussion: screening / active learning
    "chen2022activelearning": [
        "recent2023pyun",         # machine-learned wearable sensors
        "recent2025xue",          # conductive hydrogels ML-assisted
        "recent2024wang",         # MXene skin-like hydrogel sensor ML
    ],
}


def insert_into_group(src, anchor, new_keys):
    """Append new_keys to the cite group containing anchor."""
    pat = re.compile(r"\\cite\{([^}]*" + re.escape(anchor) + r"[^}]*)\}")
    m = pat.search(src)
    if not m:
        print(f"  WARNING: cite group with {anchor} not found")
        return src
    existing = m.group(1)
    # avoid duplicates
    add = [k for k in new_keys if k not in existing]
    if not add:
        print(f"  SKIP {anchor}: all keys already present")
        return src
    new_group = existing + "," + ",".join(add)
    return src[:m.start()] + "\\cite{" + new_group + "}" + src[m.end():]


count = 0
for anchor, keys in INSERT.items():
    src = insert_into_group(src, anchor, keys)
    count += len(keys)

open(TEX, "w", encoding="utf-8").write(src)
print(f"Inserted {count} new real references into cite groups.")

# verify: count cite keys now
cites = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", src):
    for k in m.group(1).split(","):
        cites.add(k.strip())
print(f"Total cited keys now: {len(cites)}")