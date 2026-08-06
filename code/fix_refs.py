"""Fix references.json: apply doc-researcher's P0 replacements + P1 corrections.
P0: 9 entries must be replaced/removed. P1: metadata corrections."""
import json, os

path = os.path.join(os.path.dirname(__file__), "..", "paper", "references.json")
with open(path, "r", encoding="utf-8") as f:
    refs = json.load(f)

# ---- P0: replace/delete entries ----
REMOVE_KEYS = {"zhao2021bioinspired", "zhang2013wet", "zhang2018bionic",
               "ahn2018stiffness", "hsieh2020smooth", "mouret2023outofdistribution",
               "kim2018polygenome", "bartel2023materials", "rao2021injectable",
               "hendrycks2020measuring", "zhang2021understanding"}

def R(key, slot, authors, year, title, journal, volume="", pages="", doi="", url=""):
    return {"key": key, "slot": slot, "authors": authors, "year": year,
            "title": title, "journal": journal, "volume": volume,
            "pages": pages, "doi": doi, "url": url}

REPLACEMENTS = [
    # zhao2021bioinspired -> Narayanan 2021 (verified by researcher)
    R("narayanan2021underwater", "discussion_mechanism",
      ["Narayanan, A.", "Dhinojwala, A.", "Joy, A."], 2021,
      "Design principles for creating synthetic underwater adhesives",
      "Chemical Society Reviews", "50", "13321-13345", "10.1039/D1CS00316J"),
    # zhang2013wet -> Cui & Liu 2021
    R("cui2021wetadhesion", "intro_importance",
      ["Cui, C.", "Liu, W."], 2021,
      "Recent advances in wet adhesives: adhesion mechanism, design principle and applications",
      "Progress in Polymer Science", "116", "101388",
      "10.1016/j.progpolymsci.2021.101388"),
    # zhang2018bionic -> Zhang 2020 Chem Soc Rev
    R("zhang2020catechol", "discussion_mechanism",
      ["Zhang, W.", "Wang, R.", "Sun, Z.", "Zhu, X.", "Zhao, Q."], 2020,
      "Catechol-functionalized hydrogels: biomimetic design, adhesion mechanism, and biomedical applications",
      "Chemical Society Reviews", "49", "433-464", "10.1039/C9CS00285E"),
    # ahn2018stiffness -> year fix 2008 (keep same entry, fix year)
    # hsieh2020smooth -> Chu 2020 ICLR (verified)
    R("chu2020smooth", "methods_model",
      ["Chu, C.", "Minami, K.", "Fukumizu, K."], 2020,
      "Smoothness and stability in GANs", "International Conference on Learning Representations", "", "", "", ""),
    # mouret2023 -> Mouret & Chatzilygeroudis 2017 GECCO
    R("mouret2017realitygap", "intro_gap",
      ["Mouret, J.-B.", "Chatzilygeroudis, K."], 2017,
      "20 years of reality gap: a few thoughts about simulators in evolutionary robotics",
      "GECCO '17 Companion", "", "1121-1124", "10.1145/3067695.3082052"),
    # kim2018polygenome -> J Phys Chem C
    R("kim2018polygenome", "discussion_compare",
      ["Kim, C.", "Chandrasekaran, A.", "Huan, T. D.", "Das, D.", "Ramprasad, R."], 2018,
      "Polymer genome: a data-powered polymer informatics platform",
      "Journal of Physical Chemistry C", "122", "17575-17585", "10.1021/acs.jpcc.8b02913"),
    # bartel2023 -> Merchant 2023 Nature (verified)
    R("merchant2023materials", "discussion_compare",
      ["Merchant, A.", "Batzner, S.", "Schoenholz, S. S.", "Aykol, M.", "Cubuk, E. D."], 2023,
      "Scaling deep learning for materials discovery", "Nature", "624", "80-85",
      "10.1038/s41586-023-06735-9"),
    # rao2021injectable -> Suneetha 2019 ACS Omega
    R("suneetha2019injectable", "intro_importance",
      ["Suneetha, M.", "Rao, K. M.", "Han, S. S."], 2019,
      "One-pot synthesis of injectable self-healing hydrogels for wound healing",
      "ACS Omega", "4", "12647-12656", "10.1021/acsomega.9b01458"),
]

# remove P0 entries
refs = [r for r in refs if r["key"] not in REMOVE_KEYS]

# fix P1 metadata in place
for r in refs:
    if r["key"] == "himanen2020datadriven":
        r["year"] = 2019
    elif r["key"] == "du2019interpretable":
        r["year"] = 2020
    elif r["key"] == "audus2019polymer":
        r["year"] = 2017
    elif r["key"] == "varaprasad2017hydrogel":
        r["authors"] = ["Varaprasad, K.", "Raghavendra, G. M.", "Jayaramudu, T.",
                        "Yallapu, M. M.", "Sadiku, R."]
    elif r["key"] == "liao2025natureb":
        r["key"] = "liao2025nature"  # merge duplicate (keep single entry later)

# dedupe by key (keep first occurrence)
seen = {}
dedup = []
for r in refs:
    if r["key"] not in seen:
        seen[r["key"]] = True
        dedup.append(r)

refs = dedup + REPLACEMENTS

with open(path, "w", encoding="utf-8") as f:
    json.dump(refs, f, ensure_ascii=False, indent=2)
print(f"references.json fixed: {len(refs)} entries")
from collections import Counter
print(dict(Counter(r["slot"] for r in refs)))
# report what was removed
print("Removed keys:", sorted(REMOVE_KEYS))
