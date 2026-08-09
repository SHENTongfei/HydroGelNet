import _runtime_guard  # noqa
import re

tex = open(r"C:/Users/TS/WorkBuddy/HydroGelNet/audit/rewrite_baseline/v7_merged.tex",
           encoding="utf-8").read()
bib = open(r"C:/Users/TS/WorkBuddy/HydroGelNet/paper/reference_final.bib",
           encoding="utf-8").read()

cites = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        cites.add(k.strip())

bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
print(f"Cited keys: {len(cites)}")
print(f"Bib keys: {len(bib_keys)}")
print(f"Cited but NOT in bib: {sorted(cites - bib_keys)}")
print(f"In bib but NOT cited: {sorted(bib_keys - cites)}")

# year distribution (recent 5y = 2021-2026)
years = {}
for m in re.finditer(r"@\w+\{([^,]+),", bib):
    key = m.group(1)
    ym = re.search(r"\byear\s*=\s*\{?(\d{4})\}?", bib[m.end():m.end()+400])
    if ym:
        years[key] = int(ym.group(1))
recent = sum(1 for y in years.values() if y >= 2021)
print(f"Entries with year: {len(years)}")
print(f"Recent 5y (2021+): {recent} ({recent/len(years)*100:.0f}%)" if years else "no years")