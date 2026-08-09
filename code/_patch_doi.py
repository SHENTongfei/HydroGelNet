import _runtime_guard  # noqa
import re
import json
import urllib.request
import urllib.parse
import time

BIB = r"C:/Users/TS/WorkBuddy/HydroGelNet/paper/reference_final.bib"
bib = open(BIB, encoding="utf-8").read()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())[:40]


def openalex_title(ttl, retries=2):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
        "filter": "title.search:" + ttl[:150],
        "per-page": 4,
        "mailto": "shen.tongfei@outlook.com",
    })
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "mailto:shen.tongfei@outlook.com"})
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = json.load(resp)
            for w in data.get("results", []):
                doi = w.get("doi") or ""
                if doi.startswith("https://doi.org/"):
                    doi = doi[len("https://doi.org/"):]
                if norm(w.get("title"))[:30] == norm(ttl)[:30] and doi:
                    return doi
            return None
        except Exception:
            time.sleep(2)
    return None


# entries missing doi
missing = []
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.DOTALL):
    key, block = m.group(1), m.group(2)
    if not re.search(r"\bdoi\s*=", block):
        ttl = re.search(r"title\s*=\s*\{([^}]+)\}", block)
        missing.append((key, ttl.group(1) if ttl else ""))

print(f"Entries missing doi: {len(missing)}")
patched = 0
for key, title in missing:
    doi = openalex_title(title)
    if doi:
        m = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\}", bib, re.DOTALL)
        if m:
            block = m.group(1).rstrip()
            bib = (bib[:m.start()] + "@article{" + key + "," + block +
                   ",\n  doi = {" + doi + "}\n}\n" + bib[m.end():])
            patched += 1
            print(f"  PATCHED {key}: {doi}")
    time.sleep(0.4)

open(BIB, "w", encoding="utf-8").write(bib)
print(f"Patched {patched}/{len(missing)}")

# final audit
bib2 = open(BIB, encoding="utf-8").read()
all_keys = set(re.findall(r"@\w+\{([^,]+),", bib2))
years = {}
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib2, re.DOTALL):
    ym = re.search(r"year\s*=\s*\{?(\d{4})\}?", m.group(2))
    if ym:
        years[m.group(1)] = int(ym.group(1))
recent = sum(1 for y in years.values() if y >= 2021)
missing_doi = []
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib2, re.DOTALL):
    if not re.search(r"\bdoi\s*=", m.group(2)):
        missing_doi.append(m.group(1))
print(f"Total entries: {len(all_keys)}")
print(f"Recent 5y (2021+): {recent} ({recent/len(all_keys)*100:.0f}%)")
print(f"Missing DOI: {len(missing_doi)} {missing_doi}")