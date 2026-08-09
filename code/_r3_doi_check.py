# -*- coding: utf-8 -*-
"""R3: Crossref DOI verification for a sample of bib entries."""
import re, json, urllib.request, urllib.parse, random

BIB = r"C:\Users\TS\WorkBuddy\HydroGelNet\paper\reference_final.bib"
bib = open(BIB, encoding="utf-8").read()

entries = []
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.DOTALL):
    key, block = m.group(1), m.group(2)
    dm = re.search(r"doi\s*=\s*\{([^}]+)\}", block)
    if dm:
        entries.append((key, dm.group(1)))

print(f"total entries: {len(entries)} with DOI")
# sample 30 spread across the file (deterministic: every Nth)
random.seed(42)
sample = random.sample(entries, min(30, len(entries)))

ok, fail = 0, []
for key, doi in sample:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:shen.tongfei@outlook.com"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
        if status == 200:
            ok += 1
        else:
            fail.append((key, doi, status))
    except Exception as e:
        fail.append((key, doi, str(e)[:60]))

print(f"verified OK: {ok}/{len(sample)}")
for k, d, s in fail:
    print(f"  FAIL {k}: {d} -> {s}")

# also verify arXiv DOIs are 10.48550 format (they resolve via data.crossref)
arxiv = [e for e in entries if "10.48550" in e[1]]
print(f"\narXiv-style DOIs: {len(arxiv)}")
for k, d in arxiv[:5]:
    print(f"  {k}: {d}")
