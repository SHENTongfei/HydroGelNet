# -*- coding: utf-8 -*-
"""R3 full DOI verification: split bib by balanced braces."""
import re, urllib.request, urllib.parse, random

BIB = r"C:\Users\TS\WorkBuddy\HydroGelNet\paper\reference_final.bib"
bib = open(BIB, encoding="utf-8").read()

def split_entries(text):
    out = []
    i = 0
    while i < len(text):
        m = re.match(r"@(\w+)\{([^,}]+),", text[i:])
        if not m:
            i += 1
            continue
        key = m.group(2)
        start = i
        depth = 0
        j = i
        while j < len(text):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((key, text[start:j + 1]))
        i = j + 1
    return out

entries = split_entries(bib)
print(f"parsed {len(entries)} entries")
with_doi = []
for key, block in entries:
    dm = re.search(r"doi\s*=\s*\{([^}]+)\}", block)
    if dm:
        with_doi.append((key, dm.group(1)))
print(f"with DOI: {len(with_doi)}")

random.seed(7)
sample = random.sample(with_doi, min(40, len(with_doi)))
ok, fail = 0, []
for key, doi in sample:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:shen.tongfei@outlook.com"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            st = resp.status
        if st == 200:
            ok += 1
        else:
            fail.append((key, doi, st))
    except Exception as e:
        fail.append((key, doi, str(e)[:50]))
print(f"verified OK: {ok}/{len(sample)}")
for k, d, s in fail:
    print(f"  FAIL {k}: {d} -> {s}")
