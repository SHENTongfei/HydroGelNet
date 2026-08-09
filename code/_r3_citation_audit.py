# -*- coding: utf-8 -*-
"""R3 cross-final audit: verify cited entries exist via citation-finder's
internal source (OpenAlex) + Crossref DOI check (H37: citation-finder authority).
Samples: all 33 recent* keys + 15 classic keys = 48 entries."""
import re, json, urllib.request, urllib.parse, time, sys

P_TEX = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex'
P_BIB = r'C:\Users\TS\WorkBuddy\HydroGelNet\paper\reference_final.bib'
tex = open(P_TEX, encoding='utf-8').read()
bib = open(P_BIB, encoding='utf-8').read()

# ---- gather cited keys ----
cited = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', tex):
    for k in m.group(1).split(','):
        cited.add(k.strip())

# ---- parse bib entries (robust: locate @type{key, then brace-count) ----
entries = {}
for m in re.finditer(r'@(\w+)\{([^,\s}]+),', bib):
    etype, key = m.group(1), m.group(2)
    start = m.end()
    depth = 1
    j = start
    while j < len(bib) and depth > 0:
        if bib[j] == '{':
            depth += 1
        elif bib[j] == '}':
            depth -= 1
        j += 1
    block = bib[start:j - 1]
    def field(f):
        fm = re.search(r'\b' + f + r'\s*=\s*\{([^}]*)\}', block)
        return fm.group(1) if fm else ''
    entries[key] = {"type": etype, "title": field('title'),
                    "doi": field('doi'), "year": field('year'),
                    "journal": field('journal')}

def openalex_doi(doi):
    """Query OpenAlex by DOI (citation-finder internal source)."""
    url = 'https://api.openalex.org/works/https://doi.org/' + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={'User-Agent': 'mailto:shen.tongfei@outlook.com'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.load(resp)
        return bool(d.get('id'))
    except Exception:
        return None

def crossref_doi(doi):
    url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={'User-Agent': 'mailto:shen.tongfei@outlook.com'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status == 200
    except Exception:
        return None

# ---- sample: recent* keys (the risky auto-generated ones) ----
recent = sorted([k for k in cited if k.startswith('recent')])
classic = sorted([k for k in cited if not k.startswith('recent')])
sample = recent + classic[:8]   # all 33 recent + 8 classic

print(f"cited total={len(cited)} | recent={len(recent)} classic={len(classic)}")
print(f"sampling {len(sample)} entries (all recent + 8 classic)\n")

results = []
for key in sample:
    e = entries.get(key)
    if not e:
        results.append((key, 'NO_BIB_ENTRY', ''))
        print(f"[NO_BIB] {key}")
        continue
    doi = e.get('doi', '')
    if not doi:
        results.append((key, 'NO_DOI', e.get('title', '')[:50]))
        print(f"[NO_DOI] {key}: {e.get('title','')[:50]}")
        continue
    # OpenAlex (citation-finder authority) then Crossref cross-check
    ok_oa = openalex_doi(doi)
    time.sleep(0.3)
    ok_cr = crossref_doi(doi)
    ok = bool(ok_oa) or bool(ok_cr)
    src = 'openalex' if ok_oa else ('crossref' if ok_cr else 'NONE')
    results.append((key, 'OK' if ok else 'FAIL', f"{src} doi={doi[:40]}"))
    print(f"[{'OK' if ok else 'FAIL'}] {key}: {src} {doi[:40]}")

fails = [r for r in results if r[1] in ('FAIL', 'NO_BIB_ENTRY', 'NO_DOI')]
print(f"\n=== R3 citation check: {len(results)-len(fails)} OK / {len(fails)} issues ===")
for f in fails:
    print(f"  {f[0]}: {f[1]} {f[2][:60]}")
