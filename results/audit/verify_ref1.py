#!/usr/bin/env python3
import json, time, urllib.parse, urllib.request

EMAIL = "research@example.com"

def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": f"citation-verify/1.0 (mailto:{EMAIL})", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            err = str(e); time.sleep(4 * (i + 1))
    return {"__error__": err}

def oa(w):
    b = w.get("biblio") or {}
    s = ((w.get("primary_location") or {}).get("source") or {})
    return {"title": w.get("title"), "year": w.get("publication_year"), "date": w.get("publication_date"),
            "doi": w.get("doi"), "venue": s.get("display_name"),
            "vol": b.get("volume"), "pages": f"{b.get('first_page')}-{b.get('last_page')}",
            "authors": [a.get("raw_author_name") for a in (w.get("authorships") or [])][:8],
            "cited_by": w.get("cited_by_count")}

OUT = {}
# Nature vol 644 first page 89
r = get("https://api.openalex.org/works?filter=primary_location.source.issn:0028-0836,"
        f"biblio.volume:644,biblio.first_page:89&per-page=10&mailto={EMAIL}")
OUT["A_nature_644_p89"] = [oa(w) for w in r.get("results", [])] if "results" in r else r
time.sleep(2)

# Title-based: the recurring Google Scholar hit
r = get("https://api.openalex.org/works?search=" +
        urllib.parse.quote("Data-driven de novo design of super-adhesive hydrogels") +
        f"&per-page=8&mailto={EMAIL}")
OUT["B_datadriven_superadhesive"] = [oa(w) for w in r.get("results", [])] if "results" in r else r
time.sleep(2)

# Other candidate collision titles
for key, q in [("C_ai_guided_wet", "AI-guided engineering of super-adhesive hydrogels for wet environments"),
               ("D_protein_seq_superadhesion", "From protein sequences to super-adhesion data-driven pipeline hydrogel"),
               ("E_ai_driven_soft_superadhesives", "AI-driven soft materials design for superadhesives")]:
    r = get(f"https://api.openalex.org/works?search={urllib.parse.quote(q)}&per-page=5&mailto={EMAIL}")
    OUT[key] = [oa(w) for w in r.get("results", [])] if "results" in r else r
    time.sleep(2)

print(json.dumps(OUT, indent=2, ensure_ascii=False))
