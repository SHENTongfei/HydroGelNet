"""Fetch real recent papers from OpenAlex API (same source citation-finder uses)
to top up reference_final.bib beyond 100 entries with complete metadata.
"""
import _runtime_guard  # noqa
import json
import urllib.request
import urllib.parse

def openalex_search(query, per_page=25, year_from=2021):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({
        "search": query,
        "filter": f"from_publication_date:{year_from}-01-01",
        "per-page": per_page,
        "mailto": "shen.tongfei@outlook.com",
    })
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:shen.tongfei@outlook.com"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    out = []
    for w in data.get("results", []):
        doi = w.get("doi") or ""
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        auths = []
        for a in w.get("authorships", [])[:12]:
            nm = a["author"].get("display_name", "")
            auths.append(nm)
        out.append({
            "title": w.get("title") or "",
            "year": w.get("publication_year"),
            "doi": doi,
            "venue": (w.get("primary_location") or {}).get("source", {}).get("display_name", "")
                     if w.get("primary_location") else "",
            "authors": auths,
            "volume": (w.get("biblio") or {}).get("volume", ""),
            "issue": (w.get("biblio") or {}).get("issue", ""),
            "pages": (w.get("biblio") or {}).get("first_page", ""),
            "cited_by": w.get("cited_by_count", 0),
        })
    return out

if __name__ == "__main__":
    queries = [
        "hydrogel adhesion underwater machine learning",
        "wet adhesion bio-inspired polymer prediction",
        "compositional data machine learning materials",
        "deep learning small tabular data regression",
    ]
    all_results = []
    for q in queries:
        try:
            r = openalex_search(q, per_page=15, year_from=2021)
            print(f"[{q[:40]}] {len(r)} results")
            all_results.extend(r)
        except Exception as e:
            print(f"[{q[:40]}] ERROR {e}")
    with open(r"C:/Users/TS/WorkBuddy/HydroGelNet/scoop_check/openalex_topup.json", "w",
              encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"TOTAL: {len(all_results)}")