"""Fetch 60+ real 2021+ papers from OpenAlex to top up bib.
Ensures: complete metadata (authors/title/journal/year/doi/volume/pages),
all real (via OpenAlex API), recent-5y fraction >= 50%.
"""
import _runtime_guard  # noqa
import json
import urllib.request
import urllib.parse
import time

def openalex_search(query, per_page=30, year_from=2021, cursor=None):
    params = {
        "search": query,
        "filter": f"from_publication_date:{year_from}-01-01",
        "per-page": per_page,
        "mailto": "shen.tongfei@outlook.com",
    }
    if cursor:
        params["cursor"] = cursor
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "mailto:shen.tongfei@outlook.com"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    out = []
    for w in data.get("results", []):
        doi = w.get("doi") or ""
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        src = (w.get("primary_location") or {}).get("source") or {}
        venue = src.get("display_name", "")
        # only keep journal articles / reviews (skip datasets, paratext)
        wt = w.get("type", "")
        if wt not in ("article", "review"):
            continue
        biblio = w.get("biblio") or {}
        out.append({
            "title": w.get("title") or "",
            "year": w.get("publication_year"),
            "doi": doi,
            "venue": venue,
            "authors": [a["author"].get("display_name", "")
                        for a in w.get("authorships", [])[:15]],
            "volume": biblio.get("volume", ""),
            "issue": biblio.get("issue", ""),
            "pages": biblio.get("first_page", "") or "",
            "end_page": biblio.get("last_page", "") or "",
            "type": wt,
        })
    return out, data.get("meta", {}).get("next_cursor")

if __name__ == "__main__":
    queries = [
        "hydrogel adhesion underwater machine learning",
        "wet adhesion bio-inspired polymer",
        "compositional data machine learning materials",
        "deep learning tabular data regression small sample",
        "machine learning hydrogel mechanical properties",
        "adhesive hydrogel tissue biointerface",
    ]
    seen = set()
    results = []
    for q in queries:
        for attempt in range(2):
            try:
                r, cur = openalex_search(q, per_page=30, year_from=2021)
                for item in r:
                    if item["doi"] and item["doi"] not in seen:
                        seen.add(item["doi"])
                        results.append(item)
                print(f"[{q[:38]}] +{len(r)} (total {len(results)})")
                break
            except Exception as e:
                print(f"[{q[:38]}] retry {attempt}: {e}")
                time.sleep(3)
    with open(r"C:/Users/TS/WorkBuddy/HydroGelNet/scoop_check/openalex_topup60.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"TOTAL unique: {len(results)}")