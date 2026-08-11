#!/usr/bin/env python3
"""Retry pass with backoff for Ref1 (Liao Nature 2025), Ref4 (Cornell), arXiv 2207.08815."""
import json, time, urllib.parse, urllib.request, ssl

EMAIL = "research@example.com"
CTX = ssl.create_default_context()

def get(url, timeout=40, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": f"citation-verify/1.0 (mailto:{EMAIL})",
                "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last = str(e)
            time.sleep(3 * (i + 1))
    return {"__error__": last}

def slim_oa(w):
    if not w or "__error__" in w: return w
    loc = (w.get("primary_location") or {}); src = (loc.get("source") or {})
    bib = w.get("biblio") or {}
    return {"title": w.get("title"), "year": w.get("publication_year"),
            "date": w.get("publication_date"), "doi": w.get("doi"),
            "venue": src.get("display_name"), "volume": bib.get("volume"),
            "issue": bib.get("issue"), "pages": f"{bib.get('first_page')}-{bib.get('last_page')}",
            "type": w.get("type"),
            "authors": [a.get("raw_author_name") for a in (w.get("authorships") or [])][:10],
            "cited_by": w.get("cited_by_count"), "id": w.get("id")}

def slim_cr(m):
    if not m or "__error__" in m: return m
    return {"title": (m.get("title") or [None])[0],
            "container": (m.get("container-title") or [None])[0],
            "volume": m.get("volume"), "page": m.get("page"),
            "year": ((m.get("issued") or {}).get("date-parts") or [[None]])[0][0],
            "doi": m.get("DOI"), "type": m.get("type"), "publisher": m.get("publisher"),
            "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in (m.get("author") or [])][:10],
            "isbn": m.get("ISBN")}

OUT = {}

# --- Ref1: Nature vol 644 pages 89-95, 2025. Query OpenAlex by biblio filter.
u = ("https://api.openalex.org/works?filter=primary_location.source.issn:0028-0836,"
     "biblio.volume:644,biblio.first_page:89&per-page=10&mailto=" + EMAIL)
r = get(u); time.sleep(2)
OUT["ref1_openalex_biblio_filter"] = ([slim_oa(w) for w in r.get("results", [])]
                                      if isinstance(r, dict) and "results" in r else r)

r = get("https://api.openalex.org/works?search=" + urllib.parse.quote(
    "hydrogel dry crosslinking underwater adhesion") +
    "&filter=publication_year:2025&per-page=10&mailto=" + EMAIL); time.sleep(2)
OUT["ref1_openalex_topic_2025"] = ([slim_oa(w) for w in r.get("results", [])]
                                   if isinstance(r, dict) and "results" in r else r)

r = get("https://api.crossref.org/journals/0028-0836/works?query.bibliographic=" +
        urllib.parse.quote("hydrogel adhesion") +
        "&filter=from-pub-date:2025-01-01&rows=15&mailto=" + EMAIL); time.sleep(3)
OUT["ref1_crossref_nature_2025"] = ([slim_cr(m) for m in r.get("message", {}).get("items", [])]
                                    if isinstance(r, dict) and "message" in r else r)

# --- arXiv 2207.08815 (Grinsztajn)
try:
    req = urllib.request.Request("http://export.arxiv.org/api/query?id_list=2207.08815",
                                 headers={"User-Agent": "citation-verify/1.0"})
    with urllib.request.urlopen(req, timeout=30) as rr:
        xml = rr.read().decode("utf-8")
    import re
    OUT["arxiv_2207_08815"] = {
        "title": re.findall(r"<title>(.*?)</title>", xml, re.S)[-1].strip(),
        "authors": re.findall(r"<name>(.*?)</name>", xml),
        "published": re.findall(r"<published>(.*?)</published>", xml),
        "comment": re.findall(r'<arxiv:comment[^>]*>(.*?)</arxiv:comment>', xml, re.S),
    }
except Exception as e:
    OUT["arxiv_2207_08815"] = {"__error__": str(e)}
time.sleep(2)

# --- Ref4: Cornell, Experiments with Mixtures
r = get("https://api.crossref.org/works?query.bibliographic=" + urllib.parse.quote(
    "Experiments with Mixtures Designs Models and the Analysis of Mixture Data Cornell") +
    "&rows=10&mailto=" + EMAIL); time.sleep(3)
OUT["ref4_crossref_query"] = ([slim_cr(m) for m in r.get("message", {}).get("items", [])]
                              if isinstance(r, dict) and "message" in r else r)

for doi in ["10.1002/9780470907443", "10.1002/9781118204221", "10.1002/0471471631"]:
    OUT[f"ref4_doi_{doi.replace('/','_')}"] = slim_cr(get(
        f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={EMAIL}").get("message", {})
        if isinstance(get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={EMAIL}"), dict) else {})
    time.sleep(3)

r = get("https://api.openalex.org/works?search=" + urllib.parse.quote(
    "Experiments with Mixtures Cornell mixture data") + "&per-page=10&mailto=" + EMAIL)
OUT["ref4_openalex"] = ([slim_oa(w) for w in r.get("results", [])]
                        if isinstance(r, dict) and "results" in r else r)

print(json.dumps(OUT, indent=2, ensure_ascii=False))
