#!/usr/bin/env python3
"""H37 compliance: verify the 4 cited references against OpenAlex + Crossref."""
import json, sys, time
import urllib.parse
import urllib.request

EMAIL = "research@example.com"
OUT = {}

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": f"citation-verify/1.0 (mailto:{EMAIL})"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def openalex_search(q, per_page=5, extra=""):
    url = f"https://api.openalex.org/works?search={urllib.parse.quote(q)}&per-page={per_page}&mailto={EMAIL}{extra}"
    try:
        return get(url).get("results", [])
    except Exception as e:
        return [{"__error__": str(e)}]

def openalex_doi(doi):
    url = f"https://api.openalex.org/works/https://doi.org/{doi}?mailto={EMAIL}"
    try:
        return get(url)
    except Exception as e:
        return {"__error__": str(e)}

def crossref_doi(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={EMAIL}"
    try:
        return get(url).get("message", {})
    except Exception as e:
        return {"__error__": str(e)}

def crossref_query(q, rows=5):
    url = f"https://api.crossref.org/works?query.bibliographic={urllib.parse.quote(q)}&rows={rows}&mailto={EMAIL}"
    try:
        return get(url).get("message", {}).get("items", [])
    except Exception as e:
        return [{"__error__": str(e)}]

def slim_oa(w):
    if not w or "__error__" in w:
        return w
    loc = (w.get("primary_location") or {})
    src = (loc.get("source") or {})
    bib = w.get("biblio") or {}
    return {
        "title": w.get("title"),
        "year": w.get("publication_year"),
        "date": w.get("publication_date"),
        "doi": w.get("doi"),
        "venue": src.get("display_name"),
        "volume": bib.get("volume"), "issue": bib.get("issue"),
        "first_page": bib.get("first_page"), "last_page": bib.get("last_page"),
        "type": w.get("type"),
        "authors": [a.get("raw_author_name") or (a.get("author") or {}).get("display_name")
                    for a in (w.get("authorships") or [])][:12],
        "cited_by": w.get("cited_by_count"),
        "id": w.get("id"),
    }

def slim_cr(m):
    if not m or "__error__" in m:
        return m
    return {
        "title": (m.get("title") or [None])[0],
        "container": (m.get("container-title") or [None])[0],
        "volume": m.get("volume"), "issue": m.get("issue"), "page": m.get("page"),
        "year": ((m.get("issued") or {}).get("date-parts") or [[None]])[0][0],
        "doi": m.get("DOI"), "type": m.get("type"), "publisher": m.get("publisher"),
        "authors": [f"{a.get('given','')} {a.get('family','')}".strip()
                    for a in (m.get("author") or [])][:12],
        "isbn": m.get("ISBN"),
    }

# ---- Ref 1: Liao et al. Nature 644, 89-95 (2025)
OUT["ref1_liao_nature"] = {
    "oa_search_hydrogel": [slim_oa(w) for w in openalex_search(
        "hydrogel underwater adhesion", per_page=8,
        extra="&filter=publication_year:2025,primary_location.source.display_name.search:Nature")],
    "oa_search_liao": [slim_oa(w) for w in openalex_search(
        "Liao hydrogel adhesive Nature 2025", per_page=8)],
    "cr_query": [slim_cr(m) for m in crossref_query("Nature 2025 volume 644 pages 89-95", rows=8)],
}

# ---- Ref 2: Grinsztajn et al.
OUT["ref2_grinsztajn"] = {
    "oa_search": [slim_oa(w) for w in openalex_search(
        "Why do tree-based models still outperform deep learning on typical tabular data", per_page=6)],
    "cr_query": [slim_cr(m) for m in crossref_query(
        "Why do tree-based models still outperform deep learning on typical tabular data Grinsztajn", rows=6)],
}

# ---- Ref 3: Shwartz-Ziv & Armon
OUT["ref3_shwartzziv"] = {
    "oa_doi": slim_oa(openalex_doi("10.1016/j.inffus.2021.11.011")),
    "cr_doi": slim_cr(crossref_doi("10.1016/j.inffus.2021.11.011")),
}

# ---- Ref 4: Cornell, Experiments with Mixtures
OUT["ref4_cornell"] = {
    "cr_query": [slim_cr(m) for m in crossref_query(
        "Cornell Experiments with Mixtures Designs Models and the Analysis of Mixture Data", rows=8)],
    "oa_search": [slim_oa(w) for w in openalex_search(
        "Experiments with Mixtures Designs Models Analysis of Mixture Data Cornell", per_page=8)],
    "cr_doi_3rd": slim_cr(crossref_doi("10.1002/9781118204221")),
    "cr_doi_alt": slim_cr(crossref_doi("10.1002/0471471631")),
}

print(json.dumps(OUT, indent=2, ensure_ascii=False))
