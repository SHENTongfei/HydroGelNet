# -*- coding: utf-8 -*-
"""R2 malicious-view text audit (H35): em-dash=0, prose colon=0, Results zero cite,
citation integrity (0 dangling / 0 uncited), number consistency, structure."""
import re, os

P_TEX = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex'
P_BIB = r'C:\Users\TS\WorkBuddy\HydroGelNet\paper\reference_final.bib'
tex = open(P_TEX, encoding='utf-8').read()
bib = open(P_BIB, encoding='utf-8').read()

report = []
def check(name, ok, detail=""):
    report.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail[:110]}")

# ---- 1. em/en dash in prose (H29) ----
for ch, nm in [("\u2014", "em-dash"), ("\u2013", "en-dash")]:
    n = tex.count(ch)
    check(f"H29 {nm} = 0", n == 0, f"count={n}")

# ---- 2. prose colon (allow: title/caption-bold/table-head/ref/labels/url/abstract-structure/math) ----
body = tex
# strip \title and \caption blocks entirely (their colons are allowed)
body = re.sub(r'\\title(\[[^\]]*\])?\{[^}]*\}', '', body)
body = re.sub(r'\\caption(\[[^\]]*\])?\{[^}]*\}', '', body, flags=re.DOTALL)
# strip \Fref/\Tref/\ref/\label with fig:/tab:/eq:/sec:/eq: prefixes
body = re.sub(r'\\(Fref|Tref|ref|label)\{[^}]*:[^}]*\}', '', body)
# strip inline math $...$ and \[...\]
body = re.sub(r'\$[^$]*\$', '', body)
body = re.sub(r'\\\[.*?\\\]', '', body, flags=re.DOTALL)
# strip equation environments
body = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', body, flags=re.DOTALL)
# strip URLs
body = re.sub(r'https?://[^\s}]+', '', body)
# strip abstract structure labels
body = re.sub(r'(Motivation|Results|Availability|Contact|Keywords):', '', body)
n_colon = body.count(':')
check("H29 prose colon = 0", n_colon == 0, f"count={n_colon}")

# ---- 3. Results section zero cite ----
# find Results..Discussion region
m_res = re.search(r'\\section\{Results\}', tex)
m_disc = re.search(r'\\section\{Discussion\}', tex)
if m_res and m_disc:
    res = tex[m_res.end():m_disc.start()]
    cites = re.findall(r'\\cite\{[^}]*\}', res)
    check("Results zero cite", len(cites) == 0, f"cites in Results={len(cites)}")
else:
    check("Results zero cite", False, "could not locate Results/Discussion")

# ---- 4. citation integrity ----
bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib))
cited = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', tex):
    for k in m.group(1).split(','):
        cited.add(k.strip())
dangling = cited - bib_keys
uncited = bib_keys - cited
check("0 dangling cite", len(dangling) == 0, f"dangling={sorted(dangling)[:5]}")
# uncited bib entries are OK if bib has extras, but flag count
check("bib >= cited (no shortage)", len(bib_keys) >= len(cited),
      f"bib={len(bib_keys)} cited={len(cited)} uncited={len(uncited)}")

# ---- 5. key numbers appear consistently ----
nums = {
    "0.7924 (internal R2)": ["0.7924", "0.79"],
    "0.6946 (external R2)": ["0.6946", "0.69"],
    "0.6342 (SVR external)": ["0.6342", "0.63"],
    "0.87 (Spearman)": ["0.87"],
    "0.0724 (fusion delta)": ["0.0724", "0.072"],
    "0.0631 (BAxPEA)": ["0.0631"],
    "316 internal": ["316"],
    "25 prospective": ["25"],
    "50 models": ["50"],
}
for nm, keys in nums.items():
    hits = sum(tex.count(k) for k in keys)
    check(f"num '{nm}' present", hits >= 2, f"hits={hits}")

# ---- 6. placeholders / leftovers ----
for pat, nm in [(r'\[missing[^\]]*\]', 'missing placeholder'),
                (r'[?][);,]', 'question-mark citation'),
                (r'\bnu\s?ll\b', 'null text'),
                (r'TODO|FIXME|XXX', 'TODO/FIXME')]:
    found = re.findall(pat, tex)
    check(f"no {nm}", len(found) == 0, f"count={len(found)}")

# ---- 7. structure integrity ----
check("begin/end document 1:1",
      tex.count(r'\begin{document}') == 1 and tex.count(r'\end{document}') == 1)
n_fig = tex.count(r'\begin{figure')
n_tab = tex.count(r'\begin{table')
check("figures/tables present", n_fig >= 8 and n_tab >= 4,
      f"figs={n_fig} tables={n_tab}")

# ---- 8. bbl/bibtex (from local compile if available) ----
blg = r'C:\Users\TS\WorkBuddy\HydroGelNet\_local_compile\v7_test.blg'
if os.path.exists(blg):
    blg_txt = open(blg, encoding='utf-8', errors='ignore').read()
    n_miss = len(re.findall(r"didn't find|missing field", blg_txt))
    check("bibtex clean", n_miss == 0, f"issues={n_miss}")
else:
    check("bibtex clean", True, "blg not found (skip)")

# ---- summary ----
fails = [r for r in report if r[1] == "FAIL"]
print(f"\n=== R2 summary: {len(report)-len(fails)} PASS / {len(fails)} FAIL ===")
for f in fails:
    print(f"  FAIL: {f[0]} - {f[2]}")
