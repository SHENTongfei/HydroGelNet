# -*- coding: utf-8 -*-
"""R2b: corrected text audit (exclude math env, \Tref/\Fref macros, ?} in
subsection titles)."""
import re

TEX = r"C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex"
BIB = r"C:\Users\TS\WorkBuddy\HydroGelNet\paper\reference_final.bib"
tex = open(TEX, encoding="utf-8").read()
bib = open(BIB, encoding="utf-8").read()

issues = []
def chk(name, ok, detail):
    issues.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

# ---------- 1. dashes ----------
dash_count = (tex.count("\u2014") + tex.count("\u2013") +
              tex.count("---") + tex.count("--"))
chk("H29 em/en-dash = 0", dash_count == 0, f"dash count = {dash_count}")

# ---------- 2. prose colons (mask math env + macros + titles) ----------
lines = tex.split("\n")
code_lines = [re.sub(r"(?<!\\)%.*$", "", ln).rstrip() for ln in lines]
body = "\n".join(code_lines)

# mask inline/display math ($...$ and \[...\])
masked = re.sub(r"\$[^$]*\$", "MATH", body)
masked = re.sub(r"\\\[.*?\\\]", "MATH", masked, flags=re.DOTALL)

# mask allowed patterns
allowed_pat = re.compile(
    r"\\title\[[^\]]*\]\{[^}]*\}"
    r"|\\ref\{[^}]*\}"
    r"|\\label\{(sec|eq|fig|tab):[^}]*\}"
    r"|\\cite\{[^}]*\}"
    r"|\\Fref\{[^}]*\}\{[^}]*\}"
    r"|\\Tref\{[^}]*\}"
    r"|https?://[^\s)}]+"
    r"|\\subsection\{[^}]*\}"   # titles may contain ?
)
masked = allowed_pat.sub("ALLOWED", masked)

real_colons = []
for i, ln in enumerate(masked.split("\n"), 1):
    for m in re.finditer(":", ln):
        ctx = ln[max(0, m.start()-30):m.end()+25]
        real_colons.append((i, ctx.strip()))
chk("H29 prose colons = 0 (math/labels/macros/titles masked)",
    len(real_colons) == 0, f"{len(real_colons)} hits: {real_colons[:5]}")

# ---------- 3. Results cites ----------
m_res = re.search(r"\\section\{Results\}(.*?)\\section\{Discussion\}", tex, re.DOTALL)
res_cites = len(re.findall(r"\\cite\{", m_res.group(1))) if m_res else -1
chk("Results section \\cite = 0", res_cites == 0, f"{res_cites} cites")

# ---------- 4. refs ----------
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cite_keys = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        cite_keys.add(k.strip())
dangling = cite_keys - bib_keys
chk("no dangling cite keys", len(dangling) == 0, f"dangling: {sorted(dangling)[:5]}")
chk("bib >= 100", len(bib_keys) >= 100, f"{len(bib_keys)} bib entries")
chk("cited >= 100", len(cite_keys) >= 100, f"{len(cite_keys)} cited keys")
years = {}
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.DOTALL):
    ym = re.search(r"year\s*=\s*\{?(\d{4})\}?", m.group(2))
    if ym:
        years[m.group(1)] = int(ym.group(1))
recent = sum(1 for y in years.values() if y >= 2021)
chk("recent-5y >= 50%", len(years) > 0 and recent / len(years) >= 0.5,
    f"{recent}/{len(years)} = {recent/max(len(years),1)*100:.0f}%")

# ---------- 5. locked numbers ----------
locked = ["0.7924", "0.8067", "0.6946", "0.6342", "0.87", "0.0724", "0.0631",
          "0.0026", "316", "25"]
missing_nums = [n for n in locked if n not in tex]
chk("locked numbers present", len(missing_nums) == 0, f"missing: {missing_nums}")

# ---------- 6. placeholders (exclude ?} in subsection titles) ----------
body_wo_titles = re.sub(r"\\subsection\{[^}]*\}", "", tex)
bad_found = []
for pat in [r"\[missing", r"\[待填写", r"TODO", r"XXX", r"unknown citation",
            r"\?\}", r"\?,"]:
    hits = re.findall(pat, body_wo_titles)
    if hits:
        bad_found.append((pat, len(hits)))
chk("no placeholders / broken cites", len(bad_found) == 0, f"{bad_found}")

# ---------- 7. structure ----------
n_begin = tex.count(r"\begin{document}")
n_end = tex.count(r"\end{document}")
chk("begin/end 1:1", n_begin == 1 and n_end == 1, f"{n_begin}/{n_end}")
n_fig = len(re.findall(r"\\begin\{figure\*?\}", tex))
n_tab = len(re.findall(r"\\begin\{table\*?\}", tex))
chk("figures >= 8", n_fig >= 8, f"{n_fig} figures")
chk("tables >= 4", n_tab >= 4, f"{n_tab} tables")

# ---------- 8. term consistency ----------
for pat, bad in [(r"HydroGelNet", "old repo name"),
                 (r"youthful", "old term"),
                 (r"\bthe the\b", "dup word"),
                 (r"\ba a\b", "dup word")]:
    hits = re.findall(pat, tex, re.IGNORECASE)
    chk(f"no '{bad}'", len(hits) == 0, f"{len(hits)} hits")

print("\n=== R2 SUMMARY ===")
fails = [i for i in issues if i[1] == "FAIL"]
print(f"PASS: {len(issues)-len(fails)}/{len(issues)}, FAIL: {len(fails)}")
for f in fails:
    print(f"  FAIL {f[0]}: {f[2]}")
