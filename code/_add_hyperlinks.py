# -*- coding: utf-8 -*-
"""Add hyperref macros to preamble, then replace (Fig. NX)/(Table N) with hyperlinks."""
import re

P = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex'
tex = open(P, encoding='utf-8').read()

# 1) add macros after \usepackage line
MACROS = (
    '% clickable figure/table references\n'
    r'\newcommand{\Fref}[2]{Fig.~\hyperref[#1]{\ref{#1}#2}}' + '\n'
    r'\newcommand{\Tref}[1]{Table~\hyperref[#1]{\ref{#1}}}' + '\n'
)
anchor = r'\usepackage{url,hyperref,lineno,microtype,subcaption,amsmath,booktabs,multirow}'
if r'\newcommand{\Fref}' not in tex:
    tex = tex.replace(anchor, anchor + '\n' + MACROS, 1)
    print('macros added')
else:
    print('macros already present')

# 2) label mapping: figure/table number -> label
fig_label = {1: 'figpipeline', 2: 'figmodelarch', 3: 'fig:dataset', 4: 'fig:cv',
             5: 'fig:bench', 6: 'fig:ext', 7: 'fig:abl', 8: 'fig:interp'}
tab_label = {1: 'tab:int', 2: 'tab:ext', 3: 'tab:gap', 4: 'tab:abl'}

# 3) replace (Fig. N) or (Fig. NX) or (Fig. NX, NY) ... -> (\Fref{label}{X}, \Fref{label}{Y})
# pattern: (Fig. 5A, 5F) or (Fig. 3D, 3E) or (Fig. 7A, 7D) or (Fig. 4G, 4I) or (Fig. 5A, 5F)
def repl_fig(m):
    n = int(m.group(1))
    letters = m.group(2)
    # letters like "A" or "A, 5F" -> split
    # handle "3D, 3E" style: letters part may be "A, 5F"
    label = fig_label.get(n)
    if not label:
        return m.group(0)
    parts = re.split(r',\s*', letters)
    # each part may be "A" or "5F" (second fig num)
    out = []
    for pt in parts:
        pm = re.match(r'(\d*)([A-I])$', pt.strip())
        if pm:
            ln = pm.group(1) or str(n)
            L = pm.group(2)
            out.append(r'\Fref{%s}{%s}' % (fig_label[int(ln)], L))
        else:
            out.append(pt.strip())
    return '(' + ', '.join(out) + ')'

# replace (Fig. N...) forms
tex2 = re.sub(r'\(Fig\.\s*(\d+)\s*([A-I](?:\s*,\s*\d*[A-I])*)\)', repl_fig, tex)

# 4) replace (Table N) -> (\Tref{label})
def repl_tab(m):
    n = int(m.group(1))
    label = tab_label.get(n)
    return r'(\Tref{%s})' % label if label else m.group(0)
tex2 = re.sub(r'\(Table\s*(\d+)\)', repl_tab, tex2)

# 5) handle combined (Table 1; Fig. 5A) -> (\Tref{tab:int}; \Fref{fig:bench}{A})
def repl_combo(m):
    inner = m.group(1)
    # Table N; Fig. NX  OR  Table N; Fig. NX, NY
    out = []
    for seg in inner.split(';'):
        seg = seg.strip()
        tm = re.match(r'Table\s*(\d+)$', seg)
        fm = re.match(r'Fig\.\s*(\d+)\s*([A-I](?:\s*,\s*\d*[A-I])*)$', seg)
        if tm:
            out.append(r'\Tref{%s}' % tab_label.get(int(tm.group(1)), seg))
        elif fm:
            out.append(repl_fig(fm))
        else:
            out.append(seg)
    return '(' + '; '.join(out) + ')'
tex2 = re.sub(r'\((Table\s*\d+;\s*Fig\.\s*\d+\s*[A-I](?:\s*,\s*\d*[A-I])*)\)', repl_combo, tex2)

open(P, 'w', encoding='utf-8').write(tex2)
# count
n_fref = tex2.count(r'\Fref')
n_tref = tex2.count(r'\Tref')
print(f'Fref: {n_fref}, Tref: {n_tref}')
