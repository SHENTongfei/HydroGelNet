# -*- coding: utf-8 -*-
"""Analyze all Fig/Table refs in tex, report whether at sentence end."""
import re

P = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex'
tex = open(P, encoding='utf-8').read()
lines = tex.split('\n')

pat = re.compile(r'\((?:Fig\.|Table)[^)]*\)')
for i, ln in enumerate(lines, 1):
    for m in pat.finditer(ln):
        start = m.start()
        # check char after the match: sentence end if '.' then space/EOL, or ';' inside later
        after = ln[m.end():m.end() + 6]
        # find preceding sentence boundary
        end_of_sentence = bool(re.match(r'^\.\s', after) or after.startswith('.'))
        print(f'L{i} {"END" if end_of_sentence else "MID"}: {m.group(0)} ...{after[:10]!r}')
