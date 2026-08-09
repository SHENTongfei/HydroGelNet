# -*- coding: utf-8 -*-
"""Re-audit refs after hyperlink conversion: mark MID (not at sentence end)."""
import re

P = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex'
tex = open(P, encoding='utf-8').read()
lines = tex.split('\n')

# match ((\Fref{...}{...}) or (\Tref{...}) at any position
pat = re.compile(r'\(\\(?:Fref|Tref)\{[^}]*\}\{[^}]*\}\)')
for i, ln in enumerate(lines, 1):
    for m in pat.finditer(ln):
        after = ln[m.end():m.end() + 8]
        # END if followed by '.' then whitespace/EOL, or by ';' (part of list) then later '.'
        end_of_sentence = bool(re.match(r'^\.\s', after) or after.startswith('.'))
        tag = 'END' if end_of_sentence else 'MID'
        if tag == 'MID':
            # show context
            before = ln[max(0, m.start() - 40):m.start()]
            print(f'L{i} MID: ...{before}...{m.group(0)}...{after[:15]!r}')
