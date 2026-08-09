# -*- coding: utf-8 -*-
"""Fix bib entries with garbage prefix before @article{...}."""
import re

P = r'C:\Users\TS\WorkBuddy\HydroGelNet\paper\reference_final.bib'
bib = open(P, encoding='utf-8').read()
lines = bib.split('\n')

# Find lines that contain @article{ but do NOT start with @ (after optional whitespace)
fixed = 0
out_lines = []
for i, ln in enumerate(lines):
    stripped = ln.strip()
    if '@article{' in stripped and not stripped.startswith('@'):
        # garbage prefix before @article{key,
        m = re.match(r'^(.*?)@(article|book|inproceedings|misc|techreport|phdthesis)\{', stripped)
        if m:
            garbage = m.group(1)
            # sanity: garbage should be short junk (DOI/line-no/number), not real text
            print(f'L{i+1}: FIX  garbage={garbage!r}  line={stripped[:80]}')
            ln = ln.replace(garbage + '@', '@', 1)
            fixed += 1
    out_lines.append(ln)

bib = '\n'.join(out_lines)
open(P, 'w', encoding='utf-8').write(bib)
print(f'\nFixed {fixed} entries')
