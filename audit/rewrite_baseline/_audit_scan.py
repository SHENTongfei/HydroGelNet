# -*- coding: utf-8 -*-
import re

lines = open(r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex', encoding='utf-8').read().split('\n')

print('LINE 305 (Conclusion):')
print(lines[304])
print()

print('=== colons in body-ish lines ===')
for i, l in enumerate(lines, 1):
    if re.match(r'\s*\\begin{tabular}', l):
        continue
    if '\\\\' in l:
        continue
    if 'http' in l:
        continue
    if '\\label' in l:
        continue
    if l.strip().startswith('%'):
        continue
    if ':' in l:
        # filter: table header like "Model & R^2 & ..." no colon. URLs already filtered.
        print(f'{i}: {l[:220]}')

print()
print('=== Figure~/ref / ref{fig usage ===')
for i, l in enumerate(lines, 1):
    if '\\ref' in l and 'Fig' in l:
        print(f'{i}: {l[:220]}')

print()
print('=== includegraphics calls ===')
for i, l in enumerate(lines, 1):
    if '\\includegraphics' in l:
        print(f'{i}: {l.strip()[:220]}')

print()
print('=== seed protocol mentions ===')
for i, l in enumerate(lines, 1):
    if re.search(r'\b\d+\s*seed', l, re.I) or '50 models' in l or '50 fold' in l:
        print(f'{i}: {l[:220]}')

print()
print('=== placeholders ===')
for i, l in enumerate(lines, 1):
    if re.search(r'\[待填写\]|TODO|TBD|XXX|placeholder|FIXME|\[L5\]|\?\?', l, re.I):
        print(f'{i}: {l[:220]}')

print()
print('=== begin/end balance ===')
print('begin document:', len(re.findall(r'\\begin\{document\}', '\n'.join(lines))))
print('end document:', len(re.findall(r'\\end\{document\}', '\n'.join(lines))))
print('begin figure:', len(re.findall(r'\\begin\{figure', '\n'.join(lines))))
print('end figure:', len(re.findall(r'\\end\{figure', '\n'.join(lines))))
print('begin table:', len(re.findall(r'\\begin\{table', '\n'.join(lines))))
print('end table:', len(re.findall(r'\\end\{table', '\n'.join(lines))))
print('begin abstract:', len(re.findall(r'\\begin\{abstract\}', '\n'.join(lines))))
print('end abstract:', len(re.findall(r'\\end\{abstract\}', '\n'.join(lines))))

print()
print('=== external R2 values across doc ===')
for i, l in enumerate(lines, 1):
    if re.search(r'0\.69|0\.6946|0\.71|0\.63|0\.6342', l):
        print(f'{i}: {l[:220]}')
