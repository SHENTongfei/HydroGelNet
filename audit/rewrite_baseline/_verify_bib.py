# -*- coding: utf-8 -*-
import re, os

tex = open(r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex', encoding='utf-8').read()
lines = tex.split('\n')

# 1. citations in Results region (lines 110-271 = Results)
print('=== \\cite occurrences per line (whole doc) ===')
for i, l in enumerate(lines, 1):
    if '\\cite' in l:
        keys = re.findall(r'\\cite\{([^}]*)\}', l)
        flat = [k.strip() for ks in keys for k in ks.split(',')]
        print(f'{i}: {flat}')

print()
print('=== bib file check ===')
bibpath = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\reference_final.bib'
if os.path.exists(bibpath):
    bib = open(bibpath, encoding='utf-8').read()
    bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
    print('bib entries:', len(bibkeys))
    cited = set()
    for m in re.finditer(r'\\cite\{([^}]*)\}', tex):
        for k in m.group(1).split(','):
            cited.add(k.strip())
    print('cited keys:', len(cited))
    missing = sorted(c for c in cited if c not in bibkeys)
    print('cited but MISSING from bib:', missing if missing else 'NONE')
else:
    print('bib file NOT FOUND at', bibpath)
