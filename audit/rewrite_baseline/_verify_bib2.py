# -*- coding: utf-8 -*-
import re
bib = open(r'C:\Users\TS\WorkBuddy\HydroGelNet\paper\reference_final.bib', encoding='utf-8').read()
bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
print('bib entries:', len(bibkeys))
tex = open(r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex', encoding='utf-8').read()
cited = set()
for m in re.finditer(r'\\cite\{([^}]*)\}', tex):
    for k in m.group(1).split(','):
        cited.add(k.strip())
print('cited keys:', len(cited))
missing = sorted(c for c in cited if c not in bibkeys)
print('cited but MISSING from bib:', missing if missing else 'NONE')
