# -*- coding: utf-8 -*-
import re
t = open(r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex', encoding='utf-8').read()
# strip comments
t2 = re.sub(r'(?<![\\])%.*', '', t)
print('open braces:', t2.count('{'), 'close braces:', t2.count('}'))
print('open parens:', t2.count('('), 'close parens:', t2.count(')'))
print('dollar signs:', t2.count('$'))
print('has 0.71:', '0.71' in t2)
print('has 0.69:', '0.69' in t2)
print('has FigureN_:', t2.count('FigureN_'))
print('has FigN_ old:', len(re.findall(r'Fig[0-9]_', t2)))
print('has Figure~ref:', 'Figure~' in t2)
print('has ref{fig:', len(re.findall(r'\\ref\{fig', t2)))
