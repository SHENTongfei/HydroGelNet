# -*- coding: utf-8 -*-
import re
short = 'Characteristics of the internal and prospective cohorts'
label = 'fig:dataset'
pat_str = r'(\caption\[' + re.escape(short) + r'\]\{)(.*?)}\label\{' + re.escape(label) + r'\}'
print(repr(pat_str))
p = re.compile(pat_str, re.DOTALL)
print('compiles OK')
src = open(r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex', encoding='utf-8').read()
m = p.search(src)
print('match found:', bool(m))
