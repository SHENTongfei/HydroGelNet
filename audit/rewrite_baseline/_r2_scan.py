# -*- coding: utf-8 -*-
import re

t = open(r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex', encoding='utf-8').read()
lines = t.split('\n')
n = len(lines)

print('total chars:', len(t), 'total lines:', n)
print()
print('=== 1. 0.71 residual ===')
hits = [i+1 for i, l in enumerate(lines) if '0.71' in l]
print('lines with 0.71:', hits if hits else 'NONE (0 residual)')
print()
print('=== 2. en-dash / em-dash / triple-hyphen / double-hyphen ===')
for i, l in enumerate(lines, 1):
    for ch, name in [('\u2014', 'EM'), ('\u2013', 'EN'), ('---', 'TRIPLE'), ('--', 'DOUBLE')]:
        if ch in l:
            print(f'{i}: [{name}] {l[:150]}')
print('(no output above = clean)')
print()
print('=== 3. colons in body text (excluding \\label, URLs, table specs) ===')
for i, l in enumerate(lines, 1):
    if '\\label' in l: continue
    if 'http' in l: continue
    if re.match(r'\s*\\begin{tabular}', l): continue
    if l.strip().startswith('%'): continue
    if '\\\\' in l: continue
    if ':' in l:
        print(f'{i}: {l[:220]}')
print('(no output above = clean)')
print()
print('=== 4. Figure~\\ref / FigN_ old naming ===')
print('Figure~ occurrences:', t.count('Figure~'))
print('FigN_ old naming:', len(re.findall(r'Fig[0-9]_', t)))
print('FigureN_ naming count:', len(re.findall(r'Figure[0-9]_', t)))
print()
print('=== 5. environment balance ===')
for env in ['document', 'abstract', 'figure*', 'table', 'figure']:
    b = len(re.findall(r'\\begin\{' + env + r'\}', t))
    e = len(re.findall(r'\\end\{' + env + r'\}', t))
    print(f'{env}: begin={b} end={e} {"OK" if b==e else "MISMATCH"}')
print()
print('=== 6. brace/paren/dollar balance ===')
t2 = re.sub(r'(?<![\\])%.*', '', t)
print('braces:', t2.count('{'), t2.count('}'))
print('parens:', t2.count('('), t2.count(')'))
print('dollars:', t2.count('$'))
print()
print('=== 7. placeholders ===')
ph = [(i+1) for i, l in enumerate(lines) if re.search(r'\[待填写\]|TODO|TBD|FIXME|placeholder|\?\?', l, re.I)]
print('placeholders:', ph if ph else 'NONE')
print()
print('=== 8. includegraphics files ===')
for i, l in enumerate(lines, 1):
    if '\\includegraphics' in l:
        m = re.search(r'\{([^}]*)\}', l)
        print(f'{i}: {m.group(1) if m else l.strip()}')
print()
print('=== 9. seeds protocol ===')
for i, l in enumerate(lines, 1):
    if re.search(r'\b5 seeds\b', l, re.I):
        print(f'LINE {i}: 5 seeds FOUND -> {l[:120]}')
    if '10 seeds' in l or '50 models' in l:
        print(f'{i}: {l[:110]}')
print()
print('=== 10. external R2 values (0.69 / 0.6946 / 0.6342) ===')
for i, l in enumerate(lines, 1):
    if re.search(r'0\.6946|0\.6342', l):
        print(f'{i}: ...{l[max(0,l.find("0.69")-40):l.find("0.69")+80] if "0.69" in l else l[:120]}')
