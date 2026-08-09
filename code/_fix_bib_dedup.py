# -*- coding: utf-8 -*-
"""Fix bib: double commas and duplicate entries."""
import re

P = r'C:\Users\TS\WorkBuddy\HydroGelNet\paper\reference_final.bib'
bib = open(P, encoding='utf-8').read()

# 1) fix double comma after field values: '},,' -> '},'
n1 = bib.count(',,')
bib = bib.replace(',,', ',')
print(f'double commas fixed: {n1}')

# 2) remove duplicate entries (keep first occurrence)
keys = re.findall(r'@\w+\{([^,]+),', bib)
from collections import Counter
dups = {k: v for k, v in Counter(keys).items() if v > 1}
print(f'duplicate keys to dedupe: {dups}')

# parse entries: find each @type{key, ... } balanced block
def parse_entries(text):
    entries = []
    for m in re.finditer(r'@(\w+)\{([^,}]+),(.*?)\n\}', text, re.DOTALL):
        entries.append((m.start(), m.end(), m.group(1), m.group(2), m.group(3)))
    return entries

# do it iteratively: find spans of each @..{} block with proper brace counting
def split_entries(text):
    """Return list of (start, end, type, key) for each top-level @type{...} entry."""
    out = []
    i = 0
    while i < len(text):
        m = re.match(r'@(\w+)\{([^,}]+),', text[i:])
        if not m:
            i += 1
            continue
        typ, key = m.group(1), m.group(2)
        start = i
        depth = 0
        j = i
        while j < len(text):
            ch = text[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((start, j + 1, typ, key))
        i = j + 1
    return out

entries = split_entries(bib)
print(f'total entries parsed: {len(entries)}')

# dedupe: keep first occurrence
seen = set()
keep = []
for start, end, typ, key in entries:
    if key in seen:
        print(f'  REMOVE dup: {key} (span {start}-{end})')
        continue
    seen.add(key)
    keep.append((start, end))

# rebuild: keep only kept spans in order
parts = []
last = 0
for start, end in keep:
    if start > last:
        pass  # gaps handled by keeping spans directly
# simpler: build from original text by removing dup spans (from back to front)
drop_spans = []
seen2 = set()
for start, end, typ, key in entries:
    if key in seen2:
        drop_spans.append((start, end))
    else:
        seen2.add(key)
for start, end in sorted(drop_spans, reverse=True):
    bib = bib[:start] + bib[end:]
    print(f'  removed span {start}-{end}')

open(P, 'w', encoding='utf-8').write(bib)
final_keys = re.findall(r'@\w+\{([^,]+),', bib)
print(f'final entries: {len(final_keys)}, unique: {len(set(final_keys))}')
