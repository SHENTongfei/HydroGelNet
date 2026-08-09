# -*- coding: utf-8 -*-
"""Normalize bib: ensure every @article{key, starts at line start and entries are properly closed."""
import re

P = r'C:\Users\TS\WorkBuddy\HydroGelNet\paper\reference_final.bib'
bib = open(P, encoding='utf-8').read()
lines = bib.split('\n')

# Strategy: find all @type{key, positions. Everything before first @ is dropped.
# Then walk: an entry starts at '@' and ends at the matching '}' at brace-depth 0 on its own line.

# First, merge into single stream (keep line structure minimal): join with newline marker
text = '\n'.join(lines)

# Find entry starts: @word{key,
entry_starts = []
for m in re.finditer(r'@(article|book|inproceedings|misc|techreport|phdthesis|incollection)\s*\{\s*([^,\s}]+)\s*,', text):
    entry_starts.append((m.start(), m.group(1), m.group(2)))

print(f'found {len(entry_starts)} entry starts')

# Walk entries by brace depth from each start
entries = []
for idx, (start, etype, key) in enumerate(entry_starts):
    end = entry_starts[idx + 1][0] if idx + 1 < len(entry_starts) else len(text)
    block = text[start:end]
    # block may contain a leading field remnant (e.g. "pages = {...},\n" before @article)
    # remove anything before the '@'
    at = block.find('@')
    if at > 0:
        remnant = block[:at]
        if remnant.strip():
            print(f'  [{key}] dropped remnant before @: {remnant.strip()[:60]!r}')
        block = block[at:]
    # ensure the entry has closing '}' : find brace balance
    depth = 0
    closed_at = -1
    in_str = False
    for i, ch in enumerate(block):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                closed_at = i
                break
    if closed_at == -1:
        print(f'  [WARN {key}] no balanced close, appending }}')
        block = block.rstrip() + '\n}\n'
    else:
        # truncate to closed_at+1, but only if the remainder is whitespace or belongs to next
        block = block[:closed_at + 1]
    entries.append((etype, key, block))

# rebuild bib: entries separated by blank line
out = '\n\n'.join(b for _, _, b in entries) + '\n'
open(P, 'w', encoding='utf-8').write(out)
print(f'normalized to {len(entries)} entries, {len(out)} chars')
