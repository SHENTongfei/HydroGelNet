# -*- coding: utf-8 -*-
"""Build local natbib test file from v7_merged.tex."""
import re

SRC = r'C:\Users\TS\WorkBuddy\HydroGelNet\audit\rewrite_baseline\v7_merged.tex'
OUT = r'C:\Users\TS\WorkBuddy\HydroGelNet\_local_compile\v7_test.tex'

src = open(SRC, encoding='utf-8').read()
body_start = src.find(r'\begin{document}')
preamble = (
    r'\documentclass[10pt]{article}' + '\n'
    r'\usepackage[margin=0.6in]{geometry}' + '\n'
    r'\usepackage{graphicx}' + '\n'
    r'\usepackage{caption}' + '\n'
    r'\usepackage{amsmath,amssymb}' + '\n'
    r'\usepackage{booktabs}' + '\n'
    r'\usepackage{multirow}' + '\n'
    r'\usepackage{url}' + '\n'
    r'\usepackage[hidelinks]{hyperref}' + '\n'
    r'\usepackage{subcaption}' + '\n'
    r'\usepackage[round,authoryear]{natbib}' + '\n'
    r'\setlength\textwidth{6.5in}' + '\n'
    r'\begin{document}' + '\n'
)
body = src[body_start:]
body = re.sub(r'\\def\\keyFont.*?\\def\\firstAuthorLast', '', body, flags=re.DOTALL)
body = re.sub(r'\\def\\(firstAuthorLast|Authors|Address|corrAuthor|corrAddress|corrEmail)\{.*?\}\n', '', body, flags=re.DOTALL)
for cmd in ['onecolumn', 'firstpage', 'author', 'address', 'correspondance', 'extraAuth', 'maketitle']:
    body = re.sub(r'\\' + cmd + r'(\{[^}]*\})?', '', body)
out = preamble + body
out = out.replace(r'\end{document}',
                  r'\bibliographystyle{plainnat}\bibliography{reference_final}\end{document}')
open(OUT, 'w', encoding='utf-8').write(out)
print('test tex written, len', len(out))
