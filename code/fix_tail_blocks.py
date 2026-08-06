"""Rewrite SIMPLEX tex tail blocks per original TransMICRO PDF structure."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
src = open(p, encoding="utf-8").read()

marker = "\\section*{Data Availability Statement}"
bib_marker = "\\bibliographystyle"
start = src.index(marker)
end = src.index(bib_marker)

new_tail = """\\section*{Data Availability Statement}

The raw experimental dataset analysed in this study is publicly available from the repository \\emph{sheng-hu/hydrogels} (MIT licence) accompanying Liao et al., \\textit{Nature} 644, 89--95 (2025), DOI 10.1038/s41586-025-09269-4. The training code, the processed tensors used in this work, the per-formulation predictions underlying every reported table and figure, and the regeneration scripts are all available from the GitHub repository \\url{https://github.com/SHENTongfei/HydroGelNet}. No additional materials were generated.

\\section*{Author Contributions}

T.S., X.L., and H.K.L. contributed equally to this work and are jointly listed as co-first authors. T.S. and X.L. conceived the study, designed the experimental workflow, and curated the public hydrogel adhesion dataset. H.K.L. developed the SIMPLEX architecture and implemented the training and inference pipeline. H.L. and Z.P. executed the downstream interpretability analyses, including permutation importance, attention attribution, and partial dependence profiling. X.F. supervised the project, secured funding, and wrote the manuscript. All authors reviewed the final version and approved its submission.

\\section*{Funding}

This work was supported by the National Natural Science Foundation of China (No. 62505285).

\\section*{Acknowledgments}

The authors gratefully acknowledge Liao et al. and the maintainers of the open hydrogel dataset for making the experimental data publicly available under the MIT licence. Constructive discussions with colleagues at the participating institutions are also appreciated.

\\section*{Conflict of Interest Statement}

The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

"""

src = src[:start] + new_tail + src[end:]
open(p, "w", encoding="utf-8").write(src)
print("tail blocks rewritten per original PDF")
