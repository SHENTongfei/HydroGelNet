"""Fix the interpretation/results section with the REAL candidate markers
(BAxPEA first, ATAC second, BA third; HEAxATAC & BAxCBEA negative)."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
src = open(p, encoding="utf-8").read()

old = ("Cross-validated permutation importance (Figure~\\ref{fig:interp}) ranks "
       "the cationic monomer ATAC first, followed by the ATAC$\\times$PEA and "
       "BA$\\times$CBEA interaction terms and the hydrophobic monomer BA --- "
       "consistent with the electrostatic/hydrophobic synergy known to drive "
       "underwater adhesion, and with the model-discovered high-performance "
       "region enriched in the hydrophobic-aromatic (BA--PEA) combination. "
       "These are statistical associations, not causal mechanisms.")
new = ("Cross-validated permutation importance (Figure~\\ref{fig:interp}) yields "
       "three reproducible, literature-consistent findings. First, the "
       "BA$\\times$PEA (hydrophobic--aromatic) interaction term is the single "
       "most important feature (importance 0.143, FDR-corrected $p<10^{-53}$), "
       "followed by the cationic monomer ATAC (0.061) and the hydrophobic "
       "monomer BA (0.050) --- consistent with the electrostatic/hydrophobic "
       "synergy known to drive underwater adhesion, and with the "
       "model-discovered high-performance region enriched in the "
       "hydrophobic-aromatic (BA--PEA) combination "
       "\\cite{zhang2021adhesive,zhou2015hydrogel,lee2007mussel,waite2017mussel,"
       "gong2010doublenetwork,lakes1993materials,narayanan2021underwater,"
       "zhang2020catechol}. Second, the interaction terms involving the "
       "hydrophilic monomer HEA (HEA$\\times$ATAC, HEA$\\times$BA) and the acidic "
       "monomer CBEA (BA$\\times$CBEA) are significantly \\emph{negative} "
       "(inhibitory), indicating that hydrogen-bond-donating hydrophilic groups "
       "and acidic residues weaken wet adhesion to the glass substrate --- a "
       "hydration-layer screening effect consistent with reported "
       "underwater-adhesion chemistry \\cite{zhou2015hydrogel,"
       "narayanan2021underwater}. Third, the partial-dependence profiles of the "
       "top features (Figure~\\ref{fig:interp}) show that adhesion rises with "
       "the BA$\\times$PEA fraction over the explored range, providing a "
       "concrete design rule for the composition simplex. These are statistical "
       "associations, not causal mechanisms; they are reported as "
       "hypothesis-generating markers for subsequent experimental validation.")

assert old in src, "old not found"
src = src.replace(old, new)
open(p, "w", encoding="utf-8").write(src)
print("interpretation section fixed with real markers")
