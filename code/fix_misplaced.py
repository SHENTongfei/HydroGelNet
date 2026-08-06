"""Fix two misplaced cite groups (before \\subsection and \\begin{figure*})."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
t = open(p, encoding="utf-8").read()

# 1) \cite{tibshirani1996lasso} before \subsection{Limitations}
t = t.replace(
    "\\cite{tibshirani1996lasso}\n\n\\subsection{Limitations}",
    "\\subsection{Limitations}")

# 2) \cite{ward2018matminer} before \begin{figure*}
t = t.replace(
    "\\cite{ward2018matminer}\n\n\\begin{figure*}",
    "\\begin{figure*}")

# re-attach removed cites to the preceding paragraph
# find the paragraph before each affected heading and append the cite
t = t.replace(
    "range-restricted Spearman $\\rho$ is a conservative estimate of ranking ability.",
    "range-restricted Spearman $\\rho$ is a conservative estimate of ranking ability "
    "\\cite{tibshirani1996lasso}.")
t = t.replace(
    "exposed which measured quantities actually carry information.",
    "exposed which measured quantities actually carry information "
    "\\cite{ward2018matminer}.")

open(p, "w", encoding="utf-8").write(t)
print("fixed 2 misplaced cites")
