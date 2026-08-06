"""Close two truncated \\cite blocks."""
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper",
                 "frontiers_SIMPLEX.tex")
t = open(p, encoding="utf-8").read()

# 1) close the truncated cite at end of workflow paragraph
t = t.replace(
    "\\cite{jain2013materials,audus2019polymer,ward2017magpie,ward2018matminer\n",
    "\\cite{jain2013materials,audus2019polymer,ward2017magpie,ward2018matminer}.\n")

# 2) remove the dangling duplicate tibshirani cite fragment (already cited at L66)
t = t.replace(
    " under target-value extrapolation. \\cite{tibshirani1996lasso\n",
    " under target-value extrapolation.\n")

open(p, "w", encoding="utf-8").write(t)
print("fixed truncated cites")
