"""Quick check of all figure captions - panel coverage and consistency."""
import re

with open(r"C:/Users/TS/WorkBuddy/HydroGelNet/paper/frontiers_SIMPLEX.tex", "r", encoding="utf-8") as f:
    tex = f.read()

# Find all \caption[short]{long} blocks
caps = re.findall(r"\\caption\[(.*?)\]\{(.*?)\}", tex, re.DOTALL)
print(f"=== {len(caps)} figure captions ===")
for i, (short, full) in enumerate(caps, 1):
    panels = re.findall(r"\(([A-I])\)", full)
    n_unique = len(set(panels))
    missing = sorted(set("ABCDEFGHI") - set(panels))
    extra = sorted(set(panels) - set("ABCDEFGHI"))
    print(f"[{i}] {short[:50]}")
    print(f"    panels: {panels}")
    print(f"    unique: {n_unique}, missing: {missing or '-'}, extras: {extra or '-'}")
    print()
