"""Patch figures.py: insert SIMPLEX schematic helpers + fix merged comment/def."""
import ast

src = open("figures.py", encoding="utf-8").read()

# 1) split merged comment+def line (from previous bad patch)
src = src.replace(
    "# ================================def fig1(ctx: Ctx) -> None:",
    "# =========================================================================== #\n"
    "# Figure 1 -- SIMPLEX pipeline (schematic)\n"
    "# =========================================================================== #\n"
    "def fig1(ctx: Ctx) -> None:")

# 2) helper definitions inserted before the Figure-1 banner
helpers = '''
PAL = {
    "blue":   ("#CCE4FC", "#2E6DA4"),
    "green":  ("#E4FCFC", "#2E8B57"),
    "red":    ("#FCE4E4", "#C0392B"),
    "purple": ("#FCE4FC", "#8E44AD"),
    "orange": ("#FCE4CC", "#E67E22"),
    "bp":     ("#E4E4FC", "#5B6EE1"),
    "grey":   ("#E4E4E4", "#555555"),
}

class Layout:
    def __init__(self):
        self.boxes = []
    def add(self, x0, y0, x1, y1, label=""):
        self.boxes.append((x0, y0, x1, y1, label))
    def check(self) -> list:
        bad = []
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                a, b = self.boxes[i], self.boxes[j]
                ix = min(a[2], b[2]) - max(a[0], b[0])
                iy = min(a[3], b[3]) - max(a[1], b[1])
                if ix > 0.4 and iy > 0.4:
                    bad.append((a[4], b[4], round(ix, 2), round(iy, 2)))
        return bad

def _shadow(ax, x, y, w, h, r):
    ax.add_patch(FancyBboxPatch((x + 0.28, y - 0.30), w, h,
                 boxstyle=f"round,pad=0.12,rounding_size={r}",
                 facecolor="#00000010", edgecolor="none", zorder=2))

def _pbox(ax, L, x, y, w, h, label, key, fs=7.0, bold=False, lw=1.4, r=0.5,
          sub=None):
    face, edge = PAL[key]
    _shadow(ax, x, y, w, h, r)
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.12,rounding_size={r}",
                       facecolor=face, edgecolor=edge, lw=lw, zorder=3)
    ax.add_patch(p)
    text = label if sub is None else label + "\\n" + sub
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1a1a1a", zorder=4,
            linespacing=1.35)
    L.add(x - 0.07, y - 0.07, x + w + 0.07, y + h + 0.07, label.split("\\n")[0])
    return (x, y, w, h)

def _parrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=2.5))

def _stage_label(ax, L, x, text):
    ax.text(x, 41.6, text, fontsize=7.5, ha="center", color="#333333",
            fontweight="bold")
    ax.plot([x - 6, x + 6], [40.6, 40.6], color="#BBBBBB", lw=0.8)

def _psave(fig, name, L, outdir):
    bad = L.check()
    print(f"  {name}: {len(L.boxes)} elements, overlaps = {len(bad)}")
    for b in bad:
        print(f"    OVERLAP: {b[0]} <-> {b[1]} (ix={b[2]}, iy={b[3]})")
    fig.savefig(f"{outdir}/{name}.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(f"{outdir}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

'''

anchor = "# Figure 1 -- SIMPLEX pipeline (schematic)"
src = src.replace(anchor, helpers + anchor)

# 3) rename helper calls inside fig1/fig2 to local names
src = src.replace("box(ax, L,", "_pbox(ax, L,")
src = src.replace("arrow(ax,", "_parrow(ax,")
src = src.replace("stage_label(ax, L,", "_stage_label(ax, L,")
src = src.replace("save(fig,", "_psave(fig,")

open("figures.py", "w", encoding="utf-8").write(src)
try:
    ast.parse(src)
    print("syntax OK")
except SyntaxError as e:
    print(f"SYNTAX line {e.lineno}: {e.msg}")
