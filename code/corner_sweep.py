"""Corner sweep: for every legend in the figures, test candidate positions,
report the one with minimal data-under-legend. Guides manual fixes."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures
import sci_style as ss

ss.apply_style()

def measure(ax, leg, pos):
    """Return fraction of data pixels under legend at given loc."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            leg.set_bbox_to_anchor(None)
            leg.set_loc(pos)
        except Exception:
            return 1.0
    fig = ax.figure
    if not isinstance(fig.canvas, FigureCanvasAgg):
        fig.canvas = FigureCanvasAgg(fig)
    fig.canvas.draw()
    bb = leg.get_window_extent()
    x0, y0, x1, y1 = int(bb.x0), int(bb.y0), int(bb.x1), int(bb.y1)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return 1.0
    buf_with = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    leg.set_visible(False)
    fig.canvas.draw()
    buf_no = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    leg.set_visible(True)
    fig.canvas.draw()
    H = buf_with.shape[0]
    y0b, y1b = max(0, H - y1), min(H, H - y0)
    x0b, x1b = max(0, x0), min(buf_with.shape[1], x1)
    if y1b <= y0b or x1b <= x0b:
        return 1.0
    return (buf_no[y0b:y1b, x0b:x1b].sum(axis=2) < 720).mean()

CAND = ["upper left", "upper right", "lower left", "lower right",
        "center right", "center left", "upper center", "lower center"]

captured = []
orig_save = plt.Figure.savefig
def patched(self, fname, *a, **k):
    captured.append((self, os.path.basename(str(fname))))
    return orig_save(self, fname, *a, **k)
plt.Figure.savefig = patched

ctx = figures.Ctx()
for k in sorted(figures.FIGURES):
    try:
        figures.FIGURES[k](ctx)
    except Exception as e:
        print(f"  [ERROR] fig{k}: {e}")

print("\n=== Corner sweep results (best loc per legend) ===")
for fig, fname in captured:
    if not fname.endswith(".png"):
        continue
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        title = (ax.get_title() or (ax.texts[0].get_text() if ax.texts else ""))[:22]
        best_pos, best_frac = None, 1.0
        for pos in CAND:
            try:
                f = measure(ax, leg, pos)
            except Exception:
                f = 1.0
            if f < best_frac:
                best_frac, best_pos = f, pos
        print(f"  {fname:<30} [{title}]: best={best_pos}  data-under-legend={best_frac*100:.1f}%")
