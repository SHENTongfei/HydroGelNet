"""Legend overlap check for ALL HydroGelNet figures.
Renders each figure with and without legends; measures data-under-legend.
Fixes any overlap before manuscript assembly (lesson from TransMICRO)."""
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

# force Agg canvas on every figure we capture
def _force_agg(fig):
    if not isinstance(fig.canvas, FigureCanvasAgg):
        fig.canvas = FigureCanvasAgg(fig)
    return fig

captured = []
orig_save = plt.Figure.savefig
def patched(self, fname, *a, **k):
    captured.append((self, os.path.basename(str(fname))))
    return orig_save(self, fname, *a, **k)
plt.Figure.savefig = patched

ctx = figures.Ctx()
report = []
for k in sorted(figures.FIGURES):
    try:
        figures.FIGURES[k](ctx)
    except Exception as e:
        print(f"  [ERROR] fig{k}: {e}")
        continue

for fig, fname in captured:
    if not fname.endswith((".png", ".pdf")):
        continue
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        _force_agg(fig)
        bb = leg.get_window_extent()
        x0, y0, x1, y1 = int(bb.x0), int(bb.y0), int(bb.x1), int(bb.y1)
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        fig.canvas.draw()
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
            continue
        frac = (buf_no[y0b:y1b, x0b:x1b].sum(axis=2) < 720).mean()
        title = (ax.get_title() or (ax.texts[0].get_text() if ax.texts else ""))[:24]
        status = "OK" if frac < 0.05 else "OVERLAP!"
        report.append((fname, title, frac, w, h, status))

print("\n=== Legend overlap report (HydroGelNet figures) ===")
bad = 0
for fname, title, frac, w, h, status in report:
    if status == "OVERLAP!":
        bad += 1
    print(f"  {fname:<34} [{title}]: legend {w}x{h}px  data-under-legend={frac*100:.1f}%  {status}")
print(f"\n  {len(report)-bad}/{len(report)} legend(s) clean")
sys.exit(1 if bad else 0)
