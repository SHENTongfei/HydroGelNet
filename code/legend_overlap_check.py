"""Legend overlap detector (difference method) for SCI figures.

Renders each figure once WITH legend and once WITHOUT, then measures what
fraction of non-white pixels inside the legend bbox is actual DATA (i.e.
pixels that exist in the no-legend render). Used to verify no legend/data
overlap in the final figures (lesson from TransMICRO Fig3B/4D/7H incident).

Usage:
    python legend_overlap_check.py <figure_script.py> [--panel P1,P2,...]

If --panel is omitted, checks all axes that contain a legend.
"""
import sys, os, subprocess
import numpy as np

def check_script(script: str, panels=None) -> int:
    """Import the figure script, monkey-patch save to intercept figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = []

    orig_save = plt.Figure.savefig
    def patched_save(self, fname, *a, **k):
        captured.append((self, str(fname)))
        return orig_save(self, fname, *a, **k)
    plt.Figure.savefig = patched_save

    # run the script (it must be self-contained, saving via sci_style.save_figure)
    spec = importlib.util.spec_from_file_location("figmod", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()

    # for each captured figure, measure legend overlap
    results = []
    for fig, fname in captured:
        if not fname.endswith((".png", ".pdf")):
            continue
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            # bbox in display coords
            bb = leg.get_window_extent()
            x0, y0 = int(bb.x0), int(bb.y0)
            x1, y1 = int(bb.x1), int(bb.y1)
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue
            # render with legend
            fig.canvas.draw()
            buf_with = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            # hide legend, render again
            leg.set_visible(False)
            fig.canvas.draw()
            buf_no = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            leg.set_visible(True)
            fig.canvas.draw()

            # invert y for buffer (buffer origin top-left, window origin bottom-left)
            H = buf_with.shape[0]
            y0b, y1b = H - y1, H - y0
            y0b, y1b = max(0, y0b), min(H, y1b)
            x0b, x1b = max(0, x0), min(buf_with.shape[1], x1)
            if y1b <= y0b or x1b <= x0b:
                continue
            region_with = buf_with[y0b:y1b, x0b:x1b]
            region_no = buf_no[y0b:y1b, x0b:x1b]
            nonwhite_no = (region_no.sum(axis=2) < 720).mean()  # not near-white
            # data pixels = non-white in no-legend render
            title = ax.get_title() or (ax.texts[0].get_text() if ax.texts else "")
            results.append((fname, ax, nonwhite_no, w, h, title))
    return results

if __name__ == "__main__":
    import importlib.util
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    script = sys.argv[1]
    results = check_script(script)
    print(f"\n=== Legend overlap report for {os.path.basename(script)} ===")
    bad = 0
    for fname, ax, frac, w, h, title in results:
        status = "OK" if frac < 0.05 else "OVERLAP!"
        if frac >= 0.05:
            bad += 1
        print(f"  {os.path.basename(fname)} [{title[:30]}]: "
              f"legend {w}x{h}px, data-under-legend={frac*100:.1f}%  {status}")
    print(f"\n  {len(results)-bad}/{len(results)} panels clean")
    sys.exit(1 if bad else 0)
