"""Replace fig1/fig2 in figures.py with redesigned implementations."""
import re

SRC = "figures.py"
with open(SRC, encoding="utf-8") as f:
    txt = f.read()

with open("redraw_fig12.py", encoding="utf-8") as f:
    redraw_lines = f.read().splitlines()

# split redraw source into function blocks by top-level def lines
def split_blocks(lines):
    blocks, cur, cur_name = [], [], None
    for ln in lines:
        if ln.startswith("def ") and cur_name is not None:
            blocks.append((cur_name, "\n".join(cur)))
            cur = []
        if ln.startswith("def "):
            cur_name = ln.split("(")[0].replace("def ", "").strip()
            cur = [ln]
        elif cur_name is not None:
            cur.append(ln)
    if cur_name:
        blocks.append((cur_name, "\n".join(cur)))
    return dict(blocks)

blocks = split_blocks(redraw_lines)
new_fig1 = blocks["fig1"].replace("def fig1() -> None:",
                                  "def fig1(ctx: Ctx) -> None:")
new_fig2 = blocks["fig2"].replace("def fig2() -> None:",
                                  "def fig2(ctx: Ctx) -> None:")

start = txt.index("def fig1(ctx")
end = txt.index("def fig3(ctx")
new_txt = txt[:start] + new_fig1 + "\n\n\n" + new_fig2 + "\n\n\n" + txt[end:]

with open(SRC, "w", encoding="utf-8") as f:
    f.write(new_txt)
import ast
ast.parse(new_txt)
print("figures.py updated: fig1/fig2 replaced (syntax OK)")
