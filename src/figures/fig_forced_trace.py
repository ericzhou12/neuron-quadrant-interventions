"""Picture of forced continuation on one real trace.

The text is an excerpt (lightly trimmed) of the thinking block for GSM8K problem
idx 1 (answer 3) in q3_grow@0.75__forced__eos__seed1.jsonl: Qwen3-1.7B with Q3
activations x1.75. Each time the model emitted </think> (or ended the text)
before reaching the intact model's thinking length, the token was replaced with "Wait" and
generation carried on. Only the word "Wait" is inserted; everything else is the
model's own output.

Usage: python src/figures/fig_forced_trace.py [results/q3__Qwen__Qwen3-1.7B]
Writes figures/fig10_forced_trace.png.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

style.apply()
folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q3__Qwen__Qwen3-1.7B")
STOP = "<STOP>"  # marks a place where the model emitted </think> and it was replaced

LINES = [
    "Yes, that's correct. So the answer is 3.",
    "",
    "**Final Answer**",
    "\\boxed{3}",
    STOP,
    'Wait, but hold on. Wait, the problem says "half that much white',
    'fiber." Is "that much" referring to the blue fiber? Yes. So white',
    "is half of blue. So 2 blue, 1 white. Total 3.",
    "",
    "Yes, that's correct. So the answer is 3.",
    "",
    "**Final Answer**",
    "\\boxed{3}",
    STOP,
    'Wait, but the original problem says "half that much white fiber."',
    "So if blue is 2, then white is 1. So total 3.",
    "",
    "Yes. So the answer is 3.",
    "",
    "**Final Answer**",
    "\\boxed{3}",
    STOP,
    "Wait...",
]
# lines that START with the inserted word (the model also says "Wait" on its own mid-line)
INSERTED = {i for i, line in enumerate(LINES) if i > 0 and LINES[i - 1] == STOP}

MONO = "DejaVu Sans Mono"
FS, LH = 11.5, 0.31          # font size (pt), line height (inches)
W = 9.4
H = 1.55 + LH * len(LINES) + 0.75
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

fig.text(0.035, 1 - 0.42 / H, "Forcing the model to keep thinking", fontsize=14.5, fontweight="bold", color="#0b0b0b", va="top")
fig.text(0.035, 1 - 0.78 / H, "Qwen3-1.7B with Q3 activations ×1.75, one GSM8K problem (correct answer: 3). "
         "Excerpt from the model's thinking.", fontsize=9.5, color=style.INK2, va="top")

x0, top = 0.75, H - 1.55
card = FancyBboxPatch((0.38, top - LH * len(LINES) - 0.12), W - 0.76, LH * len(LINES) + 0.42,
                      boxstyle="round,pad=0,rounding_size=0.12", fc="#ffffff", ec=style.GRID, lw=1.2)
ax.add_patch(card)

# width of one monospace character, in data units (inches here)
probe = ax.text(0, 0, "M" * 50, family=MONO, fontsize=FS, alpha=0)
fig.canvas.draw()
cw = probe.get_window_extent().width / fig.dpi / 50
probe.remove()

for i, line in enumerate(LINES):
    y = top - LH * i
    if line == STOP:
        tok = "<stop>"  # the file does not record whether it was </think> or end-of-text
        ax.text(x0, y, tok, family=MONO, fontsize=FS, color=style.Q4_COL, va="center")
        ax.plot([x0 - 0.03, x0 + cw * len(tok) + 0.03], [y, y], color=style.Q4_COL, lw=1.6, solid_capstyle="round")
        ax.annotate("", xy=(x0 + cw * len(tok) + 0.62, y), xytext=(x0 + cw * len(tok) + 0.16, y),
                    arrowprops=dict(arrowstyle="-|>", color=style.MUTED, lw=1.3))
        ax.text(x0 + cw * len(tok) + 0.74, y, "the model tries to stop; its stop token is replaced with “Wait”",
                fontsize=9.3, color=style.INK2, va="center", style="italic")
        continue
    ax.text(x0, y, line, family=MONO, fontsize=FS, color=style.INK, va="center")
    if i in INSERTED:  # cover the model-ink "Wait" with a highlighted one
        ax.add_patch(FancyBboxPatch((x0 - 0.04, y - LH * 0.36), cw * 4 + 0.08, LH * 0.72,
                                    boxstyle="round,pad=0,rounding_size=0.05", fc="#dbe9fa", ec="none", zorder=2))
        ax.text(x0, y, "Wait", family=MONO, fontsize=FS, color=style.Q3_COL, fontweight="bold", va="center", zorder=3)

ly = 0.38
ax.add_patch(FancyBboxPatch((0.42, ly - 0.1), 0.52, 0.2, boxstyle="round,pad=0,rounding_size=0.05", fc="#dbe9fa", ec="none"))
ax.text(0.68, ly, "Wait", family=MONO, fontsize=9.5, color=style.Q3_COL, fontweight="bold", va="center", ha="center")
ax.text(1.05, ly, "inserted by us", fontsize=9.3, color=style.INK2, va="center")
ax.text(2.45, ly, "<stop>", family=MONO, fontsize=9.5, color=style.Q4_COL, va="center")
ax.plot([2.42, 2.45 + cw * 6 * 9.5 / FS + 0.03], [ly, ly], color=style.Q4_COL, lw=1.4)
ax.text(2.45 + cw * 6 * 9.5 / FS + 0.14, ly, "</think> or end of text, removed", fontsize=9.3, color=style.INK2, va="center")
ax.text(W - 0.42, ly, "all other text is the model's own", fontsize=9.3, color=style.INK2, va="center", ha="right")

out = folder / "figures" / "fig10_forced_trace.png"
fig.savefig(out, dpi=160)
print("wrote", out)
