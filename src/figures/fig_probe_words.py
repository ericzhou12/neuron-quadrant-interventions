"""Probe with a wider set of words: urge to keep going vs urge to end, by leak size.

Left: summed probability of ANY continuation word (Wait, But, Alternatively, Hmm,
maybe, check/verify/confirm) along the intact model's reasoning. Middle: "Wait"
alone, the old measure. Right: urge to end early by either route (start the
Final Answer block, or close </think> directly). All as log ratios vs intact,
95% bootstrap bands over traces.

Usage: python src/figures/fig_probe_words.py [folder] [tag]
Writes figures/fig8_probe_continuation_words[__tag].png
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

style.apply()
folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q3__Qwen__Qwen3-1.7B")
tag = sys.argv[2] if len(sys.argv) > 2 else "cont"
S = json.loads((folder / f"probe_summary__{tag}.json").read_text())["per_condition"]
GAINS = {0.5: "shrink@0.5", 0.75: "shrink@0.25", 1.25: "grow@0.25", 1.5: "grow@0.5", 1.75: "grow@0.75"}
PANELS = [("continue_any", "a  Any continuation word"), ("wait", 'b  "Wait" alone (the old measure)'),
          ("end_either", "c  Urge to end early (either route)")]
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
n = None
for ax, (key, title) in zip(axes, PANELS):
    for quad, colour in (("q3", style.Q3_COL), ("q4", style.Q4_COL)):
        xs, y, lo, hi = [1.0], [0.0], [0.0], [0.0]
        for g, fill in sorted(GAINS.items()):
            s = S.get(f"{quad}_{fill}", {}).get(key)
            if s:
                xs.append(g); y.append(s["mean"]); lo.append(s["ci"][0]); hi.append(s["ci"][1]); n = s["n"]
        o = np.argsort(xs); xs, y, lo, hi = (np.array(v)[o] for v in (xs, y, lo, hi))
        style.line_with_band(ax, xs, y, lo, hi, colour, f"{quad.upper()}")
    ax.axhline(0, color=style.MUTED, lw=1.1, ls=":", zorder=1)
    ax.set_title(title); ax.set_xlabel("leak size multiplier (1 = intact)")
    ax.set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75]); ax.set_xlim(0.42, 1.83)
axes[0].set_ylabel("log ratio vs intact (0 = no change)")
axes[1].set_ylim(axes[0].get_ylim()[0] - 0.1, max(axes[1].get_ylim()[1], axes[0].get_ylim()[1]))
axes[0].set_ylim(axes[1].get_ylim())
fig.suptitle(f"Qwen3-1.7B, teacher-forced probe on {n} intact traces ({tag}). Shaded: 95% interval. "
             "Panels a and b share a scale; c does not.", fontsize=9.5, color=style.INK2, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=(0, 0, 0.985, 0.95), w_pad=5)
out = folder / "figures" / ("fig8_probe_continuation_words" + ("" if tag == "cont" else f"__{tag}") + ".png")
fig.savefig(out, dpi=160)
print("wrote", out)
