"""Plot KL damage against the share of neurons ablated.

    python src/sparsity_sweep/plot_results.py results/Qwen__Qwen3-0.6B.csv
    python src/sparsity_sweep/plot_results.py results/Qwen__Qwen3-0.6B.csv results/Qwen__Qwen3-1.7B.csv   # overlay
    python src/sparsity_sweep/plot_results.py --log results/...csv   # log-log version, saved as *_log.png

Linear (default): both axes linear. The right panel is a zoom on 85-100%
ablated, where the curves separate; it is a crop, not a rescale.

Log (--log): x is % of neurons KEPT on a log axis, y is KL on a log axis.
A power law shows up as a straight line. Points with KL = 0 (nothing ablated)
cannot be drawn on a log axis and are left out.

Shaded bands are 95% bootstrap intervals over evaluation batches (only for
CSVs from the current run_sweep.py).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

args = sys.argv[1:]
log = "--log" in args
paths = [Path(p) for p in args if p != "--log"]
if not paths:
    sys.exit(__doc__)

labels = {
    "top": "keep strongest (the test)",
    "random": "keep random (control)",
    "bottom": "keep weakest (control)",
}
colors = {"top": "tab:blue", "random": "tab:orange", "bottom": "tab:green"}
linestyles = ["-", "--", ":", "-."]
ZOOM_FROM = 85
GOODFIRE_ABLATED = 92.2  # top-320 of 4096


def load(path):
    """{mode: sorted list of (ablated_pct, kl, kl_lo, kl_hi)}"""
    series = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            kl = float(row["kl"])
            # 95% bootstrap interval, if the sweep recorded one (older CSVs did not).
            lo = float(row["kl_lo"]) if row.get("kl_lo") else kl
            hi = float(row["kl_hi"]) if row.get("kl_hi") else kl
            series[row["mode"]].append((float(row["ablated_pct"]), kl, lo, hi))
    for pts in series.values():
        pts.sort()
    return series


def draw(ax, x, y, lo, hi, mode, style, label):
    ax.plot(x, y, marker="o", markersize=4, color=colors.get(mode), linestyle=style, label=label)
    ax.fill_between(x, lo, hi, color=colors.get(mode), alpha=0.2, linewidth=0)


if log:
    fig, ax = plt.subplots(figsize=(8, 5))
    axes = [ax]
else:
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    axes = [ax_full, ax_zoom]

for i, path in enumerate(paths):
    model = path.stem.replace("__", "/")
    for mode, pts in load(path).items():
        if log:
            pts = [p for p in pts if p[1] > 0]  # log axis cannot show KL = 0
            if not pts:
                continue
        x, y, lo, hi = zip(*pts)
        if log:
            x = [100 - a for a in x]  # % kept
        label = labels.get(mode, mode)
        if len(paths) > 1:
            label = f"{model.split('/')[-1]}: {label}"
        for j, ax in enumerate(axes):
            draw(ax, x, y, lo, hi, mode, linestyles[i % len(linestyles)], label if j == 0 else None)

if log:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # more ablation to the right, matching the linear plot
    ticks = [50, 25, 12.5, 8, 4, 2, 1]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.axvline(100 - GOODFIRE_ABLATED, color="gray", linestyle=":", linewidth=1)
    ax.text(100 - GOODFIRE_ABLATED, ax.get_ylim()[0] * 1.3, " Goodfire top-320", color="gray",
            fontsize=8, va="bottom", ha="left")
    ax.set_xlabel("Neurons KEPT per token (%, log scale)")
    ax.set_ylabel("KL divergence from intact model (nats, log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
else:
    ax_full.set_xlim(0, 100)
    ax_zoom.set_xlim(ZOOM_FROM, 100)
    ax_full.axvspan(ZOOM_FROM, 100, color="gray", alpha=0.08)
    ax_full.set_ylim(bottom=0)  # KL is never negative
    for ax in axes:
        ax.axvline(GOODFIRE_ABLATED, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Neurons mean-ablated per token (%)")
        ax.grid(alpha=0.3)
    ax_zoom.text(GOODFIRE_ABLATED, ax_zoom.get_ylim()[1] * 0.97, " Goodfire top-320", color="gray",
                 fontsize=8, ha="left", va="top")
    ax_full.set_ylabel("KL divergence from intact model (nats)")
    ax_full.set_title("full range")
    ax_zoom.set_title(f"zoom: {ZOOM_FROM}-100% (same scale)")
    ax_zoom.legend(*ax_full.get_legend_handles_labels(), fontsize=8, loc="center left",
                   bbox_to_anchor=(0.01, 0.45))

fig.suptitle(" vs ".join(p.stem.replace("__", "/") for p in paths))
fig.tight_layout()

suffix = "_log" if log else ""
out = (paths[0].with_name(paths[0].stem + suffix + ".png") if len(paths) == 1
       else paths[0].parent / f"comparison{suffix}.png")
fig.savefig(out, dpi=160)
print(f"saved {out}")
