"""Ablation tests on GSM8K (thinking on): what happens to reasoning length and
accuracy when Q3 or Q4 activations are flattened or partly removed.

Only the partial edits that pass the damage gate are shown. Each edited model
is compared with the intact model on the same problems and the same seed.
  qmean      every firing replaced by that neuron's typical value in the quadrant (seed 1)
  zero@0.25  a random 25% of the quadrant's firings set to zero (seeds 0 and 1 pooled)
Full removal of a quadrant is in fig6_full_ablation.png.

Usage: python src/figures/fig_ablation.py [results/q3__Qwen__Qwen3-1.7B]
Writes figures/fig8_ablation_gsm8k.png and ablation_summary.json (pooled and per-seed numbers).
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
rng = np.random.default_rng(0)
CAP = 4096

# (label under the axis, fill, seeds pooled into this column)
EDITS = [("flatten to typical\nvalue (qmean)", "qmean", (1,)),
         ("remove 25%\n(zero@0.25)", "zero@0.25", (0, 1))]
QUADS = (("q3", style.Q3_COL, "Q3 activations", -0.16), ("q4", style.Q4_COL, "Q4 activations (control)", 0.16))


def load(name):
    p = folder / f"{name}.jsonl"
    return {r["idx"]: r for r in map(json.loads, p.open())} if p.exists() else None


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def errbar(ax, x, y, lo, hi, colour):
    ax.plot([x, x], [lo, hi], color=colour, lw=2.4, solid_capstyle="round", zorder=2)
    ax.plot([x], [y], "o", color=colour, ms=8, zorder=3)


def stats(pairs):
    """pairs: list of (intact record, edited record) for the same problem and seed."""
    r = np.array([e["thinking_tokens"] / max(1, b["thinking_tokens"]) for b, e in pairs])
    boots = [np.median(rng.choice(r, len(r))) for _ in range(4000)]
    k = sum(e["correct"] for _, e in pairs)
    return {"n": len(pairs), "length_ratio": float(np.median(r)),
            "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "accuracy": k / len(pairs), "intact_accuracy": sum(b["correct"] for b, _ in pairs) / len(pairs),
            "lost": sum(b["correct"] and not e["correct"] for b, e in pairs),
            "gained": sum(e["correct"] and not b["correct"] for b, e in pairs),
            "median_thinking_tokens": float(np.median([e["thinking_tokens"] for _, e in pairs])),
            "cap_hits": float(np.mean([e["total_tokens"] >= CAP for _, e in pairs]))}


summary = {}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4))

for gi, (xlabel, fill, seeds) in enumerate(EDITS):
    bases = {s: load(f"none__seed{s}") for s in seeds}
    intact = [bases[s][i] for s in seeds for i in sorted(bases[s])]
    k = sum(b["correct"] for b in intact); lo, hi = wilson(k, len(intact))
    errbar(a2, gi - 0.3, k / len(intact), lo, hi, style.INTACT_COL)
    for quad, colour, _, dx in QUADS:
        pairs = []
        for s in seeds:
            run = load(f"{quad}_{fill}__seed{s}")
            if run is None:
                continue
            per_seed = [(bases[s][i], run[i]) for i in sorted(bases[s])]
            summary[f"{quad}_{fill}__seed{s}"] = stats(per_seed)
            pairs += per_seed
        if not pairs:
            continue
        st = stats(pairs)
        if len(seeds) > 1:
            summary[f"{quad}_{fill}__pooled"] = st
        errbar(a1, gi + dx, st["length_ratio"], *st["ci"], colour)
        a1.annotate(f"×{st['length_ratio']:.2f}", (gi + dx, st["ci"][1]), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=9, color=style.INK2)
        alo, ahi = wilson(round(st["accuracy"] * st["n"]), st["n"])
        errbar(a2, gi + dx * 0.9 + 0.08, st["accuracy"], alo, ahi, colour)

for ax in (a1, a2):
    ax.set_xticks(range(len(EDITS))); ax.set_xticklabels([e[0] for e in EDITS])
    ax.set_xlim(-0.6, len(EDITS) - 0.4); ax.grid(axis="x", visible=False)
a1.axhline(1, color=style.MUTED, lw=1.1, ls=":", zorder=1)
a1.annotate("intact", (len(EDITS) - 0.4, 1), xytext=(-4, 5), textcoords="offset points",
            ha="right", color=style.MUTED, fontsize=9.5)
a1.set_ylim(0.9, 1.6)
a1.set_ylabel("thinking length vs intact (median ratio)")
a1.set_title("a  Partial ablation makes reasoning longer")
a2.set_ylim(0.6, 1.0); a2.set_ylabel("GSM8K accuracy")
a2.set_title("b  Accuracy decreases slightly")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=l)
           for c, l in ((style.INTACT_COL, "intact"), (style.Q3_COL, "Q3 activations"), (style.Q4_COL, "Q4 activations (control)"))]
a2.legend(handles=handles, loc="lower left", ncol=3, handletextpad=0.2, columnspacing=1.2)

fig.suptitle("Qwen3-1.7B on GSM8K, thinking on. Each edit is paired with the intact model on the same problems; "
             "bars are 95% intervals.",
             fontsize=9.5, color=style.INK2, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=3)
out = folder / "figures" / "fig8_ablation_gsm8k.png"
fig.savefig(out, dpi=160)
(folder / "ablation_summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", out)
for name, st in summary.items():
    print(f"{name}: n={st['n']} length x{st['length_ratio']:.2f} [{st['ci'][0]:.2f}, {st['ci'][1]:.2f}] "
          f"acc {st['accuracy']:.3f} (intact {st['intact_accuracy']:.3f}) lost {st['lost']} gained {st['gained']}")
