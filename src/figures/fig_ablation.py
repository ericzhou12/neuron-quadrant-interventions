"""Ablation tests on GSM8K (thinking on): what happens to reasoning length and
accuracy when Q3 or Q4 leaks are flattened or removed.

Panels a and b: the partial edits that pass the damage gate. Each edited model
is compared with the intact model on the same problems and the same seed.
  qmean      every firing replaced by that neuron's typical value in the quadrant
  zero@0.25  a random 25% of the quadrant's firings set to zero (two seeds)
Panel c: removing ALL firings of a quadrant (fails the damage gate). Every dot
is one problem.

Usage: python src/figures/fig_ablation.py [results/q3__Qwen__Qwen3-1.7B]
Writes figures/fig8_ablation_gsm8k.png and ablation_summary.json.
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

# (label under the axis, fill, seed)
EDITS = [("flatten to typical\nvalue (qmean)\nseed 1", "qmean", 1),
         ("remove 25%\n(zero@0.25)\nseed 1", "zero@0.25", 1),
         ("remove 25%\n(zero@0.25)\nseed 0", "zero@0.25", 0)]
QUADS = (("q3", style.Q3_COL, "Q3 activations", -0.16), ("q4", style.Q4_COL, "Q4 activations (control)", 0.16))


def load(name):
    p = folder / f"{name}.jsonl"
    return {r["idx"]: r for r in map(json.loads, p.open())} if p.exists() else None


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def errbar(ax, x, y, lo, hi, colour, hollow=False):
    ax.plot([x, x], [lo, hi], color=colour, lw=2.4, solid_capstyle="round", zorder=2)
    ax.plot([x], [y], "o", color=colour, ms=8, zorder=3,
            mfc=style.SURFACE if hollow else colour, mew=2)


summary = {}
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 5.4), gridspec_kw={"width_ratios": [1, 1, 1.05]})

for gi, (xlabel, fill, seed) in enumerate(EDITS):
    base = load(f"none__seed{seed}")
    ids = sorted(base)
    k = sum(base[i]["correct"] for i in ids); lo, hi = wilson(k, len(ids))
    errbar(a2, gi - 0.3, k / len(ids), lo, hi, style.INTACT_COL)
    for quad, colour, _, dx in QUADS:
        run = load(f"{quad}_{fill}__seed{seed}")
        if run is None:
            continue
        r = np.array([run[i]["thinking_tokens"] / max(1, base[i]["thinking_tokens"]) for i in ids])
        boots = [np.median(rng.choice(r, len(r))) for _ in range(4000)]
        med, rlo, rhi = float(np.median(r)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        errbar(a1, gi + dx, med, rlo, rhi, colour)
        a1.annotate(f"×{med:.2f}", (gi + dx, rhi), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=9, color=style.INK2)
        kk = sum(run[i]["correct"] for i in ids); alo, ahi = wilson(kk, len(ids))
        errbar(a2, gi + dx * 0.9 + 0.08, kk / len(ids), alo, ahi, colour)
        lost = sum(base[i]["correct"] and not run[i]["correct"] for i in ids)
        gained = sum(run[i]["correct"] and not base[i]["correct"] for i in ids)
        summary[f"{quad}_{fill}__seed{seed}"] = {
            "n": len(ids), "length_ratio": med, "ci": [rlo, rhi], "accuracy": kk / len(ids),
            "intact_accuracy": k / len(ids), "lost": lost, "gained": gained,
            "median_thinking_tokens": float(np.median([run[i]["thinking_tokens"] for i in ids])),
            "cap_hits": float(np.mean([run[i]["total_tokens"] >= CAP for i in ids]))}

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
a2.set_title("b  Accuracy barely moves")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=8, label=l)
           for c, l in ((style.INTACT_COL, "intact"), (style.Q3_COL, "Q3 activations"), (style.Q4_COL, "Q4 activations (control)"))]
a2.legend(handles=handles, loc="lower left", ncol=3, handletextpad=0.2, columnspacing=1.2)

# Panel c: the whole quadrant zeroed. One dot per problem.
base = load("none__seed1")
rows = [("intact", base, style.INTACT_COL)]
for name, lab, colour in (("q3_zero__seed1", "all Q3\nremoved", style.Q3_COL),
                          ("q4_zero__first25__seed1", "all Q4\nremoved", style.Q4_COL)):
    run = load(name)
    if run is not None:
        rows.append((lab, run, colour))
shared = sorted(set.intersection(*[set(r[1]) for r in rows[1:]])) if len(rows) > 1 else sorted(base)
for xi, (lab, run, colour) in enumerate(rows):
    ids = sorted(run) if lab != "intact" else sorted(rows[1][1]) if len(rows) > 1 else sorted(run)
    t = np.array([run[i]["thinking_tokens"] for i in ids])
    a3.scatter(xi + rng.uniform(-0.2, 0.2, len(t)), t + 1, s=26, color=colour, alpha=0.6, linewidths=0, zorder=3)
    near0, capped = int((t <= 4).sum()), int((t >= CAP - 16).sum())  # capped = still thinking at the cap
    if lab != "intact":
        acc = np.mean([run[i]["correct"] for i in ids])
        a3.annotate(f"{near0} of {len(t)} skip\nthinking,\n{capped} of {len(t)} run\nto the cap\naccuracy {acc:.0%}",
                    (xi, 22), ha="center", va="center", fontsize=8.6, color=style.INK2)
        summary[name] = {"n": len(t), "near_zero": near0, "capped": capped, "accuracy": float(acc)}
    else:
        a3.annotate(f"median\n{int(np.median(t)):,} tokens\n(same {len(t)}\nproblems)", (xi, 22),
                    ha="center", va="center", fontsize=8.6, color=style.INK2)
a3.set_yscale("log"); a3.set_ylim(0.7, 7000)
a3.set_yticks([1, 10, 100, 1000, 4096]); a3.set_yticklabels(["0", "10", "100", "1,000", "cap"]); a3.minorticks_off()
a3.set_xticks(range(len(rows))); a3.set_xticklabels([r[0] for r in rows]); a3.set_xlim(-0.6, len(rows) - 0.4)
a3.grid(axis="x", visible=False)
a3.set_ylabel("thinking tokens per problem (log scale)")
a3.set_title("c  Full ablation breaks the model")

fig.suptitle("Qwen3-1.7B on GSM8K, thinking on. a, b: 200 problems, each edit paired with the intact model on the same "
             "seed; bars are 95% intervals.  c: one dot per problem, seed 1; this edit fails the damage gate.",
             fontsize=9.5, color=style.INK2, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=3)
out = folder / "figures" / "fig8_ablation_gsm8k.png"
fig.savefig(out, dpi=160)
(folder / "ablation_summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", out)
print(json.dumps(summary, indent=1))
