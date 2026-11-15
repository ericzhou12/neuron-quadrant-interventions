"""Dose-response for the gain knob: reasoning length and accuracy vs leak size.

Every Q3 (or Q4) firing is multiplied by a gain: shrink@0.5 = x0.5, none = x1,
grow@0.25 = x1.25, and so on. Left: median per-problem thinking length relative
to the intact model (bootstrap 95% band). Right: GSM8K accuracy (Wilson 95% band).

Usage: python src/figures/fig_gain_dose.py [results/q3__Qwen__Qwen3-1.7B] [seed]
Writes figures/fig7_gain_dose_response.png and gain_dose_summary__seed<k>.json.
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
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
GAINS = {0.5: "shrink@0.5", 1.25: "grow@0.25", 1.5: "grow@0.5", 1.75: "grow@0.75"}
rng = np.random.default_rng(0)


def load(name):
    p = folder / f"{name}__seed{seed}.jsonl"
    return {r["idx"]: r for r in map(json.loads, p.open())} if p.exists() else None


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


base = load("none")
ids = sorted(base)
summary = {}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.9))
for quad, colour, dy in (("q3", style.Q3_COL, 0), ("q4", style.Q4_COL, 0)):
    xs, ratio, rlo, rhi, acc, alo, ahi = [1.0], [1.0], [1.0], [1.0], [], [], []
    k = sum(base[i]["correct"] for i in ids); lo, hi = wilson(k, len(ids))
    acc.append(k / len(ids)); alo.append(lo); ahi.append(hi)
    for g, fill in sorted(GAINS.items()):
        run = load(f"{quad}_{fill}")
        if run is None:
            continue
        r = np.array([run[i]["thinking_tokens"] / max(1, base[i]["thinking_tokens"]) for i in ids])
        boots = [np.median(rng.choice(r, len(r))) for _ in range(2000)]
        k = sum(run[i]["correct"] for i in ids); lo, hi = wilson(k, len(ids))
        xs.append(g); ratio.append(float(np.median(r))); rlo.append(float(np.percentile(boots, 2.5)))
        rhi.append(float(np.percentile(boots, 97.5))); acc.append(k / len(ids)); alo.append(lo); ahi.append(hi)
        summary[f"{quad}_{fill}"] = {"gain": g, "length_ratio": ratio[-1], "ci": [rlo[-1], rhi[-1]],
                                     "accuracy": acc[-1], "cap_hits": float(np.mean([run[i]["total_tokens"] >= 4096 for i in ids]))}
    o = np.argsort(xs); xs, ratio, rlo, rhi, acc, alo, ahi = (np.array(v)[o] for v in (xs, ratio, rlo, rhi, acc, alo, ahi))
    name = f"{quad.upper()} leaks"
    style.line_with_band(a1, xs, ratio, rlo, rhi, colour, name)
    style.line_with_band(a2, xs, acc, alo, ahi, colour, name, dy=-9 if quad == "q3" else 9)

a1.set_yscale("log"); a1.set_yticks([0.4, 0.5, 0.7, 1, 1.5, 2])
a1.set_yticklabels(["0.4", "0.5", "0.7", "1", "1.5", "2"]); a1.minorticks_off()
style.reference_line(a1, 1, "intact", 1.83)
a1.set_ylabel("thinking length vs intact (median ratio)")
a1.set_title("Bigger Q3 leaks, shorter reasoning")
a2.set_ylabel("GSM8K accuracy"); a2.set_ylim(0.4, 1.0)
a2.set_title("Accuracy holds from x1 to x1.5")
for ax in (a1, a2):
    ax.set_xlabel("leak size multiplier (1 = intact)")
    ax.set_xticks([0.5, 0.75, 1, 1.25, 1.5, 1.75]); ax.set_xlim(0.42, 1.83)
fig.suptitle(f"Qwen3-1.7B, 200 GSM8K problems, seed {seed}. Shaded: 95% interval.",
             fontsize=9.5, color=style.INK2, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=(0, 0, 0.985, 0.96), w_pad=7)
out = folder / "figures" / "fig7_gain_dose_response.png"
fig.savefig(out, dpi=160)
(folder / f"gain_dose_summary__seed{seed}.json").write_text(json.dumps(summary, indent=1))
print("wrote", out)
