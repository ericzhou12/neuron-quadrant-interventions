"""One-figure overview of a Q3 results folder.

    python plot_q3.py results/q3__Qwen__Qwen3-1.7B

Reads summary.csv (run analyze_q3.py first), damage.json and the per-problem
jsonl files. Draws one row per condition and four panels:

  math accuracy | MMLU accuracy | damage on WikiText (KL) | answer length vs intact

Conditions that broke the model are drawn in gray and labeled with what they
actually output, so they are not misread as effects on reasoning.
Saves overview.png in the folder.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else sys.exit(__doc__)

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
USABLE, BROKEN, INTACT = "#2a78d6", "#b9b8b2", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK})

rows = list(csv.DictReader((folder / "summary.csv").open()))
damage = json.loads((folder / "damage.json").read_text()) if (folder / "damage.json").exists() else {}
f = lambda x: float(x) if x not in ("", None) else float("nan")


def load(variant):
    path = folder / f"{variant}__seed0.jsonl"
    return [json.loads(l) for l in path.open()] if path.exists() else []


def diagnose(variant, length_key):
    """Short description of what a broken condition outputs, or None if it looks usable."""
    recs = load(variant)
    if not recs:
        return None
    acc = statistics.mean(r["correct"] for r in recs)
    lengths = [r[length_key] for r in recs]
    text_key = "answer_text" if length_key == "answer_tokens" else "thinking"
    cap = max(lengths)
    blank = statistics.mean(len(r[text_key].strip()) == 0 or
                            sum(c.isspace() for c in r[text_key]) / max(1, len(r[text_key])) > 0.8
                            for r in recs)
    if acc > 0.05:
        return None
    if statistics.median(lengths) < 10:
        return "stops at once (outputs ~nothing)"
    if blank > 0.5:
        return "outputs blank space to the limit"
    if statistics.mean(l >= cap for l in lengths) > 0.5:
        return f"never finishes ({cap}-token limit)"
    return "0% accuracy"


# Order: intact, then by damage (mildest first).
mode = "off" if all(r["thinking_mode"] == "off" for r in rows) else "on"
length_key = "answer_tokens" if mode == "off" else "thinking_tokens"
rows.sort(key=lambda r: f(r["damage_kl"]))
base_acc = f(rows[0]["baseline_accuracy"])
base_mmlu = damage.get("none", {}).get("mmlu_accuracy", float("nan"))

names = ["intact model"] + [r["variant"].replace("__nothink", "") for r in rows]
notes = [None] + [diagnose(r["variant"], length_key) for r in rows]
acc = [base_acc] + [f(r["accuracy"]) for r in rows]
mmlu = [base_mmlu] + [f(r["mmlu_accuracy"]) for r in rows]
kl = [0.0] + [f(r["damage_kl"]) for r in rows]
length = [1.0] + [f(r["length_ratio"]) for r in rows]
colors = [INTACT] + [BROKEN if n else USABLE for n in notes[1:]]

fig, axes = plt.subplots(1, 4, figsize=(14, 0.6 * len(names) + 1.9), sharey=True)
y = list(range(len(names)))[::-1]
panels = [
    (acc, "Math accuracy", "{:.0%}", (0, 1), base_acc, "intact"),
    (mmlu, "MMLU accuracy", "{:.0%}", (0, 0.6), 0.25, "chance"),
    (kl, "Damage: KL on WikiText (nats)", "{:.2f}", None, None, None),
    (length, f"{'Answer' if mode == 'off' else 'Thinking'} length vs intact", "{:.2f}×", None, 1.0, "no change"),
]
for ax, (vals, title, fmt, lim, ref, ref_label) in zip(axes, panels):
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="white", linewidth=2)
    top = lim[1] if lim else max(v for v in vals if v == v) * 1.25
    ax.set_xlim(0, top)
    for yi, v in zip(y, vals):
        if v == v:
            ax.text(v + top * 0.015, yi, fmt.format(v), va="center", fontsize=8, color=INK_2,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
    if ref is not None and ref == ref:
        ax.axvline(ref, color=INK_2, linestyle=(0, (3, 3)), linewidth=1)
        ax.text(ref, len(names) - 0.45, f" {ref_label}", fontsize=7.5, color=INK_2, va="bottom")
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=14)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

labels = [n if not note else f"{n}\n{note}" for n, note in zip(names, notes)]
axes[0].set_yticks(y, labels)
for tick, note in zip(axes[0].get_yticklabels(), notes):
    if note:
        tick.set_color(INK_2)
        tick.set_fontsize(8)

model = folder.name.removeprefix("q3__").replace("__", "/")
fig.suptitle(f"{model}: Q3 interventions, thinking {mode}  "
             f"(n={rows[0]['n']} problems; blue = usable, gray = model broken)",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.subplots_adjust(left=0.15, right=0.98, top=0.82, bottom=0.1, wspace=0.12)
out = folder / "overview.png"
fig.savefig(out, dpi=160, facecolor="white")
print(f"saved {out}")
