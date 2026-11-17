"""Publication-style figures for the Q3 reasoning project.

    python src/figures/make_figures.py [results/q3__Qwen__Qwen3-1.7B]
    python src/figures/make_figures.py --selftest      # synthetic data for figs 3-6

Writes PNGs (dpi 160) into <results>/figures/. Any figure whose input data does
not exist yet is skipped with a printed note.

  fig1_damage_dose.png    damage and direct-answer rate vs share of firings zeroed (damage.json)
  fig2_putback_ladder.png WikiText loss for each put-back edit, Q3 vs Q4 (damage.json)
  fig3_generation.png     GSM8K thinking length, accuracy, cap hits (thinking-on runs)
  fig4_probe.png          teacher-forced stop / wait urge vs damage (probe_stop*.jsonl)
  fig5_probe_conclude.png gain dial: conclude / wait urge vs leak size x0.5-x1.5 (probe_summary__v3seed0/v2.json)
  fig6_full_ablation.png  per-problem thinking length when a whole quadrant is mean-filled or zeroed (seed 1)

Colours: Q3 edits blue, Q4 edits (the matched control) orange, intact model grey.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys as _sys, pathlib as _pl  # make the sibling src/ folders importable
_sys.path[:0] = [str(_p) for _p in _pl.Path(__file__).resolve().parents[1].iterdir() if _p.is_dir()]
from analyze_q3 import load_runs  # noqa: E402

DEFAULT_FOLDER = "results/q3__Qwen__Qwen3-1.7B"
DPI = 160

# Palette (validated categorical slots 1, 2 and 7 of the dataviz reference palette).
Q3_COL = "#2a78d6"
Q4_COL = "#e2703a"
OTHER_COL = "#6250d6"
INTACT_COL = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"

FILL_ORDER = ["qmean", "resample", "const", "pool", "shrink", "grow", "zero", "mean"]
FILL_DESC = {
    "qmean": "qmean\nneuron's typical\nvalue, same tokens",
    "resample": "resample\nneuron's own values,\nshuffled in time",
    "const": "const\naverage push\non every token",
    "pool": "pool\nany neuron's\nvalues (layer)",
    "zero": "zero\nremoved\nentirely",
}

import style  # noqa: E402  (shared look: see style.py)

style.apply()
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "lines.linewidth": 2.0})


# ----------------------------------------------------------------------------- helpers

def parse_condition(name: str) -> tuple[str, str, float | None]:
    """'q3_zero@0.25' -> ('q3', 'zero', 0.25). Unknown names -> ('other', name, None)."""
    m = re.fullmatch(r"(q[1-4])_([a-z_]+?)(?:@([0-9.]+))?", name)
    if not m:
        return "other", name, None
    return m.group(1), m.group(2), float(m.group(3)) if m.group(3) else None


def quad_colour(name: str) -> str:
    if name == "none":
        return INTACT_COL
    q = parse_condition(name)[0]
    return {"q3": Q3_COL, "q4": Q4_COL}.get(q, OTHER_COL)


def condition_sort_key(name: str):
    """Group each edit with its Q3/Q4 twin: (fill, dose, quadrant)."""
    if name == "none":
        return (-1, 0, 0, "")
    q, fill, dose = parse_condition(name)
    fi = FILL_ORDER.index(fill) if fill in FILL_ORDER else len(FILL_ORDER)
    qi = {"q3": 0, "q4": 1}.get(q, 2)
    return (fi, -(dose if dose is not None else 1.0), qi, name)


def binom_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, centre - half, centre + half


def bootstrap(stat, n: int, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat(indices) over resampled indices."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def add_header(fig, title: str, caption: str, top: float = 0.985):
    fig.text(0.012, top, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, top - 0.045 * (6.0 / fig.get_figheight()), caption, fontsize=8.5,
             color=INK2, va="top", wrap=True)


def add_footer(fig, text: str):
    fig.text(0.012, 0.008, text, fontsize=7.8, color=INK2, va="bottom", ha="left")


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {path}")


def skip(name: str, why: str):
    print(f"  skip {name}: {why}")


def load_damage(folder: Path) -> dict | None:
    p = folder / "damage.json"
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- fig 1

def fig1_damage_dose(folder: Path, out_dir: Path):
    name = "fig1_damage_dose.png"
    dmg = load_damage(folder)
    if dmg is None or "none" not in dmg:
        return skip(name, "no damage.json (or no 'none' entry)")
    doses = [0.05, 0.1, 0.25, 0.5, 1.0]  # 1.0 = the whole quadrant (key "<q>_zero")
    base = dmg["none"]
    n = base.get("mmlu_n", 300)

    def series(q: str, field: str, base_val):
        xs, ys = [0.0], [base_val]
        for d in doses:
            r = dmg.get(f"{q}_zero" if d == 1.0 else f"{q}_zero@{d:g}")
            if r is not None and field in r:
                xs.append(d * 100)
                ys.append(r[field])
        return np.array(xs), np.array(ys)

    if len(series("q3", "loss_increase_pct", 0.0)[0]) < 2:
        return skip(name, "no q3_zero@p entries in damage.json")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), sharex=True)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.78, bottom=0.27, wspace=0.3)
    add_header(fig, "Removing Q3 or Q4 firings does matched damage, but only removing Q3 raises the direct-answer rate",
               "Qwen3-1.7B, thinking off. x = share of the quadrant's firings set to zero (random choice of firings). "
               "Loss on WikiText; MMLU n = %d questions, one seed; bands = ±1 binomial standard error." % n)

    panels = [
        ("loss_increase_pct", 0.0, "a  General damage: WikiText loss", "Loss increase vs intact (%)", False),
        ("mmlu_accuracy", base.get("mmlu_accuracy"), "b  Knowledge: MMLU accuracy", "Accuracy (letter-probability score)", True),
        ("mmlu_direct_rate", base.get("mmlu_direct_rate"), "c  Behaviour: answers at once",
         "Share of replies giving a letter\nwithin 16 tokens (no reasoning)", True),
    ]
    for ax, (field, bval, title, ylabel, is_rate) in zip(axes, panels):
        ax.set_title(title)
        if bval is None:
            ax.text(0.5, 0.5, "no intact value", transform=ax.transAxes, ha="center", color=INK2)
            continue
        if is_rate:
            se = binom_se(bval, n)
            ax.axhspan(bval - se, bval + se, color=INTACT_COL, alpha=0.15, lw=0)
            ax.axhline(bval, color=INTACT_COL, lw=1.2, ls="--")
        ends = []
        for q, col, lab in (("q3", Q3_COL, "Q3 (gate < 0, up < 0)"), ("q4", Q4_COL, "Q4 matched control (gate < 0, up > 0)")):
            xs, ys = series(q, field, bval)
            if is_rate:
                se = np.sqrt(ys * (1 - ys) / n)
                ax.fill_between(xs, ys - se, ys + se, color=col, alpha=0.16, lw=0)
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=5.5, mec=style.SURFACE, mew=1.2, label=lab, zorder=3)
            ends.append((xs[-1], ys[-1], q.upper(), col))
        # Direct labels at line ends, nudged apart if they collide.
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        ends.sort(key=lambda e: e[1])
        if len(ends) == 2 and ends[1][1] - ends[0][1] < gap:
            mid = (ends[0][1] + ends[1][1]) / 2
            ends[0] = (*ends[0][:1], mid - gap / 2, *ends[0][2:])
            ends[1] = (*ends[1][:1], mid + gap / 2, *ends[1][2:])
        for x, y, lab, col in ends:
            ax.annotate(lab, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, fontweight="bold", color=col)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Share of quadrant's firings zeroed (%)")
        ax.set_xticks([0, 10, 25, 50, 100])
        ax.set_xlim(-3, 112)
        if is_rate:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    if "mmlu_direct_rate" in base:
        axes[2].set_ylim(0.38, 0.85)
    axes[1].set_ylim(0.2, 0.65)
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color=INTACT_COL, ls="--", lw=1.2))
    labels.append("intact model (± SE band)")
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.055))
    save(fig, out_dir, name)


# ----------------------------------------------------------------------------- fig 2

def fig2_putback_ladder(folder: Path, out_dir: Path):
    name = "fig2_putback_ladder.png"
    dmg = load_damage(folder)
    if dmg is None:
        return skip(name, "no damage.json")
    fills = ["qmean", "resample", "const", "pool", "zero"]
    whole = [f for f in fills if f"q3_{f}" in dmg or f"q4_{f}" in dmg]
    part = [f for f in fills if f"q3_{f}@0.25" in dmg or f"q4_{f}@0.25" in dmg]
    if not whole:
        return skip(name, "no whole-quadrant put-back entries in damage.json")

    panels = [("", whole, "a  Whole quadrant edited")]
    if part:
        panels.append(("@0.25", part, "b  Only 25% of the quadrant's firings edited"))
    widths = [len(p[1]) for p in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(12.5, 5.0), gridspec_kw={"width_ratios": widths},
                             squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.06, right=0.985, top=0.79, bottom=0.25, wspace=0.12)
    add_header(fig, "Damage grows as each edit throws away more of the leak's structure, alike for Q3 and Q4",
               "Qwen3-1.7B. Bars = WikiText loss increase vs intact model (log scale); edits ordered left to right "
               "from least to most structure destroyed. Single evaluation, no error bars.")
    bw = 0.38
    for ax, (suffix, fl, title) in zip(axes, panels):
        ax.set_title(title)
        x = np.arange(len(fl))
        for off, q, col in ((-bw / 2 - 0.01, "q3", Q3_COL), (bw / 2 + 0.01, "q4", Q4_COL)):
            for i, f in enumerate(fl):
                r = dmg.get(f"{q}_{f}{suffix}")
                if r is None or "loss_increase_pct" not in r:
                    ax.text(x[i] + off, 0.13, "not run", rotation=90, ha="center", va="bottom", fontsize=7.5, color=INK2)
                    continue
                v = r["loss_increase_pct"]
                ax.bar(x[i] + off, v, bw, color=col, bottom=0.1 if False else None,
                       label=None, edgecolor=style.SURFACE, linewidth=0)
                ax.text(x[i] + off, v * 1.08, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7.8, color=INK)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 200)
        ax.set_xticks(x)
        ax.set_xticklabels([FILL_DESC.get(f, f) for f in fl], fontsize=8)
        ax.grid(axis="x", visible=False)
        for gate in (10,):
            ax.axhline(gate, color=INK2, lw=0.9, ls=":")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    axes[0].set_ylabel("WikiText loss increase (%, log scale)")
    axes[-1].text(1.0, 10, "damage gate (+10%) ", transform=axes[-1].get_yaxis_transform(), ha="right",
                  va="bottom", fontsize=7.8, color=INK2)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=Q3_COL), Patch(color=Q4_COL)], ["Q3 edited", "Q4 edited (matched control)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    missing = [f"q{q}_{f}@0.25" for f in fills for q in (3, 4) if f"q{q}_{f}@0.25" not in dmg]
    if missing and part:
        add_footer(fig, "Panel b omits edits not yet run: " + ", ".join(missing))
    save(fig, out_dir, name)


# ----------------------------------------------------------------------------- fig 3

def _is_main_thinking_on(variant: str) -> bool:
    if any(t in variant for t in ("__forced", "waitsof", "pilot", "__nothink", "__genonly")):
        return False
    return True


def fig3_generation(folder: Path, out_dir: Path):
    name = "fig3_generation.png"
    runs = load_runs(folder)
    variants = {v: recs for v, recs in runs.items() if _is_main_thinking_on(v)}
    # Keep only genuinely thinking-on records.
    variants = {v: {k: r for k, r in recs.items() if r.get("thinking_mode", "on") == "on"}
                for v, recs in variants.items()}
    base = variants.get("none")
    if not base:
        return skip(name, "no thinking-on baseline (none__seed<k>.jsonl)")
    # Edits that fail the damage gate (broken model) are shown in fig6, not here.
    dmg_path = folder / "damage.json"
    dmg = json.loads(dmg_path.read_text()) if dmg_path.exists() else {}
    broken = {c for c, d in dmg.items() if d.get("passes_gate") is False}
    conds = sorted([v for v in variants if v != "none" and variants[v]
                    and v.split("__")[0] not in broken], key=condition_sort_key)
    if not conds:
        return skip(name, "baseline exists but no thinking-on condition runs yet")

    paired = {}
    for c in conds:
        keys = sorted(set(variants[c]) & set(base))
        if keys:
            paired[c] = keys
    conds = [c for c in conds if c in paired]
    if not conds:
        return skip(name, "no condition shares problems with the baseline")
    all_keys = sorted(set().union(*paired.values()))
    print("  fig3 paired n per condition: " + ", ".join(f"{c}={len(paired[c])}" for c in conds))
    seeds = sorted({k[0] for k in all_keys})

    rows = ["none"] + conds
    fig = plt.figure(figsize=(12.5, 4.6 + 0.32 * len(rows) + 0.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.18 + 0.075 * len(rows)], hspace=0.42, wspace=0.14,
                          left=0.11, right=0.985, top=0.86, bottom=0.08)
    ns = sorted({len(paired[c]) for c in conds})
    n_txt = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}-{ns[-1]}"
    add_header(fig, "GSM8K with thinking on: how long the model reasons, and whether it gets the answer",
               f"Qwen3-1.7B, {n_txt} problems per condition (seeds {', '.join(map(str, seeds))} pooled), each paired "
               "with the intact model on the same problem and seed. Thinking capped at 4096 tokens.")

    # (a) survival curves
    ax = fig.add_subplot(gs[0, :])
    ax.set_title("a  Share of problems still thinking after t tokens (leaks halved, dashed, vs 50% larger, solid)")
    # Only the clearest contrast is drawn here (leaks halved vs made 50% larger);
    # panels b-d list every edit.
    SHOWN = ["none", "q3_shrink@0.5", "q4_shrink@0.5", "q3_grow@0.5", "q4_grow@0.5"]
    shown = [c for c in SHOWN if c in rows] or rows

    def line_style(c):
        if c == "none":
            return dict(color=INTACT_COL, ls="-", lw=2.6)
        return dict(color=quad_colour(c), ls="--" if "shrink" in c else "-", lw=2.2)

    for c in shown:
        keys = all_keys if c == "none" else paired[c]
        recs = base if c == "none" else variants[c]
        t = np.sort(np.array([max(recs[k]["thinking_tokens"], 1) for k in keys]))
        alive = 1.0 - np.arange(1, len(t) + 1) / len(t)
        xs = np.concatenate([[1], np.repeat(t, 2)])
        ys = np.concatenate([[1.0], np.column_stack([np.r_[1.0, alive[:-1]], alive]).ravel()])
        ax.plot(xs, ys, label=f"{c} (n={len(keys)})", **line_style(c))
    ax.set_xlim(0, 4300)  # linear: a plain token count with a hard cap
    ax.set_ylim(-0.02, 1.02)
    ax.axvline(4096, color=INK2, lw=0.8, ls=":")
    ax.text(4096, 1.0, " cap", fontsize=7.8, color=INK2, va="top")
    ax.set_xlabel("Thinking tokens so far")
    ax.set_ylabel("Share still thinking")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper right", ncol=1, fontsize=8.5)

    y = np.arange(len(rows))[::-1]
    axb, axc, axd = (fig.add_subplot(gs[1, i]) for i in range(3))

    # (b) paired ratio
    axb.set_title("b  Thinking length vs intact, same problem")
    for yi, c in zip(y, rows):
        if c == "none":
            axb.plot(1, yi, "o", color=INTACT_COL, ms=7)
            continue
        keys = paired[c]
        ratio = np.array([max(variants[c][k]["thinking_tokens"], 1) / max(base[k]["thinking_tokens"], 1) for k in keys])
        med = float(np.median(ratio))
        lo, hi = bootstrap(lambda ix: np.median(ratio[ix]), len(ratio))
        col = quad_colour(c)
        axb.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        axb.plot(med, yi, "o", color=col, ms=7, mec=style.SURFACE, mew=1.2)
        axb.annotate(f"{med:.2f}", (hi, yi), xytext=(4, 0), textcoords="offset points", va="center",
                     fontsize=7.5, color=INK2)
    axb.axvline(1, color=INTACT_COL, lw=1.1, ls="--")
    axb.set_xscale("log")
    axb.set_xlabel("Median per-problem ratio\n(condition ÷ intact), log; 95% bootstrap CI")
    lo_x, hi_x = axb.get_xlim()
    axb.set_xlim(min(lo_x, 0.5), max(hi_x, 2.0))
    lo_x, hi_x = axb.get_xlim()
    ticks = [t for t in (0.1, 0.2, 0.25, 0.5, 1, 2, 4, 8) if lo_x <= t <= hi_x]
    axb.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axb.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axb.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # (c) accuracy, (d) cap-hit rate
    for axx, field, title, xlabel in (
            (axc, "correct", "c  Accuracy", "Share correct; 95% Wilson interval"),
            (axd, "capped", "d  Hit the 4096-token cap", "Share that never closed </think>;\n95% Wilson interval")):
        axx.set_title(title)
        for yi, c in zip(y, rows):
            keys = all_keys if c == "none" else paired[c]
            recs = base if c == "none" else variants[c]
            if field == "correct":
                k = sum(bool(recs[kk]["correct"]) for kk in keys)
            else:
                k = sum(not recs[kk]["finished_thinking"] for kk in keys)
            p, lo, hi = wilson(k, len(keys))
            col = quad_colour(c)
            axx.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
            axx.plot(p, yi, "o", color=col, ms=7, mec=style.SURFACE, mew=1.2)
            if c == "none":
                axx.axvline(p, color=INTACT_COL, lw=1.1, ls="--")
        axx.set_xlabel(xlabel)
        axx.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axc.set_xlim(0, 1)
    axd.set_xlim(0, max(0.25, axd.get_xlim()[1]))

    for axx in (axb, axc, axd):
        axx.set_yticks(y)
        axx.set_ylim(-0.6, len(rows) - 0.4)
        axx.grid(axis="y", visible=False)
    axb.set_yticklabels([f"{c}" for c in rows])
    for lab, c in zip(axb.get_yticklabels(), rows):
        lab.set_color(INK)
    axc.set_yticklabels([])
    axd.set_yticklabels([])
    save(fig, out_dir, name)


# ----------------------------------------------------------------------------- fig 4

def load_probe(folder: Path) -> dict[str, dict[int, dict]]:
    """condition -> {idx: record}, merged over all probe_stop*.jsonl (first occurrence wins)."""
    out: dict[str, dict[int, dict]] = {}
    for p in sorted(folder.glob("probe_stop*.jsonl")):
        for line in p.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.setdefault(r["condition"], {}).setdefault(r["idx"], r)
    return out


def fig4_probe(folder: Path, out_dir: Path):
    name = "fig4_probe.png"
    probe = load_probe(folder)
    if "none" not in probe:
        return skip(name, "no probe_stop*.jsonl with a 'none' condition")
    base = probe["none"]
    conds = sorted([c for c in probe if c != "none" and set(probe[c]) & set(base)], key=condition_sort_key)
    if not conds:
        return skip(name, "probe file has only the intact model")

    stats = {}
    for c in conds:
        idx = sorted(set(probe[c]) & set(base))
        s = {"n": len(idx), "kl": float(np.mean([probe[c][i]["kl"] for i in idx]))}
        for m in ("stop_hazard", "wait_hazard"):
            a = np.array([probe[c][i][m] for i in idx])
            b = np.array([base[i][m] for i in idx])
            s[m] = float(a.mean() / b.mean())
            s[m + "_ci"] = bootstrap(lambda ix: a[ix].mean() / b[ix].mean(), len(idx))
        d = np.array([probe[c][i]["logp_stop_end"] - base[i]["logp_stop_end"] for i in idx])
        s["dlogp"] = float(d.mean())
        s["dlogp_ci"] = bootstrap(lambda ix: d[ix].mean(), len(idx))
        stats[c] = s
    print("  fig4 paired n per condition: " + ", ".join(f"{c}={stats[c]['n']}" for c in conds))
    ns = sorted({s["n"] for s in stats.values()})
    n_txt = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}-{ns[-1]}"

    fig = plt.figure(figsize=(13, 5.4 + 0.12 * len(conds)))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.32, left=0.06, right=0.985, top=0.8, bottom=0.12)
    add_header(fig, "Teacher-forced probe: does an edit raise the urge to stop thinking more than its matched control?",
               f"Qwen3-1.7B reading the intact model's own reasoning traces ({n_txt} traces, paired by trace). "
               "Error bars = 95% bootstrap CI over traces. Points to the right did more damage.\nSuperseded in part: "
               "</think> and \"Wait\" turned out to be narrow targets. See fig5 for the urge to end by either "
               "route and a wider set of continuation words.")

    def scatter(ax, metric, title, ylabel):
        ax.set_title(title)
        for c in conds:
            s = stats[c]
            col = quad_colour(c)
            lo, hi = s[metric + "_ci"]
            ax.errorbar(s["kl"], s[metric], yerr=[[s[metric] - lo], [hi - s[metric]]], fmt="o", color=col,
                        ms=7, mec=style.SURFACE, mew=1.2, elinewidth=1.6, capsize=0, zorder=3)
            ax.annotate(c, (s["kl"], s[metric]), xytext=(5, 4), textcoords="offset points", fontsize=7.3,
                        color=INK2)
        ax.axhline(1, color=INTACT_COL, lw=1.1, ls="--")
        ax.text(0.0, 1, " intact", transform=ax.get_yaxis_transform(), fontsize=7.8, color=INTACT_COL, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Damage on the reasoning text: mean KL from intact (log)")
        ax.set_ylabel(ylabel)
        fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(fmt)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(fmt)
        ax.tick_params(axis="x", which="minor", labelsize=7)
        ax.tick_params(axis="y", which="minor", labelsize=7)

    scatter(fig.add_subplot(gs[0, 0]), "stop_hazard", "a  Urge to stop thinking",
            "Summed P(</think>) over the trace,\nratio to intact (log)")
    scatter(fig.add_subplot(gs[0, 1]), "wait_hazard", "b  Urge to say \"Wait\"",
            "Summed P(\"Wait\") over the trace,\nratio to intact (log)")

    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("c  Urge to stop at the real end")
    y = np.arange(len(conds))[::-1]
    for yi, c in zip(y, conds):
        s = stats[c]
        lo, hi = s["dlogp_ci"]
        col = quad_colour(c)
        ax.barh(yi, s["dlogp"], 0.62, color=col, edgecolor=style.SURFACE, linewidth=0)
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1.2)
    ax.axvline(0, color=INTACT_COL, lw=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels(conds, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in log P(</think>) at the real end\nvs intact (nats); 95% bootstrap CI")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=Q3_COL, ms=7, label="Q3 edits"),
               Line2D([], [], marker="o", ls="", color=Q4_COL, ms=7, label="Q4 edits (matched control)")]
    if any(quad_colour(c) == OTHER_COL for c in conds):
        handles.append(Line2D([], [], marker="o", ls="", color=OTHER_COL, ms=7, label="other quadrants"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
    save(fig, out_dir, name)


# ----------------------------------------------------------------------------- fig 5

GAIN_STEPS = [("shrink@0.5", 0.5), ("shrink@0.25", 0.75), ("grow@0.25", 1.25), ("grow@0.5", 1.5)]
RATIO_TICKS = [0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4, 5, 7]


def _ratio_axis(ax, sparse=False):
    """Y axis holds natural-log ratios; label the ticks as plain 'x times' multipliers.
    sparse: a handful of round ticks (1-2-5 steps for a wide range, quarters for a narrow one)."""
    lo, hi = ax.get_ylim()
    if sparse:
        wide = hi - lo > math.log(4)
        ticks = [t for t in ((0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if wide else (0.5, 0.75, 1, 1.25, 1.5, 2))
                 if lo <= math.log(t) <= hi]
    else:
        ticks = [t for t in RATIO_TICKS if lo <= math.log(t) <= hi]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([math.log(t) for t in ticks]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"×{math.exp(v):.3g}"))


def fig5_probe_conclude(folder: Path, out_dir: Path):
    """Leak size as a dial, seen by the probe: urge to end, urge to keep going
    (a SET of continuation words), and "Wait" alone for comparison."""
    name = "fig5_probe_conclude.png"
    main_p, rep_p = folder / "probe_summary__cont.json", folder / "probe_summary__contseed0.json"
    old = False
    if not main_p.exists():  # fall back to the earlier probe (no continuation-word set)
        main_p, rep_p, old = folder / "probe_summary__v3seed0.json", folder / "probe_summary__v2.json", True
    if not main_p.exists():
        return skip(name, f"no {main_p.name}")
    main = json.loads(main_p.read_text())["per_condition"]
    rep = json.loads(rep_p.read_text())["per_condition"] if rep_p.exists() else {}
    steps = GAIN_STEPS + [("grow@0.75", 1.75)]
    if not any(f"q3_{s}" in main for s, _ in steps):
        return skip(name, "no q3_shrink/grow entries in the probe summary")
    n_main = sorted({main[f"{q}_{s}"]["n"] for q in ("q3", "q4") for s, _ in steps if f"{q}_{s}" in main})
    n_rep = sorted({rep[f"{q}_{s}"]["n"] for q in ("q3", "q4") for s, _ in steps if f"{q}_{s}" in rep})
    if old:
        panels = [("conclude", "a  Urge to conclude early", "ratio to intact (log scale)"),
                  ("wait", 'b  Urge to say "Wait"', "ratio to intact (log scale)")]
    else:
        panels = [("end_either", "a  Urge to end", "Ratio to intact, log scale"),
                  ("continue_any", "b  Urge to keep going", "")]

    # One row per set of traces: the main set on top, the second set underneath.
    sets = [(main, n_main, "first set of traces")] + ([(rep, n_rep, "second set of traces")] if n_rep else [])
    fig, grid = plt.subplots(len(sets), len(panels), figsize=(5.6 * len(panels), 1.3 + 4.3 * len(sets)),
                             sharex=True, sharey="col", squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.95, top=1 - 1.25 / (1.3 + 4.3 * len(sets)),
                        bottom=0.95 / (1.3 + 4.3 * len(sets)), wspace=0.24, hspace=0.3)
    add_header(fig, "Leak size as a dial: bigger Q3 leaks raise the urge to end and lower the urge to keep going",
               "Qwen3-1.7B replaying the intact model's own reasoning. Every firing in the quadrant is multiplied "
               "by the leak size.\nLines = mean ratio to intact, shaded = 95% interval.")
    letters = iter("abcdefgh")
    for row, (data, n_set, set_name) in zip(grid, sets):
        for ax, (metric, title, ylabel) in zip(row, panels):
            ax.set_title(f"{next(letters)}  {title.split('  ', 1)[1]}, {set_name}")
            ax.axhline(0, color=INTACT_COL, lw=1.1, ls=":", zorder=1)
            ends = []
            for q, col in (("q3", Q3_COL), ("q4", Q4_COL)):
                xs, ys, los, his = [1.0], [0.0], [0.0], [0.0]
                for s_, x in steps:
                    r = data.get(f"{q}_{s_}")
                    if r is None or r.get(metric) is None:
                        continue
                    xs.append(x); ys.append(r[metric]["mean"]); los.append(r[metric]["ci"][0]); his.append(r[metric]["ci"][1])
                o = np.argsort(xs)
                xs, ys, los, his = (np.array(v)[o] for v in (xs, ys, los, his))
                style.line_with_band(ax, xs, ys, los, his, col, None)
                ends.append((xs[-1], ys[-1], q.upper(), col))
            ends.sort(key=lambda e: e[1])
            ax.set_xlim(0.4, 1.9)
            ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75])
            ax.set_xticklabels(["×0.5", "×0.75", "×1", "×1.25", "×1.5", "×1.75"])
            ax.set_ylabel(ylabel)
            ax._ends = ends
    for ax in grid[-1]:
        ax.set_xlabel("Leak size")
    for ax in grid.ravel():
        lo, hi = ax.get_ylim()
        gap = 0.07 * (hi - lo)
        ends = ax._ends
        if len(ends) == 2 and ends[1][1] - ends[0][1] < gap:
            mid = (ends[0][1] + ends[1][1]) / 2
            ends = [(ends[0][0], mid - gap / 2, *ends[0][2:]), (ends[1][0], mid + gap / 2, *ends[1][2:])]
        for x, y, lab, col in ends:
            ax.annotate(lab, (x, y), xytext=(11, 0), textcoords="offset points", va="center", fontsize=10.5, color=col)
        _ratio_axis(ax, sparse=True)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=Q3_COL, lw=2.4, marker="o", ms=6, label="Q3 leaks scaled"),
               Line2D([], [], color=Q4_COL, lw=2.4, marker="o", ms=6, label="Q4 leaks scaled"),
               Line2D([], [], color=INTACT_COL, ls=":", lw=1.1, label="intact model")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, out_dir, name)


# ----------------------------------------------------------------------------- fig 6

FIG6_ROWS = [("none", "none__seed1.jsonl", "Intact model"),
             ("q3_mean", "q3_mean__seed1.jsonl", "Q3 set to neuron mean\n(Goodfire-style fill)"),
             ("q4_mean", "q4_mean__seed1.jsonl", "Q4 set to neuron mean"),
             ("q3_zero", "q3_zero__seed1.jsonl", "Q3 removed (zero)"),
             ("q4_zero", "q4_zero__first25__seed1.jsonl", "Q4 removed (zero)")]
CAP = 4096
NEAR_ZERO = 5  # 1-5 thinking tokens is "\n\n</think>"-style skipping; pile it with exact zeros


def _pile(k: int, x0: float, direction: int, y0: float, dx: float, dy: float, per_col: int):
    """Grid positions for k identical values, stacked in columns growing away from x0."""
    xs, ys = [], []
    for i in range(k):
        c, r = divmod(i, per_col)
        n_in_col = min(per_col, k - c * per_col)
        xs.append(x0 + direction * c * dx)
        ys.append(y0 + (r - (n_in_col - 1) / 2) * dy)
    return xs, ys


def fig6_full_ablation(folder: Path, out_dir: Path):
    name = "fig6_full_ablation.png"
    data = []
    for cond, fn, label in FIG6_ROWS:
        p = folder / fn
        if not p.exists():
            print(f"  fig6: missing {fn}, row left out")
            continue
        recs = [json.loads(l) for l in p.open() if l.strip()]
        recs = [r for r in recs if r.get("thinking_mode", "on") == "on" and r["idx"] < 50]
        if recs:
            data.append((cond, label, recs))
    if not any(c == "none" for c, _, _ in data) or len(data) < 2:
        return skip(name, "need none__seed1.jsonl plus at least one full-ablation run")
    dmg = load_damage(folder) or {}

    fig, ax = plt.subplots(figsize=(12, 1.15 * len(data) + 2.4))
    fig.subplots_adjust(left=0.22, right=0.985, top=1 - 1.25 / fig.get_figheight(),
                        bottom=0.65 / fig.get_figheight())
    add_header(fig, "Editing a whole quadrant breaks the model: it skips thinking or loops to the cap, and solves nothing",
               "Qwen3-1.7B, GSM8K, thinking on, seed 1, first 50 problems (Q4 zero: first 25). One dot per problem; "
               "filled = correct, hollow = wrong. Piles at ~0 (0-5 tokens) and at the\n4096-token cap are "
               "stacked so every problem is visible; bold number = pile size. Damage = WikiText loss increase vs intact (single evaluation). No error bars: every "
               "problem is shown.", top=0.985)
    rng = np.random.default_rng(0)
    y = np.arange(len(data))[::-1]
    dx, dy, per_col = 52, 0.075, 9
    labels = []
    for yi, (cond, label, recs) in zip(y, data):
        col = quad_colour(cond)
        t = np.array([r["thinking_tokens"] for r in recs])
        ok = np.array([bool(r["correct"]) for r in recs])
        zero, cap = t <= NEAR_ZERO, t >= CAP
        mid = ~zero & ~cap
        pts = []  # (x, y, correct)
        for mask, x0, d in ((zero, -30, -1), (cap, CAP + 30, 1)):
            idx = np.flatnonzero(mask)
            xs, ys = _pile(len(idx), x0, d, yi, dx, dy, per_col)
            pts += list(zip(xs, ys, ok[idx]))
        for i in np.flatnonzero(mid):
            pts.append((t[i], yi + rng.uniform(-0.3, 0.3), ok[i]))
        for x, yy, c in pts:
            ax.plot(x, yy, "o", ms=5.2, mec=col, mew=1.1, mfc=col if c else style.SURFACE, zorder=3)
        for mask, x, ha in ((zero, -30 - dx * math.ceil(zero.sum() / per_col) + dx / 2 - 20, "right"),
                            (cap, CAP + 30 + dx * math.ceil(cap.sum() / per_col) - dx / 2 + 20, "left")):
            if mask.sum():
                ax.text(x, yi, f"{mask.sum()}", ha=ha, va="center", fontsize=8.5, color=INK, fontweight="bold")
        if cond == "none":
            med = float(np.median(t))
            ax.plot([med, med], [yi - 0.38, yi + 0.38], color=INK2, lw=1.4, zorder=4)
            ax.text(med, yi + 0.4, f"median {med:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        d = dmg.get(cond, {}).get("loss_increase_pct")
        dtxt = "damage —" if cond == "none" else (f"damage +{d:.0f}%" if d is not None else "damage n/a")
        labels.append(f"{label}\nn = {len(recs)}, accuracy {ok.mean():.0%}, {dtxt}")
        tot = np.array([r.get("total_tokens", -1) for r in recs])
        empty = int((tot == 0).sum())
        looped = int((zero & (tot >= CAP)).sum())
        note = []
        if empty:
            note.append(f"{empty} of {len(recs)} emitted nothing at all (stopped at once)")
        if looped:
            of = "all" if looped == zero.sum() else f"{looped} of the"
            note.append(f"{of} {int(zero.sum())} that skipped thinking then looped in the answer up to the cap")
        if note:
            ax.text(200, yi, "\n".join(note), fontsize=7.8, color=INK2, va="center")
    ax.axvline(CAP, color=INK2, lw=0.9, ls=":", zorder=1)
    ax.text(CAP, len(data) - 0.42, "4096-token cap\n(never closed </think>)", ha="center", va="bottom",
            fontsize=7.8, color=INK2)
    ax.axvline(0, color=INK2, lw=0.9, ls=":", zorder=1)
    ax.text(0, len(data) - 0.42, "~0 thinking\ntokens", ha="center", va="bottom", fontsize=7.8, color=INK2)
    ax.set_xlim(-560, CAP + 560)
    ax.set_ylim(-0.6, len(data) - 0.05)
    ax.set_xticks([0, 1000, 2000, 3000, 4096])
    ax.set_xlabel("Thinking tokens per problem")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    for lab in ax.get_yticklabels():
        lab.set_color(INK)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for yi in y[:-1]:
        ax.axhline(yi - 0.5, color=GRID, lw=0.7)
    save(fig, out_dir, name)


# ----------------------------------------------------------------------------- self-test

def _write_synthetic(folder: Path):
    rng = random.Random(0)
    conds = {"none": 1.0, "q3_qmean": 0.6, "q4_qmean": 0.95, "q3_zero@0.25": 0.7, "q4_zero@0.25": 1.0,
             "q3_resample": 0.8}
    for seed in (0, 1):
        base_len = {i: int(math.exp(rng.gauss(6.8, 0.6))) for i in range(40)}
        for c, scale in conds.items():
            with (folder / f"{c}__seed{seed}.jsonl").open("w") as f:
                for i in range(40 if c != "q3_resample" else 30):
                    t = min(4096, int(base_len[i] * scale * math.exp(rng.gauss(0, 0.3))))
                    f.write(json.dumps({"condition": c, "seed": seed, "idx": i, "thinking_mode": "on",
                                        "correct": rng.random() < 0.85 * min(1, scale + 0.2),
                                        "thinking_tokens": t, "answer_tokens": 120,
                                        "finished_thinking": t < 4096, "wait_count": t // 300}) + "\n")
    # Distractors that fig 3 must ignore.
    with (folder / "none__nothink__seed0.jsonl").open("w") as f:
        f.write(json.dumps({"condition": "none", "seed": 0, "idx": 0, "thinking_mode": "off", "correct": True,
                            "thinking_tokens": 0, "answer_tokens": 300, "finished_thinking": True,
                            "wait_count": 0}) + "\n")
    with (folder / "q3_qmean__forced__seed0.jsonl").open("w") as f:
        f.write(json.dumps({"condition": "q3_qmean", "seed": 0, "idx": 0, "thinking_mode": "on", "correct": True,
                            "thinking_tokens": 4000, "answer_tokens": 100, "finished_thinking": True,
                            "wait_count": 3}) + "\n")
    # Probe: two tagged files with an overlapping condition to test de-duplication.
    pconds = {"none": (1.0, 0.0), "q3_qmean": (1.8, 0.25), "q4_qmean": (1.1, 0.28), "q3_zero@0.25": (1.4, 0.13),
              "q4_zero@0.25": (1.05, 0.16), "q3_const": (2.2, 0.54), "q4_const": (1.3, 0.62), "q2_zero@0.05": (0.9, 0.4)}
    for tag, names in (("", ["none", "q3_qmean", "q4_qmean", "q3_zero@0.25", "q4_zero@0.25"]),
                       ("__putback", ["none", "q3_qmean", "q3_const", "q4_const", "q2_zero@0.05"])):
        with (folder / f"probe_stop{tag}.jsonl").open("w") as f:
            for i in range(50):
                bs, bw = math.exp(rng.gauss(0, 0.5)), math.exp(rng.gauss(1, 0.4))
                for c in names:
                    mult, kl = pconds[c]
                    f.write(json.dumps({"condition": c, "idx": i, "n_positions": 800,
                                        "stop_hazard": bs * mult * math.exp(rng.gauss(0, 0.2)),
                                        "logp_stop_end": -0.5 + math.log(mult) + rng.gauss(0, 0.3),
                                        "wait_hazard": bw * (2 - mult / 2) * math.exp(rng.gauss(0, 0.2)),
                                        "nll": 0.4 + kl, "kl": kl * math.exp(rng.gauss(0, 0.2))}) + "\n")
    # Fig 5 summary (gain dial) and fig 6 full-ablation runs (seed 1).
    def entry(m):
        return {"mean": m, "ci": [m - 0.3, m + 0.3], "n": 40}
    summ = {f"{q}_{s}": {"kl": 0.1, "n": 40, "conclude": entry(sg * (x - 1) * 2), "wait": entry(-(x - 1))}
            for q, sg in (("q3", 1), ("q4", -1)) for s, x in GAIN_STEPS}
    (folder / "probe_summary__v3seed0.json").write_text(json.dumps({"per_condition": summ}))
    for c, fn in (("none", "none__seed1.jsonl"), ("q3_mean", "q3_mean__seed1.jsonl"), ("q3_zero", "q3_zero__seed1.jsonl")):
        with (folder / fn).open("w") as f:
            for i in range(60):
                t = int(math.exp(rng.gauss(7, 0.5))) if c == "none" else rng.choice([0, 4096])
                t = min(t, 4096)
                f.write(json.dumps({"condition": c, "seed": 1, "idx": i, "thinking_mode": "on", "correct": c == "none",
                                    "thinking_tokens": t, "answer_tokens": 0, "total_tokens": t,
                                    "finished_thinking": t < 4096}) + "\n")


def selftest():
    tmp = Path(tempfile.mkdtemp(prefix="make_figures_selftest_"))
    print(f"selftest: synthetic data in {tmp}")
    _write_synthetic(tmp)
    out = tmp / "figures"
    for f in (fig1_damage_dose, fig2_putback_ladder, fig3_generation, fig4_probe, fig5_probe_conclude,
              fig6_full_ablation):
        f(tmp, out)
    for n in ("fig3_generation.png", "fig4_probe.png", "fig5_probe_conclude.png", "fig6_full_ablation.png"):
        assert (out / n).exists(), f"{n} was not produced"
    print("selftest ok (figs 1 and 2 are expected to skip: no damage.json in the synthetic folder)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", default=DEFAULT_FOLDER)
    p.add_argument("--selftest", action="store_true", help="render figs 3-6 from synthetic data in a temp dir")
    args = p.parse_args(argv)
    if args.selftest:
        selftest()
        return
    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"no such folder: {folder}")
    out = folder / "figures"
    print(f"figures from {folder} -> {out}")
    for f in (fig1_damage_dose, fig2_putback_ladder, fig3_generation, fig4_probe, fig5_probe_conclude,
              fig6_full_ablation):
        f(folder, out)


if __name__ == "__main__":
    main()
