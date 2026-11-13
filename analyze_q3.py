"""Analyse stage 0 results.

    python analyze_q3.py results/q3__Qwen__Qwen3-1.7B

Every condition is compared against the intact model (`none`) on the SAME
problems with the SAME seed, problem by problem. Answers two questions:

  1. Does the intervention shorten reasoning?
     -> median thinking length, as a ratio to baseline, with a 95% bootstrap interval
  2. Does accuracy drop, and does it drop *where reasoning got shorter*?
     -> accuracy change with an exact McNemar test, plus accuracy change split by
        how much each problem's reasoning shrank

Writes summary.csv and survival.png into the results folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load_runs(folder: Path) -> dict[str, dict[tuple[int, int], dict]]:
    """variant name -> {(seed, problem idx): record}"""
    runs: dict[str, dict] = defaultdict(dict)
    for path in sorted(folder.glob("*__seed*.jsonl")):
        variant = path.stem.rsplit("__seed", 1)[0]
        for line in path.open():
            r = json.loads(line)
            runs[variant][(r["seed"], r["idx"])] = r
    return runs


def mcnemar_exact(lost: int, gained: int) -> float:
    """Two-sided exact test on discordant pairs (base right/new wrong vs the reverse)."""
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def median_ratio(base: list[int], new: list[int]) -> float:
    mb = statistics.median(base)
    return statistics.median(new) / mb if mb else float("nan")


def bootstrap_ci(base: list[int], new: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for median(new)/median(base), resampling problems in pairs."""
    rng = random.Random(seed)
    n = len(base)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = median_ratio([base[i] for i in idx], [new[i] for i in idx])
        if not math.isnan(r):
            stats.append(r)
    if not stats:
        return float("nan"), float("nan")
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]


def length_of(r: dict, key: str) -> int:
    return r.get(key, r.get("total_tokens", 0))


def is_nothink(variant: str) -> bool:
    return "__nothink" in variant


def baseline_for(variant: str) -> str:
    return "none__nothink" if is_nothink(variant) else "none"


def repetition(text: str, n: int = 4) -> float:
    """Share of repeated word 4-grams in the reasoning (0 = no repeats, near 1 = looping)."""
    words = text.split()
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def accuracy_on(runs: dict, keys) -> float:
    return statistics.mean(runs[k]["correct"] for k in keys)


def forced_report(runs: dict) -> list[dict]:
    """For each forced run, how much accuracy came back, relative to the
    dose-matched forcing of the intact model."""
    out = []
    for variant in sorted(runs):
        if not variant.endswith("__forced"):
            continue
        unforced = variant[: -len("__forced")]
        control = f"none__waitsof-{variant.replace('__', '-')}"
        if unforced not in runs:
            continue
        keys = set(runs["none"]) & set(runs[unforced]) & set(runs[variant])
        row = {
            "variant": variant,
            "n": len(keys),
            "acc_intact": accuracy_on(runs["none"], keys),
            "acc_ablated": accuracy_on(runs[unforced], keys),
            "acc_ablated_forced": accuracy_on(runs[variant], keys),
            "median_waits_forced": statistics.median(runs[variant][k].get("n_forced", 0) for k in keys),
            "repetition_ablated_forced": statistics.mean(repetition(runs[variant][k]["thinking"]) for k in keys),
            "repetition_intact": statistics.mean(repetition(runs["none"][k]["thinking"]) for k in keys),
        }
        gap = row["acc_intact"] - row["acc_ablated"]
        row["recovered_share"] = (row["acc_ablated_forced"] - row["acc_ablated"]) / gap if gap > 0 else float("nan")
        if control in runs:
            ck = keys & set(runs[control])
            row["intact_gain_from_forcing"] = accuracy_on(runs[control], ck) - accuracy_on(runs["none"], ck)
        else:
            row["intact_gain_from_forcing"] = float("nan")
        row["ablated_gain_from_forcing"] = row["acc_ablated_forced"] - row["acc_ablated"]
        out.append(row)
    return out


def compare(base_runs: dict, runs: dict, long_threshold: int, length_key: str = "thinking_tokens") -> dict:
    """length_key: thinking_tokens for thinking-on runs, answer_tokens for thinking-off."""
    keys = sorted(set(base_runs) & set(runs))
    b = [base_runs[k] for k in keys]
    v = [runs[k] for k in keys]
    bl = [length_of(r, length_key) for r in b]
    vl = [length_of(r, length_key) for r in v]
    lost = sum(x["correct"] and not y["correct"] for x, y in zip(b, v))
    gained = sum(y["correct"] and not x["correct"] for x, y in zip(b, v))
    lo, hi = bootstrap_ci(bl, vl)

    out = {
        "n": len(keys),
        "accuracy": statistics.mean(r["correct"] for r in v),
        "baseline_accuracy": statistics.mean(r["correct"] for r in b),
        "accuracy_change": statistics.mean(r["correct"] for r in v) - statistics.mean(r["correct"] for r in b),
        "mcnemar_p": mcnemar_exact(lost, gained),
        "lost": lost,
        "gained": gained,
        "median_thinking": statistics.median(vl),
        "baseline_median_thinking": statistics.median(bl),
        "length_ratio": median_ratio(bl, vl),
        "length_ratio_ci_low": lo,
        "length_ratio_ci_high": hi,
        "cap_hit_rate": statistics.mean(not r["finished_thinking"] for r in v),
        "mean_wait_count": statistics.mean(r["wait_count"] for r in v),
        "repetition": statistics.mean(repetition(r.get("thinking", "")) for r in v),
    }

    # Problems the baseline reasoned about at length (Goodfire's success measure used these).
    long_pairs = [(x, y) for x, y in zip(b, v) if length_of(x, length_key) > long_threshold]
    out["n_long"] = len(long_pairs)
    if long_pairs:
        out["long_length_ratio"] = median_ratio([length_of(x, length_key) for x, _ in long_pairs],
                                                [length_of(y, length_key) for _, y in long_pairs])
        out["long_accuracy_change"] = (statistics.mean(y["correct"] for _, y in long_pairs)
                                       - statistics.mean(x["correct"] for x, _ in long_pairs))

    # Does accuracy drop where reasoning shrank? Split problems by their own length ratio.
    # If accuracy also drops where length barely changed, something besides length is at work.
    groups = {"shrank_a_lot": [], "shrank_some": [], "barely_changed": []}
    for x, y in zip(b, v):
        if length_of(x, length_key) == 0:
            continue
        ratio = length_of(y, length_key) / length_of(x, length_key)
        key = "shrank_a_lot" if ratio < 0.5 else "shrank_some" if ratio < 0.8 else "barely_changed"
        groups[key].append(y["correct"] - x["correct"])
    for key, diffs in groups.items():
        out[f"n_{key}"] = len(diffs)
        out[f"acc_change_{key}"] = statistics.mean(diffs) if diffs else float("nan")
    return out


def survival_plot(runs: dict, path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping survival plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, recs in sorted(runs.items(), key=lambda kv: kv[0] != "none"):
        if is_nothink(variant):
            continue
        lengths = sorted(r["thinking_tokens"] for r in recs.values())
        n = len(lengths)
        xs = [0] + lengths
        ys = [1.0] + [1 - (i + 1) / n for i in range(n)]
        style = dict(color="black", linewidth=2.2) if variant == "none" else dict(linewidth=1.4)
        ax.step(xs, ys, where="post", label=variant, **style)
    ax.set_xlabel("Thinking tokens")
    ax.set_ylabel("Share of runs still thinking")
    ax.set_title("How long does the model keep reasoning?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"saved {path}")


def fmt(x, digits=3):
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{digits}f}"
    return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--long-threshold", type=int, default=1000,
                   help="Baseline thinking length that counts as 'long reasoning'.")
    args = p.parse_args()

    runs = load_runs(args.folder)
    if "none" not in runs and "none__nothink" not in runs:
        raise SystemExit("no baseline (`none` or `none__nothink`) results found; run it first")
    damage_path = args.folder / "damage.json"
    damage = json.loads(damage_path.read_text()) if damage_path.exists() else {}

    rows = []
    mmlu_base = damage.get("none", {}).get("mmlu_accuracy", float("nan"))
    for variant, recs in sorted(runs.items()):
        base_name = baseline_for(variant)
        if variant == base_name:
            continue
        if base_name not in runs:
            print(f"skipping {variant}: no {base_name} run to compare with")
            continue
        length_key = "answer_tokens" if is_nothink(variant) else "thinking_tokens"
        row = {"variant": variant, "thinking_mode": "off" if is_nothink(variant) else "on",
               **compare(runs[base_name], recs, args.long_threshold, length_key)}
        d = damage.get(variant.split("__")[0], {})
        row["damage_kl"] = d.get("kl", float("nan"))
        row["damage_loss_increase_pct"] = d.get("loss_increase_pct", float("nan"))
        row["mmlu_accuracy"] = d.get("mmlu_accuracy", float("nan"))
        row["mmlu_change"] = row["mmlu_accuracy"] - mmlu_base
        row["mmlu_letter_mass"] = d.get("mmlu_letter_mass", float("nan"))
        rows.append(row)
    on_rows = [r for r in rows if r["thinking_mode"] == "on"]

    for base_name, length_key, what in (("none", "thinking_tokens", "thinking"),
                                        ("none__nothink", "answer_tokens", "answer")):
        if base_name not in runs:
            continue
        base = list(runs[base_name].values())
        print(f"baseline ({base_name}): n={len(base)}, accuracy {statistics.mean(r['correct'] for r in base):.3f}, "
              f"median {what} {statistics.median(length_of(r, length_key) for r in base):.0f} tokens, "
              f"cap hits {statistics.mean(not r['finished_thinking'] for r in base):.3f}, "
              f"MMLU {fmt(mmlu_base)}")
    print()

    # Thinking-on runs measure the length of the reasoning; thinking-off runs
    # measure the length of the shown answer.
    for mode, what in (("on", "thinking"), ("off", "answer")):
        mode_rows = [r for r in rows if r["thinking_mode"] == mode]
        if not mode_rows:
            continue
        print(f"=== thinking {mode.upper()} (length = {what} tokens) ===\n")
        print(f"1. Does it shorten the {what}?  (length ratio = median {what} length vs baseline; 1.0 = no change)")
        print("   Damage columns are measured without reasoning: high values mean the model is broadly broken.")
        print(f"{'variant':<30}{'ratio':>7}{'95% CI':>16}{'long-only':>11}{'wiki KL':>9}{'MMLU chg':>10}")
        for r in mode_rows:
            ci = f"[{fmt(r['length_ratio_ci_low'], 2)}, {fmt(r['length_ratio_ci_high'], 2)}]"
            print(f"{r['variant']:<30}{fmt(r['length_ratio'], 2):>7}{ci:>16}"
                  f"{fmt(r.get('long_length_ratio', float('nan')), 2):>11}{fmt(r['damage_kl'], 4):>9}"
                  f"{fmt(r['mmlu_change'], 3):>10}")

        print(f"\n2. Does accuracy drop, and where?  (change in accuracy, by how much each problem's {what} shrank)")
        print(f"{'variant':<30}{'acc':>7}{'change':>8}{'p':>8}{'<0.5x':>14}{'0.5-0.8x':>14}{'>0.8x':>14}")
        for r in mode_rows:
            cells = [f"{fmt(r[f'acc_change_{k}'], 2)} (n={r[f'n_{k}']})"
                     for k in ("shrank_a_lot", "shrank_some", "barely_changed")]
            print(f"{r['variant']:<30}{fmt(r['accuracy']):>7}{fmt(r['accuracy_change'], 3):>8}"
                  f"{fmt(r['mcnemar_p'], 3):>8}" + "".join(f"{c:>14}" for c in cells))
        print()

    print("\nHow to read part 2: if accuracy drops mostly in the '<0.5x' column and stays flat in")
    print("'>0.8x', the drop goes with shorter reasoning. If it drops in '>0.8x' too, something")
    print("besides length is hurting accuracy. This is suggestive, not proof: harder problems")
    print("may shrink differently. Compare with the __genonly runs to rule out misreading the prompt.")

    forced = forced_report(runs) if "none" in runs else []
    if forced:
        print("\n3. Forced continuation: does accuracy come back when the ablated model is made to keep thinking?")
        print(f"{'variant':<28}{'intact':>8}{'ablated':>9}{'forced':>8}{'recovered':>11}"
              f"{'abl gain':>10}{'intact gain':>13}{'repetition':>12}")
        for r in forced:
            print(f"{r['variant']:<28}{fmt(r['acc_intact']):>8}{fmt(r['acc_ablated']):>9}"
                  f"{fmt(r['acc_ablated_forced']):>8}{fmt(r['recovered_share'], 2):>11}"
                  f"{fmt(r['ablated_gain_from_forcing'], 3):>10}{fmt(r['intact_gain_from_forcing'], 3):>13}"
                  f"   {fmt(r['repetition_ablated_forced'], 2)} vs {fmt(r['repetition_intact'], 2)}")
        print("\nHow to read part 3: 'recovered' is the share of the lost accuracy that forcing brings back")
        print("(1.0 = all of it). Compare 'abl gain' with 'intact gain': forcing alone can help a little,")
        print("so only the gain beyond the intact model's counts. High repetition means the forced text")
        print("is looping, so a failure to recover may just mean the extra reasoning was junk.")
        with (args.folder / "forced_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(forced[0]))
            w.writeheader()
            w.writerows(forced)

    off = {r["variant"].replace("__nothink", ""): r for r in rows if r["thinking_mode"] == "off"}
    paired = [(r, off[r["variant"]]) for r in on_rows if r["variant"] in off]
    if paired:
        print("\n4. Thinking on vs off: does the accuracy drop need reasoning?")
        print(f"{'condition':<24}{'acc chg (on)':>14}{'acc chg (off)':>15}{'p (off)':>9}{'answer len (off)':>18}")
        for on, o in paired:
            print(f"{on['variant']:<24}{fmt(on['accuracy_change'], 3):>14}{fmt(o['accuracy_change'], 3):>15}"
                  f"{fmt(o['mcnemar_p'], 3):>9}{fmt(o['length_ratio'], 2) + 'x':>18}")
        print("\nHow to read part 4: a drop with thinking on but not off means the accuracy loss acts")
        print("through reasoning. A drop in both means Q3 also helps compute answers directly.")
        print("'answer len (off)' should stay near 1.0x; if answers shrink with thinking off, length")
        print("has crept back in through the shown working.")

    out = args.folder / "summary.csv"
    with out.open("w", newline="") as f:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")
    if any(not is_nothink(v) for v in runs):
        survival_plot(runs, args.folder / "survival.png")


if __name__ == "__main__":
    main()
