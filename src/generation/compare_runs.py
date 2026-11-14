"""Quick paired comparison of thinking-on runs against the intact model.

    python src/generation/compare_runs.py results/q3__Qwen__Qwen3-1.7B [--seed 1]

For each finished (or .partial) run: accuracy with lost/gained counts, median
thinking tokens, the median per-problem length ratio with a bootstrap 95%
interval, cap hits, "wait" count, and the stop hazard recorded during
generation. Also the Q3-vs-Q4 contrast on the same problems where both exist.
Complements analyze_q3.py (the full report); this one also reads partial runs.
"""
import json, random, statistics as st, sys
from pathlib import Path

folder = Path(sys.argv[1])
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
load = lambda p: {r["idx"]: r for r in map(json.loads, p.open())}
base = load(folder / f"none__seed{seed}.jsonl")


def ci(xs, f=st.median, n=2000):
    rng = random.Random(0)
    b = sorted(f(rng.choices(xs, k=len(xs))) for _ in range(n))
    return b[int(.025 * n)], b[int(.975 * n)]


runs = {}
for p in sorted(folder.glob(f"*__seed{seed}.jsonl*")):
    name = p.name.split("__seed")[0] + (" (partial)" if p.suffix == ".partial" else "")
    if name.startswith("none") or "nothink" in name:
        continue
    runs[name] = load(p)

print(f"seed {seed}; intact: acc {st.mean(r['correct'] for r in base.values()):.3f}, median thinking "
      f"{st.median(r['thinking_tokens'] for r in base.values()):.0f}, cap {st.mean(not r['finished_thinking'] for r in base.values()):.3f}, "
      f"waits {st.mean(r['wait_count'] for r in base.values()):.1f}\n")
print(f"{'run':>28} |   n | acc (base) lost/gained | median think | length ratio [95% CI] | cap (base) | waits (base) | stop hazard")
for name, run in runs.items():
    idx = sorted(set(run) & set(base))
    r, b = [run[i] for i in idx], [base[i] for i in idx]
    ratios = [x["thinking_tokens"] / max(1, y["thinking_tokens"]) for x, y in zip(r, b)]
    lo, hi = ci(ratios)
    lost = sum(y["correct"] and not x["correct"] for x, y in zip(r, b))
    gained = sum(x["correct"] and not y["correct"] for x, y in zip(r, b))
    print(f"{name:>28} | {len(idx):3d} | {st.mean(x['correct'] for x in r):.3f} ({st.mean(y['correct'] for y in b):.3f}) {lost:2d}/{gained:2d}     | "
          f"{st.median(x['thinking_tokens'] for x in r):7.0f}      | {st.median(ratios):.2f} [{lo:.2f}, {hi:.2f}]     | "
          f"{st.mean(not x['finished_thinking'] for x in r):.3f} ({st.mean(not y['finished_thinking'] for y in b):.3f}) | "
          f"{st.mean(x['wait_count'] for x in r):5.1f} ({st.mean(y['wait_count'] for y in b):.1f}) | "
          f"{st.mean(x.get('stop_hazard', float('nan')) for x in r):.2f}")

print("\nQ3 vs its Q4 match, same problems (ratio of thinking tokens Q3 edit / Q4 edit):")
for name, run in runs.items():
    if not name.startswith("q3_"):
        continue
    other = runs.get("q4_" + name[3:])
    if other is None:
        continue
    idx = sorted(set(run) & set(other))
    ratios = [run[i]["thinking_tokens"] / max(1, other[i]["thinking_tokens"]) for i in idx]
    lo, hi = ci(ratios)
    dw = [run[i]["wait_count"] - other[i]["wait_count"] for i in idx]
    wlo, whi = ci(dw, st.mean)
    print(f"  {name:>24}: n {len(idx)}, length ratio {st.median(ratios):.2f} [{lo:.2f}, {hi:.2f}], "
          f"waits diff {st.mean(dw):+.1f} [{wlo:+.1f}, {whi:+.1f}]")
