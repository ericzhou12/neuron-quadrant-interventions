"""Where does reasoning get longer or shorter: before the model first reaches its answer, or after?

    python src/generation/analyze_phases.py results/q3__Qwen__Qwen3-1.7B [--seed 1]

For each finished trace we find the first place in the thinking where the
model's final answer (the number it ends up giving) appears as a number.
    solve  = thinking up to that point (getting to the answer the first time)
    check  = thinking after it (re-reading, re-deriving, second-guessing)
Lengths are in characters of thinking text, converted to approximate tokens by
the trace's own tokens-per-character rate. Traces that hit the cap, gave no
answer, or never state their answer in the thinking are left out, so this
describes finished reasoning only (it cannot see runaway traces).
Compared with the intact model on the same problems, medians with bootstrap
95% intervals of the paired ratio.
"""
import json, random, re, statistics as st, sys
from pathlib import Path

folder = Path(sys.argv[1])
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
load = lambda p: {r["idx"]: r for r in map(json.loads, p.open())}


def phases(r):
    if not r["finished_thinking"] or r.get("pred") is None or not r["thinking"]:
        return None
    try:
        target = float(r["pred"])
    except ValueError:
        return None
    text = r["thinking"]
    for m in re.finditer(r"-?\d[\d,]*\.?\d*", text):
        try:
            if abs(float(m.group().replace(",", "").rstrip(".")) - target) < 1e-6:
                rate = r["thinking_tokens"] / len(text)
                return m.end() * rate, (len(text) - m.end()) * rate
        except ValueError:
            pass
    return None


def ci(xs, n=2000):
    rng = random.Random(0)
    b = sorted(st.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return b[int(.025 * n)], b[int(.975 * n)]


base = load(folder / f"none__seed{seed}.jsonl")
bp = {i: phases(r) for i, r in base.items()}
ok = [p for p in bp.values() if p]
print(f"seed {seed}; intact: {len(ok)}/{len(base)} usable traces, median solve {st.median(p[0] for p in ok):.0f} tokens, "
      f"median check {st.median(p[1] for p in ok):.0f} tokens\n")
print(f"{'run':>16} |  n | solve: median, ratio vs intact [95% CI] | check: median, ratio vs intact [95% CI] | usable")
out = {}
for p in sorted(folder.glob(f"*__seed{seed}.jsonl")):
    name = p.name.split("__seed")[0]
    if name.startswith("none") or "nothink" in name or "__" in name:
        continue
    run = load(p)
    pairs = [(phases(run[i]), bp[i]) for i in run if i in bp]
    usable = sum(1 for a, _ in pairs if a)
    pairs = [(a, b) for a, b in pairs if a and b]
    if len(pairs) < 20:
        continue
    rs = [a[0] / max(b[0], 1) for a, b in pairs]
    rc = [(a[1] + 20) / (b[1] + 20) for a, b in pairs]
    (sl, sh), (cl, chh) = ci(rs), ci(rc)
    out[name] = {"n": len(pairs), "solve_median": st.median(a[0] for a, _ in pairs), "solve_ratio": st.median(rs), "solve_ci": [sl, sh],
                 "check_median": st.median(a[1] for a, _ in pairs), "check_ratio": st.median(rc), "check_ci": [cl, chh]}
    print(f"{name:>16} | {len(pairs):3d} | {out[name]['solve_median']:5.0f}  {st.median(rs):.2f} [{sl:.2f}, {sh:.2f}]            | "
          f"{out[name]['check_median']:5.0f}  {st.median(rc):.2f} [{cl:.2f}, {chh:.2f}]            | {usable}/{len(run)}")
(folder / f"phases_summary__seed{seed}.json").write_text(json.dumps(out, indent=2))
