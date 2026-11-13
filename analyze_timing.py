"""Does Q3 activity change right before the model stops thinking or says "Wait"?

Reads probe_timing.jsonl (from `probe_stop.py --timing`): the intact model's
per-token Q3/Q4 activity along its own thinking traces.

The prediction of token t is computed at position t-1, so we look at activity at
t-1. A fair comparison needs the same token at t-1: "Wait" almost always comes
right after a paragraph break, and the break token itself has its own typical
activity. So for each event we compare against CONTROL positions in the same
trace whose t-1 token is identical but where the model wrote something else:

    wait  : t-1 token is X, next token is "Wait"     vs   t-1 token is X, next is not "Wait"
    end   : t-1 token is X, next token is </think>   vs   t-1 token is X, next is not </think>

Activity is z-scored within each trace (subtract the trace's mean, divide by
its std) so traces with generally higher activity don't dominate.

H2 (token-specific signal) predicts a consistent difference: e.g. Q3 activity
drops before </think> and/or rises before "Wait", and Q4 doesn't show the same.
H3 (steady bias) predicts no difference.

    python analyze_timing.py results/q3__Qwen__Qwen3-1.7B/probe_timing.jsonl
"""

from __future__ import annotations

import json
import random
import statistics
import sys

from transformers import AutoTokenizer


def zscore(xs):
    m, s = statistics.mean(xs), statistics.pstdev(xs) or 1.0
    return [(x - m) / s for x in xs]


def boot_ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    b = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return b[int(0.025 * n)], b[int(0.975 * n)]


def main(path, model="Qwen/Qwen3-1.7B"):
    tok = AutoTokenizer.from_pretrained(model)
    think_end = tok.convert_tokens_to_ids("</think>")
    wait_ids = {tok(s, add_special_tokens=False)["input_ids"][0] for s in ("Wait", " Wait")}
    recs = [json.loads(l) for l in open(path)]
    series = [k for k in recs[0] if k.startswith(("q3_", "q4_"))]

    events = {"wait": lambda t: t in wait_ids, "end": lambda t: t == think_end}
    print(f"{len(recs)} traces. Values: mean within-trace z-score at the token BEFORE the event, minus the same at")
    print("matched control positions (same preceding token, different next token). Bootstrap 95% CI over traces.\n")
    for ev, is_ev in events.items():
        print(f"== before {ev!r}")
        for key in series:
            diffs, n_ev = [], 0
            for r in recs:
                toks, z = r["tokens"], zscore(r[key])
                ev_pos = [t for t in range(1, len(toks)) if is_ev(toks[t])]
                if not ev_pos:
                    continue
                per_trace = []
                for t in ev_pos:
                    prev = toks[t - 1]
                    ctrl = [z[s - 1] for s in range(1, len(toks)) if toks[s - 1] == prev and not is_ev(toks[s])]
                    if ctrl:
                        per_trace.append(z[t - 1] - statistics.mean(ctrl))
                if per_trace:
                    diffs.append(statistics.mean(per_trace))
                    n_ev += len(per_trace)
            if len(diffs) < 5:
                print(f"  {key:>18}: too few events ({len(diffs)} traces)")
                continue
            lo, hi = boot_ci(diffs)
            flag = "  *" if lo > 0 or hi < 0 else ""
            print(f"  {key:>18}: {statistics.mean(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"({len(diffs)} traces, {n_ev} events){flag}")
        print()
    print("* = interval excludes 0. Many series are tested, so treat lone stars with caution.")


if __name__ == "__main__":
    main(*sys.argv[1:])
