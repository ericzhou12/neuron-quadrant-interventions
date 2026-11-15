"""Summarise the teacher-forced probe (probe_stop*.jsonl), paired against the intact model.

    python src/probe/analyze_probe.py results/q3__Qwen__Qwen3-1.7B [tag ...]

Per condition, over the same intact traces:
    KL            damage on the reasoning text (use it to match Q3 and Q4 edits)
    wait          mean log ratio of wait_hazard (edited / intact): >0 = more urge to say "Wait"
    conclude      mean log ratio of final_hazard: >0 = more urge to start "**Final Answer**" early
    at end        mean change in log P(start Final Answer) where the intact model really did
Bootstrap 95% intervals over traces. Writes probe_summary.json and prints Q3-minus-Q4 contrasts.
"""
import collections, json, math, random, statistics as st, sys
from pathlib import Path

folder = Path(sys.argv[1])
tags = sys.argv[2:]
files = sorted(folder.glob("probe_stop*.jsonl")) if not tags else [folder / f"probe_stop__{t}.jsonl" for t in tags]
rows = collections.defaultdict(dict)
for f in files:
    for line in f.open():
        r = json.loads(line)
        rows[(f.stem, r["condition"])][r["idx"]] = r


def ci(d, n=2000):
    rng = random.Random(0)
    b = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return [b[int(.025 * n)], b[int(.975 * n)]]


def stat(d):
    return {"mean": st.mean(d), "ci": ci(d), "n": len(d)} if len(d) >= 5 else None


EPS = 1e-6
per_trace, summary = {}, {}
for (stem, cond), rs in rows.items():
    if cond == "none":
        continue
    base = rows[(stem, "none")]
    idx = sorted(set(rs) & set(base))
    d = {"wait": {i: math.log((rs[i]["wait_hazard"] + EPS) / (base[i]["wait_hazard"] + EPS)) for i in idx}}
    if all("cont_hazard" in rs[i] and "cont_hazard" in base[i] for i in idx):
        d["continue_any"] = {i: math.log((rs[i]["cont_hazard"] + EPS) / (base[i]["cont_hazard"] + EPS)) for i in idx}
        for m in rs[idx[0]]["marker_hazards"]:
            d["m_" + m] = {i: math.log((rs[i]["marker_hazards"][m] + 1e-3) / (base[i]["marker_hazards"][m] + 1e-3))
                           for i in idx}
    fin = [i for i in idx if rs[i].get("final_hazard") is not None]
    if fin:
        # eps of 1e-3: hazards below one-in-a-thousand per trace are treated as equal
        d["conclude"] = {i: math.log((rs[i]["final_hazard"] + 1e-3) / (base[i]["final_hazard"] + 1e-3)) for i in fin}
        d["at_end"] = {i: rs[i]["logp_final_end"] - base[i]["logp_final_end"] for i in fin}
        if all(rs[i].get("end_hazard_pre") is not None and base[i].get("end_hazard_pre") is not None for i in fin):
            d["close_direct"] = {i: math.log((rs[i]["close_hazard_pre"] + 1e-3) / (base[i]["close_hazard_pre"] + 1e-3)) for i in fin}
            d["end_either"] = {i: math.log((rs[i]["end_hazard_pre"] + 1e-3) / (base[i]["end_hazard_pre"] + 1e-3)) for i in fin}
    per_trace[cond] = d
    summary[cond] = {"kl": st.mean(rs[i]["kl"] for i in idx), "n": len(idx),
                     **{k: stat(list(v.values())) for k, v in d.items()}}

f = lambda s: "        -          " if s is None else f"{s['mean']:+.2f} [{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}]"
print(f"{'condition':>20} |  n |    KL | wait (log ratio)    | conclude early (log ratio) | log P(conclude) at true end")
for c, s in summary.items():
    print(f"{c:>20} | {s['n']:2d} | {s['kl']:.3f} | {f(s['wait'])} | {f(s.get('conclude'))}        | {f(s.get('at_end'))}")

print("\nQ3 edit minus matching Q4 edit (same traces):")
contrasts = {}
for c in summary:
    if not c.startswith("q3_") or "q4_" + c[3:] not in per_trace:
        continue
    o = "q4_" + c[3:]
    out = {}
    for k in per_trace[c]:
        if k in per_trace[o]:
            ids = sorted(set(per_trace[c][k]) & set(per_trace[o][k]))
            out[k] = stat([per_trace[c][k][i] - per_trace[o][k][i] for i in ids])
    contrasts[c] = out
    print(f"{c:>20} | KL {summary[c]['kl']:.3f} vs {summary[o]['kl']:.3f} | wait {f(out.get('wait'))} | "
          f"conclude {f(out.get('conclude'))} | at end {f(out.get('at_end'))}")
mk = [k for k in next(iter(per_trace.values()), {})
      if k in ("continue_any", "conclude", "close_direct", "end_either") or k.startswith("m_")]
if mk:
    g = lambda s: "   -   " if s is None else f"{s['mean']:+.2f}"
    print("\nContinuation words, mean log ratio vs intact (summed probability along the trace):")
    print(f"{'condition':>20} | " + " | ".join(f"{k.replace('m_', ''):>13}" for k in mk))
    for c, sm in summary.items():
        print(f"{c:>20} | " + " | ".join(f"{g(sm.get(k)):>13}" for k in mk))
    print("\nQ3 minus Q4, continuation words [95% CI]:")
    for c, out in contrasts.items():
        print(f"{c:>20} | " + " | ".join(f"{k.replace('m_', '')} {f(out.get(k))}" for k in mk))
name = "probe_summary" + ("__" + "_".join(tags) if tags else "") + ".json"
(folder / name).write_text(json.dumps({"per_condition": summary, "q3_minus_q4": contrasts}, indent=2))
print(f"\nwrote {folder / name}")
