"""Teacher-forced probe: how much does each edit change the model's urge to stop thinking?

Generation is slow and noisy: one sampled trace per problem, hours per condition.
This probe instead takes the intact model's own finished thinking traces (from
the `none` thinking-on run), feeds each one back through the model under every
condition, and reads off at each position:

    p_stop  = P(next token is </think>)    the urge to end the thinking
    p_wait  = P(next token is "Wait")      the urge to second-guess

Same text for every condition, so the comparison is exactly paired and has no
sampling noise. One forward pass per trace per condition: minutes, not hours.

Per trace we record (all summed or averaged over the thinking positions):
    stop_hazard      sum of p_stop before the real end. Roughly "how many times
                     would the model have tried to stop along this path". Higher
                     under an edit = the edit pushes toward shorter reasoning.
    logp_stop_end    log p_stop at the position where the intact model did stop.
    wait_hazard      sum of p_wait over all positions.
    nll              mean -log P(actual next token): damage on reasoning text itself.
    kl               mean KL(intact || edited) over thinking positions: the same,
                     as a distribution distance. Use it to match damage between
                     Q3 and Q4 edits.

What this can and can't show: it measures the local urge to stop along the
intact model's path, not the length the edited model would actually produce
(it would wander off that path). It is a mechanism probe; the generation runs
are the confirmatory test.

With --timing, also saves per-position Q3/Q4 activity from the intact model
(share of neurons in each quadrant, and their summed activation, per layer
group) so you can see whether Q3 activity changes right before </think> or "Wait".

    python src/probe/probe_stop.py --model Qwen/Qwen3-1.7B --n-traces 100 \
        --conditions none,q3_qmean,q4_qmean,q3_zero@0.25,q4_zero@0.25,q3_resample,q3_pool,q3_const --timing
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys as _sys, pathlib as _pl  # make the sibling src/ folders importable
_sys.path[:0] = [str(_p) for _p in _pl.Path(__file__).resolve().parents[1].iterdir() if _p.is_dir()]
from neuron_sparsity import get_mlps, pick_device
from quadrants import compute_fills, describe, intervene, parse_condition, quadrant_masks
from run_q3_reasoning import PROMPT_SUFFIX, calibration_batches, load_gsm8k, read_jsonl


def build_sequence(tokenizer, question: str, thinking: str, think_end_id: int) -> tuple[list[int], int]:
    """Prompt + thinking + </think>, as token ids. Returns (ids, index of first thinking token)."""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question + PROMPT_SUFFIX}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=True)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not thinking.lstrip().startswith("<think>"):
        thinking = "<think>\n" + thinking.lstrip("\n")
    t_ids = tokenizer(thinking, add_special_tokens=False)["input_ids"]
    return p_ids + t_ids + [think_end_id], len(p_ids)


class QuadrantTimer:
    """Hooks that record, per token, each quadrant's share of neurons and summed
    activation, averaged within layer groups (early / middle / late thirds)."""

    def __init__(self, model, quads=("q3", "q4")):
        self.mlps = get_mlps(model)
        self.quads = quads
        self.stash, self.rows, self.handles = {}, {}, []
        n = len(self.mlps)
        self.group = [min(2, 3 * i // n) for i in range(n)]

    def __enter__(self):
        for i, mlp in enumerate(self.mlps):
            self.handles.append(mlp.gate_proj.register_forward_hook(
                lambda _m, _i, o, i=i: self.stash.__setitem__((i, "g"), o)))
            self.handles.append(mlp.up_proj.register_forward_hook(
                lambda _m, _i, o, i=i: self.stash.__setitem__((i, "u"), o)))

            def rec(_m, args, i=i):
                acts = args[0][0].float()  # batch of 1: [seq, d]
                masks = quadrant_masks(self.stash.pop((i, "g"))[0], self.stash.pop((i, "u"))[0])
                for q in self.quads:
                    m = masks[q]
                    key = (q, self.group[i])
                    share = m.float().mean(-1)
                    mass = (acts * m).sum(-1)
                    s, ms, c = self.rows.get(key, (0, 0, 0))
                    self.rows[key] = (s + share, ms + mass, c + 1)

            self.handles.append(mlp.down_proj.register_forward_pre_hook(rec))
        return self

    def result(self) -> dict:
        out = {}
        for (q, grp), (s, ms, c) in self.rows.items():
            name = ("early", "mid", "late")[grp]
            out[f"{q}_share_{name}"] = (s / c).cpu().tolist()
            out[f"{q}_mass_{name}"] = (ms / c).cpu().tolist()
        self.rows = {}
        return out

    def __exit__(self, *_):
        for h in self.handles:
            h.remove()
        self.stash.clear()


def paired_summary(base: list[float], other: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean paired difference with a bootstrap 95% interval, and share of traces where it went up."""
    diffs = [o - b for b, o in zip(base, other)]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    return {"mean_diff": statistics.mean(diffs), "ci_low": boots[int(0.025 * n_boot)],
            "ci_high": boots[int(0.975 * n_boot)], "share_up": statistics.mean(d > 0 for d in diffs)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--conditions", default="none,q3_qmean,q4_qmean")
    p.add_argument("--traces", help="jsonl of intact thinking-on traces (default: the none__seed0 run)")
    p.add_argument("--n-traces", type=int, default=100)
    p.add_argument("--max-thinking-tokens", type=int, default=3500,
                   help="Skip traces longer than this (memory: logits are seq x 150k).")
    p.add_argument("--timing", action="store_true", help="Also save per-token Q3/Q4 activity of the intact model.")
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--tag", default="", help="Suffix for the output file names.")
    args = p.parse_args()
    conditions = args.conditions.split(",")
    for c in conditions:
        parse_condition(c)
    if "none" not in conditions:
        conditions = ["none"] + conditions

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device).eval()
    think_end_id = tok.convert_tokens_to_ids("</think>")
    wait_ids = sorted({tok(s, add_special_tokens=False)["input_ids"][0] for s in ("Wait", " Wait")})
    # "Wait" is only about a sixth of the model's backtracking phrases, so also track a
    # wider set of continuation / second-guessing words (each is a single token here).
    MARKERS = {"wait": ["Wait"], "but": ["But"], "alternatively": ["Alternatively"], "hmm": ["Hmm"],
               "maybe": ["Maybe", "maybe"], "check": ["check", "verify", "confirm"]}
    marker_ids = {}
    for name, words in MARKERS.items():
        ids = set()
        for w in words:
            for v in (w, " " + w):
                enc = tok(v, add_special_tokens=False)["input_ids"]
                assert len(enc) == 1, f"{v!r} is not a single token"
                ids.add(enc[0])
        marker_ids[name] = sorted(ids)
    all_marker_ids = sorted({i for ids in marker_ids.values() for i in ids})
    # The intact model nearly always ends its thinking with a "**Final Answer**" block,
    # after which </think> is almost certain. So the real decision to stop is the
    # decision to START that block: P(first token of "**Final Answer**").
    # First three tokens only ("**", "Final", " Answer"): the closing "**" merges with the newline.
    final_marker = tok("**Final Answer**", add_special_tokens=False)["input_ids"][:3]
    final_id = final_marker[0]

    out_dir = Path("results") / f"q3__{args.model.replace('/', '__')}"
    trace_path = Path(args.traces) if args.traces else out_dir / f"none__seed{args.seed}.jsonl"
    traces = [r for r in read_jsonl(trace_path).values() if r["finished_thinking"]
              and r.get("thinking_mode", "on") == "on" and r["thinking_tokens"] <= args.max_thinking_tokens]
    traces = sorted(traces, key=lambda r: r["idx"])[: args.n_traces]
    questions = {r["idx"]: r["question"] for r in load_gsm8k(max(t["idx"] for t in traces) + 1)}
    print(f"{len(traces)} intact traces from {trace_path.name}")

    print("computing fill values...")
    fills, _ = compute_fills(model, calibration_batches(tok, args.calib_batches))

    tag = f"__{args.tag}" if args.tag else ""
    rec_path = out_dir / f"probe_stop{tag}.jsonl"
    timing_path = out_dir / f"probe_timing{tag}.jsonl"
    rec_f = rec_path.open("w")
    tim_f = timing_path.open("w") if args.timing else None

    per_cond = {c: [] for c in conditions}
    with torch.no_grad():
        for n, tr in enumerate(traces):
            ids, start = build_sequence(tok, questions[tr["idx"]], tr["thinking"], think_end_id)
            x = torch.tensor([ids], device=device)
            # Positions whose NEXT token is a thinking token or the final </think>.
            pos = torch.arange(start - 1, len(ids) - 1, device=device)
            targets = x[0, pos + 1]
            # Where (if anywhere) the intact trace starts its last "**Final Answer**" block,
            # as an index into `pos`/`lp` (the position that predicts the block's first token).
            k_final = None
            m = len(final_marker)
            for j in range(len(ids) - m, start - 1, -1):
                if ids[j : j + m] == final_marker:
                    k_final = j - start
                    break
            base_lp = None
            for c in conditions:
                if c == "none" and tim_f is not None:
                    with QuadrantTimer(model) as timer:
                        logits = model(x).logits[0, pos].float()
                    tim = timer.result()
                    tim_f.write(json.dumps({"idx": tr["idx"], "start": start,
                                            "tokens": ids[start:], **{k: v[start:] for k, v in tim.items()}}) + "\n")
                else:
                    with intervene(model, c, fills, seed=args.seed + n):
                        logits = model(x).logits[0, pos].float()
                lp = F.log_softmax(logits, -1)
                del logits
                if c == "none":
                    base_lp = lp
                p_stop = lp[:, think_end_id].exp()
                p_wait = lp[:, wait_ids].exp().sum(-1)
                r = {
                    "condition": c, "idx": tr["idx"], "n_positions": len(pos),
                    "stop_hazard": float(p_stop[:-1].sum()),
                    "logp_stop_end": float(lp[-1, think_end_id]),
                    "wait_hazard": float(p_wait.sum()),
                    # summed P(word) along the trace for each continuation word, and for all of them
                    "marker_hazards": {m: float(lp[:, ids].exp().sum()) for m, ids in marker_ids.items()},
                    "cont_hazard": float(lp[:, all_marker_ids].exp().sum()),
                    # conclude hazard: summed P(start the Final Answer block) BEFORE the model really did,
                    # and log P(start it) at the point where the intact model did start it.
                    "final_hazard": None if k_final is None else float(lp[:k_final, final_id].exp().sum()),
                    "logp_final_end": None if k_final is None else float(lp[k_final, final_id]),
                    "final_hazard_all": float(lp[:, final_id].exp().sum()),
                    # The edited model may end its thinking WITHOUT a Final Answer block (straight to
                    # </think>). So also record the urge to close thinking directly before the real
                    # end, and the two routes combined ("end by either route").
                    "close_hazard_pre": None if k_final is None else float(lp[:k_final, think_end_id].exp().sum()),
                    "end_hazard_pre": None if k_final is None else float(
                        lp[:k_final, [final_id, think_end_id]].exp().sum()),
                    "nll": float(-lp.gather(1, targets[:, None]).mean()),
                    "kl": float((base_lp.exp() * (base_lp - lp)).sum(-1).mean()),
                }
                per_cond[c].append(r)
                rec_f.write(json.dumps(r) + "\n")
                if c != "none":
                    del lp
            rec_f.flush()
            if (n + 1) % 10 == 0:
                print(f"  {n + 1}/{len(traces)} traces", flush=True)
    rec_f.close()
    if tim_f:
        tim_f.close()

    # Summary: paired differences vs the intact model on the same traces.
    metrics = ("stop_hazard", "logp_stop_end", "wait_hazard", "nll", "kl")
    base = per_cond["none"]
    print(f"\nintact model, mean per trace: " + ", ".join(
        f"{m} {statistics.mean(r[m] for r in base):.3f}" for m in metrics[:4]))
    summary = {}
    print(f"\n{'condition':>16} | {'KL':>6} | {'NLL +':>7} | stop hazard change [95% CI] (share up) | "
          f"log p(stop) at end | wait hazard change")
    for c in conditions:
        if c == "none":
            continue
        rows = per_cond[c]
        s = {m: paired_summary([b[m] for b in base], [r[m] for r in rows]) for m in metrics[:4]}
        s["kl"] = statistics.mean(r["kl"] for r in rows)
        summary[c] = s
        sh, le, wh, nl = s["stop_hazard"], s["logp_stop_end"], s["wait_hazard"], s["nll"]
        print(f"{c:>16} | {s['kl']:6.3f} | {nl['mean_diff']:+7.3f} | "
              f"{sh['mean_diff']:+7.3f} [{sh['ci_low']:+.3f}, {sh['ci_high']:+.3f}] ({sh['share_up']:.0%}) | "
              f"{le['mean_diff']:+6.2f} [{le['ci_low']:+.2f}, {le['ci_high']:+.2f}] | "
              f"{wh['mean_diff']:+7.3f} [{wh['ci_low']:+.3f}, {wh['ci_high']:+.3f}]")
    (out_dir / f"probe_stop_summary{tag}.json").write_text(json.dumps(
        {"n_traces": len(traces), "conditions": {c: describe(c) for c in conditions},
         "intact_means": {m: statistics.mean(r[m] for r in base) for m in metrics[:4]}, "paired": summary},
        indent=2))
    print(f"\nper-trace records: {rec_path}")


if __name__ == "__main__":
    main()
