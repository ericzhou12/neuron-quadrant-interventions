"""Which way do the leaks push? Three lenses on the average Q3 / Q4 "push".

Every neuron writes into the residual stream along a fixed direction (its column
of the output weights). So a quadrant's AVERAGE contribution per token is one
fixed vector per layer:   push_L = W_down_L @ (mean activation from that quadrant).
This script measures that vector on the intact model's own reasoning text and
asks what it does, three ways:

  1. Logit lens (direct).  Multiply the push by the unembedding matrix: which
     tokens does it raise or lower if written straight to the output? Cheap,
     but ignores everything later layers do with it.
  2. Jacobian lens.  Take the gradient of a behaviour (the urge to end, the
     urge to keep going) with respect to the residual stream at each layer,
     and dot it with the push. A first-order prediction of what adding the
     push does AFTER the rest of the network has processed it.
  3. Steering (causal check).  Actually add alpha x push at every layer and
     token, and measure the same two urges. If adding +0.5 x the Q3 push
     reproduces `q3_grow@0.5`, the leaks act as a constant push; if not, it
     matters which tokens they land on.

The two urges are the probe's (see probe_stop.py), teacher-forced on intact traces:
  end       log of summed P(start "**Final Answer**" or close </think>) before the real end
  continue  log of summed P(Wait, But, Alternatively, Hmm, maybe, check/verify/confirm)

Usage:  python src/probe/leak_lens.py [--n-traces 16] [--seed 1]
Writes results/q3__<model>/leak_lens.json and leak_lens.txt.
"""
import argparse, json, statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys as _sys, pathlib as _pl  # make the sibling src/ folders importable
_sys.path[:0] = [str(_p) for _p in _pl.Path(__file__).resolve().parents[1].iterdir() if _p.is_dir()]
from neuron_sparsity import get_mlps, pick_device
from quadrants import compute_fills
from run_q3_reasoning import load_gsm8k, read_jsonl
from probe_stop import build_sequence

MARKERS = ["Wait", "But", "Alternatively", "Hmm", "Maybe", "maybe", "check", "verify", "confirm"]
ALPHAS = [-0.5, -0.25, 0.25, 0.5, 0.75]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--n-traces", type=int, default=16)
    p.add_argument("--max-thinking-tokens", type=int, default=1200)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    out_dir = Path("results") / f"q3__{args.model.replace('/', '__')}"
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    mlps = get_mlps(model)
    n_layers = len(mlps)
    think_end = tok.convert_tokens_to_ids("</think>")
    final_marker = tok("**Final Answer**", add_special_tokens=False)["input_ids"][:3]
    final_id = final_marker[0]
    end_ids = [final_id, think_end]
    cont_ids = sorted({tok(v, add_special_tokens=False)["input_ids"][0] for w in MARKERS for v in (w, " " + w)})

    traces = [r for r in read_jsonl(out_dir / f"none__seed{args.seed}.jsonl").values()
              if r["finished_thinking"] and r["thinking_tokens"] <= args.max_thinking_tokens]
    traces = sorted(traces, key=lambda r: r["idx"])
    questions = {r["idx"]: r["question"] for r in load_gsm8k(max(t["idx"] for t in traces) + 1)}
    seqs = []
    for tr in traces:
        ids, start = build_sequence(tok, questions[tr["idx"]], tr["thinking"], think_end)
        k_final = None
        for j in range(len(ids) - 3, start - 1, -1):
            if ids[j:j + 3] == final_marker:
                k_final = j - start
                break
        if k_final is not None and k_final > 20:
            seqs.append((tr["idx"], ids, start, k_final))
        if len(seqs) == args.n_traces:
            break
    print(f"{len(seqs)} intact traces (thinking <= {args.max_thinking_tokens} tokens) from none__seed{args.seed}.jsonl")

    # ---- the push vectors, measured on the reasoning text itself
    with torch.no_grad():
        fills, _ = compute_fills(model, [torch.tensor([ids]) for _, ids, _, _ in seqs])
    push = {q: [mlps[i].down_proj.weight.float() @ fills[i][f"{q}_contrib"].float() for i in range(n_layers)]
            for q in ("q3", "q4")}
    third = lambda i: min(2, 3 * i // n_layers)
    cos = lambda a, b: F.cosine_similarity(a, b, 0).item()
    report = {"n_traces": len(seqs), "layers": []}
    for i in range(n_layers):
        report["layers"].append({"layer": i, "norm_q3": push["q3"][i].norm().item(),
                                 "norm_q4": push["q4"][i].norm().item(), "cos_q3_q4": cos(push["q3"][i], push["q4"][i])})

    # ---- 1. direct logit lens
    W_U = model.lm_head.weight.float()
    watch = {"** (start Final Answer)": final_id, "</think>": think_end}
    for w in ("Wait", "But", "Alternatively", "Hmm", "maybe", "check"):
        watch[w] = tok(" " + w, add_special_tokens=False)["input_ids"][0]
    lens = {}
    for q in ("q3", "q4"):
        for name, layers in (("early", [i for i in range(n_layers) if third(i) == 0]),
                             ("mid", [i for i in range(n_layers) if third(i) == 1]),
                             ("late", [i for i in range(n_layers) if third(i) == 2]),
                             ("all", list(range(n_layers)))):
            d = W_U @ torch.stack([push[q][i] for i in layers]).sum(0)
            z = (d - d.mean()) / d.std()
            top = [tok.decode([t]) for t in z.topk(12).indices.tolist()]
            bot = [tok.decode([t]) for t in (-z).topk(12).indices.tolist()]
            lens[f"{q}_{name}"] = {"top": top, "bottom": bot, "z": {k: z[v].item() for k, v in watch.items()}}
    report["logit_lens"] = lens

    # ---- shared: the two urges for one trace, optionally with a steering vector added
    def urges(ids, start, k_final, add=None, grad=False):
        x = torch.tensor([ids], device=device)
        handles, outs = [], [None] * n_layers
        for i, mlp in enumerate(mlps):
            def hook(_m, _inp, out, i=i):
                if add is not None:
                    out = out + add[i]
                if grad:
                    out.retain_grad()
                    outs[i] = out
                return out
            handles.append(mlp.down_proj.register_forward_hook(hook))
        try:
            lp = F.log_softmax(model(x).logits[0, start - 1:len(ids) - 1].float(), -1)
        finally:
            for h in handles:
                h.remove()
        end = torch.logsumexp(lp[:k_final][:, end_ids].reshape(-1), 0)
        cont = torch.logsumexp(lp[:, cont_ids].reshape(-1), 0)
        return end, cont, outs

    # ---- 2. Jacobian lens: d(urge) / d(alpha) for adding alpha x push at every token of layer L
    jac = {f"{q}_{u}": torch.zeros(n_layers) for q in ("q3", "q4") for u in ("end", "continue")}
    base = []
    for n, (idx, ids, start, k_final) in enumerate(seqs):
        with torch.enable_grad():
            # a dummy leaf so the graph is built even though the weights are frozen
            leaf = [torch.zeros(1, device=device, requires_grad=True) for _ in range(n_layers)]
            end, cont, outs = urges(ids, start, k_final, add=[l * 0 for l in leaf], grad=True)
            for u, obj in (("end", end), ("continue", cont)):
                for o in outs:
                    o.grad = None
                obj.backward(retain_graph=(u == "end"))
                for i, o in enumerate(outs):
                    g = o.grad[0].float().sum(0)  # summed over tokens: the push is added at every token
                    for q in ("q3", "q4"):
                        jac[f"{q}_{u}"][i] += (g @ push[q][i]).item()
        base.append((end.item(), cont.item()))
        del outs, end, cont
        print(f"  jacobian {n + 1}/{len(seqs)}")
    jac = {k: (v / len(seqs)).tolist() for k, v in jac.items()}
    report["jacobian_per_layer"] = jac
    report["jacobian_total"] = {k: sum(v) for k, v in jac.items()}
    report["jacobian_by_third"] = {k: [sum(x for i, x in enumerate(v) if third(i) == t) for t in range(3)]
                                   for k, v in jac.items()}

    # ---- 3. steering: actually add alpha x push everywhere and re-measure
    steer = {}
    with torch.no_grad():
        for q in ("q3", "q4"):
            for a in ALPHAS:
                add = [a * push[q][i] for i in range(n_layers)]
                d_end, d_cont = [], []
                for (idx, ids, start, k_final), (b_end, b_cont) in zip(seqs, base):
                    end, cont, _ = urges(ids, start, k_final, add=add)
                    d_end.append(end.item() - b_end)
                    d_cont.append(cont.item() - b_cont)
                steer[f"{q}@{a:+.2f}"] = {"alpha": a, "end": st.mean(d_end), "continue": st.mean(d_cont),
                                          "end_sd": st.pstdev(d_end), "continue_sd": st.pstdev(d_cont),
                                          "end_linear": a * report["jacobian_total"][f"{q}_end"],
                                          "continue_linear": a * report["jacobian_total"][f"{q}_continue"]}
                print(f"  steer {q} alpha {a:+.2f}: end {steer[f'{q}@{a:+.2f}']['end']:+.2f}  "
                      f"continue {steer[f'{q}@{a:+.2f}']['continue']:+.2f}")
    report["steering"] = steer

    # ---- write
    (out_dir / "leak_lens.json").write_text(json.dumps(report, indent=1))
    L = [f"Leak push lenses, {args.model}, {len(seqs)} intact traces (seed {args.seed})", "",
         "Size and alignment of the average push per layer (norm Q3, norm Q4, cosine between them):"]
    for r in report["layers"]:
        L.append(f"  layer {r['layer']:2d}  {r['norm_q3']:.3f}  {r['norm_q4']:.3f}  {r['cos_q3_q4']:+.2f}")
    L += ["", "1. Direct logit lens (z-score of each watched token among the whole vocabulary):"]
    for k, v in lens.items():
        L.append(f"  {k:9s} " + "  ".join(f"{t} {z:+.1f}" for t, z in v["z"].items()))
        L.append(f"            raises: {v['top']}")
        L.append(f"            lowers: {v['bottom']}")
    L += ["", "2. Jacobian lens: predicted change in each urge per unit of push (alpha = 1), by third of the stack and total:"]
    for k, v in report["jacobian_by_third"].items():
        L.append(f"  {k:12s} early {v[0]:+.2f}  mid {v[1]:+.2f}  late {v[2]:+.2f}  total {report['jacobian_total'][k]:+.2f}")
    L += ["", "3. Steering: measured change when alpha x push is really added (and the Jacobian's straight-line prediction):"]
    for k, v in steer.items():
        L.append(f"  {k:9s} end {v['end']:+.2f} (linear {v['end_linear']:+.2f})   "
                 f"continue {v['continue']:+.2f} (linear {v['continue_linear']:+.2f})")
    (out_dir / "leak_lens.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
