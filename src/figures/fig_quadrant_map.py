"""Map of neuron firings in the (gate, up) plane for Qwen3-1.7B, in the style of
the quadrant figure in Goodfire's "Neurons as an Interpretable Basis".

Every MLP neuron on every token is one firing with two pre-activations: gate
and up. Its output is act_fn(gate) * up. The signs of gate and up put the
firing in one of four quadrants (Q1..Q4).

Panel a: a random sample of firings, one dot each, coloured by quadrant.
Panel b: the same sampled firings, dot area proportional to |output|,
         colour by the sign of the output.
Panel c: each quadrant's share of firings and of total |activation|.

The text is the intact model's own GSM8K reasoning (prompt + thinking block of
the first --n-traces problems in none__seed1.jsonl), replayed teacher-forced.
One short forward pass per trace; the histogram is cached in quadrant_map.npz,
so re-plotting needs no model (--replot).

Usage: python src/figures/fig_quadrant_map.py [--n-traces 24] [--max-tokens 768] [--replot]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm

sys.path[:0] = [str(p) for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]
import style  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--folder", default="results/q3__Qwen__Qwen3-1.7B")
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--n-traces", type=int, default=24)
p.add_argument("--max-tokens", type=int, default=768)
p.add_argument("--replot", action="store_true")
p.add_argument("--dots-per-layer", type=int, default=40, help="random firings kept per layer per trace for the dot plot")
args = p.parse_args()
folder = Path(args.folder)
cache = folder / "quadrant_map.npz"
LIM, BINS = 6.0, 240  # histogram covers gate, up in [-LIM, LIM]
QUADS = ["q1", "q2", "q3", "q4"]

if not (args.replot and cache.exists() and "sample_g" in np.load(cache).files):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neuron_sparsity import get_mlps, pick_device
    from run_q3_reasoning import PROMPT_SUFFIX, load_gsm8k

    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device).eval()
    mlps = get_mlps(model)
    runs = {r["idx"]: r for r in map(json.loads, (folder / "none__seed1.jsonl").open())}
    problems = load_gsm8k(args.n_traces)

    hist = torch.zeros(BINS * BINS, dtype=torch.float64)
    counts = torch.zeros(len(mlps), 4, dtype=torch.float64)   # firings per layer, quadrant
    mass = torch.zeros(len(mlps), 4, dtype=torch.float64)     # sum |activation|
    outside, stash, n_tokens = 0, {}, 0
    sample_g, sample_u = [], []
    gen = torch.Generator().manual_seed(0)

    def keep(name, i):
        return lambda _m, _inp, out: stash.__setitem__((i, name), out.detach())

    handles = []
    for i, mlp in enumerate(mlps):
        handles += [mlp.gate_proj.register_forward_hook(keep("g", i)), mlp.up_proj.register_forward_hook(keep("u", i))]

    with torch.no_grad():
        for prob in problems:
            prompt = tok.apply_chat_template([{"role": "user", "content": prob["question"] + PROMPT_SUFFIX}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
            thinking = runs[prob["idx"]]["thinking"]
            text = prompt + (thinking if thinking.lstrip().startswith("<think>") else "<think>\n" + thinking)
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, : n_prompt + args.max_tokens]
            model(ids.to(device))
            for i, mlp in enumerate(mlps):
                g = stash.pop((i, "g"))[0, n_prompt:].float()   # thinking tokens only
                u = stash.pop((i, "u"))[0, n_prompt:].float()
                a = (mlp.act_fn(g) * u).abs()
                for k, m in enumerate(((g > 0) & (u > 0), (g > 0) & (u < 0), (g < 0) & (u < 0), (g < 0) & (u > 0))):
                    counts[i, k] += m.sum().item()
                    mass[i, k] += a[m].sum().item()
                pick = torch.randint(0, g.numel(), (args.dots_per_layer,), generator=gen).to(g.device)  # uniform over firings
                sample_g.append(g.reshape(-1)[pick].cpu()); sample_u.append(u.reshape(-1)[pick].cpu())
                inside = (g.abs() < LIM) & (u.abs() < LIM)
                outside += (~inside).sum().item()
                gi = ((g[inside] + LIM) / (2 * LIM) * BINS).long().clamp_(0, BINS - 1)
                ui = ((u[inside] + LIM) / (2 * LIM) * BINS).long().clamp_(0, BINS - 1)
                hist += torch.bincount((ui * BINS + gi).cpu(), minlength=BINS * BINS).double()
            n_tokens += ids.shape[1] - n_prompt
            print(f"problem {prob['idx']}: {ids.shape[1] - n_prompt} thinking tokens", flush=True)
    for h in handles:
        h.remove()
    # act_fn on a grid, for panel b (so the plot uses the model's own activation function)
    grid = torch.linspace(-LIM, LIM, 481)
    act = mlps[0].act_fn(grid).numpy()
    np.savez_compressed(cache, hist=hist.view(BINS, BINS).numpy(), counts=counts.numpy(), mass=mass.numpy(),
                        outside=outside, n_tokens=n_tokens, sample_g=torch.cat(sample_g).numpy(), sample_u=torch.cat(sample_u).numpy(), grid=grid.numpy(), act=act,
                        act_name=type(mlps[0].act_fn).__name__, n_traces=len(problems))
    del model

d = np.load(cache, allow_pickle=True)
hist, counts, mass = d["hist"], d["counts"], d["mass"]
total = counts.sum()
share_f = counts.sum(0) / total
share_a = mass.sum(0) / mass.sum()
summary = {"model": args.model, "text": f"intact thinking traces, first {int(d['n_traces'])} GSM8K problems, seed 1",
           "thinking_tokens": int(d["n_tokens"]), "firings": float(total), "activation_fn": str(d["act_name"]),
           "share_outside_plot_range": float(d["outside"]) / float(total),
           **{f"{q}_share_of_firings": float(share_f[k]) for k, q in enumerate(QUADS)},
           **{f"{q}_share_of_abs_activation": float(share_a[k]) for k, q in enumerate(QUADS)},
           **{f"{q}_mean_abs_activation": float(mass.sum(0)[k] / counts.sum(0)[k]) for k, q in enumerate(QUADS)},
           "per_layer_share_of_firings": {q: [float(x) for x in counts[:, k] / counts.sum(1)] for k, q in enumerate(QUADS)}}
(folder / "quadrant_map_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "per_layer_share_of_firings"}, indent=1))

# ---------------------------------------------------------------- plot
style.apply()
VIEW = 10.0  # shows ~99% of firings; +-4 showed only 80%
density = LinearSegmentedColormap.from_list("ink", [style.SURFACE, "#cfcbe6", "#7f74c9", "#3a2f8f", "#17123f"])
diverge = LinearSegmentedColormap.from_list("leak", ["#7a2f0d", style.Q4_COL, "#f3ede6", style.Q3_COL, "#0f3a73"])
# where each quadrant's label sits: (x, y, ha, va), with gate on x and up on y
PAD = 0.04 * VIEW
CORNER = {"q1": (VIEW - PAD, VIEW - PAD, "right", "top"), "q2": (VIEW - PAD, -VIEW + PAD, "right", "bottom"),
          "q3": (-VIEW + PAD, -VIEW + PAD, "left", "bottom"), "q4": (-VIEW + PAD, VIEW - PAD, "left", "top")}
NOTE = {"q1": "gate open, large +", "q2": "gate open, large −", "q3": "gate closed, small +",
        "q4": "gate closed, small −"}

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16.2, 6.3), gridspec_kw={"width_ratios": [1, 1, 0.78]})

cols = {"q1": "#3a2f8f", "q2": "#8f8ab8", "q3": style.Q3_COL, "q4": style.Q4_COL}
sg, su = d["sample_g"], d["sample_u"]
quad_of = np.where(sg > 0, np.where(su > 0, 0, 1), np.where(su < 0, 2, 3))
order = np.random.default_rng(0).permutation(len(sg))  # so no quadrant is drawn on top of another
a1.scatter(sg[order], su[order], s=4, c=[cols[QUADS[k]] for k in quad_of[order]], alpha=0.55, linewidths=0, zorder=2,
           rasterized=True)
a1.set_aspect("equal")
a1.set_title(f"a  {len(sg):,} randomly sampled firings")
for k, q in enumerate(QUADS):
    x, y, ha, va = CORNER[q]
    a1.text(x, y, f"{q.upper()}\n{share_f[k]:.0%} of firings", ha=ha, va=va, fontsize=10.5, color=cols[q],
            fontweight="bold", linespacing=1.35, zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc=style.SURFACE, ec="none", alpha=0.9))

assert "SiLU" in str(d["act_name"])
so = sg / (1 + np.exp(-sg)) * su  # each sampled firing's output, SiLU(gate) * up
SIZE_PER_UNIT, SIZE_MIN, OUT_CAP = 8.0, 0.5, 12.0  # dot AREA is proportional to |output|, capped at OUT_CAP


def dot_size(v):
    return SIZE_MIN + SIZE_PER_UNIT * np.minimum(np.abs(v), OUT_CAP)


big_first = np.argsort(-np.abs(so))  # draw large dots first so the small leaks stay visible on top
a2.scatter(sg[big_first], su[big_first], s=dot_size(so[big_first]),
           c=np.where(so[big_first] > 0, style.Q3_COL, style.Q4_COL), alpha=0.45, linewidths=0, zorder=2, rasterized=True)
a2.set_aspect("equal")
a2.set_title("b  The same firings, sized by output")
for q in QUADS:
    x, y, ha, va = CORNER[q]
    a2.text(x, y, f"{q.upper()}\n{NOTE[q]}", ha=ha, va=va, fontsize=9.6, color=style.INK, linespacing=1.35, zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc=style.SURFACE, ec="none", alpha=0.9))
handles = [plt.scatter([], [], s=dot_size(v), color=style.INK2, alpha=0.6, linewidths=0, label=f"{v:g}") for v in (0.15, 1.5, 10)]
handles += [plt.scatter([], [], s=40, color=c, linewidths=0, label=l)
            for c, l in ((style.Q3_COL, "positive output"), (style.Q4_COL, "negative output"))]
a2.legend(handles=handles, title="|output| = |SiLU(gate) × up|", loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=5, columnspacing=1.3, handletextpad=0.4, title_fontsize=9)
for ax in (a1, a2):
    ax.axhline(0, color=style.INK, lw=0.9); ax.axvline(0, color=style.INK, lw=0.9)
    ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW, VIEW); ax.grid(False)
    ax.set_xlabel("gate pre-activation"); ax.set_ylabel("up pre-activation")
    ax.set_xticks(range(-10, 11, 5)); ax.set_yticks(range(-10, 11, 5))

y = np.arange(4)[::-1]
for k, q in enumerate(QUADS):
    a3.barh(y[k] + 0.19, share_f[k], height=0.34, color=cols[q])
    a3.barh(y[k] - 0.19, share_a[k], height=0.34, color=cols[q], alpha=0.45)
    a3.text(share_f[k] + 0.008, y[k] + 0.19, f"{share_f[k]:.0%} of firings", va="center", fontsize=9.5, color=style.INK)
    a3.text(share_a[k] + 0.008, y[k] - 0.19, f"{share_a[k]:.0%} of activation", va="center", fontsize=9.5, color=style.INK2)
a3.set_yticks(y); a3.set_yticklabels([q.upper() for q in QUADS], fontsize=11)
a3.set_xlim(0, 0.62); a3.set_xticks([0, 0.2, 0.4]); a3.set_xticklabels(["0%", "20%", "40%"])
a3.grid(axis="y", visible=False)
a3.set_title("c  Q3 and Q4: most firings, little activation")
a3.set_xlabel("share of the total")

fig.suptitle(f"{args.model.split('/')[-1]}: every MLP neuron on every token of the intact model's own GSM8K reasoning "
             f"({int(d['n_traces'])} problems, {int(d['n_tokens']):,} thinking tokens, all {counts.shape[0]} layers, "
             f"{total / 1e9:.1f} billion firings). Activation share uses |output|.",
             fontsize=9.5, color=style.INK2, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=(0, 0.07, 1, 0.955), w_pad=2.5)
out_path = folder / "figures" / "fig9_quadrant_map.png"
fig.savefig(out_path, dpi=160)
print("wrote", out_path)
