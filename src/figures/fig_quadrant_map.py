"""Map of neuron firings in the (gate, up) plane for Qwen3-1.7B, in the style of
the quadrant figure in Goodfire's "Neurons as an Interpretable Basis".

Every MLP neuron on every token is one firing with two pre-activations: gate
and up. Its output is act_fn(gate) * up. The signs of gate and up put the
firing in one of four quadrants (Q1..Q4).

Panel a: where the firings are (2-D histogram over all layers and neurons).
Panel b: the output each point of the plane produces, on a scale that keeps
         the small leaks visible.
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
args = p.parse_args()
folder = Path(args.folder)
cache = folder / "quadrant_map.npz"
LIM, BINS = 6.0, 240  # histogram covers gate, up in [-LIM, LIM]
QUADS = ["q1", "q2", "q3", "q4"]

if not (args.replot and cache.exists()):
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
                        outside=outside, n_tokens=n_tokens, grid=grid.numpy(), act=act,
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
VIEW = 4.0
density = LinearSegmentedColormap.from_list("ink", [style.SURFACE, "#cfcbe6", "#7f74c9", "#3a2f8f", "#17123f"])
diverge = LinearSegmentedColormap.from_list("leak", ["#7a2f0d", style.Q4_COL, "#f3ede6", style.Q3_COL, "#0f3a73"])
# where each quadrant's label sits: (x, y, ha, va), with gate on x and up on y
CORNER = {"q1": (VIEW - 0.15, VIEW - 0.15, "right", "top"), "q2": (VIEW - 0.15, -VIEW + 0.15, "right", "bottom"),
          "q3": (-VIEW + 0.15, -VIEW + 0.15, "left", "bottom"), "q4": (-VIEW + 0.15, VIEW - 0.15, "left", "top")}
NOTE = {"q1": "gate open, large +", "q2": "gate open, large −", "q3": "gate closed, small + leak",
        "q4": "gate closed, small − leak"}

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16.2, 5.9), gridspec_kw={"width_ratios": [1, 1, 0.78]})

frac = hist / total
im = a1.imshow(np.where(frac > 0, frac, np.nan), origin="lower", extent=(-LIM, LIM, -LIM, LIM), cmap=density,
               norm=LogNorm(vmin=frac[frac > 0].max() / 3000, vmax=frac.max()), interpolation="nearest", aspect="equal")
cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
cb.set_label("share of all firings per cell (log scale)"); cb.outline.set_visible(False)
a1.set_title("a  Where firings land")
for k, q in enumerate(QUADS):
    x, y, ha, va = CORNER[q]
    a1.text(x, y, f"{q.upper()}\n{share_f[k]:.0%} of firings", ha=ha, va=va, fontsize=10.5, color=style.INK,
            fontweight="bold", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", fc=style.SURFACE, ec="none", alpha=0.85))

G, U = np.meshgrid(d["grid"], d["grid"])
out = np.interp(G, d["grid"], d["act"]) * U
im2 = a2.imshow(out, origin="lower", extent=(-LIM, LIM, -LIM, LIM), cmap=diverge,
                norm=SymLogNorm(linthresh=0.5, vmin=-12, vmax=12), aspect="equal")
cb2 = fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.03, ticks=[-10, -3, -1, -0.3, 0, 0.3, 1, 3, 10])
cb2.ax.set_yticklabels(["−10", "−3", "−1", "−0.3", "0", "+0.3", "+1", "+3", "+10"])
cb2.set_label("neuron output = SiLU(gate) × up"); cb2.outline.set_visible(False)
a2.set_title("b  What each firing outputs")
for q in QUADS:
    x, y, ha, va = CORNER[q]
    a2.text(x, y, f"{q.upper()}\n{NOTE[q]}", ha=ha, va=va, fontsize=9.6, color=style.INK, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", fc=style.SURFACE, ec="none", alpha=0.85))
for ax in (a1, a2):
    ax.axhline(0, color=style.INK, lw=0.9); ax.axvline(0, color=style.INK, lw=0.9)
    ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW, VIEW); ax.grid(False)
    ax.set_xlabel("gate pre-activation"); ax.set_ylabel("up pre-activation")
    ax.set_xticks(range(-4, 5, 2)); ax.set_yticks(range(-4, 5, 2))

cols = {"q1": "#3a2f8f", "q2": "#8f8ab8", "q3": style.Q3_COL, "q4": style.Q4_COL}
y = np.arange(4)[::-1]
for k, q in enumerate(QUADS):
    a3.barh(y[k] + 0.19, share_f[k], height=0.34, color=cols[q])
    a3.barh(y[k] - 0.19, share_a[k], height=0.34, color=cols[q], alpha=0.45)
    a3.text(share_f[k] + 0.008, y[k] + 0.19, f"{share_f[k]:.0%} of firings", va="center", fontsize=9.5, color=style.INK)
    a3.text(share_a[k] + 0.008, y[k] - 0.19, f"{share_a[k]:.0%} of activation", va="center", fontsize=9.5, color=style.INK2)
a3.set_yticks(y); a3.set_yticklabels([q.upper() for q in QUADS], fontsize=11)
a3.set_xlim(0, 0.62); a3.set_xticks([0, 0.2, 0.4]); a3.set_xticklabels(["0%", "20%", "40%"])
a3.grid(axis="y", visible=False)
a3.set_title("c  Leaks: most firings, little activation")
a3.set_xlabel("share of the total")

fig.suptitle(f"{args.model.split('/')[-1]}: every MLP neuron on every token of the intact model's own GSM8K reasoning "
             f"({int(d['n_traces'])} problems, {int(d['n_tokens']):,} thinking tokens, all {counts.shape[0]} layers, "
             f"{total / 1e9:.1f} billion firings). Activation share uses |output|.",
             fontsize=9.5, color=style.INK2, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=2.5)
out_path = folder / "figures" / "fig9_quadrant_map.png"
fig.savefig(out_path, dpi=160)
print("wrote", out_path)
