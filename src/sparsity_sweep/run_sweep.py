"""Experiment 1: how many neurons can you ablate before the model breaks?

Usage (start small to check everything runs):

    python src/sparsity_sweep/run_sweep.py --model Qwen/Qwen3-0.6B --eval-batches 8

Then the real run:

    python src/sparsity_sweep/run_sweep.py --model Qwen/Qwen3-1.7B

Writes results/<model-name>.csv with one row per (mode, keep_frac).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys as _sys, pathlib as _pl  # make the sibling src/ folders importable
_sys.path[:0] = [str(_p) for _p in _pl.Path(__file__).resolve().parents[1].iterdir() if _p.is_dir()]
from neuron_sparsity import (AblationSpec, bootstrap_ci, compute_neuron_means, evaluate, get_mlps,
                             pick_device)


def make_batches(tokenizer, split: str, n_batches: int, batch_size: int, seq_len: int) -> list[torch.Tensor]:
    """Tokenize WikiText into fixed-length chunks: a list of [batch, seq_len] tensors."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    need = n_batches * batch_size * seq_len
    ids: list[int] = []
    for row in ds:
        text = row["text"]
        if len(text.strip()) < 50:  # skip headings and blank lines
            continue
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(ids) >= need:
            break
    if len(ids) < need:
        raise ValueError(f"Only got {len(ids)} tokens from split {split!r}, needed {need}.")
    t = torch.tensor(ids[:need]).view(n_batches, batch_size, seq_len)
    return [t[i] for i in range(n_batches)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--keep-fracs", default="1.0,0.5,0.25,0.125,0.08,0.04,0.02,0.01",
                   help="Fractions of neurons to KEEP per token per layer. 0.08 is roughly the Goodfire top-320/4096 setting.")
    p.add_argument("--modes", default="top,random,bottom")
    p.add_argument("--calib-batches", type=int, default=32, help="Batches for computing neuron means.")
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                   help="float32 is the safest on Apple Silicon; use bfloat16 if memory is tight.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(device).eval()

    mlps = get_mlps(model)
    d_mlp = mlps[0].down_proj.in_features
    print(f"{len(mlps)} layers x {d_mlp} neurons = {len(mlps) * d_mlp:,} neurons")

    # Calibration and evaluation text come from different splits, so the
    # means are never computed on the text we evaluate on.
    calib = make_batches(tokenizer, "train", args.calib_batches, args.batch_size, args.seq_len)
    evalb = make_batches(tokenizer, "test", args.eval_batches, args.batch_size, args.seq_len)

    print("computing neuron means...")
    means = compute_neuron_means(model, calib)

    base = evaluate(model, evalb)
    print(f"intact model: loss {base['loss']:.4f} over {base['tokens']:,} tokens")

    out = Path("results") / f"{args.model.replace('/', '__')}.csv"
    out_batches = out.with_suffix(".batches.csv")  # per-batch values, for re-analysis
    out.parent.mkdir(exist_ok=True)
    rows = []
    batch_rows = [{"model": args.model, "mode": "intact", "keep_frac": 1.0, "batch": i,
                   "loss": round(l, 6), "kl": 0.0} for i, l in enumerate(base["batch_loss"])]
    for mode in args.modes.split(","):
        for frac in [float(x) for x in args.keep_fracs.split(",")]:
            t0 = time.time()
            r = evaluate(model, evalb, means, AblationSpec(mode=mode, keep_frac=frac))
            kl_lo, kl_hi = bootstrap_ci(r["batch_kl"])
            li_lo, li_hi = bootstrap_ci(r["batch_loss"], baseline=base["batch_loss"])
            row = {
                "model": args.model,
                "mode": mode,
                "keep_frac": frac,
                "ablated_pct": round(100 * (1 - frac), 2),
                "loss": round(r["loss"], 5),
                "loss_increase_pct": round(100 * (r["loss"] / base["loss"] - 1), 2),
                "loss_increase_pct_lo": round(100 * li_lo, 2),
                "loss_increase_pct_hi": round(100 * li_hi, 2),
                "kl": round(r["kl"], 5),
                "kl_lo": round(kl_lo, 5),
                "kl_hi": round(kl_hi, 5),
                "baseline_loss": round(base["loss"], 5),
            }
            rows.append(row)
            batch_rows += [{"model": args.model, "mode": mode, "keep_frac": frac, "batch": i,
                            "loss": round(l, 6), "kl": round(k, 6)}
                           for i, (l, k) in enumerate(zip(r["batch_loss"], r["batch_kl"]))]
            print(f"{mode:>6} keep {frac:<6} loss {row['loss']:.4f} "
                  f"({row['loss_increase_pct']:+}%)  KL {row['kl']:.4f} "
                  f"[95% CI {row['kl_lo']:.4f}-{row['kl_hi']:.4f}]  [{time.time() - t0:.0f}s]")

            # Write after every setting so a crash never loses finished work.
            for path, data in ((out, rows), (out_batches, batch_rows)):
                with path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(data[0]))
                    w.writeheader()
                    w.writerows(data)

    print(f"\nsaved {out} and {out_batches}")


if __name__ == "__main__":
    main()
