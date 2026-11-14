"""Offline sanity check on a tiny random model. No downloads needed.

    python tests/test_smoke.py

Checks the properties the real experiment relies on:
  1. keeping 100% of neurons changes nothing (KL = 0)
  2. hooks are removed afterwards (model is intact again)
  3. the number of neurons kept per token is exactly k
  4. ablating more neurons does more damage
  5. bootstrap intervals contain the estimate and per-batch values average to it
"""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

import sys as _sys, pathlib as _pl  # make the src/ folders importable
_sys.path[:0] = [str(_p) for _p in (_pl.Path(__file__).resolve().parents[1] / "src").iterdir() if _p.is_dir()]
from neuron_sparsity import AblationSpec, _keep_mask, bootstrap_ci, compute_neuron_means, evaluate, get_mlps

torch.manual_seed(0)
config = LlamaConfig(vocab_size=500, hidden_size=64, intermediate_size=256,
                     num_hidden_layers=3, num_attention_heads=4, max_position_embeddings=64)
model = LlamaForCausalLM(config).eval()
batches = [torch.randint(0, 500, (4, 32)) for _ in range(4)]

assert len(get_mlps(model)) == 3
means = compute_neuron_means(model, batches)
assert means[0].shape == (256,)

base = evaluate(model, batches)
assert base["kl"] == 0.0

full = evaluate(model, batches, means, AblationSpec("top", 1.0))
assert abs(full["kl"]) < 1e-6, full

again = evaluate(model, batches)
assert abs(again["loss"] - base["loss"]) < 1e-6, "hooks were not removed"

acts = torch.randn(2, 5, 256)
for mode in ("top", "bottom", "random"):
    mask = _keep_mask(acts, means[0], AblationSpec(mode, 0.25))
    assert (mask.sum(-1) == 64).all(), mode

kls = [evaluate(model, batches, means, AblationSpec("top", f))["kl"] for f in (0.5, 0.1, 0.02)]
assert kls[0] < kls[1] < kls[2], kls

top = evaluate(model, batches, means, AblationSpec("top", 0.1))["kl"]
bottom = evaluate(model, batches, means, AblationSpec("bottom", 0.1))["kl"]
assert top < bottom, (top, bottom)

r = evaluate(model, batches, means, AblationSpec("top", 0.1))
assert abs(sum(r["batch_kl"]) / len(r["batch_kl"]) - r["kl"]) < 1e-6
lo, hi = bootstrap_ci(r["batch_kl"])
assert lo <= r["kl"] <= hi, (lo, r["kl"], hi)
li_lo, li_hi = bootstrap_ci(r["batch_loss"], baseline=base["batch_loss"])
inc = r["loss"] / base["loss"] - 1
assert li_lo <= inc <= li_hi, (li_lo, inc, li_hi)

print("all checks passed")
print(f"  KL keeping 50% / 10% / 2% (top): {kls[0]:.4f} / {kls[1]:.4f} / {kls[2]:.4f}")
print(f"  KL keeping 10%: top {top:.4f} vs bottom {bottom:.4f}")
print(f"  KL keeping 10% (top): {r['kl']:.4f}, 95% CI {lo:.4f}-{hi:.4f}")
