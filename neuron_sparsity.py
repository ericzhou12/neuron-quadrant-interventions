"""Core tools for neuron-level sparsification experiments on gated-MLP transformers.

Works with any Hugging Face causal LM whose MLP blocks look like

    out = down_proj( act_fn(gate_proj(x)) * up_proj(x) )

(Qwen, Llama, Gemma, Mistral...). The "neuron activation" is the product
act_fn(gate) * up, i.e. exactly the input to `down_proj`. We intervene there
with a forward pre-hook, so no model code is modified.

Experiment: for every token and every layer, KEEP a subset of neurons and
replace the rest with their dataset-mean activation (mean ablation). Then
measure how much the model's next-token predictions change.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Finding the MLP blocks
# ---------------------------------------------------------------------------


def get_mlps(model) -> list[torch.nn.Module]:
    """Return every gated MLP block (one per layer), in layer order."""
    mlps = [
        m
        for m in model.modules()
        if all(hasattr(m, a) for a in ("gate_proj", "up_proj", "down_proj"))
    ]
    if not mlps:
        raise ValueError(
            "No gated MLP blocks found. This code expects modules with "
            "gate_proj / up_proj / down_proj (Qwen, Llama, Gemma, Mistral)."
        )
    return mlps


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Step 1: per-neuron mean activations (what we ablate *to*)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_neuron_means(model, batches: list[torch.Tensor]) -> list[torch.Tensor]:
    """Mean activation of every neuron over a calibration set.

    Returns one tensor of shape [d_mlp] per layer.
    """
    mlps = get_mlps(model)
    sums = [None] * len(mlps)
    count = 0

    def make_hook(i):
        def hook(_module, args):
            acts = args[0]  # [batch, seq, d_mlp]
            s = acts.float().sum(dim=(0, 1))
            sums[i] = s if sums[i] is None else sums[i] + s

        return hook

    handles = [m.down_proj.register_forward_pre_hook(make_hook(i)) for i, m in enumerate(mlps)]
    try:
        for ids in batches:
            model(ids.to(model.device))
            count += ids.numel()
    finally:
        for h in handles:
            h.remove()
    return [s / count for s in sums]


# ---------------------------------------------------------------------------
# Step 2: the ablation itself
# ---------------------------------------------------------------------------


@dataclass
class AblationSpec:
    """Which neurons to keep at each (token, layer).

    mode:
      "top"    keep the k neurons that deviate MOST from their mean  (the real test)
      "bottom" keep the k neurons that deviate LEAST                 (control)
      "random" keep k random neurons                                 (control)
    keep_frac: fraction of the layer's neurons to keep, in (0, 1].
    """

    mode: str = "top"
    keep_frac: float = 0.1


def _keep_mask(acts: torch.Tensor, mean: torch.Tensor, spec: AblationSpec) -> torch.Tensor:
    d_mlp = acts.shape[-1]
    k = max(1, round(spec.keep_frac * d_mlp))
    if k >= d_mlp:
        return torch.ones_like(acts, dtype=torch.bool)

    # Score = how far the neuron is from its typical value on this token.
    # NOTE: this ranking rule is a design choice. Alternatives worth trying:
    # raw |activation|, or a per-neuron percentile (closer to the Goodfire post).
    if spec.mode == "random":
        score = torch.rand(acts.shape, device=acts.device)
    else:
        score = (acts.float() - mean).abs()
        if spec.mode == "bottom":
            score = -score
        elif spec.mode != "top":
            raise ValueError(f"unknown mode {spec.mode!r}")

    idx = score.topk(k, dim=-1).indices
    mask = torch.zeros_like(acts, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


@contextmanager
def ablate_neurons(model, means: list[torch.Tensor], spec: AblationSpec):
    """Context manager: inside the `with` block the model runs sparsified."""
    mlps = get_mlps(model)

    def make_hook(i):
        def hook(_module, args):
            acts = args[0]
            mean = means[i].to(acts.device)
            mask = _keep_mask(acts, mean, spec)
            new = torch.where(mask, acts, mean.to(acts.dtype))
            return (new,) + tuple(args[1:])

        return hook

    handles = [m.down_proj.register_forward_pre_hook(make_hook(i)) for i, m in enumerate(mlps)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Step 3: measuring the damage
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, batches: list[torch.Tensor], means=None, spec: AblationSpec | None = None) -> dict:
    """Compare the (optionally ablated) model against the intact model.

    Returns average next-token cross-entropy loss, and KL(intact || ablated)
    averaged over token positions. With spec=None, returns the intact baseline.
    Also returns per-batch averages ("batch_loss", "batch_kl"), used for
    bootstrap error bars.
    """
    total_loss = 0.0
    total_kl = 0.0
    n_tokens = 0
    batch_loss: list[float] = []
    batch_kl: list[float] = []

    for ids in batches:
        ids = ids.to(model.device)
        clean_logits = model(ids).logits.float()

        if spec is None:
            logits = clean_logits
        else:
            with ablate_neurons(model, means, spec):
                logits = model(ids).logits.float()

        # Predict token t+1 from position t.
        pred = logits[:, :-1].reshape(-1, logits.shape[-1])
        target = ids[:, 1:].reshape(-1)
        loss_sum = F.cross_entropy(pred, target, reduction="sum").item()

        clean_logp = F.log_softmax(clean_logits[:, :-1], dim=-1)
        abl_logp = F.log_softmax(logits[:, :-1], dim=-1)
        kl_sum = (clean_logp.exp() * (clean_logp - abl_logp)).sum(-1).sum().item()

        total_loss += loss_sum
        total_kl += kl_sum
        n_tokens += target.numel()
        batch_loss.append(loss_sum / target.numel())
        batch_kl.append(kl_sum / target.numel())

    return {"loss": total_loss / n_tokens, "kl": total_kl / n_tokens, "tokens": n_tokens,
            "batch_loss": batch_loss, "batch_kl": batch_kl}


# ---------------------------------------------------------------------------
# Step 4: error bars
# ---------------------------------------------------------------------------


def bootstrap_ci(values: list[float], baseline: list[float] | None = None,
                 n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Bootstrap confidence interval over evaluation batches.

    Resamples batches with replacement and recomputes the average each time.
    With `baseline` (per-batch intact loss), it returns the interval for the
    relative increase mean(values)/mean(baseline) - 1, resampling the same
    batches for both so the comparison stays paired.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.tensor(values, dtype=torch.float64)
    idx = torch.randint(0, len(v), (n_boot, len(v)), generator=g)
    stat = v[idx].mean(1)
    if baseline is not None:
        b = torch.tensor(baseline, dtype=torch.float64)
        stat = stat / b[idx].mean(1) - 1
    alpha = (1 - level) / 2
    lo, hi = torch.quantile(stat, torch.tensor([alpha, 1 - alpha], dtype=torch.float64)).tolist()
    return lo, hi
