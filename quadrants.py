"""Quadrant-level interventions on gated-MLP neurons.

Each neuron computes  act = act_fn(g) * u,  where
    g = gate_proj(x)   (gate pre-activation)
    u = up_proj(x)     (up pre-activation)

Goodfire label each firing by the signs of g and u:

    Q1: g > 0, u > 0  -> gate open,   large positive output
    Q2: g > 0, u < 0  -> gate open,   large negative output
    Q3: g < 0, u < 0  -> gate closed, small positive output (SiLU leak)
    Q4: g < 0, u > 0  -> gate closed, small negative output (SiLU leak)

We capture g and u with forward hooks on gate_proj and up_proj, then edit the
activation right before down_proj. The model's code is never modified.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from neuron_sparsity import get_mlps

QUADRANTS = ("q1", "q2", "q3", "q4")
SAMPLE_QUADS = ("q3", "q4")  # quadrants we keep raw values for (resample / pool fills)

# Condition names are "<quadrant>_<fill>", optionally with "@<dose>": the share of
# that quadrant's firings to edit, chosen at random per firing. Examples:
#   q3_qmean        every Q3 firing replaced by that neuron's typical Q3 value
#   q3_zero@0.1     a random 10% of Q3 firings set to zero
# Fills:
#   zero      set to 0
#   mean      the neuron's mean over ALL tokens (Goodfire-style; injects large values)
#   qmean     the neuron's mean over its firings IN THAT QUADRANT ("typical value")
# Put-back fills, to tell the hypotheses apart (see CLAUDE.md):
#   resample  the same neuron's value at a random other firing in that quadrant
#             (calibration text). Keeps each neuron's size distribution,
#             destroys which token gets which value.
#   pool      a value from a random firing of ANY neuron in that quadrant and
#             layer. Keeps the overall size distribution only (H1: is it just noise?).
#   const     remove the quadrant's firings and add the neuron's average
#             contribution from that quadrant (share of tokens x typical value)
#             to EVERY token. Keeps the average push exactly and nothing else,
#             not even which tokens fired (H3: is it a steady bias?).
FILLS = {
    "zero": "set to zero",
    "mean": "replaced by the neuron's mean over all tokens (Goodfire-style)",
    "qmean": "replaced by the neuron's typical value in that quadrant",
    "resample": "replaced by the same neuron's value from a random other firing in that quadrant",
    "pool": "replaced by a random value from any neuron's firings in that quadrant (same layer)",
    "const": "removed, and the neuron's average contribution from them added to every token",
    # Gain knobs: here @p is NOT a share of firings. Every firing in the quadrant is
    # multiplied by (1 - p) for shrink, (1 + p) for grow. shrink@0.25 removes the
    # same average amount as zero@0.25 but without the random per-firing noise.
    "shrink": "all multiplied by (1 - dose)",
    "grow": "all multiplied by (1 + dose)",
}
GAIN_FILLS = ("shrink", "grow")
SPECIAL = {
    "none": "intact model (baseline)",
    "random_matched": "same number of random non-Q3 firings per token set to zero. HARSH: also hits "
                      "large gate-open firings, so it is not size-matched; prefer q4_* as the control",
    "q3_boost": "Q3 firings multiplied by --boost",
}

# Default stage 0: nothing here broke Qwen3-1.7B in the thinking-off runs.
STAGE0 = ["none", "q3_qmean", "q4_qmean"]


def split_layers(condition: str) -> tuple[str, str | None]:
    """"q3_zero@0.25#late" -> ("q3_zero@0.25", "late"). No "#": (condition, None)."""
    cond, _, spec = condition.partition("#")
    return cond, (spec or None)


def layer_set(spec: str | None, n_layers: int) -> set[int] | None:
    """Which layers a "#spec" restricts the edit to: early/mid/late thirds, or "a-b" (inclusive)."""
    if spec is None:
        return None
    thirds = {"early": 0, "mid": 1, "late": 2}
    if spec in thirds:
        return {i for i in range(n_layers) if min(2, 3 * i // n_layers) == thirds[spec]}
    try:
        a, _, b = spec.partition("-")
        lo, hi = int(a), int(b or a)
    except ValueError:
        raise ValueError(f"bad layer spec {spec!r}: use early, mid, late or a-b") from None
    return set(range(lo, hi + 1))


def parse_condition(condition: str) -> tuple[str, str | None, float]:
    """Return (base, quadrant or None, dose). Raises ValueError if malformed.

    An optional "#layers" suffix (see layer_set) restricts the edit to some layers.
    """
    condition, spec = split_layers(condition)
    layer_set(spec, 3)  # validates the spec
    base, _, dose_s = condition.partition("@")
    dose = 1.0
    if dose_s:
        dose = float(dose_s)
        if not 0 < dose <= 1:
            raise ValueError(f"dose must be in (0, 1]: {condition!r}")
    if base in SPECIAL:
        if base == "none" and dose_s:
            raise ValueError("'none' takes no dose")
        return base, None, dose
    quad, _, fill = base.partition("_")
    if fill in ("resample", "pool") and quad not in SAMPLE_QUADS:
        raise ValueError(f"{fill} is only available for {SAMPLE_QUADS}: {condition!r}")
    if quad not in QUADRANTS or fill not in FILLS:
        raise ValueError(f"unknown condition {condition!r}: use none, random_matched, q3_boost, "
                         f"or <q1-q4>_<{'/'.join(FILLS)}>[@dose], e.g. q3_qmean or q3_zero@0.1")
    return base, quad, dose


def describe(condition: str) -> str:
    condition, spec = split_layers(condition)
    if spec:
        return f"{describe(condition)} [layers: {spec} only]"
    base, quad, dose = parse_condition(condition)
    if base in SPECIAL:
        text = SPECIAL[base]
    else:
        fill = base.split("_")[1]
        if fill in GAIN_FILLS:
            factor = 1 - dose if fill == "shrink" else 1 + dose
            return f"every {quad.upper()} firing multiplied by {factor:g}"
        text = f"{quad.upper()} firings {FILLS[fill]}"
    return text if dose == 1 else f"{text} (random {dose:.0%} of them)"


# Kept for the command-line help and tests.
CONDITIONS = {**SPECIAL, **{f"{q}_{f}": describe(f"{q}_{f}") for q in QUADRANTS for f in FILLS
                            if f not in ("resample", "pool") or q in SAMPLE_QUADS}}


def quadrant_masks(g: torch.Tensor, u: torch.Tensor) -> dict[str, torch.Tensor]:
    gate_open = g > 0
    up_pos = u > 0
    return {
        "q1": gate_open & up_pos,
        "q2": gate_open & ~up_pos,
        "q3": ~gate_open & ~up_pos,
        "q4": ~gate_open & up_pos,
    }


def _random_matched_mask(q3: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Per token, pick as many random non-Q3 neurons as there are Q3 neurons."""
    # Random on CPU with our own generator, so we don't disturb the sampling RNG.
    score = torch.rand(q3.shape, generator=generator).to(q3.device)
    score = score.masked_fill(q3, -1.0)  # Q3 positions are never picked
    n = q3.sum(-1, keepdim=True)  # how many to pick, per token
    rank = score.argsort(-1, descending=True).argsort(-1)  # 0 = highest score
    return rank < n


def _thin(mask: torch.Tensor, dose: float, generator: torch.Generator) -> torch.Tensor:
    """Keep a random `dose` share of the True entries."""
    if dose >= 1:
        return mask
    keep = torch.rand(mask.shape, generator=generator).to(mask.device) < dose
    return mask & keep


def apply_condition(acts, g, u, condition, fills=None, boost=2.0, generator=None):
    """Return (edited activations, mask of entries that were changed).

    fills: per-neuron vectors for this layer: "all" (mean over all tokens) and
    "q1".."q4" (mean over that quadrant's firings), as the condition needs.
    """
    base, quad, dose = parse_condition(condition)
    if base == "none":
        return acts, torch.zeros_like(acts, dtype=torch.bool)
    if dose < 1 and generator is None:
        generator = torch.Generator().manual_seed(0)

    masks = quadrant_masks(g, u)

    if base == "q3_boost":
        mask = _thin(masks["q3"], dose, generator)
        return torch.where(mask, acts * boost, acts), mask

    if base == "random_matched":
        mask = _thin(_random_matched_mask(masks["q3"], generator), dose, generator)
        return torch.where(mask, torch.zeros_like(acts), acts), mask

    fill_kind = base.split("_")[1]
    if fill_kind in GAIN_FILLS:
        mask = masks[quad]
        factor = 1 - dose if fill_kind == "shrink" else 1 + dose
        return torch.where(mask, acts * factor, acts), mask
    mask = _thin(masks[quad], dose, generator)
    if fill_kind == "const":
        # Zero the firings, then add back their average contribution everywhere.
        # With a dose p, a share p of firings is removed, so p x the average is added.
        bias = (dose * fills[f"{quad}_contrib"]).to(device=acts.device, dtype=acts.dtype)
        return torch.where(mask, torch.zeros_like(acts), acts) + bias, mask
    if fill_kind in ("resample", "pool"):
        r = torch.rand(acts.shape, generator=generator).to(acts.device)
        if fill_kind == "resample":
            vals, counts = fills[f"{quad}_samples"], fills[f"{quad}_counts"]  # [S, d], [d]
            vals, counts = vals.to(acts.device), counts.to(acts.device)
            idx = (r * counts).long().clamp(max=vals.shape[0] - 1).reshape(-1, acts.shape[-1])
            fill = vals.gather(0, idx).reshape(acts.shape)
            fill = torch.where(counts > 0, fill, torch.zeros_like(fill))  # never seen firing: 0
        else:
            pool = fills[f"{quad}_pool"].to(acts.device)
            fill = pool[(r * len(pool)).long().clamp(max=len(pool) - 1)]
        return torch.where(mask, fill.to(acts.dtype), acts), mask
    if fill_kind == "zero":
        fill = torch.zeros_like(acts)
    else:
        vec = fills["all" if fill_kind == "mean" else quad]
        fill = vec.to(device=acts.device, dtype=acts.dtype).expand_as(acts)
    return torch.where(mask, fill, acts), mask


def needed_fills(condition: str) -> set[str]:
    base, quad, _ = parse_condition(condition)
    if base.endswith("_qmean"):
        return {quad}
    for kind, keys in (("resample", ("samples", "counts")), ("pool", ("pool",)), ("const", ("contrib",))):
        if base.endswith("_" + kind):
            return {f"{quad}_{k}" for k in keys}
    if base.endswith("_mean"):
        return {"all"}
    return set()


class InterventionStats:
    """Counts how many neuron firings an intervention touched."""

    def __init__(self):
        self.changed = 0
        self.total = 0
        self._pending = []

    def add(self, mask):
        self._pending.append(mask.sum())  # stays on GPU; no sync per layer
        self.total += mask.numel()

    def fraction(self) -> float:
        if self._pending:
            self.changed += int(torch.stack(self._pending).sum().item())
            self._pending = []
        return self.changed / self.total if self.total else 0.0


@contextmanager
def intervene(model, condition="none", fills=None, boost=2.0, seed=0, gen_only=False,
              prefill_gen_mask=None, verify=False):
    """Inside the `with` block, every MLP layer runs with the intervention.

    fills: list (one per layer) of dicts {"all": vec, "q1": vec, ..., "q4": vec}.
    gen_only: don't intervene on the prompt. Single-token passes (decoding
        steps) are always edited. Multi-token passes are left untouched,
        unless prefill_gen_mask is given.
    prefill_gen_mask: bool tensor [batch, seq]; True marks generated tokens
        inside a multi-token pass. Used when a forced-continuation round
        re-feeds earlier reasoning, so it is still edited.
    verify: check that the captured g and u reproduce the activation (tests).
    """
    parse_condition(condition)  # raises on a malformed name
    condition, layer_spec = split_layers(condition)
    only_layers = layer_set(layer_spec, len(get_mlps(model)))
    for key in needed_fills(condition):
        if fills is None or key not in fills[0]:
            raise ValueError(f"condition {condition!r} needs fill values {key!r}")

    mlps = get_mlps(model)
    stash: dict = {}
    stats = InterventionStats()
    generator = torch.Generator().manual_seed(seed)
    handles = []

    for i, mlp in enumerate(mlps):
        if only_layers is not None and i not in only_layers:
            continue

        def save_g(_m, _inp, out, i=i):
            stash[(i, "g")] = out

        def save_u(_m, _inp, out, i=i):
            stash[(i, "u")] = out

        def edit(_m, args, i=i, mlp=mlp):
            acts = args[0]
            g = stash.pop((i, "g"))
            u = stash.pop((i, "u"))
            if verify:
                expected = mlp.act_fn(g) * u
                assert torch.allclose(expected, acts, atol=1e-5), "captured g/u don't match activation"
            allowed = None
            if gen_only and acts.shape[1] != 1:
                if prefill_gen_mask is None or tuple(prefill_gen_mask.shape) != tuple(acts.shape[:2]):
                    return None  # prompt pass: leave untouched
                allowed = prefill_gen_mask.to(acts.device).unsqueeze(-1)
            new, mask = apply_condition(
                acts, g, u, condition, None if fills is None else fills[i], boost, generator
            )
            if allowed is not None:
                mask = mask & allowed
                new = torch.where(allowed, new, acts)
            stats.add(mask)
            return (new,) + tuple(args[1:])

        handles.append(mlp.gate_proj.register_forward_hook(save_g))
        handles.append(mlp.up_proj.register_forward_hook(save_u))
        handles.append(mlp.down_proj.register_forward_pre_hook(edit))

    try:
        yield stats
    finally:
        for h in handles:
            h.remove()
        stash.clear()


# ---------------------------------------------------------------------------
# Statistics gathered once per model
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_fills(model, batches, n_sample_tokens: int = 512) -> tuple[list[dict], dict]:
    """One pass over calibration text. Returns:

    fills:  per layer, {"all": mean activation per neuron,
                        "q1".."q4": mean activation per neuron over its firings in that quadrant,
                        "<q>_contrib": mean over ALL tokens of the neuron's activation when in <q>,
                        and for q3/q4, from the first n_sample_tokens tokens:
                        "<q>_samples" [S, d]: each neuron's values at its firings in <q>,
                            packed at the top of the column; "<q>_counts" [d]: how many;
                        "<q>_pool": all those values from all neurons, flattened}
    census: share of firings, and of total |activation|, in each quadrant.
    """
    mlps = get_mlps(model)
    stash: dict = {}
    sums = [None] * len(mlps)
    q_sums = [{q: None for q in QUADRANTS} for _ in mlps]
    q_counts = [{q: None for q in QUADRANTS} for _ in mlps]
    n_tokens = 0
    kept = [[] for _ in mlps]  # (acts, masks) from the first n_sample_tokens tokens
    kept_n = [0] * len(mlps)
    counts = {q: 0 for q in QUADRANTS}
    mass = {q: 0.0 for q in QUADRANTS}
    handles = []

    def add(store, key, value):
        store[key] = value if store[key] is None else store[key] + value

    for i, mlp in enumerate(mlps):
        handles.append(mlp.gate_proj.register_forward_hook(lambda _m, _i, o, i=i: stash.__setitem__((i, "g"), o)))
        handles.append(mlp.up_proj.register_forward_hook(lambda _m, _i, o, i=i: stash.__setitem__((i, "u"), o)))

        def tally(_m, args, i=i):
            acts = args[0].float()
            masks = quadrant_masks(stash.pop((i, "g")), stash.pop((i, "u")))
            add(sums, i, acts.sum(dim=(0, 1)))
            if kept_n[i] < n_sample_tokens:
                flat = acts.reshape(-1, acts.shape[-1])[: n_sample_tokens - kept_n[i]]
                fm = {q: masks[q].reshape(-1, acts.shape[-1])[: len(flat)] for q in SAMPLE_QUADS}
                kept[i].append((flat.half().cpu(), {q: v.cpu() for q, v in fm.items()}))
                kept_n[i] += len(flat)
            for q, m in masks.items():
                add(q_sums[i], q, (acts * m).sum(dim=(0, 1)))
                add(q_counts[i], q, m.sum(dim=(0, 1)).float())
                counts[q] += int(m.sum().item())
                mass[q] += float(acts.abs()[m].sum().item())

        handles.append(mlp.down_proj.register_forward_pre_hook(tally))

    try:
        for ids in batches:
            model(ids.to(model.device))
            n_tokens += ids.numel()
    finally:
        for h in handles:
            h.remove()

    fills = []
    for i in range(len(mlps)):
        layer = {"all": sums[i] / n_tokens}
        for q in QUADRANTS:
            s, c = q_sums[i][q], q_counts[i][q]
            layer[q] = torch.where(c > 0, s / c.clamp(min=1), torch.zeros_like(s))
            layer[f"{q}_contrib"] = s / n_tokens
        A = torch.cat([a for a, _ in kept[i]])
        for q in SAMPLE_QUADS:
            M = torch.cat([m[q] for _, m in kept[i]])
            order = (~M).to(torch.int8).argsort(dim=0, stable=True)  # firings first, per neuron
            dev = sums[i].device
            layer[f"{q}_samples"] = A.gather(0, order).to(dev)
            layer[f"{q}_counts"] = M.sum(0).to(dev)
            layer[f"{q}_pool"] = A[M].to(dev)
        fills.append(layer)

    n, total_mass = sum(counts.values()), sum(mass.values())
    census = {}
    for q in QUADRANTS:
        census[f"{q}_share_of_firings"] = counts[q] / n
        census[f"{q}_share_of_abs_activation"] = mass[q] / total_mass
    return fills, census


@torch.no_grad()
def general_damage(model, batches, condition, fills, seed=0) -> dict:
    """How much the condition changes predictions on ordinary text.

    Returns mean KL(intact || intervened) per token and the loss increase.
    Applied to every token (gen_only does not apply to plain text).
    """
    kl_sum, loss_clean, loss_int, n = 0.0, 0.0, 0.0, 0
    for b, ids in enumerate(batches):
        ids = ids.to(model.device)
        clean = model(ids).logits.float()
        with intervene(model, condition, fills, seed=seed + b):
            inter = model(ids).logits.float()
        target = ids[:, 1:].reshape(-1)
        lp_c = F.log_softmax(clean[:, :-1], -1)
        lp_i = F.log_softmax(inter[:, :-1], -1)
        kl_sum += (lp_c.exp() * (lp_c - lp_i)).sum(-1).sum().item()
        loss_clean += F.nll_loss(lp_c.reshape(-1, lp_c.shape[-1]), target, reduction="sum").item()
        loss_int += F.nll_loss(lp_i.reshape(-1, lp_i.shape[-1]), target, reduction="sum").item()
        n += target.numel()
    return {
        "kl": kl_sum / n,
        "loss_clean": loss_clean / n,
        "loss_intervened": loss_int / n,
        "loss_increase_pct": 100 * (loss_int / loss_clean - 1),
    }
