"""Faster generation loop for the Q3 experiments.

`model.generate` keeps computing every row of a batch until the slowest row
finishes, so one runaway trace (4,096 tokens) makes the whole batch pay for
4,096 steps. This loop does the decoding itself and DROPS rows from the batch
(and from the attention cache) as soon as they finish, so cost follows the
total number of tokens actually generated.

It also does forced continuation in the same pass: when a row tries to close
its thinking and should be forced, the </think> token is replaced by "\n\nWait"
and decoding simply carries on (no re-feeding rounds).

Sampling matches Hugging Face's order: temperature, then top-k, then top-p.
Randomness comes from a CPU generator per batch, so a run is reproducible.

Extra per-problem measurements (all from the model's own next-token
probabilities BEFORE temperature/top-k, on the path it actually took):
    stop_hazard   sum of P(</think>) over thinking positions
    wait_hazard   sum of P("Wait")  over thinking positions
"""

from __future__ import annotations

import torch

from quadrants import intervene


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, gen: torch.Generator):
    """logits [B, V] on device -> token ids [B] (CPU). Greedy if temperature == 0."""
    if temperature == 0:
        return logits.argmax(-1).cpu()
    vals, idx = (logits.float() / temperature).topk(top_k, dim=-1)  # sorted, descending
    vals, idx = vals.cpu(), idx.cpu()
    probs = vals.softmax(-1)
    # top-p: keep the smallest set of tokens whose probability reaches top_p
    cum = probs.cumsum(-1)
    drop = (cum - probs) >= top_p  # tokens that start after top_p is already reached
    probs = probs.masked_fill(drop, 0)
    probs = probs / probs.sum(-1, keepdim=True)
    choice = torch.multinomial(probs, 1, generator=gen)
    return idx.gather(1, choice).squeeze(1)


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], condition: str, fills, *,
                   max_new_tokens: int, think_end_id: int | None, eos_ids: set[int],
                   seed: int, temperature: float = 0.6, top_k: int = 20, top_p: float = 0.95,
                   gen_only: bool = False, boost: float = 2.0,
                   force_fn=None, wait_ids: list[int] | None = None, max_forced: int = 20,
                   wait_token_ids: list[int] | None = None):
    """Generate for a batch of prompts. Returns (list of dicts, share of firings changed).

    Each dict: {"ids": generated token ids, "n_forced": int, "stop_hazard": float, "wait_hazard": float}.

    force_fn(row, thinking_len, n_forced) -> bool: called when row `row` emits
    </think> while thinking; True = replace it with `wait_ids` and keep going.
    """
    dev = model.device
    enc = tokenizer(prompts, return_tensors="pt", padding=True)  # left-padded
    B = len(prompts)
    rng = torch.Generator().manual_seed(seed)

    out = [{"ids": [], "n_forced": 0, "stop_hazard": 0.0, "wait_hazard": 0.0} for _ in range(B)]
    thinking = [think_end_id is not None] * B  # still inside the thinking block?
    pending = [[] for _ in range(B)]  # forced tokens waiting to be fed
    active = list(range(B))  # original row index of each row still in the batch

    attn = enc["attention_mask"].to(dev)
    with intervene(model, condition, fills, boost=boost, seed=seed, gen_only=gen_only) as stats:
        res = model(input_ids=enc["input_ids"].to(dev), attention_mask=attn, use_cache=True)
        cache = res.past_key_values
        logits = res.logits[:, -1]

        while active:
            probs = None
            if think_end_id is not None:
                probs = logits.float().softmax(-1)
                p_stop = probs[:, think_end_id].cpu().tolist()
                p_wait = probs[:, wait_token_ids].sum(-1).cpu().tolist() if wait_token_ids else [0.0] * len(active)
            sampled = _sample(logits, temperature, top_k, top_p, rng).tolist()

            next_tok, keep = [], []
            for row, i in enumerate(active):
                o = out[i]
                if pending[i]:
                    tok = pending[i].pop(0)  # forced token; the model's own choice is ignored
                else:
                    tok = sampled[row]
                    if thinking[i]:
                        o["stop_hazard"] += p_stop[row]
                        o["wait_hazard"] += p_wait[row]
                    if thinking[i] and tok == think_end_id:
                        if (force_fn is not None and o["n_forced"] < max_forced
                                and force_fn(i, len(o["ids"]), o["n_forced"])):
                            o["n_forced"] += 1
                            tok, *rest = wait_ids
                            pending[i] = list(rest)
                        else:
                            thinking[i] = False
                o["ids"].append(tok)
                finished = (tok in eos_ids and not pending[i]) or len(o["ids"]) >= max_new_tokens
                if not finished:
                    keep.append(row)
                    next_tok.append(tok)

            if not keep:
                break
            if len(keep) < len(active):
                sel = torch.tensor(keep, device=dev)
                cache.batch_select_indices(sel)
                attn = attn[sel]
                active = [active[r] for r in keep]
            attn = torch.cat([attn, attn.new_ones(len(active), 1)], dim=1)
            res = model(input_ids=torch.tensor(next_tok, device=dev)[:, None], attention_mask=attn,
                        past_key_values=cache, use_cache=True)
            cache = res.past_key_values
            logits = res.logits[:, -1]
        changed = stats.fraction()
    return out, changed
