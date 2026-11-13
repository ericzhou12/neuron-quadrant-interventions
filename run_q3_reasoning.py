"""Stage 0: does ablating Q3 firings shorten reasoning in a small model, and
does accuracy drop with it?

Goodfire found that ablating Q3 firings in Qwen 3.6 35B-A3B cut average
reasoning on 24 GSM8K problems from 1,550 to 176 tokens. This script runs the
same kind of intervention, with controls, on a small reasoning model.

Every condition is first checked for general damage (WikiText loss, MMLU).
Conditions that break the model are skipped for GSM8K automatically.

1. Find doses that don't break the model (minutes, no GSM8K):

    python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --damage-only \
        --conditions q3_qmean,q4_qmean,q3_zero@0.05,q3_zero@0.1,q3_zero@0.25,q4_zero@0.05,q4_zero@0.1,q4_zero@0.25

2. Main runs, thinking on (hours; finished runs are skipped on restart). Add the
   largest q3_zero / q4_zero doses that passed step 1:

    python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --n-problems 200 --conditions none,q3_qmean,q4_qmean

3. Thinking off, same conditions (cheap):

    python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --n-problems 200 --conditions none,q3_qmean,q4_qmean \
        --no-thinking --max-new-tokens 1024

4. Only if step 2 shows an effect: forced continuation and the dose-matched control:

    python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --n-problems 200 --conditions q3_qmean --force-to-baseline
    python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --n-problems 200 --conditions none --force-like q3_qmean__forced

Then:

    python analyze_q3.py results/q3__Qwen__Qwen3-1.7B

Outputs in results/q3__<model>/:
    <condition>[__genonly]__seed<k>.jsonl   one record per problem, with full text
    damage.json                             general damage per condition, and whether it passed the gate
    census.json                             share of firings in each quadrant
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_sparsity import get_mlps, pick_device
from quadrants import STAGE0, compute_fills, describe, general_damage, intervene, parse_condition

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_gsm8k(n: int, offset: int = 0):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i in range(offset, min(offset + n, len(ds))):
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({"idx": i, "question": ds[i]["question"], "gold": gold})
    return rows


def _chunk(ids: list[int], n_batches: int, batch_size: int, seq_len: int):
    need = n_batches * batch_size * seq_len
    if len(ids) < need:
        raise ValueError(f"only {len(ids)} tokens, needed {need}")
    t = torch.tensor(ids[:need]).view(n_batches, batch_size, seq_len)
    return [t[i] for i in range(n_batches)]


def calibration_batches(tokenizer, n_batches: int, batch_size: int = 4, seq_len: int = 256):
    """Fill values are computed on GSM8K *train* solutions, never the eval problems."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    need = n_batches * batch_size * seq_len
    ids: list[int] = []
    for row in ds:
        ids.extend(tokenizer(row["question"] + "\n" + row["answer"] + "\n\n", add_special_tokens=False)["input_ids"])
        if len(ids) >= need:
            break
    return _chunk(ids, n_batches, batch_size, seq_len)


def ordinary_text_batches(tokenizer, n_batches: int, batch_size: int = 4, seq_len: int = 256):
    """WikiText test split, for measuring general damage."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    need = n_batches * batch_size * seq_len
    ids: list[int] = []
    for row in ds:
        if len(row["text"].strip()) < 50:
            continue
        ids.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        if len(ids) >= need:
            break
    return _chunk(ids, n_batches, batch_size, seq_len)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    candidate = boxed[-1] if boxed else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", candidate)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def parse_generation(token_ids: list[int], tokenizer, think_end_id: int | None,
                     no_thinking: bool = False, max_new_tokens: int | None = None) -> dict:
    pad = tokenizer.pad_token_id
    while token_ids and token_ids[-1] == pad:
        token_ids = token_ids[:-1]

    if no_thinking:
        # Thinking is switched off in the prompt, so everything generated is the answer.
        capped = max_new_tokens is not None and len(token_ids) >= max_new_tokens
        thinking_ids, answer_ids, finished = [], token_ids, not capped
    elif think_end_id is not None and think_end_id in token_ids:
        cut = token_ids.index(think_end_id)
        thinking_ids, answer_ids, finished = token_ids[:cut], token_ids[cut + 1 :], True
    else:
        # Never closed its thinking: hit the cap. Counted as wrong, as in Goodfire's eval.
        thinking_ids, answer_ids, finished = token_ids, [], False

    thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True)
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return {
        "thinking_tokens": len(thinking_ids),
        "answer_tokens": len(answer_ids),
        "total_tokens": len(token_ids),
        "finished_thinking": finished,
        "wait_count": len(re.findall(r"\bwait\b", thinking, flags=re.IGNORECASE)),
        "thinking": thinking,
        "answer_text": answer,
    }


def run_name(condition: str, gen_only: bool, seed: int, force_tag: str = "", no_thinking: bool = False) -> str:
    suffix = "__genonly" if gen_only and condition != "none" else ""
    if no_thinking:
        suffix += "__nothink"
    if force_tag:
        suffix += f"__{force_tag}"
    return f"{condition}{suffix}__seed{seed}"


def read_jsonl(path: Path) -> dict[int, dict]:
    return {r["idx"]: r for r in map(json.loads, path.open())}


# ---------------------------------------------------------------------------
# MMLU: a verifiable check of knowledge that needs no reasoning trace
# ---------------------------------------------------------------------------

LETTERS = "ABCD"


def load_mmlu(n: int, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)  # a fixed random sample across subjects
    return [ds[i] for i in order[:n]]


MMLU_PROMPT_VERSION = 4  # bump when scoring changes, so old scores are recomputed
MMLU_ANSWER_TOKENS = 16
MMLU_PREFILL = "The answer is ("


def mmlu_prompt(tokenizer, row, prefill: str = "") -> str:
    options = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, row["choices"]))
    content = f"{row['question']}\n\n{options}\n\nAnswer with the letter of the correct option only."
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) + prefill


def extract_letter(text: str) -> str | None:
    """First standalone A-D in the answer, whatever the formatting ("B", "**B**", "(B)", "B. Paris")."""
    m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text)
    return m.group(1) if m else None


def _letter_ids(tokenizer) -> list[list[int]]:
    """Token ids for each letter, with and without a leading space."""
    first = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"][:1]
    return [first(s) + first(" " + s) for s in LETTERS]


@torch.no_grad()
def mmlu_score(model, tokenizer, rows, condition, fills, batch_size=8, seed=0) -> dict:
    """Two measures, thinking off, intervention on every token.

    1. Knowledge (mmlu_accuracy, used by the gate): the reply is pre-filled
       with "The answer is (" and we compare the model's probability of A/B/C/D
       as the next token. No generation and no parsing, so how willing the
       model is to answer directly can't leak into it.
    2. Direct-answer rate (mmlu_direct_rate): the model replies freely, greedy,
       16 tokens; share of replies with a standalone A-D letter. Low = it
       started working the problem instead of answering. This is a behaviour
       measure, not a knowledge one. mmlu_direct_accuracy is v3's old score.
    """
    letter_ids = _letter_ids(tokenizer)
    flat = torch.tensor(sum(letter_ids, []))
    correct, mass, direct, direct_correct = [], [], [], []
    for b in range(0, len(rows), batch_size):
        batch = rows[b : b + batch_size]
        enc = tokenizer([mmlu_prompt(tokenizer, r, MMLU_PREFILL) for r in batch], return_tensors="pt",
                        padding=True).to(model.device)
        with intervene(model, condition, fills, seed=seed + b):
            probs = model(**enc).logits[:, -1].float().softmax(-1).cpu()
        per_letter = probs[:, flat].view(len(batch), len(LETTERS), -1).sum(-1)
        for r, p in zip(batch, per_letter):
            correct.append(int(p.argmax()) == r["answer"])
            mass.append(float(p.sum()))

        enc = tokenizer([mmlu_prompt(tokenizer, r) for r in batch], return_tensors="pt",
                        padding=True).to(model.device)
        with intervene(model, condition, fills, seed=seed + b):
            out = model.generate(**enc, max_new_tokens=MMLU_ANSWER_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for r, ids in zip(batch, out[:, enc["input_ids"].shape[1]:].tolist()):
            letter = extract_letter(tokenizer.decode(ids, skip_special_tokens=True))
            direct.append(letter is not None)
            direct_correct.append(letter == LETTERS[r["answer"]])
    return {
        "mmlu_accuracy": statistics.mean(correct),
        # Should be near 1: the pre-fill makes a letter the obvious next token.
        "mmlu_letter_mass": statistics.mean(mass),
        "mmlu_direct_rate": statistics.mean(direct),
        "mmlu_direct_accuracy": statistics.mean(direct_correct),
        "mmlu_n": len(rows),
        "mmlu_prompt_version": MMLU_PROMPT_VERSION,
    }


def gate_check(d: dict, base: dict, max_loss_pct: float, max_mmlu_drop: float) -> tuple[bool, str]:
    """Does this condition leave the model broadly working? Returns (passes, reason)."""
    problems = []
    if d.get("loss_increase_pct", 0) > max_loss_pct:
        problems.append(f"WikiText loss +{d['loss_increase_pct']:.1f}% > {max_loss_pct}%")
    if "mmlu_accuracy" in d and "mmlu_accuracy" in base:
        drop = base["mmlu_accuracy"] - d["mmlu_accuracy"]
        if drop > max_mmlu_drop:
            problems.append(f"MMLU drop {drop:.3f} > {max_mmlu_drop}")
    return (not problems, "; ".join(problems) or "ok")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_condition(model, tokenizer, problems, condition, fills, args, think_end_id):
    records = []
    changed_fracs = []
    for b in range(0, len(problems), args.batch_size):
        batch = problems[b : b + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p["question"] + PROMPT_SUFFIX}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not args.no_thinking,
            )
            for p in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Same seed per batch in every condition, so conditions get comparable sampling noise.
        torch.manual_seed(args.seed * 100_000 + b)
        gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        if args.greedy:
            gen_kwargs.update(do_sample=False)
        elif args.no_thinking:  # Qwen's recommended non-thinking settings
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        else:  # Qwen's recommended thinking-mode settings; greedy tends to loop
            gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)

        with intervene(model, condition, fills, boost=args.boost,
                       seed=args.seed * 100_000 + b, gen_only=args.gen_only) as stats:
            out = model.generate(**enc, **gen_kwargs)
        changed_fracs.append(stats.fraction())

        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for p, ids in zip(batch, new_tokens.tolist()):
            r = parse_generation(ids, tokenizer, think_end_id, args.no_thinking, args.max_new_tokens)
            pred = extract_answer(r["answer_text"]) if r["finished_thinking"] else None
            records.append({
                "condition": condition,
                "thinking_mode": "off" if args.no_thinking else "on",
                "gen_only": bool(args.gen_only and condition != "none"),
                "seed": args.seed,
                "idx": p["idx"],
                "gold": p["gold"],
                "pred": pred,
                "correct": is_correct(pred, p["gold"]),
                **r,
            })
        done = min(b + args.batch_size, len(problems))
        print(f"  {done}/{len(problems)} problems", flush=True)

    return records, statistics.mean(changed_fracs)


# ---------------------------------------------------------------------------
# Forced continuation ("budget forcing", as in the s1 paper)
# ---------------------------------------------------------------------------


def _stop_ids(model, think_end_id) -> set[int]:
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    return eos | ({think_end_id} if think_end_id is not None else set())


def run_condition_forced(model, tokenizer, problems, condition, fills, args, think_end_id,
                         targets: dict[int, int] | None = None, quotas: dict[int, int] | None = None):
    """Generate, but when the model tries to stop thinking, replace </think> with
    "Wait" and let it continue.

    targets: per problem, keep forcing until thinking reaches this many tokens
             (the intact model's length on the same problem and seed).
    quotas:  per problem, force exactly this many times (the dose-matched
             control for the intact model).
    Generation runs in rounds; each round re-feeds the text so far. With
    --gen-only, earlier generated text is still edited on re-feed.
    """
    if think_end_id is None:
        raise SystemExit("forced continuation needs a model with a </think> token")
    stop = _stop_ids(model, think_end_id)
    wait_ids = tokenizer("\n\nWait", add_special_tokens=False)["input_ids"]
    pad = tokenizer.pad_token_id
    records, changed_fracs = [], []

    for b in range(0, len(problems), args.batch_size):
        batch = problems[b : b + args.batch_size]
        prompt_ids = [
            tokenizer(tokenizer.apply_chat_template(
                [{"role": "user", "content": p["question"] + PROMPT_SUFFIX}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            ), add_special_tokens=False)["input_ids"]
            for p in batch
        ]
        gen = [[] for _ in batch]
        phase = ["think"] * len(batch)
        n_forced = [0] * len(batch)
        done = [False] * len(batch)
        rnd = 0

        while not all(done):
            active = [i for i in range(len(batch)) if not done[i]]
            budget = {i: args.max_new_tokens - len(gen[i]) for i in active}
            seqs = [prompt_ids[i] + gen[i] for i in active]
            width = max(map(len, seqs))
            input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            attn = torch.zeros_like(input_ids)
            gen_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for row, (i, s) in enumerate(zip(active, seqs)):
                input_ids[row, width - len(s):] = torch.tensor(s)
                attn[row, width - len(s):] = 1
                if gen[i]:
                    gen_mask[row, width - len(gen[i]):] = True

            torch.manual_seed(args.seed * 100_000 + b + 7919 * rnd)
            gen_kwargs = dict(max_new_tokens=max(budget.values()), pad_token_id=pad,
                              eos_token_id=sorted(stop))
            if args.greedy:
                gen_kwargs.update(do_sample=False)
            else:
                gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
            with intervene(model, condition, fills, boost=args.boost, seed=args.seed * 100_000 + b + 7919 * rnd,
                           gen_only=args.gen_only, prefill_gen_mask=gen_mask) as stats:
                out = model.generate(input_ids=input_ids.to(model.device),
                                     attention_mask=attn.to(model.device), **gen_kwargs)
            changed_fracs.append(stats.fraction())

            for row, i in enumerate(active):
                new = out[row, width:].tolist()
                cut = next((k for k, t in enumerate(new) if t in stop), None)
                new = new[: cut + 1] if cut is not None else [t for t in new if t != pad]
                new = new[: budget[i]]
                gen[i] += new
                last = gen[i][-1] if gen[i] else None

                if len(gen[i]) >= args.max_new_tokens:
                    done[i] = True  # hit the cap
                elif phase[i] == "think" and last == think_end_id:
                    idx = batch[i]["idx"]
                    thinking_len = len(gen[i]) - 1
                    if targets is not None:
                        force = thinking_len < targets[idx]
                    else:
                        force = n_forced[i] < quotas[idx]
                    if force and n_forced[i] < args.max_forced:
                        gen[i] = gen[i][:-1] + wait_ids
                        n_forced[i] += 1
                    else:
                        phase[i] = "answer"
                else:
                    done[i] = True  # finished the answer, or stopped without closing thinking
            rnd += 1

        for i, p in enumerate(batch):
            r = parse_generation(gen[i], tokenizer, think_end_id)
            pred = extract_answer(r["answer_text"]) if r["finished_thinking"] else None
            records.append({
                "condition": condition,
                "thinking_mode": "on",
                "gen_only": bool(args.gen_only and condition != "none"),
                "seed": args.seed,
                "idx": p["idx"],
                "gold": p["gold"],
                "pred": pred,
                "correct": is_correct(pred, p["gold"]),
                "n_forced": n_forced[i],
                "target": None if targets is None else targets[p["idx"]],
                **r,
            })
        print(f"  {min(b + args.batch_size, len(problems))}/{len(problems)} problems", flush=True)

    return records, statistics.mean(changed_fracs)


# ---------------------------------------------------------------------------
# Fast path (default): our own decoding loop, see fast_gen.py
# ---------------------------------------------------------------------------


def run_condition_fast(model, tokenizer, problems, condition, fills, args, think_end_id,
                       targets: dict[int, int] | None = None, quotas: dict[int, int] | None = None,
                       partial_path: Path | None = None):
    """Plain or forced generation with fast_gen.generate_batch.

    Finished rows leave the batch at once, and records are appended to
    `partial_path` after every batch, so an interrupted run resumes where it
    stopped (whole batches only, so seeds stay the same).
    """
    from fast_gen import generate_batch

    forcing = targets is not None or quotas is not None
    if forcing and think_end_id is None:
        raise SystemExit("forced continuation needs a model with a </think> token")
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    wait_ids = tokenizer("\n\nWait", add_special_tokens=False)["input_ids"]
    wait_token_ids = sorted({tokenizer(s, add_special_tokens=False)["input_ids"][0] for s in ("Wait", " Wait")})
    if args.greedy:
        sampling = dict(temperature=0.0)
    elif args.no_thinking:  # Qwen's recommended non-thinking settings
        sampling = dict(temperature=0.7, top_p=0.8, top_k=20)
    else:  # Qwen's recommended thinking-mode settings; greedy tends to loop
        sampling = dict(temperature=0.6, top_p=0.95, top_k=20)

    records = []
    if partial_path is not None and partial_path.exists():
        records = [json.loads(l) for l in partial_path.open()]
        print(f"  resuming: {len(records)} problems already done")
    have = {r["idx"] for r in records}

    for b in range(0, len(problems), args.batch_size):
        batch = problems[b : b + args.batch_size]
        if all(p["idx"] in have for p in batch):
            continue
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p["question"] + PROMPT_SUFFIX}],
                tokenize=False, add_generation_prompt=True, enable_thinking=not args.no_thinking,
            )
            for p in batch
        ]
        force_fn = None
        if targets is not None:
            force_fn = lambda row, thinking_len, n: thinking_len < targets[batch[row]["idx"]]
        elif quotas is not None:
            force_fn = lambda row, thinking_len, n: n < quotas[batch[row]["idx"]]
        outs, changed = generate_batch(
            model, tokenizer, prompts, condition, fills,
            max_new_tokens=args.max_new_tokens, think_end_id=None if args.no_thinking else think_end_id,
            eos_ids=eos, seed=args.seed * 100_000 + b, gen_only=args.gen_only, boost=args.boost,
            force_fn=force_fn, wait_ids=wait_ids, max_forced=args.max_forced,
            wait_token_ids=wait_token_ids, **sampling)
        new = []
        for p, o in zip(batch, outs):
            ids = [t for t in o["ids"] if t not in eos]
            r = parse_generation(ids, tokenizer, think_end_id, args.no_thinking, args.max_new_tokens)
            if args.no_thinking:
                r["finished_thinking"] = len(o["ids"]) < args.max_new_tokens or o["ids"][-1] in eos
            pred = extract_answer(r["answer_text"]) if r["finished_thinking"] else None
            rec = {
                "condition": condition,
                "thinking_mode": "off" if args.no_thinking else "on",
                "gen_only": bool(args.gen_only and condition != "none"),
                "seed": args.seed,
                "idx": p["idx"],
                "gold": p["gold"],
                "pred": pred,
                "correct": is_correct(pred, p["gold"]),
                "stop_hazard": round(o["stop_hazard"], 4),
                "wait_hazard": round(o["wait_hazard"], 4),
                "pct_firings_changed": round(100 * changed, 3),
                **r,
            }
            if forcing:
                rec["n_forced"] = o["n_forced"]
                rec["target"] = None if targets is None else targets[p["idx"]]
            new.append(rec)
        records += new
        if partial_path is not None:
            with partial_path.open("a") as f:
                for rec in new:
                    f.write(json.dumps(rec) + "\n")
        print(f"  {min(b + args.batch_size, len(problems))}/{len(problems)} problems", flush=True)

    changed_all = statistics.mean(r["pct_firings_changed"] for r in records) / 100
    return records, changed_all


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--conditions", default=",".join(STAGE0),
                   help="Comma-separated: none, random_matched, q3_boost, or <q1-q4>_<zero|mean|qmean>, "
                        "optionally with @dose to edit a random share, e.g. q3_qmean,q4_qmean,q3_zero@0.1")
    p.add_argument("--damage-only", action="store_true",
                   help="Only measure general damage (WikiText + MMLU) for the conditions, then stop. "
                        "Takes minutes; use it to find doses that don't break the model.")
    p.add_argument("--max-loss-increase", type=float, default=10.0,
                   help="Damage gate: skip GSM8K runs for conditions raising WikiText loss by more than this %%.")
    p.add_argument("--max-mmlu-drop", type=float, default=0.05,
                   help="Damage gate: skip GSM8K runs for conditions dropping MMLU accuracy by more than this.")
    p.add_argument("--allow-broken", action="store_true", help="Run GSM8K even for conditions that fail the gate.")
    p.add_argument("--gen-only", action="store_true",
                   help="Intervene only while generating, not on the prompt (rules out 'misread the question').")
    p.add_argument("--force-to-baseline", action="store_true",
                   help="Forced continuation: when the model tries to stop thinking, append 'Wait' until "
                        "it reaches the intact model's length on that problem. Needs the `none` run first.")
    p.add_argument("--force-like", metavar="VARIANT",
                   help="Dose-matched control: force the same number of 'Wait's per problem as VARIANT's "
                        "forced run, e.g. --conditions none --force-like q3_zero__forced")
    p.add_argument("--max-forced", type=int, default=20, help="Never force more than this many times per problem.")
    p.add_argument("--no-thinking", action="store_true",
                   help="Answer directly with thinking switched off, so reasoning length plays no part. "
                        "Answer length is still recorded.")
    p.add_argument("--mmlu-questions", type=int, default=300,
                   help="MMLU questions scored per condition (letter likelihood, thinking off). 0 to skip.")
    p.add_argument("--n-problems", type=int, default=200)
    p.add_argument("--offset", type=int, default=0, help="Start at this GSM8K test index.")
    p.add_argument("--batch-size", type=int, default=25,
                   help="Problems decoded together. Finished rows leave the batch, so large is fine.")
    p.add_argument("--legacy-generate", action="store_true",
                   help="Use model.generate (slow: every row waits for the slowest) instead of fast_gen.")
    p.add_argument("--tag", default="", help="Extra suffix for the output file name (e.g. a pilot run).")
    p.add_argument("--max-new-tokens", type=int, default=4096, help="Goodfire capped thinking at 4,096.")
    p.add_argument("--boost", type=float, default=2.0, help="Multiplier for q3_boost.")
    p.add_argument("--calib-batches", type=int, default=16)
    p.add_argument("--damage-batches", type=int, default=8, help="Batches of WikiText for the general-damage measure.")
    p.add_argument("--greedy", action="store_true", help="Greedy decoding (Goodfire's figure used greedy).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="Re-run conditions that already have results.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"],
                   help="bfloat16 is faster; try float32 if you see errors or NaNs.")
    args = p.parse_args()
    conditions = args.conditions.split(",")
    for c in conditions:
        try:
            parse_condition(c)
        except ValueError as e:
            p.error(str(e))
    if args.no_thinking and (args.force_to_baseline or args.force_like):
        p.error("forced continuation needs thinking on")

    device = pick_device()
    print(f"device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
    model.to(device).eval()
    mlps = get_mlps(model)
    print(f"{len(mlps)} layers x {mlps[0].down_proj.in_features} neurons")

    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_end_id is None or think_end_id == tokenizer.unk_token_id:
        print("WARNING: tokenizer has no </think> token; thinking length will equal total length.")
        think_end_id = None

    out_dir = Path("results") / f"q3__{args.model.replace('/', '__')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("computing fill values and quadrant census...")
    fills, census = compute_fills(model, calibration_batches(tokenizer, args.calib_batches))
    (out_dir / "census.json").write_text(json.dumps(census, indent=2))
    print("  share of firings:", {q: round(census[f"{q}_share_of_firings"], 3) for q in ("q1", "q2", "q3", "q4")})

    # General damage, once per condition: WikiText KL, plus MMLU accuracy.
    # `none` is always scored, since the MMLU drop is measured against it.
    damage_path = out_dir / "damage.json"
    damage = json.loads(damage_path.read_text()) if damage_path.exists() else {}
    wiki = mmlu = None
    print("\ngeneral damage (measured without any reasoning):")
    for c in ["none"] + [c for c in conditions if c != "none"]:
        d = damage.get(c, {}) if not args.overwrite else {}
        if c != "none" and "kl" not in d:
            if wiki is None:
                wiki = ordinary_text_batches(tokenizer, args.damage_batches)
            d.update(general_damage(model, wiki, c, fills, seed=args.seed))
        if args.mmlu_questions and (d.get("mmlu_n") != args.mmlu_questions
                                    or d.get("mmlu_prompt_version") != MMLU_PROMPT_VERSION):
            if mmlu is None:
                mmlu = load_mmlu(args.mmlu_questions)
            d.update(mmlu_score(model, tokenizer, mmlu, c, fills, args.batch_size, seed=args.seed))
        if c != "none":
            d["passes_gate"], d["gate_reason"] = gate_check(d, damage["none"], args.max_loss_increase,
                                                            args.max_mmlu_drop)
        damage[c] = d
        parts = []
        if "kl" in d:
            parts.append(f"WikiText KL {d['kl']:.3f}, loss +{d['loss_increase_pct']:.1f}%")
        if "mmlu_accuracy" in d:
            parts.append(f"MMLU {d['mmlu_accuracy']:.3f} (letter mass {d['mmlu_letter_mass']:.2f}) | "
                         f"answers directly {d['mmlu_direct_rate']:.0%}")
        if c != "none":
            parts.append("OK" if d["passes_gate"] else f"BREAKS MODEL ({d['gate_reason']})")
        print(f"  {c:>16}: " + " | ".join(parts))
        damage_path.write_text(json.dumps(damage, indent=2))

    if damage["none"].get("mmlu_letter_mass", 1) < 0.8:
        print(f"\nWARNING: after the pre-fill the intact model puts only {damage['none']['mmlu_letter_mass']:.0%} "
              "of its probability on A-D, so the MMLU score is unreliable. Treat the gate's MMLU check with care.")

    if args.damage_only:
        print(f"\n--damage-only: stopping. Results in {damage_path}")
        return

    blocked = [c for c in conditions if c != "none" and not damage[c]["passes_gate"]]
    if blocked and not args.allow_broken:
        print(f"\nSkipping GSM8K runs for {blocked}: they break the model on ordinary text, so any "
              "reasoning result would just reflect a broken model. Use a smaller @dose, or --allow-broken.")
        conditions = [c for c in conditions if c not in blocked]

    problems = load_gsm8k(args.n_problems, args.offset)

    targets = quotas = None
    force_tag = ""
    if args.force_to_baseline and args.force_like:
        p.error("use --force-to-baseline or --force-like, not both")
    if args.force_to_baseline:
        base_path = out_dir / f"none__seed{args.seed}.jsonl"
        if not base_path.exists():
            p.error(f"--force-to-baseline needs {base_path}; run --conditions none first")
        base = read_jsonl(base_path)
        targets = {pr["idx"]: base[pr["idx"]]["thinking_tokens"] for pr in problems}
        force_tag = "forced"
    elif args.force_like:
        src = out_dir / f"{args.force_like}__seed{args.seed}.jsonl"
        if not src.exists():
            p.error(f"--force-like needs {src}")
        recs = read_jsonl(src)
        quotas = {pr["idx"]: recs[pr["idx"]].get("n_forced", 0) for pr in problems}
        force_tag = f"waitsof-{args.force_like.replace('__', '-')}"

    for condition in conditions:
        name = run_name(condition, args.gen_only, args.seed, force_tag, args.no_thinking)
        if args.tag:
            name = name.replace(f"__seed{args.seed}", f"__{args.tag}__seed{args.seed}")
        path = out_dir / f"{name}.jsonl"
        if path.exists() and not args.overwrite:
            print(f"\n== {name}: already done, skipping (use --overwrite to redo)")
            continue
        print(f"\n== {name}: {describe(condition)}")
        t0 = time.time()
        partial = path.with_suffix(".jsonl.partial")
        if args.overwrite and partial.exists():
            partial.unlink()
        if not args.legacy_generate:
            records, changed = run_condition_fast(model, tokenizer, problems, condition, fills, args,
                                                  think_end_id, targets=targets, quotas=quotas,
                                                  partial_path=partial)
        elif force_tag:
            records, changed = run_condition_forced(model, tokenizer, problems, condition, fills, args,
                                                    think_end_id, targets=targets, quotas=quotas)
        else:
            records, changed = run_condition(model, tokenizer, problems, condition, fills, args, think_end_id)
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps({"pct_firings_changed": round(100 * changed, 3), **r}) + "\n")
        if partial.exists():
            partial.unlink()
        length_key = "answer_tokens" if args.no_thinking else "thinking_tokens"
        lengths = [r[length_key] for r in records]
        print(f"   accuracy {statistics.mean(r['correct'] for r in records):.3f} | "
              f"median {'answer' if args.no_thinking else 'thinking'} {statistics.median(lengths):.0f} tokens | "
              f"cap hits {statistics.mean(not r['finished_thinking'] for r in records):.3f} | "
              f"{100 * changed:.1f}% of firings changed | {(time.time() - t0) / 60:.1f} min")

    print(f"\ndone. analyse with:  python analyze_q3.py {out_dir}")


if __name__ == "__main__":
    main()
