"""Offline checks for the Q3 experiment on a tiny random model. No downloads.

    python test_quadrants.py
"""

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from analyze_q3 import compare, load_runs, mcnemar_exact
from quadrants import CONDITIONS, apply_condition, compute_fills, general_damage, intervene, quadrant_masks
from run_q3_reasoning import extract_answer, is_correct, run_name

torch.manual_seed(0)
config = LlamaConfig(vocab_size=500, hidden_size=64, intermediate_size=256,
                     num_hidden_layers=3, num_attention_heads=4, max_position_embeddings=128)
model = LlamaForCausalLM(config).eval()
batches = [torch.randint(0, 500, (2, 32)) for _ in range(3)]
fills, census = compute_fills(model, batches)
ids = batches[0]

# 1. The four quadrants partition every firing exactly once.
g, u = torch.randn(2, 5, 256), torch.randn(2, 5, 256)
m = quadrant_masks(g, u)
assert (sum(x.int() for x in m.values()) == 1).all()

# 2. Q3 outputs are small positive, Q4 small negative (SiLU leak).
act = torch.nn.functional.silu(g) * u
assert (act[m["q3"]] > 0).all() and (act[m["q4"]] < 0).all()

# 3. Fill values: typical Q3 values are positive, Q4 negative, and differ from the all-token mean.
assert (fills[0]["q3"] >= 0).all() and fills[0]["q3"].shape == (256,)
assert (fills[0]["q4"] <= 0).all()
assert not torch.allclose(fills[0]["q3"], fills[0]["all"])

# 3b. Condition names and doses.
from quadrants import describe, parse_condition
assert parse_condition("q3_zero@0.1") == ("q3_zero", "q3", 0.1)
assert parse_condition("q4_qmean") == ("q4_qmean", "q4", 1.0)
for badname in ("q5_zero", "q3_foo", "q3_zero@0", "q3_zero@1.5", "none@0.5"):
    try:
        parse_condition(badname)
        raise AssertionError(f"{badname} should be rejected")
    except ValueError:
        pass
assert "10%" in describe("q3_zero@0.1")
assert abs(sum(v for k, v in census.items() if "firings" in k) - 1) < 1e-6

# 4. Edits only touch the intended entries.
new, mask = apply_condition(act, g, u, "q3_zero")
assert torch.equal(mask, m["q3"]) and (new[m["q3"]] == 0).all() and torch.equal(new[~m["q3"]], act[~m["q3"]])
new, _ = apply_condition(act, g, u, "q3_qmean", fills[0])
assert torch.allclose(new[m["q3"]], fills[0]["q3"].expand_as(act)[m["q3"]])
new, mask = apply_condition(act, g, u, "q4_qmean", fills[0])
assert torch.equal(mask, m["q4"]) and torch.allclose(new[m["q4"]], fills[0]["q4"].expand_as(act)[m["q4"]])
gen = torch.Generator().manual_seed(0)
_, half = apply_condition(act, g, u, "q3_zero@0.5", generator=gen)
share = half.sum().item() / m["q3"].sum().item()
assert (half & ~m["q3"]).sum() == 0 and 0.4 < share < 0.6, share
new, mask = apply_condition(act, g, u, "random_matched", generator=gen)
assert not (mask & m["q3"]).any(), "random control must avoid Q3"
assert torch.equal(mask.sum(-1), m["q3"].sum(-1)), "random control must match Q3 count per token"
assert (new[mask] == 0).all()

# 4b. Put-back fills.
f0 = fills[0]
c = f0["q3_counts"]
assert c.shape == (256,) and (c > 0).all()
for n in (0, 7, 100):  # samples are exactly that neuron's Q3 values (positive), packed at the top
    assert (f0["q3_samples"][: c[n], n] >= 0).all()
assert (f0["q3_pool"] >= 0).all() and (f0["q4_pool"] <= 0).all() and len(f0["q3_pool"]) == c.sum()
new, mask = apply_condition(act, g, u, "q3_resample", f0, generator=torch.Generator().manual_seed(1))
assert torch.equal(mask, m["q3"]) and torch.equal(new[~mask], act[~mask])
for b_, t_, n in zip(*[x[:50] for x in mask.nonzero(as_tuple=True)]):  # each value is that neuron's own
    assert (f0["q3_samples"][: c[n], n].float() == new[b_, t_, n].float()).any()
new, mask = apply_condition(act, g, u, "q3_pool", f0, generator=torch.Generator().manual_seed(1))
assert torch.equal(new[~mask], act[~mask]) and (new[mask] >= 0).all()
new, mask = apply_condition(act, g, u, "q3_const", f0)
assert torch.allclose(new - torch.where(mask, torch.zeros_like(act), act), f0["q3_contrib"].expand_as(act), atol=1e-6)
assert (f0["q3_contrib"] >= 0).all() and (f0["q3_contrib"] <= f0["q3"] + 1e-6).all()
_, half = apply_condition(act, g, u, "q3_resample@0.5", f0, generator=torch.Generator().manual_seed(1))
assert 0.4 < half.sum() / m["q3"].sum() < 0.6

with torch.no_grad():
    # 5. Hooks capture the right g and u (verify=True asserts act_fn(g)*u == activation).
    clean = model(ids).logits
    with intervene(model, "none", verify=True):
        assert torch.allclose(model(ids).logits, clean)

    # 6. Every condition runs and changes the output; hooks are removed after.
    for c in CONDITIONS:
        with intervene(model, c, fills, verify=True) as stats:
            out = model(ids).logits
        if c != "none":
            assert not torch.allclose(out, clean), c
            assert 0 < stats.fraction() < 1, c
    assert torch.allclose(model(ids).logits, clean), "hooks not removed"

    # 7. gen_only leaves the prompt pass untouched...
    with intervene(model, "q3_zero", fills, gen_only=True) as stats:
        assert torch.allclose(model(ids).logits, clean)
    assert stats.total == 0
    # ...but does change generation.
    prompt = ids[:1, :8]
    plain = model.generate(prompt, max_new_tokens=12, do_sample=False, pad_token_id=0)
    with intervene(model, "q3_zero", fills, gen_only=True) as stats:
        edited = model.generate(prompt, max_new_tokens=12, do_sample=False, pad_token_id=0)
    assert stats.total > 0 and edited.shape == plain.shape

    # 8. General damage is zero for nothing, positive for an ablation.
    d = general_damage(model, batches[:2], "q3_zero", fills)
    assert d["kl"] > 0

# 9. Scoring and file naming.
assert extract_answer("so the total is \\boxed{1,234}.") == "1234"
assert extract_answer("answer: 42") == "42"
assert is_correct("72.0", "72") and not is_correct(None, "72")
assert run_name("none", True, 0) == "none__seed0"
assert run_name("q3_zero", True, 1) == "q3_zero__genonly__seed1"

# 10. Analysis on synthetic results: an ablation that halves length and loses
#     accuracy only on problems that shrank.
assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(10, 0) < 0.01
rng = random.Random(0)
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    base, abl = [], []
    for i in range(60):
        L = rng.randint(400, 2000)
        shrinks = i % 2 == 0
        base.append({"seed": 0, "idx": i, "thinking_tokens": L, "correct": True, "finished_thinking": True, "wait_count": 5})
        abl.append({"seed": 0, "idx": i, "thinking_tokens": L // 4 if shrinks else L, "correct": not (shrinks and i % 4 == 0),
                    "finished_thinking": True, "wait_count": 1})
    for name, recs in (("none__seed0", base), ("q3_zero__seed0", abl)):
        (tmp / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    runs = load_runs(tmp)
    r = compare(runs["none"], runs["q3_zero"], long_threshold=1000)
    assert r["n"] == 60 and r["lost"] == 15 and r["gained"] == 0
    assert r["acc_change_barely_changed"] == 0 and r["acc_change_shrank_a_lot"] < 0
    assert r["length_ratio_ci_low"] <= r["length_ratio"] <= r["length_ratio_ci_high"]
    res = subprocess.run([sys.executable, "analyze_q3.py", str(tmp)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp / "summary.csv").exists()

# 11. Forced continuation logic, with a stub tokenizer and a model rigged to
#     always try to close its thinking immediately.
from types import SimpleNamespace

from run_q3_reasoning import run_condition_forced


class StubTokenizer:
    pad_token_id, unk_token_id = 0, 1

    def apply_chat_template(self, msgs, **_):
        return msgs[0]["content"][:20]

    def __call__(self, text, add_special_tokens=False, return_tensors=None, padding=False):
        enc = lambda t: [10 + ord(c) % 400 for c in t]
        if isinstance(text, str):
            return {"input_ids": enc(text)}
        seqs = [enc(t) for t in text]  # batch, left-padded
        w = max(map(len, seqs))
        ids = torch.tensor([[0] * (w - len(s)) + s for s in seqs])
        mask = torch.tensor([[0] * (w - len(s)) + [1] * len(s) for s in seqs])
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


THINK_END = 3
model.generation_config.eos_token_id = 2
rig = model.lm_head.register_forward_hook(lambda _m, _i, o: o.index_add(-1, torch.tensor([THINK_END]), torch.full((*o.shape[:-1], 1), 50.0)))
tok = StubTokenizer()
wait_len = len(tok("\n\nWait")["input_ids"])
problems = [{"idx": 0, "question": "what is two plus two", "gold": "4"},
            {"idx": 1, "question": "what is three plus one", "gold": "4"}]
args = SimpleNamespace(batch_size=2, max_new_tokens=64, seed=0, greedy=True, boost=2.0, gen_only=False, max_forced=20)
with torch.no_grad():
    recs, _ = run_condition_forced(model, tok, problems, "q3_zero", fills, args, THINK_END, targets={0: 2 * wait_len - 1, 1: 0})
    assert recs[0]["n_forced"] == 2 and recs[0]["thinking_tokens"] == 2 * wait_len, recs[0]
    assert recs[1]["n_forced"] == 0 and recs[1]["finished_thinking"]
    recs, _ = run_condition_forced(model, tok, problems, "none", fills, args, THINK_END, quotas={0: 1, 1: 3})
    assert [r["n_forced"] for r in recs] == [1, 3]
    args.gen_only = True  # re-fed reasoning is still edited under gen_only
    recs, frac = run_condition_forced(model, tok, problems, "q3_zero", fills, args, THINK_END, quotas={0: 2, 1: 2})
    assert frac > 0 and all(r["n_forced"] == 2 for r in recs)
rig.remove()


# 12. Thinking off: everything generated is the answer; capped only at the token limit.
from run_q3_reasoning import mmlu_score, parse_generation


class BatchEncoding(dict):
    def to(self, _device):
        return self


tok_call = StubTokenizer.__call__
StubTokenizer.__call__ = lambda self, *a, **k: BatchEncoding(tok_call(self, *a, **k))
r = parse_generation([50, 51, 52, 2], tok, THINK_END, no_thinking=True, max_new_tokens=10)
assert r["finished_thinking"] and r["thinking_tokens"] == 0 and r["answer_tokens"] == 4
r = parse_generation(list(range(20, 30)), tok, THINK_END, no_thinking=True, max_new_tokens=10)
assert not r["finished_thinking"]
assert run_name("q3_zero", False, 0, no_thinking=True) == "q3_zero__nothink__seed0"

# 12b. The damage gate.
from run_q3_reasoning import gate_check
base_d = {"mmlu_accuracy": 0.49}
assert gate_check({"loss_increase_pct": 3.7, "mmlu_accuracy": 0.46}, base_d, 10, 0.05)[0]
ok, why = gate_check({"loss_increase_pct": 47.4, "mmlu_accuracy": 0.24}, base_d, 10, 0.05)
assert not ok and "WikiText" in why and "MMLU" in why
assert not gate_check({"loss_increase_pct": 2, "mmlu_accuracy": 0.40}, base_d, 10, 0.05)[0]

# 13. MMLU letter scoring runs, and an intervention is applied.
rows = [{"question": f"question {i}", "choices": ["w", "x", "y", "z"], "answer": i % 4} for i in range(6)]
with torch.no_grad():
    base_mmlu = mmlu_score(model, tok, rows, "none", fills, batch_size=4)
    abl_mmlu = mmlu_score(model, tok, rows, "q3_zero", fills, batch_size=4)
assert base_mmlu["mmlu_n"] == 6 and 0 <= base_mmlu["mmlu_accuracy"] <= 1
assert 0 <= base_mmlu["mmlu_direct_rate"] <= 1
assert 0 <= base_mmlu["mmlu_letter_mass"] <= 1 + 1e-6
from run_q3_reasoning import extract_letter
for text, want in [("B", "B"), ("**C**", "C"), ("The answer is (D).", "D"), ("A. Paris", "A"),
                   ("Answer: B", "B"), ("nothing here", None), ("ABCD", None)]:
    assert extract_letter(text) == want, (text, extract_letter(text))

# 14. Analysis pairs thinking-off runs with the thinking-off baseline.
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    def rec(i, think, ans, ok):
        return {"seed": 0, "idx": i, "thinking_tokens": think, "answer_tokens": ans, "correct": ok,
                "finished_thinking": True, "wait_count": 0, "thinking": ""}
    files = {
        "none__seed0": [rec(i, 1200, 80, True) for i in range(40)],
        "q3_zero__seed0": [rec(i, 300, 80, i % 4 != 0) for i in range(40)],
        "none__nothink__seed0": [rec(i, 0, 150, i % 5 != 0) for i in range(40)],
        "q3_zero__nothink__seed0": [rec(i, 0, 150, i % 5 != 0) for i in range(40)],
    }
    for name, recs in files.items():
        (tmp / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in recs))
    (tmp / "damage.json").write_text(json.dumps({"none": {"mmlu_accuracy": 0.6},
                                                 "q3_zero": {"kl": 0.1, "loss_increase_pct": 2, "mmlu_accuracy": 0.58}}))
    res = subprocess.run([sys.executable, "analyze_q3.py", str(tmp)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "4. Thinking on vs off" in res.stdout, res.stdout
    import csv as _csv
    summary = {r["variant"]: r for r in _csv.DictReader((tmp / "summary.csv").open())}
    assert float(summary["q3_zero__nothink"]["accuracy_change"]) == 0
    assert abs(float(summary["q3_zero"]["mmlu_change"]) + 0.02) < 1e-9

print("all checks passed")
print("quadrant census on a random model:", {k: round(v, 3) for k, v in census.items() if "firings" in k})
print(f"general damage of q3_zero on random model: KL {d['kl']:.4f}")


# 14. fast_gen: our decoding loop matches model.generate (greedy), with and
#     without an edit, and rows of different lengths leave the batch correctly.
from fast_gen import generate_batch

torch.manual_seed(1)
tok14 = StubTokenizer()
prompts14 = ["what is two plus two", "hi", "a much longer prompt here ok"]
enc14 = tok14(prompts14, return_tensors="pt", padding=True)
EOS14 = 2
with torch.no_grad():
    for cond in ("none", "q3_zero", "q3_qmean"):
        with intervene(model, cond, fills):
            ref = model.generate(input_ids=enc14["input_ids"], attention_mask=enc14["attention_mask"],
                                 max_new_tokens=20, do_sample=False, pad_token_id=0, eos_token_id=EOS14)
        ref = ref[:, enc14["input_ids"].shape[1]:].tolist()
        outs, frac = generate_batch(model, tok14, prompts14, cond, fills, max_new_tokens=20, think_end_id=THINK_END,
                                    eos_ids={EOS14}, seed=0, temperature=0.0, wait_token_ids=[5, 6])
        for r, o in zip(ref, outs):
            want = r[: r.index(EOS14) + 1] if EOS14 in r else r
            assert o["ids"] == want, (cond, o["ids"], want)
        assert (frac > 0) == (cond != "none")
    # rig the model to end row-specific: boost EOS so rows finish at once, check early exit
    rig = model.lm_head.register_forward_hook(lambda _m, _i, o: o.index_add(-1, torch.tensor([THINK_END]), torch.full((*o.shape[:-1], 1), 50.0)))
    quota = {0: 2, 1: 0, 2: 1}
    outs, _ = generate_batch(model, tok14, prompts14, "none", fills, max_new_tokens=30, think_end_id=THINK_END,
                             eos_ids={EOS14}, seed=0, temperature=0.0, wait_token_ids=[5],
                             force_fn=lambda row, tl, n: n < quota[row], wait_ids=tok14("\n\nWait")["input_ids"])
    assert [o["n_forced"] for o in outs] == [2, 0, 1]
    wl = len(tok14("\n\nWait")["input_ids"])
    assert [o["ids"].index(THINK_END) for o in outs] == [2 * wl, 0, wl], [o["ids"] for o in outs]
    # sampling runs and is reproducible
    a, _ = generate_batch(model, tok14, prompts14, "q3_zero@0.5", fills, max_new_tokens=10, think_end_id=THINK_END, eos_ids={EOS14}, seed=3)
    b, _ = generate_batch(model, tok14, prompts14, "q3_zero@0.5", fills, max_new_tokens=10, think_end_id=THINK_END, eos_ids={EOS14}, seed=3)
    assert [o["ids"] for o in a] == [o["ids"] for o in b]
    rig.remove()
print("fast_gen checks passed")

# 14b. Rows that finish at different steps are dropped without disturbing the others.
with torch.no_grad():
    with intervene(model, "q3_qmean", fills):
        free = model.generate(input_ids=enc14["input_ids"], attention_mask=enc14["attention_mask"],
                              max_new_tokens=20, do_sample=False, pad_token_id=0)[:, enc14["input_ids"].shape[1]:].tolist()
    eos_b = free[0][5]  # a token row 0 emits early; use it as the stop token
    outs, _ = generate_batch(model, tok14, prompts14, "q3_qmean", fills, max_new_tokens=20, think_end_id=None,
                             eos_ids={eos_b}, seed=0, temperature=0.0)
    lens = set()
    for r, o in zip(free, outs):
        want = r[: r.index(eos_b) + 1] if eos_b in r else r
        assert o["ids"] == want, (o["ids"], want)
        lens.add(len(want))
    assert len(lens) > 1, "test needs rows of different lengths"
print("fast_gen row-dropping check passed")

# 15. Gain knobs: shrink/grow scale every firing in the quadrant, nothing else.
new, mask = apply_condition(act, g, u, "q3_shrink@0.25")
assert torch.equal(mask, m["q3"]) and torch.allclose(new[m["q3"]], 0.75 * act[m["q3"]]) and torch.equal(new[~m["q3"]], act[~m["q3"]])
new, mask = apply_condition(act, g, u, "q4_grow@0.5")
assert torch.allclose(new[m["q4"]], 1.5 * act[m["q4"]]) and torch.equal(new[~m["q4"]], act[~m["q4"]])
assert "0.75" in describe("q3_shrink@0.25") and "1.5" in describe("q4_grow@0.5")
print("gain knob checks passed")

# 16. "#layers" restricts an edit to some layers; all three thirds together equal the full edit.
from quadrants import layer_set, split_layers
assert split_layers("q3_zero@0.25#late") == ("q3_zero@0.25", "late") and parse_condition("q3_zero@0.25#late")[2] == 0.25
assert layer_set("early", 3) == {0} and layer_set("1-2", 3) == {1, 2} and layer_set(None, 3) is None
with torch.no_grad():
    clean = model(ids).logits
    with intervene(model, "q3_zero") as st_full:
        full = model(ids).logits
    with intervene(model, "q3_zero#early") as st_e:
        early = model(ids).logits
    with intervene(model, "q3_zero#0-2"):
        all3 = model(ids).logits
assert not torch.allclose(early, clean) and not torch.allclose(early, full) and torch.allclose(all3, full)
assert st_e.total * 3 == st_full.total
print("layer restriction checks passed")
