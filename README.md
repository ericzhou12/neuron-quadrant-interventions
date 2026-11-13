# neuron-sparsity

Experiments building on Goodfire's [Neurons as an Interpretable Basis](https://www.goodfire.com/research/neurons-as-an-interpretable-basis),
run on small dense models that fit on a laptop. Goodfire studied one
mixture-of-experts model, Qwen 3.6 35B-A3B.

## Status

Work in progress. Results so far, with their caveats, are in
[FINDINGS.md](FINDINGS.md); figures are in
`results/q3__Qwen__Qwen3-1.7B/figures/`. Short version (Qwen3-1.7B, partial
edits only, one to two seeds): removing part of Q3 lengthened reasoning rather
than shortening it, with no accuracy change. A direct replication of
Goodfire's full ablation is still running, so treat nothing here as final.

## Layout

| file | what it does |
|---|---|
| `quadrants.py` | hooks on gate/up pre-activations, quadrant masks, all edits (`zero`, `mean`, `qmean`, `shrink`, `grow`, put-backs, layer restriction) |
| `run_q3_reasoning.py` | damage gate (WikiText, MMLU) and GSM8K generation, thinking on/off, forced continuation |
| `fast_gen.py` | decoding loop that drops finished rows and resumes interrupted runs |
| `probe_stop.py` | teacher-forced probe: replays the intact model's reasoning under each edit and measures the urge to say "Wait" or to conclude |
| `analyze_q3.py`, `compare_runs.py`, `analyze_probe.py`, `analyze_timing.py` | paired statistics |
| `make_figures.py`, `plot_q3.py` | figures |
| `neuron_sparsity.py`, `run_sweep.py`, `plot_results.py` | earlier top-k sparsity sweep |
| `test_quadrants.py`, `test_smoke.py` | offline checks on a tiny random model |
| `results/` | per-problem records (`.jsonl`, with full text), damage scans, probe output, figures; `results/old/` holds superseded runs |

## Setup

```bash
cd neuron-sparsity
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python test_quadrants.py      # offline checks, seconds
```

The first real run downloads the model, GSM8K and WikiText from Hugging Face.

## The main project: Q3 firings and reasoning length

Each neuron computes `SiLU(gate) * up`. A **Q3 firing** is one where gate and
up are both negative, so the output is a small positive leak through SiLU's
negative tail. Goodfire found that ablating Q3 firings cut reasoning on 24
GSM8K problems from 1,550 to 176 tokens, reported no accuracy numbers for that
test, and left the mechanism to future work.

**Stage 0 asks two questions:**
1. Does ablating Q3 shorten reasoning in a small dense model too?
2. Does accuracy drop, and is that because reasoning got shorter?

### Hypotheses

If editing Q3 changes reasoning length, why? Four explanations, not mutually
exclusive (expect partial support for more than one):

- **H0, artifact.** The edit doesn't touch anything reasoning-specific; it
  just damages the model, and shorter (or runaway) output is a side effect of
  a broken model. *Predicts:* large general damage (WikiText loss, MMLU), and
  the matched Q4 edit has the same effect. *Tested by:* the damage gate,
  `q4_*` controls, thinking-off runs.
- **H1, useful noise.** The leaks' content doesn't matter; small random-ish
  perturbations occasionally knock the model off a confident path, so it
  second-guesses itself ("wait") and keeps reasoning. *Predicts:* random
  values of the right size restore long reasoning. *Tested by (later stage):*
  pooled resample of Q3 values (keeps sizes, destroys which-neuron and which-token).
- **H2, token-specific signal.** The leaks carry information that varies
  token by token (something like "not done yet" or "uncertain here"), spread
  thinly across many neurons, which the model reads as a reason to continue.
  *Predicts:* `q3_qmean` (keeps the average leak, erases its token-by-token
  variation) shortens reasoning even though it barely hurts general ability;
  shuffles that break neuron identity or timing also fail to restore it.
  *Tested by:* `q3_qmean` vs `none`, later the shuffle put-backs and the
  timing analysis (Q3 activity drops before `</think>`, rises before "Wait").
- **H3, steady background bias.** The leaks act as a roughly constant push,
  the same on every token; the model needs them present at about their usual
  size, but their variation doesn't matter. *Predicts:* `q3_qmean` leaves
  reasoning length unchanged while `q3_zero@p` shortens it; adding back a
  constant per-layer vector restores it; Q3 activity over time is flat.
  *Tested by:* `q3_qmean` vs `q3_zero@p`, later the constant-vector put-back.

### What the first thinking-off runs showed (Qwen3-1.7B)

- Q3 is 40% of all firings in this dense model (Q4 another 40%), so ablating
  all of Q3 is a huge edit.
- Fully zeroing or mean-filling Q3 or Q4 **breaks the model**: WikiText loss
  +47% to +91%, MMLU at chance, GSM8K 0%. Any reasoning result from those
  conditions would just reflect a broken model.
- Replacing Q3 with each neuron's *typical Q3 value* (`q3_qmean`) barely hurts:
  loss +3.7%, GSM8K 0.83 → 0.77 with thinking off.

So every condition now goes through a **damage gate** first. If it raises
WikiText loss by more than 10% (`--max-loss-increase`) or drops MMLU by more
than 0.05 (`--max-mmlu-drop`), its GSM8K runs are skipped. `--allow-broken`
overrides this.

### Run

```bash
# 1. Find doses that don't break the model (minutes, no GSM8K)
python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --damage-only \
    --conditions q3_qmean,q4_qmean,q3_zero@0.05,q3_zero@0.1,q3_zero@0.25,q4_zero@0.05,q4_zero@0.1,q4_zero@0.25

# 2. Thinking on, the main experiment (hours; finished runs are skipped on restart).
#    Add the largest q3_zero/q4_zero doses that passed step 1, e.g. q3_zero@0.1,q4_zero@0.1
python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --n-problems 200 --conditions none,q3_qmean,q4_qmean

# 3. Thinking off, same conditions (cheap)
python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --n-problems 200 --conditions none,q3_qmean,q4_qmean \
    --no-thinking --max-new-tokens 1024

# 4. Only if step 2 shows an effect: forced continuation, then the dose-matched control
python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --n-problems 200 --conditions q3_qmean --force-to-baseline
python run_q3_reasoning.py --model Qwen/Qwen3-1.7B --n-problems 200 --conditions none --force-like q3_qmean__forced

# 5. Second seed before trusting anything (--seed 1), and --gen-only to rule out misreading the question

# Analyse everything in the folder
python analyze_q3.py results/q3__Qwen__Qwen3-1.7B
```

### Conditions

Names are `<quadrant>_<fill>`, optionally with `@<dose>` to edit only a random
share of that quadrant's firings.

| fill | meaning |
|---|---|
| `zero` | set to 0 |
| `mean` | the neuron's mean over all tokens (Goodfire-style; injects large values) |
| `qmean` | the neuron's typical value in that quadrant: keeps the average contribution, removes token-by-token detail |

| condition | role |
|---|---|
| `none` | baseline |
| `q3_qmean` | **main test**: does Q3's token-by-token detail matter for reasoning length? |
| `q4_qmean` | **matched control**: Q4 has the same share of firings (40%) and activation (15%) as Q3 |
| `q3_zero@p`, `q4_zero@p` | remove a share `p` of Q3 / Q4 firings, at doses that pass the gate |
| `random_matched` | not recommended: it hits large gate-open firings, so it isn't size-matched |

If `q3_qmean` changes reasoning length and `q4_qmean` doesn't, Q3 matters
specifically. If both do, it's any gate-closed leak.

### Thinking off

`--no-thinking` has the model answer directly. If accuracy drops with thinking
on but not off, the loss acts through reasoning; if it drops in both, Q3 also
helps compute answers directly. Answer length is recorded too: Qwen3 sometimes
shows a few lines of working even with thinking off, so check that answers
don't shrink under ablation.

### MMLU

Each condition is also scored on 300 MMLU questions (`--mmlu-questions`) with
thinking off, by which of A/B/C/D the model rates most likely: one forward
pass per question, no generation. It sits next to WikiText KL as a verifiable
check that the model isn't broadly broken. `mmlu_letter_mass` should stay near
1; if it falls, the model stopped answering in letter format.

### Forced continuation

`--force-to-baseline`: whenever the model tries to end its thinking, `</think>`
is replaced with "Wait" and it keeps going, until its reasoning is as long as
the intact model's on the same problem (at most `--max-forced` times).

`--force-like VARIANT`: forces the intact model the same number of times per
problem, because appending "Wait" can improve accuracy on its own (the s1
paper's finding). Only the ablated model's gain *beyond* that counts.

- Accuracy comes back: shorter reasoning explains the accuracy drop.
- It doesn't: Q3 contributes something besides length. Check the repetition
  score first; if the forced text is looping, the extra reasoning was junk and
  the test is inconclusive.

### What the analysis reports

`analyze_q3.py` compares every run with `none` on the same problems and seeds:

1. **Length**: median thinking as a ratio to baseline, with a 95% bootstrap
   interval, for all problems and for long-reasoning ones (baseline over 1,000
   tokens, as Goodfire did). Also general damage: KL divergence on WikiText
   and the MMLU accuracy change. If a condition damages these heavily, shorter
   reasoning may just be a broken model.
2. **Accuracy**: change from baseline with an exact McNemar test, split by how
   much each problem's reasoning shrank.
3. **Forced continuation**: share of lost accuracy recovered, gain beyond the
   intact control, and repetition.
4. **Thinking on vs off**: accuracy change in each mode, side by side.

It also saves `survival.png`: the share of runs still thinking after N tokens.

### Before believing a result

- Read 10 traces per condition in the `.jsonl` files. How reasoning ends
  (confident answer, abrupt stop, gibberish) tells you a lot.
- Check `cap_hit_rate`. Medians are robust to cap hits; means are not.
- Write down your predictions before running.

## Files

- `quadrants.py`: captures gate/up pre-activations, edits firings by quadrant, measures general damage.
- `run_q3_reasoning.py`: runs the reasoning experiment.
- `analyze_q3.py`: paired statistics, tables and the survival plot.
- `test_quadrants.py`: offline checks.
- `neuron_sparsity.py`, `run_sweep.py`, `plot_results.py`, `test_smoke.py`: the earlier
  sparsity sweep (a sanity check; mostly repeats known activation-sparsity results).

## Later stages

- Put-backs to test noise against signal: per-neuron resample, magnitude-matched
  shuffle across neurons, pooled resample, constant vector at 1/2/5x.
- Partial doses of Q3 ablation.
- Timing: Q3 activity around "Wait", against line breaks not followed by "Wait".
- Which layers carry the effect.
