# Findings

Results to date, newest observations last within each topic. Project context and hypotheses (H0-H3) are in README.md.

## Qwen3-1.7B (unless noted)

**Sparsity sweep** (keep top-k neurons per token by distance from mean, mean-ablate the rest):
KL at 92.2% ablated is ~0.95 (0.6B) and ~0.80 (1.7B) vs Goodfire's 0.140 at
top-320. Same knee shape, ~6x more damage. Can't attribute the gap (MoE vs
scale vs selection rule vs eval text). Honest status: a supporting section,
not a headline; overlaps known activation-sparsity work (TEAL, CATS).
Possible strengthening: calibrate on Qwen 3.6 35B-A3B at 8-bit; add OLMoE-1B-7B
(MoE) and Qwen3-4B (dense).

**Quadrant census:** Q3 ~40% and Q4 ~40% of firings (~15% of |activation|
each); gate-open Q1+Q2 ~20% of firings but most of the activation. Check
`census.json` for exact numbers.

**Full-quadrant ablations break the model** (thinking off, GSM8K baseline 0.83):
`q3_zero` loss +47%, `q4_zero` +59%, `q3_mean` +91% (answers 1 token),
`random_matched` +304%; all GSM8K 0.00. `q3_qmean` only +3.7%, GSM8K 0.77.
So Q3 mostly acts as a steady contribution of about its usual size (supports
H3 for general language ability); token-specific variation matters less.
Note: with thinking off, `q3_mean` stops immediately while `q3_zero` runs to
the cap, so the fill value can push length in opposite directions. Worth
remembering when interpreting Goodfire's mean-ablation result.

**Damage scan (WikiText loss increase / KL):**

| dose | Q3 zero | Q4 zero |
|---|---|---|
| 5% | +0.2% / 0.023 | +0.5% / 0.025 |
| 10% | +0.6% / 0.045 | +1.1% / 0.052 |
| 25% | +2.5% / 0.133 | +3.4% / 0.162 |
| 50% | +8.0% / 0.379 | +10.2% / 0.517 |
| qmean | +3.7% / 0.251 | +4.5% / 0.278 |

Q3 and Q4 damage is nearly identical at every dose, so the control is well matched.

**First hint of a Q3-specific effect on reasoning (2026-11-12, one seed, thinking off).**
In the MMLU check (prompt says "answer with the letter only", 16-token reply),
the intact model ignores the instruction and starts working the problem ~40% of
the time. Removing Q3 firings makes it answer directly more often, rising with
dose; removing the same share of Q4 firings (matched damage) does not:

| dose | Q3 zero: replies with a letter in 16 tokens | Q4 zero |
|---|---|---|
| 0 | 60% | 60% |
| 5% | 63% | 59% |
| 10% | 66% | 59% |
| 25% | 69% | 58% |
| 50% | 75% | 59% |
| qmean | 73% | 82% |

Points in the direction of "Q3 pushes the model to reason rather than answer".
Caveats: one seed, n=300, thinking off, a crude proxy for reasoning. And the
qmean row is not Q3-specific (Q4 qmean moves it even more), so whatever drives
it there isn't unique to Q3.

**MMLU knowledge, fixed scoring (v4, 2026-11-12).** Scored by the probability
of A/B/C/D after a pre-filled "The answer is (" (letter mass 1.00 everywhere,
so the score is clean). Intact 0.503. n=300, so one standard error is ~0.03.

| dose | Q3 zero | Q4 zero |
|---|---|---|
| 5% | 0.507 | 0.537 |
| 10% | 0.527 | 0.513 |
| 25% | 0.530 | 0.537 |
| 50% | 0.453 | 0.480 |
| qmean | 0.500 | 0.497 |

Knowledge is essentially untouched up to 25% for both quadrants. So the
direct-answer shift above is a change in *behaviour* (whether the model reasons
before answering), not in what it knows. That makes the Q3-vs-Q4 difference in
the direct-answer rate more interesting, not less. `q4_zero@0.5` fails the gate
on WikiText (+10.2%), so the matched dose pair for generation is @0.25.

**Put-back edits, general damage (2026-11-12, thinking off).** Whole quadrant
edited. WikiText loss / MMLU (v4) / direct-answer rate, Q3 vs Q4:

| edit | Q3 | Q4 |
|---|---|---|
| qmean (neuron's typical value, at the tokens that fired) | +3.7% / 0.50 / 73% | +4.5% / 0.50 / 82% |
| resample (neuron's own values, shuffled across tokens) | +6.3% / 0.47 / 72% | +6.6% / 0.45 / 75% |
| const (remove; add mean contribution to every token) | +9.7% / 0.40 / 83% | +12.3% / 0.45 / 86% |
| pool (values from any neuron in the layer) | +17.5% / 0.36 / 39% | +27.5% / 0.34 / 53% |
| zero | +47% | +59% |

Reading (general language, not reasoning): the leaks are not just noise
(against H1 for ordinary text). Each step that throws away more structure costs
more: per-neuron level at the right tokens (qmean) < shuffled timing
(resample) < no token info at all (const) < no neuron identity (pool). So
*which tokens* leak and *how much each neuron leaks* both carry information;
the exact token-by-token value carries less. Q3 and Q4 look alike on all of this.
Every flattening/shuffling edit raises the direct-answer rate in BOTH
quadrants, so that is not Q3-specific; only removing firings (zero@p) separates
Q3 from Q4 so far. Gate: q3_resample passes; q3_const fails MMLU (0.40) while
passing WikiText (+9.7%); q4_resample is borderline (MMLU drop 0.053); pool fails.

**Put-back edits at a 25% dose (2026-11-13, thinking off).** Same edits, but
applied to a random 25% of the quadrant's firings, so damage is small enough
for every one of them to pass the gate. WikiText loss / MMLU / direct-answer
rate (intact: MMLU 0.503, direct 60%):

| edit @0.25 | Q3 | Q4 |
|---|---|---|
| qmean | +0.5% / 0.54 / 59% | +0.7% / 0.54 / 65% |
| resample | +0.6% / 0.52 / 61% | +0.9% / 0.50 / 62% |
| const | +1.2% / 0.56 / 66% | +1.4% / 0.54 / 70% |
| zero | +2.5% / 0.53 / 69% | +3.4% / 0.54 / 58% |

Same damage ordering as the whole-quadrant ladder. The direct-answer rate
moves for Q3 only when firings are *removed* (zero: 69%); putting a typical or
shuffled value back (qmean, resample) leaves it at the intact 60%. So the
Q3-specific push toward reasoning seems to depend on the leak being *present
at about its usual size*, not on its exact token-by-token value (H3-like),
on this crude thinking-off proxy. `const` raises the rate in both quadrants
(66% / 70%), so that shift is not Q3-specific. One seed, n=300.

Figures: `results/q3__Qwen__Qwen3-1.7B/figures/fig1_damage_dose.png` (dose
scan: matched damage, Q3-only behaviour shift) and `fig2_putback_ladder.png`
(damage ladder, whole quadrant and @0.25). Regenerate with `python make_figures.py`.

**Infrastructure note (2026-11-13).** The first thinking-on run used
`model.generate` with batches of 8 and managed 56/200 baseline problems in two
hours (every batch waits for its slowest trace). `fast_gen.py` now drops
finished rows and saves after every batch. Seed 1 runs on the fast loop; the
legacy seed-0 run was left going.

## Thinking ON (the main experiment), Qwen3-1.7B

GSM8K test problems 0-199, sampling at Qwen's thinking settings (T=0.6,
top-p 0.95, top-k 20), cap 4,096 new tokens. Each edited run is compared with
the intact model on the same problems and seed. Sampling noise is NOT paired
(the text diverges after the first different token), so per-problem ratios are
noisy; trust the n=200 medians and their bootstrap intervals.

**Baseline (2026-11-13).** Seed 1: accuracy 0.84, median thinking 1,365
tokens, 11% of problems hit the cap, 7.5 "wait"s per trace on average. The
capped traces I read are genuine dithering ("But ... Wait ... But maybe"),
not repetition loops. Seed 0 baseline is saved too (`none__seed0.jsonl`).

**`q3_qmean` does NOT shorten reasoning; it lengthens it a little
(2026-11-13, seed 1, n=200).** Replacing every Q3 firing with that neuron's
typical Q3 value:

| | intact | q3_qmean |
|---|---|---|
| accuracy | 0.840 | 0.830 (13 lost, 11 gained: no difference) |
| median thinking tokens | 1,365 | 1,700 |
| median per-problem length ratio | 1 | 1.14 [95% CI 1.08, 1.19] |
| cap hits | 11% | 12% |
| "wait"s per trace | 7.5 | 13.0 |

Reading: Q3's token-by-token variation is not needed to keep the model
reasoning (against the simple H2 story "the variation is a keep-going
signal": erasing it should then shorten reasoning). If anything, flattening
it makes the model second-guess itself more.

**...and `q4_qmean` does exactly the same, so this is not Q3-specific
(2026-11-13, seed 1, n=200).**

| | intact | q3_qmean | q4_qmean |
|---|---|---|---|
| accuracy | 0.840 | 0.830 | 0.820 (14 lost, 10 gained) |
| median thinking tokens | 1,365 | 1,700 | 1,744 |
| length ratio vs intact | 1 | 1.14 [1.08, 1.19] | 1.13 [1.05, 1.20] |
| cap hits | 11% | 12% | 15% |
| "wait"s per trace | 7.5 | 13.0 | 11.4 |

Q3 edit vs Q4 edit on the same problems: length ratio 1.00 [0.97, 1.06],
wait-count difference +1.6 [-1.0, +5.8]. No difference.

Reading: flattening either kind of leak to its typical value makes the model
reason ~13% longer with more "wait"s, at no accuracy cost. The simplest
account is mild general perturbation (both edits cost ~4% WikiText loss) that
makes the model a bit less sure of itself, i.e. closer to H0/H1 than to a
Q3-specific signal. So for qmean: H2 gets no support (erasing the variation
does not shorten reasoning), and nothing here separates Q3 from Q4.
The `zero@0.25` pair (the edit that DID separate Q3 from Q4 on the
direct-answer proxy) is the remaining test. `python compare_runs.py <folder>`
reproduces these tables.

**Removing 25% of Q3 firings makes reasoning LONGER, and more so than
removing 25% of Q4 (2026-11-13, seed 1, n=200).**

| | intact | q3_zero@0.25 | q4_zero@0.25 |
|---|---|---|---|
| accuracy | 0.840 | 0.800 (15 lost, 7 gained) | 0.835 (11 lost, 10 gained) |
| median thinking tokens | 1,365 | 2,156 | 1,734 |
| length ratio vs intact (median of per-problem ratios) | 1 | 1.38 [1.25, 1.46] | 1.16 [1.10, 1.28] |
| cap hits | 11% | 17% | 11.5% |
| "wait"s per trace | 7.5 | 10.8 | 10.6 |
| WikiText loss (general damage) | 0 | +2.5% | +3.4% |

Q3 edit vs Q4 edit directly, same problems: median length ratio 1.14
[1.08, 1.21]; geometric-mean ratio 1.12 [1.05, 1.19].

Reading, with caveats:
- Direction is the OPPOSITE of Goodfire's report (ablating Q3 cut reasoning
  from 1,550 to 176 tokens in their MoE model). In this small dense model,
  at a dose that leaves the model working, taking Q3 away never shortened
  reasoning in any of four edits; every edit lengthened it by 13-38%.
- There is a modest Q3-specific part: zeroing Q3 lengthens reasoning ~12-14%
  more than zeroing the same share of Q4, although the Q4 edit does slightly
  MORE general damage. So general damage (H0) does not explain the gap.
  One seed so far; seed 0 is running as a replication.
- Accuracy does not move significantly in any condition (largest: q3_zero@0.25,
  15 lost vs 7 gained, exact McNemar p ~ 0.13). So there is no accuracy drop
  to explain by "shorter reasoning" here; the second half of the north-star
  question does not arise at these doses.
- Odd fit with the thinking-off proxy: there, removing Q3 made the model answer
  directly MORE often (looked like "Q3 pushes toward reasoning"); with thinking
  on, removing Q3 makes it reason LONGER. These are different behaviours
  (whether to start vs when to stop), so not a contradiction, but the simple
  slogan "Q3 = reason more" is wrong for this model.
- Possible reconciliation with Goodfire, to test next: their fill. With
  thinking off, `q3_mean` (their mean-ablation: the neuron's all-token mean,
  which injects large gate-open-sized values) stopped after 1 token while
  `q3_zero` ran to the cap. The shortening may come from what is put IN, not
  from Q3 being taken OUT. Test: `q3_mean@p` vs `q4_mean@p` at doses that pass
  the gate.

Figure: `figures/fig3_generation.png`.

## Teacher-forced probe (intact model's own traces fed back under each edit)

**The `</think>` stop hazard is the wrong target (2026-11-13).** 171 of 178
finished intact traces end with a "**Final Answer**" block, after which
`</think>` is nearly certain (log P ~ 0) and almost impossible anywhere else.
The real decision to stop is the decision to START that block. The first
probe's `stop_hazard` numbers are ratios of near-zero values; ignore them.
`probe_stop.py` now also records `final_hazard` (the urge to start the Final
Answer block early); results from that rerun (tag `v2`) are below when done.

**The urge to say "Wait" follows the SIZE of the gate-closed leaks, in both
directions, for Q3 and Q4 alike (2026-11-13, 70 traces, exactly paired).**
Change in summed P("Wait") along the trace, as a log ratio vs intact
(+0.18 = about +20%), with 95% bootstrap intervals; KL = damage on the
reasoning text.

| edit | Q3: KL | Q3: wait | Q4: KL | Q4: wait |
|---|---|---|---|---|
| grow@0.5 (all leaks x1.5) | 0.167 | -0.12 [-0.19, -0.06] | 0.145 | -0.17 [-0.25, -0.10] |
| grow@0.25 (x1.25) | 0.046 | -0.09 [-0.13, -0.05] | 0.044 | -0.12 [-0.17, -0.08] |
| shrink@0.25 (x0.75) | 0.064 | +0.18 [+0.14, +0.23] | 0.061 | +0.25 [+0.21, +0.29] |
| shrink@0.5 (x0.5) | 0.286 | +0.58 [+0.49, +0.66] | 0.270 | +0.76 [+0.67, +0.84] |
| zero@0.25 (random 25% removed) | 0.104 | +0.23 [+0.18, +0.28] | 0.103 | +0.29 [+0.23, +0.34] |
| zero@0.5 | 0.360 | +0.59 [+0.49, +0.69] | 0.358 | +0.88 [+0.78, +0.98] |
| qmean | 0.159 | +0.40 [+0.34, +0.47] | 0.141 | +0.23 [+0.18, +0.27] |
| resample | 0.295 | +0.49 | 0.273 | +0.37 |
| const | 0.422 | +1.05 | 0.434 | +0.86 |
| pool | 0.708 | +1.19 | 0.748 | +1.35 |

Reading:
- Making the leaks smaller raises the urge to second-guess; making them
  LARGER lowers it, even though growing also damages the model (KL 0.05-0.17).
  A sign that flips with the direction of the edit cannot be "any damage makes
  the model dither" (H0/H1). The leak's overall size acts like a dial on
  second-guessing: an H3-style steady-level effect.
- `zero@0.25` and `shrink@0.25` remove the same average amount and give about
  the same result (+0.23 vs +0.18 for Q3), so it is mostly the amount removed,
  with the random per-firing noise adding a little.
- It is NOT Q3-specific. Q4 edits move "Wait" at least as much as Q3 edits
  (shrink and zero: Q4 more, by 0.07-0.3 log units). So this probe does not
  explain why generation showed q3_zero@0.25 lengthening reasoning more than
  q4_zero@0.25; that gap, if real, must come from somewhere else (e.g. the
  urge to conclude, tested in probe v2).
- Edits that scramble structure (const, pool, resample, qmean) also raise
  "Wait", roughly in step with their damage, so for those the damage account
  is not excluded.
- Fits generation: every removal-type edit lengthened reasoning and raised the
  wait count.

Files: `probe_stop__main.jsonl` (per trace), `probe_summary__main.json/.txt`;
`python analyze_probe.py <folder> [tag]` regenerates the summary.

**The urge to CONCLUDE moves in opposite directions for Q3 and Q4
(2026-11-13, probe v2, 36 traces, exactly paired; the most interesting
result so far).** "Conclude early" = summed P(start the "**Final Answer**"
block) at all positions before the intact model really started it, as a log
ratio vs intact. "At true end" = change in log P(start it) at the position
where the intact model did. 95% bootstrap intervals over traces.

| edit | KL (Q3 / Q4) | Q3: conclude early | Q3: at true end | Q4: conclude early | Q4: at true end |
|---|---|---|---|---|---|
| shrink@0.25 (x0.75) | 0.061 / 0.060 | -0.57 [-0.93, -0.21] | -0.25 [-0.41, -0.12] | +0.66 [+0.41, +0.93] | +0.06 [+0.02, +0.11] |
| zero@0.25 | 0.099 / 0.098 | -0.32 [-0.67, +0.02] | -0.36 [-0.63, -0.15] | +0.86 [+0.54, +1.21] | +0.06 [+0.02, +0.12] |
| zero@0.5 | 0.339 / 0.344 | -0.86 [-1.50, -0.24] | -1.36 [-2.10, -0.76] | +1.30 [+0.73, +1.93] | +0.02 [-0.05, +0.09] |
| grow@0.25 (x1.25) | 0.045 / 0.044 | +0.51 [+0.29, +0.74] | +0.04 [+0.01, +0.08] | -0.90 [-1.26, -0.55] | -0.54 [-0.85, -0.28] |
| qmean | 0.155 / 0.139 | -0.94 [-1.47, -0.45] | -0.72 [-1.10, -0.40] | -0.82 [-1.31, -0.34] | -0.73 [-1.11, -0.37] |
| mean@0.1 (Goodfire-style fill) | 0.203 / 0.444 | +1.58 [+1.00, +2.20] | +0.00 | +0.79 [+0.03, +1.58] | -0.80 [-1.29, -0.40] |

Q3 minus Q4 on the same traces (conclude early): shrink@0.25 -1.22
[-1.66, -0.78]; zero@0.25 -1.18 [-1.66, -0.72]; zero@0.5 -2.16 [-2.94, -1.40];
grow@0.25 +1.41 [+0.93, +1.88]; qmean -0.13 [-0.40, +0.16].

Reading:
- Q3 leaks (small positive) push the model TOWARD wrapping up; Q4 leaks
  (small negative) push AGAINST it. Shrinking Q3 lowers the urge to conclude
  and growing Q3 raises it; for Q4 it is the mirror image. Four sign flips
  (quadrant x direction) at matched damage: this is not general damage (H0)
  and not noise (H1). It is a level effect (H3-like): the plain gain knob,
  which keeps every token-by-token pattern and only rescales, is enough.
- Put together with the "Wait" result (both quadrants: smaller leaks = more
  "Wait"), this explains the generation result: removing Q3 = more "Wait" AND
  less urge to conclude = clearly longer reasoning (x1.38); removing Q4 = more
  "Wait" but MORE urge to conclude = the two partly cancel (x1.16).
- So in this model Q3 is, if anything, a "you can stop now" push, the opposite
  role to the one suggested by Goodfire's ablation result.
- The Goodfire-style fill (`q3_mean@0.1`: replace 10% of Q3 firings with the
  neuron's all-token mean) RAISES the urge to conclude strongly (+1.58) and
  leaves "Wait" unchanged. That is what you would expect if the fill value is
  mostly larger than the leak it replaces (it acts like `grow`). Prediction,
  being tested in generation: `q3_grow@0.25` and `q3_mean@0.1` SHORTEN
  reasoning; `q4_grow@0.25` lengthens it. If so, Goodfire-style shortening can
  be produced here by what the fill puts in, not by taking Q3 out.
- qmean lowers the urge to conclude equally for both quadrants (and raised
  "Wait" for both), matching its equal x1.13 lengthening in generation.
- Layers (zero@0.5 in one third of the stack; "Wait" / conclude-early):
  Q3 early +0.08 / -0.71, mid +0.44 / -0.33, late -0.02 / +0.01;
  Q4 early +0.48 / +0.89, mid +0.21 / +0.53, late +0.02 / +0.08.
  The last third of layers does nothing for either quadrant (KL 0.003); the
  effects live in the first two thirds.

Caveats: 36 traces from one seed (a replication on seed-0 traces is running);
the probe measures the local urge along the intact model's path, not the
length the edited model would produce; "conclude early" sums small
probabilities, so it is noisier than "at true end" (both are shown).
Files: `probe_stop__v2.jsonl`, `probe_summary__v2.json/.txt`.

**Scope correction (2026-11-13, raised by the owner).** Everything above with
thinking on uses PARTIAL edits (25% of firings, or qmean) because full Q3
ablation breaks this model (WikiText +47%) and the damage gate skipped it.
Goodfire ablated ALL Q3 firings. So the statements above are about doses that
leave the model working; they do not show what full Q3 ablation does to
reasoning here, and they are not a replication (or refutation) of Goodfire's
test. The direct replication is now running with the gate overridden on
purpose: full `q3_mean` (their fill) and `q3_zero`, with `q4_mean` / `q4_zero`
as equally-damaging controls, thinking on, 50 problems, seed 1
(`logs/full_ablation_seed1.log`). Read those results next to their damage
numbers: a Q3-only shortening would replicate Goodfire; the same change under
Q4 would point to damage.

**The opposite-sign conclude effect replicates on different traces, with a
dose-response (2026-11-13, probe `v3seed0`, 43 traces from the seed-0
baseline; the run was stopped early at 43 of 60 to free memory).**
Conclude early, log ratio vs intact [95% CI]:

| edit | Q3 | Q4 | Q3 minus Q4 |
|---|---|---|---|
| shrink@0.5 (x0.5) | -0.63 [-1.02, -0.27] | +1.35 [+0.84, +1.90] | -1.98 [-2.67, -1.34] |
| shrink@0.25 (x0.75) | -0.38 [-0.66, -0.14] | +0.76 [+0.47, +1.08] | -1.14 [-1.59, -0.76] |
| zero@0.25 | -0.28 [-0.53, -0.06] | +0.66 [+0.40, +0.96] | -0.94 [-1.37, -0.56] |
| grow@0.25 (x1.25) | +0.50 [+0.29, +0.73] | -0.65 [-0.93, -0.41] | +1.15 [+0.77, +1.58] |
| grow@0.5 (x1.5) | +1.10 [+0.73, +1.53] | -1.12 [-1.58, -0.70] | +2.22 [+1.57, +2.95] |

Every sign matches probe v2 (36 seed-1 traces), and the effect roughly
doubles when the dose doubles, in both directions and both quadrants. "Wait"
also replicates: shrink raises it (Q3 +0.18 / +0.54, Q4 +0.24 / +0.71 at
0.25 / 0.5), grow lowers it (Q3 -0.07 / -0.09, Q4 -0.11 / -0.16).
Goodfire-style fill: `q3_mean@0.05` conclude early +0.85 [+0.37, +1.39],
`q3_mean@0.1` +1.39 [+0.81, +2.04], "Wait" unchanged; but `q4_mean` also
raises conclude-early (+0.80 at 0.05) while doing 2.4x the damage, so the
mean fill is not a clean Q3-vs-Q4 contrast.
Still a probe result: whether generated reasoning actually gets shorter under
`q3_grow` is not yet tested (that run was stopped before saving anything).
Files: `probe_stop__v3seed0.jsonl`, `probe_summary__v3seed0.json/.txt`.

## Full ablation, thinking ON (direct replication of Goodfire's test; gate overridden)

Seed 1, first 50 GSM8K problems, intact accuracy on these 0.80, intact median
thinking 1,361 tokens. All of these edits break the model on ordinary text, so
read every row next to its damage.

**Goodfire-style mean fill (2026-11-13).**

| | WikiText loss | MMLU | median thinking tokens | what the output looks like | accuracy |
|---|---|---|---|---|---|
| `q3_mean` (all Q3 firings -> neuron's all-token mean) | +91% | 0.24 (chance) | 0 | stops at once: 39 of 50 emit end-of-turn as the very first token, the rest 1-8 tokens of newlines / `</think>` | 0.00 |
| `q4_mean` (same for Q4) | +42% | 0.26 (chance) | 4,096 (cap) | starts the problem, then loops ("So 16 is 16. So 16 is 16. ...") until the cap; 48 of 50 never close thinking | 0.00 |

Reading:
- Yes, Goodfire-style full Q3 ablation "cuts reasoning" here too, all the way
  to zero tokens. But the model is not reasoning briefly and answering: it
  emits nothing and gets 0% right. In this model the effect is a broken model
  that stops immediately, not shorter reasoning.
- The Q4 control is equally broken (chance MMLU) but fails in the OPPOSITE
  direction: it cannot stop. So "damage" alone (H0) does not predict the
  direction; which quadrant gets the fill does. Q3-fill = stop, Q4-fill =
  never stop. That matches the probe: pushing the Q3 side up (grow, mean fill)
  raises the urge to conclude, pushing the Q4 side up lowers it.
- This means a length drop after mean-ablating Q3 cannot by itself be read as
  "Q3 firings drive reasoning". Here the same drop comes from an edit that
  adds a strong stop push and destroys the model. Whether Goodfire's MoE
  model was still working after their ablation is exactly the missing piece
  (they report no accuracy for that test).
- Not comparable in one respect: Goodfire's 176 tokens is short but non-zero,
  ours is zero. Their model evidently tolerates the edit better.
Files: `q3_mean__seed1.jsonl`, `q4_mean__seed1.jsonl` (50 problems each).
Full `q3_zero` / `q4_zero` with thinking on are running.

**Full zeroing, thinking on (2026-11-13, seed 1; `q3_zero` 50 problems,
`q4_zero` stopped after its first 25 because every row ran to the cap and a
batch took 48 minutes).**

| | WikiText loss | thinking tokens | what the output looks like | accuracy |
|---|---|---|---|---|
| `q3_zero` | +47% | 22 of 50 at 0-4 tokens, 28 of 50 at the 4,096 cap | either closes `<think>` at once and rambles in the answer to the cap, or loops inside thinking ("Okay, so maybe I need to think about this. Let me try.") | 0.00 |
| `q4_zero` (first 25) | +59% | 13 of 25 at 0-2 tokens, 12 of 25 at the cap | the same two failure modes | 0.00 |

On the same first 25 problems `q3_zero` gives 12 near-zero and 13 capped: no
difference from `q4_zero`. So fully zeroing either quadrant just breaks the
model in the same bimodal way (skip thinking or loop forever); nothing
Q3-specific. Only the mean FILL separates the quadrants (Q3-fill stops at
once, Q4-fill never stops).

**Where this leaves the Goodfire comparison.** In Qwen3-1.7B:
1. Full Q3 ablation can produce "reasoning drops to ~0", with either fill, but
   always with 0% accuracy and a model that is broken on ordinary text.
2. With zero fill the same thing happens for Q4, so it is damage, not Q3.
3. With mean fill the direction IS quadrant-specific (Q3 stop, Q4 runaway),
   which fits the probe's "Q3 side pushes toward concluding".
4. At doses that leave the model working, removing Q3 lengthens reasoning.
So we do not reproduce "Q3 firings are what keeps the model reasoning" as a
property of a working model. What we can say about Goodfire's own model is
limited: it may tolerate full Q3 ablation far better than a dense 1.7B model
does (their output was 176 tokens, not 0). The checkable question for them is
whether accuracy and a Q4-with-the-same-fill control were measured.
Files: `q3_zero__seed1.jsonl`, `q4_zero__first25__seed1.jsonl`.

Figures for the two sections above: `figures/fig5_probe_conclude.png` (leak
size as a dial: the urge to conclude rises with Q3 size and falls with Q4
size, an X; "Wait" falls with both; hollow points are the replication on
separate traces) and `figures/fig6_full_ablation.png` (every problem as a dot
for each full-quadrant edit). Detail visible in fig6: in the zero-fill rows
the problems with ~0 thinking tokens are not short answers; 19 of Q3's 22 and
all 13 of Q4's 13 then loop in the ANSWER up to the cap. So full zeroing is
"loop somewhere" for both quadrants, and only Q3 mean-fill actually stops.

**Timing: leak activity before "Wait" and before the Final Answer block
(2026-11-13, intact model, 85 seed-1 traces; correlational).** Per-token Q3 /
Q4 share of neurons and summed activation ("mass"; Q4 mass is negative, so
more negative = larger), z-scored within each trace, at the token before the
event, minus positions in the same trace with the same preceding token but an
ordinary next token.

| | before "Wait" (305 events) | before "**Final Answer**" (77 events) |
|---|---|---|
| Q3 mass, middle layers | +0.30 [+0.21, +0.39] | +0.75 [+0.66, +0.83] |
| Q4 mass, middle layers | -0.26 [-0.34, -0.17] | -0.59 [-0.67, -0.51] |
| Q3 mass, late layers | +0.46 [+0.34, +0.58] | +1.24 [+1.14, +1.34] |
| Q4 mass, late layers | -0.50 [-0.60, -0.38] | -1.27 [-1.37, -1.16] |
| Q3 / Q4 share, late layers | -0.30 / -0.31 | -0.98 / -0.77 |

Reading: before BOTH kinds of decision point the leaks get larger in the
middle and late layers (fewer neurons leaking in late layers, but more total
leak), and Q3 and Q4 move together, mirror images of each other. H2 predicted
something Q3-specific and directional (Q3 activity falling before the model
stops, rising before "Wait"). That is not what happens: Q3 rises before both,
and Q4 does the same with the opposite sign. So timing gives no support to a
Q3-specific "keep going" signal; it looks like a general signature of "a
decision is being made here". It is also only a correlation, and the late
layers, where it is strongest, had no causal effect in the probe (KL 0.003).
Note `analyze_timing.py` now uses the start of the Final Answer block as the
stop event (`</think>` has no usable controls). Full table:
`timing_summary__main.txt`.

## Growing the leaks, thinking ON: the first edit that SHORTENS reasoning in a working model

**`q3_grow@0.25` (every Q3 firing x1.25) shortens reasoning by ~20% with no
accuracy change; the Q4 match does much less (2026-11-13, seed 1, n=200).**
This was predicted in advance from the probe (growing Q3 raises the urge to
conclude).

| | intact | q3_grow@0.25 | q4_grow@0.25 |
|---|---|---|---|
| accuracy | 0.840 | 0.835 (8 lost, 7 gained) | 0.820 (12 lost, 8 gained) |
| median thinking tokens | 1,365 | 962 | 1,226 |
| length ratio vs intact | 1 | 0.80 [0.76, 0.86] | 0.93 [0.83, 0.99] |
| cap hits | 11% | 8.5% | 11% |
| "wait"s per trace | 7.5 | 7.0 | 6.1 |
| WikiText loss / MMLU | 0 / 0.503 | +0.2% / 0.517 | -0.1% / 0.513 |
| direct-answer rate (thinking off) | 60% | 50% | 67% |

Q3 edit vs Q4 edit on the same problems: length ratio 0.84 [0.80, 0.93].

Reading:
- Q3 size now moves real reasoning length in BOTH directions in a working
  model: 25% removed -> x1.38, 25% larger -> x0.80. Q4: x1.16 and x0.93. The
  Q3-vs-Q4 contrast is in the predicted direction both times (Q3 removal
  lengthens more; Q3 growth shortens more).
- The mild Q4 shortening fits the probe too: growing Q4 lowers the urge to
  conclude but also lowers "Wait"; the two roughly cancel.
- North-star question, second half: reasoning got 20% shorter and accuracy
  did not move (0.835 vs 0.840). So in this model a moderate cut in reasoning
  length does not by itself cost accuracy. (GSM8K is easy for this model; much
  of the intact model's extra length is re-checking. A harder task could differ.)
- The edit is tiny by general measures (WikiText +0.2%, MMLU unchanged), so
  this is not damage.
- The thinking-off proxy moves the opposite way again (Q3 larger -> answers
  directly LESS often, 50% vs 60%), as it did for removal. So Q3 seems to do
  two different things: it raises the tendency to start working a problem
  out, and, once reasoning, raises the tendency to wrap up. Worth one direct
  test later; for now just noted.
- One seed; seed 0 and a x1.5 dose are queued.
A trace read at random (idx 80) looks like normal, tidy reasoning that checks
once and stops; not truncated or degenerate.

**Where the length change happens: after the answer is first reached, not
before (2026-11-13, seed 1, finished traces only).** Each trace is split at the
first place the model's final answer appears in its thinking: "solve" (up to
there) and "check" (everything after: re-deriving, second-guessing). The
intact model spends about two thirds of its thinking in the check phase
(median solve 372 tokens, check 798). Median paired ratio vs intact [95% CI]:

| edit | solve phase | check phase |
|---|---|---|
| q3_grow@0.25 | 0.89 [0.86, 0.94] | 0.74 [0.66, 0.80] |
| q4_grow@0.25 | 0.90 [0.87, 0.96] | 0.89 [0.82, 0.95] |
| q3_zero@0.25 | 1.17 [1.10, 1.29] | 1.43 [1.33, 1.60] |
| q4_zero@0.25 | 1.20 [1.10, 1.27] | 1.19 [1.09, 1.29] |
| q3_qmean | 1.07 [1.00, 1.14] | 1.21 [1.12, 1.33] |
| q4_qmean | 1.05 [1.01, 1.11] | 1.19 [1.08, 1.28] |

Reading:
- The solve phase changes by the same amount for Q3 and Q4 edits (grow ~0.9,
  zero ~1.2). The Q3-specific part is entirely in the check phase: growing Q3
  cuts re-checking to 0.74 (Q4: 0.89), removing Q3 stretches it to 1.43
  (Q4: 1.19).
- So "Q3 pushes toward concluding" acts where you would expect: once the
  model already has its answer, Q3 size sets how long it keeps re-checking
  before committing.
- This also explains why accuracy does not move: the answer is usually
  already there before the part that gets longer or shorter.
Caveats: the split is a text heuristic (first occurrence of the final number,
which can occasionally be a coincidence, e.g. a number from the question);
traces that hit the cap or never state their answer are excluded (12-20% per
condition), so runaway reasoning is not described here.
`python analyze_phases.py <folder>`; output `phases_summary__seed1.json/.txt`.

**A mild dose of the Goodfire-style fill does not shorten reasoning
(2026-11-13, `q3_mean@0.1`, seed 1, n=200; Q4 control below).**
Replacing a random 10% of Q3 firings with the neuron's all-token mean:
accuracy 0.820 vs 0.840 (14 lost, 10 gained), median thinking 1,214 vs 1,365,
median per-problem length ratio 0.99 [0.91, 1.00], cap hits 10% vs 11%,
"wait"s 6.4 vs 7.5. General damage: WikiText +1.1%, MMLU 0.487.
So at a dose that leaves the model working, the fill barely moves reasoning
length; the collapse to zero tokens only appears at the full dose, where the
model is broken. The probe over-predicted here: it showed a large rise in the
urge to conclude EARLY (+1.4 to +1.6) but no change at the true end, and
generation sided with the latter. Lesson: of the two probe measures, "at true
end" tracked real length better than "conclude early" for this edit.
**The Q4 control for that edit hurts accuracy much more, without changing
length (2026-11-13, `q4_mean@0.1`, seed 1, n=200).**

| edit | accuracy (lost/gained vs intact 0.840) | median thinking | length ratio [95% CI] | cap hits | WikiText loss | MMLU |
|---|---|---|---|---|---|---|
| `q3_mean@0.1` | 0.820 (14/10) | 1,214 | 0.99 [0.91, 1.00] | 10% | +1.1% | 0.487 |
| `q4_mean@0.1` | 0.625 (52/9)  | 1,506 | 1.00 [0.92, 1.10] | 16% | +2.6% | 0.507 |

Q3 edit vs Q4 edit on the same problems: length ratio 0.94 [0.81, 1.00],
"wait"s -2.1 [-3.2, -1.1].

What this shows and doesn't:
- Neither mild mean-fill changes typical reasoning length. So the opposite
  full-dose behaviours (Q3 stops at once, Q4 loops to the cap) do not show up
  as a length shift at 10%; at most there is a hint in the same direction
  (Q4 has more cap hits and more "wait"s), and the paired CI touches 1.
- The Q4 edit costs 21 points of GSM8K accuracy while passing the damage gate
  (WikiText +2.6%, MMLU unchanged). Of the 52 problems it loses, 28 finish
  with a wrong number and 24 never finish. Reading the traces, the wrong ones
  are confused arithmetic that the model then talks itself into ("190 is not
  110 ... so the answer is 80"), not truncated reasoning.
- This is an accuracy drop with NO shortening, which bears on the second
  north-star question: here accuracy fell for a reason other than shorter
  reasoning. It also shows the damage gate misses harm to multi-step
  reasoning: WikiText and one-step MMLU looked fine.
- Caveat: one seed, and "mean" for a Q4 firing replaces a small negative value
  with the neuron's all-token mean, which is not the same size of change as
  for Q3. So this is not a clean "Q4 matters more for accuracy" claim.

**Seed-0 replication of the `zero@0.25` pair (2026-11-13, n=200 each).**

| seed | edit | accuracy (intact) lost/gained | median thinking | length ratio [95% CI] | cap hits (intact) |
|---|---|---|---|---|---|
| 1 | `q3_zero@0.25` | 0.800 (0.840) 15/7  | 2,156 | 1.38 [1.25, 1.46] | 17% (11%) |
| 1 | `q4_zero@0.25` | 0.835 (0.840) 11/10 | 1,734 | 1.16 [1.10, 1.28] | 11.5% (11%) |
| 0 | `q3_zero@0.25` | 0.795 (0.835) 14/6  | 1,952 | 1.27 [1.17, 1.35] | 16% (14%) |
| 0 | `q4_zero@0.25` | 0.815 (0.835) 11/7  | 1,687 | 1.18 [1.11, 1.24] | 14% (14%) |

Q3 edit vs Q4 edit, same problems: seed 1 ratio 1.14 [1.08, 1.21]; seed 0
ratio 1.07 [1.01, 1.17].

- Replicates in direction: zeroing a quarter of either leak quadrant lengthens
  reasoning, and Q3 lengthens it more than Q4 on both seeds.
- The Q3-specific part is smaller on seed 0 (7% vs 14%) and its CI only just
  excludes 1. Treat "Q3 more than Q4" as a small effect, roughly 5-15%; most
  of the lengthening (about 16-18%) is shared by both quadrants.
- Accuracy: Q3 loses about 4 points on both seeds (15/7 and 14/6 lost/gained;
  neither is significant alone by McNemar), Q4 0.5-2 points. Longer reasoning
  did not buy accuracy.

**Growing Q3 shortens reasoning in a dose-dependent way, with no loss of
accuracy (2026-11-13, `grow@0.5` pair, seed 1, n=200 each).** This is the
clearest result so far for both north-star questions.

| edit | accuracy (intact 0.840) lost/gained | median thinking (intact 1,365) | length ratio [95% CI] | cap hits (intact 11%) | WikiText loss | MMLU |
|---|---|---|---|---|---|---|
| `q3_grow@0.25` | 0.835  8/7  |   962 | 0.80 [0.76, 0.86] | 8.5% | +0.2% | 0.517 |
| `q3_grow@0.5`  | 0.855  9/12 |   690 | 0.55 [0.51, 0.61] | 3%   | +1.1% | 0.507 |
| `q4_grow@0.25` | 0.820 12/8  | 1,226 | 0.93 [0.83, 0.99] | 11%  | -0.1% | 0.513 |
| `q4_grow@0.5`  | 0.825 10/7  | 1,090 | 0.86 [0.78, 0.92] | 11.5%| +1.5% | 0.520 |

Q3 edit vs Q4 edit, same problems: `grow@0.25` 0.84 [0.80, 0.93];
`grow@0.5` 0.65 [0.61, 0.69].

By phase (`analyze_phases.py`; solve = up to the first time the final number
appears, check = after it):

| edit | solve ratio | check ratio |
|---|---|---|
| `q3_grow@0.25` | 0.89 [0.86, 0.94] | 0.74 [0.66, 0.80] |
| `q3_grow@0.5`  | 0.75 [0.72, 0.80] | 0.43 [0.41, 0.47] |
| `q4_grow@0.25` | 0.90 [0.87, 0.96] | 0.89 [0.82, 0.95] |
| `q4_grow@0.5`  | 0.85 [0.80, 0.90] | 0.80 [0.70, 0.89] |

- Dose-response: making every Q3 leak 25% larger cuts thinking by about 20%;
  50% larger cuts it by about 45%. The Q4 control shortens far less (7%, 14%),
  so most of this is specific to Q3. General damage is negligible for both.
- Where the tokens go: mostly the re-checking after the answer is first
  reached (check phase down 57% at `q3_grow@0.5`), plus a smaller cut to the
  solving itself (25%). No trace has near-zero thinking; the model still
  solves, checks once and stops. It also gets stuck far less (cap hits 3% vs
  11%).
- Accuracy does not fall: 0.855 vs 0.840 (9 lost, 12 gained; not different).
  So in this model, reasoning made ~45% shorter through Q3 costs nothing on
  GSM8K. On the question "does accuracy drop BECAUSE reasoning gets shorter":
  here it got much shorter and accuracy did not drop. Together with
  `q4_mean@0.1` (accuracy -21 points with no length change), length and
  accuracy come apart in both directions.
- Fits the "Q3 is a wrap-up push" reading from the probe (larger Q3 raises
  the urge to conclude) and the opposite sign of `q3_zero@0.25` (+27-38%).
- Limits: one seed for `grow@0.5`; GSM8K is easy for this model, so the
  re-checking may be redundant here and matter on harder problems; the growth
  direction is the opposite of Goodfire's ablation, so this supports "Q3 size
  controls reasoning length" but not their specific collapse.

**Seed-0 replication of the `grow@0.25` pair (2026-11-14, n=200 each).**

| seed | edit | accuracy (intact) lost/gained | median thinking | length ratio [95% CI] | Q3 vs Q4, same problems |
|---|---|---|---|---|---|
| 1 | `q3_grow@0.25` | 0.835 (0.840) 8/7  |   962 | 0.80 [0.76, 0.86] | 0.84 [0.80, 0.93] |
| 1 | `q4_grow@0.25` | 0.820 (0.840) 12/8 | 1,226 | 0.93 [0.83, 0.99] | |
| 0 | `q3_grow@0.25` | 0.840 (0.835) 6/7  |   966 | 0.79 [0.72, 0.85] | 0.86 [0.82, 0.92] |
| 0 | `q4_grow@0.25` | 0.860 (0.835) 9/14 | 1,176 | 0.89 [0.79, 0.97] | |

The shortening replicates almost exactly (0.80 and 0.79), as does the
Q3-over-Q4 part (0.84 and 0.86), and accuracy is unchanged on both seeds.
This is the most stable effect in the project so far.

**Doubling the leaks breaks the model (2026-11-14, `grow@1.0`, gate only).**
`q3_grow@1.0`: WikiText loss +11.7%, MMLU 0.423 (intact 0.503).
`q4_grow@1.0`: WikiText loss +10.5%, MMLU 0.450. Both fail the damage gate
(limits: +10% loss, 0.05 MMLU drop), so GSM8K was not run, per the working
agreement. The usable range for the grow knob on this model therefore ends
somewhere between +50% (loss +1.1%) and +100%.

**At `grow@0.75` reasoning shortens further and accuracy now falls, but the
fall is not well explained by the shortening (2026-11-14, seed 1, n=200).**

| edit | accuracy (intact 0.840) lost/gained | median thinking | length ratio [95% CI] | solve / check ratio | WikiText loss | MMLU (intact 0.503) |
|---|---|---|---|---|---|---|
| `q3_grow@0.25` | 0.835  8/7  |   962 | 0.80 [0.76, 0.86] | 0.89 / 0.74 | +0.2% | 0.517 |
| `q3_grow@0.5`  | 0.855  9/12 |   690 | 0.55 [0.51, 0.61] | 0.75 / 0.43 | +1.1% | 0.507 |
| `q3_grow@0.75` | 0.695 40/11 |   475 | 0.40 [0.36, 0.43] | 0.60 / 0.30 | +4.1% | 0.473 |
| `q4_grow@0.25` | 0.820 12/8  | 1,226 | 0.93 [0.83, 0.99] | 0.90 / 0.89 | -0.1% | 0.513 |
| `q4_grow@0.5`  | 0.825 10/7  | 1,090 | 0.86 [0.78, 0.92] | 0.85 / 0.80 | +1.5% | 0.520 |
| `q4_grow@0.75` | 0.750 26/8  | 1,092 | 0.82 [0.75, 0.88] | 0.75 / 0.79 | +4.4% | 0.507 |

Q3 vs Q4 on the same problems at `grow@0.75`: length ratio 0.48 [0.43, 0.54].

- Length: the Q3 dose-response continues smoothly (0.80, 0.55, 0.40) while Q4
  flattens (0.93, 0.86, 0.82). The Q3-specific shortening keeps growing.
- Accuracy: both fall at this dose. Q3 -14.5 points (McNemar p = 0.0001),
  Q4 -9 points (p = 0.003). Q3 vs Q4 directly: 28 vs 17 discordant problems,
  p = 0.14, so the extra Q3 loss is NOT established.
- Is the Q3 loss caused by the shortening? Three reasons to doubt it:
  1. Q4 loses 9 points while shortening only 18%, so most of the drop at this
     dose looks like general harm to multi-step reasoning (the gate measures,
     +4%, MMLU, do not capture it, as with `q4_mean@0.1`).
  2. Within `q3_grow@0.75`, the problems it loses were shortened LESS than the
     ones it keeps (median length ratio 0.55 for the 37 lost-but-finished vs
     0.38 for the 128 kept).
  3. At `q3_grow@0.5` reasoning was already 45% shorter with no accuracy cost.
- Reading the lost traces: confident wrong set-ups, checked briefly and
  accepted ("16 - 3 = 13 ... Yes, that seems correct", ignoring the muffins).
  The loss is concentrated in mid-difficulty problems (middle third by intact
  reasoning length: 0.95 -> 0.67 for Q3, 0.74 for Q4).
- These are correlational. The direct test is forced continuation: make
  `q3_grow@0.75` keep thinking to the intact length and see whether accuracy
  returns, against the intact model forced the same number of times.

**Queue (one job at a time).** Next: `q3_grow@0.75` with forced continuation
to the intact length (seed 1), then the intact model forced the same number
of times (control); then the `shrink@0.5` pair.