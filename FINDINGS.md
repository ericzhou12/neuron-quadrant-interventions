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
