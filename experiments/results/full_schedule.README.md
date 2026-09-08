# Does the effect survive training to completion?

Every published model stops at **2,309 steps of a 4,000-step OneCycle schedule**, about 58% of
the way through, with the learning rate still falling. That is a real uncontrolled variable and
the README says so. This is the check.

It is **not** a test of the scaling non-monotonicity, and nothing here should be read as settling
that. Settling monotonicity needs all three sizes trained to completion, which is roughly 13 hours
of CPU. This asks the narrower question that does fit: **at the sizes we could afford, does
finishing the schedule change the result?**

## n=2,048, complete schedule

Same seed, same schedule, same protocol, same 5 pinned evaluation samples over 1,280 sequences.
The only difference is `--stop-at 4000` instead of `--stop-at 2309`.

| layer, xy | 2,309 steps (published) | 4,000 steps (complete) | change |
|---|---|---|---|
| 0 | 0.8028 | 0.7808 | −0.022 |
| 1 | 1.2215 | 1.2366 | +0.015 |
| **2, the effect layer** | **1.4705** | **1.4620** | **−0.0085** |
| 3 | 0.9654 | 0.9863 | +0.021 |
| final training loss | 0.1739 | 0.1705 | −0.0034 |

The n=2,048 step-matched run's final loss of 0.1739 was read from the training console; its log was not retained. The nearest shipped evidence is train_n2048_full4000.log at step 2300: 0.1741.

**The effect survives.** Layer 2 moves by 0.0085, which is 0.6% of the value and under two
half-widths of the five-sample spread (±0.0047). Every qualitative feature is preserved: layer 0
runs backwards, layers 1 and 2 are positive, layer 2 is the peak, layer 3 is flat.

Worth noting what the loss did: **0.1739 → 0.1705 across 1,691 additional steps.** The last 42% of
the schedule buys almost nothing on this task, and when this was the only completed run we offered
that as the likely reason the ratio barely moves. **The n=8,192 run below refutes that
explanation**: its loss moves just as little and its layer 2 moves nine times as far. Whatever
holds n=2,048 still, it is not simply that the extra steps taught it nothing.

## n=8,192, complete schedule

Same construction: same seed, same 4,000-step OneCycle schedule, same protocol, same 5 pinned
evaluation samples over 1,280 sequences, `--stop-at 4000` against the published `--stop-at 2309`.
23,761 s of laptop CPU, finishing 2026-09-08 03:48.

| layer, xy | 2,309 steps (published) | 4,000 steps (complete) | change | half-width |
|---|---|---|---|---|
| 0 | 0.7990 | 0.7857 | −0.0133 | ±0.0019 |
| 1 | 1.4052 | 1.2890 | −0.1162 | ±0.0046 |
| **2, the effect layer** | **2.1201** | **2.2000** | **+0.0799** | ±0.0086 |
| 3 | 0.9531 | 1.0399 | **+0.0868** | ±0.0052 |
| final training loss | 0.1727 | 0.1703 | −0.0024 | — |

**The effect survives, and it does not barely move.** Layer 2 stays the peak and stays strongly
positive, so the headline claim is not an artefact of stopping early at this size either. But
every layer moves by far more than the sampling spread: layer 2 by **about nine half-widths**,
layer 1 by twenty-five, layer 3 by seventeen. At n=2,048 the same comparison moved layer 2 by
under two. These two models do not behave the same way when you finish the schedule, and we
cannot say why.

Two specifics worth stating rather than smoothing over:

- **Layer 2 gets stronger, not weaker.** 2.1201 to 2.2000. If stopping early biased the published
  n=8,192 number, it biased it *downward*. That is the opposite of the direction that would
  embarrass the claim, which is exactly why it should be said explicitly rather than left for a
  reader to work out.
- **Layer 3 crosses 1.** 0.9531 to 1.0399, from marginally backwards to marginally forwards. It is
  still near-flat either way, and this is a control run rather than a member of the published
  family, so it does not change the "layer 3 is flat below 16,384" statement about the step-matched
  models. It does mean that statement is a fact about those models at 2,309 steps and not a
  property of the architecture at this size.

The loss moves 0.1727 to 0.1703, which is the same near-nothing the n=2,048 pair showed. So the
explanation we reached for there — the extra steps buy little, therefore the ratio holds still —
predicts a stationary ratio here and gets a moving one. We are publishing the refutation of our own
explanation alongside the result it came from.

## Correction, 2026-09-08: these moves were measured against the wrong error bar

Everything above compares a full-schedule run against its step-matched twin using the five-sample
half-width, and calls the n=8,192 move of 0.0799 "about nine half-widths" and therefore large.

A second training seed at n=2,048, run later the same day, moves layer 2 by **0.2514**
(`seed_variance.README.md`). The five-sample half-width measures how precisely we measured one
trained model. It does not measure how much another run of the same configuration would differ,
and that is the comparison every number in this file is making.

Against the only between-run figure we have, **0.0799 is a third of a seed gap, and 0.0085 at
n=2,048 is a thirtieth of one.** So the honest reading of this file changes:

- **What stands.** Both completed models still show the effect, layer 2 well above 1 in both, and
  finishing the schedule does not destroy it. That was the question this file was built to ask.
- **What does not.** "The n=8,192 layer 2 moves nine half-widths and that is a finding" does not
  stand. Neither does the contrast drawn between n=2,048 barely moving and n=8,192 moving a lot,
  because both moves are inside the range a reseed produces at a single size.
- **What was already right, by luck rather than design.** The paragraph above arguing that the
  loss barely moves in both cases and therefore cannot explain the difference between them: there
  may be no difference between them to explain.

The measurements are unchanged and the CSV is untouched. What changed is the size of the ruler we
were holding them against.

## What this does and does not license

**Licenses:** saying the effect is not an artefact of stopping early, at n=2,048.

**Licenses, after the n=8,192 run:** saying the same at n=8,192, and saying that how much the
ratio moves when a model finishes its schedule is itself size-dependent, on two sizes.

**Does not license:** any claim about monotonicity, which needs all three sizes and a reason rather
than two points; any claim about n=16,384, which does not fit before the deadline and is not
attempted; and any explanation of *why* the two sizes differ, which we do not have.

**The published headline stays the step-matched family.** These runs are an additional control,
kept in a separate file so nothing can mix them into `measured.csv`, exactly as the two-family
confusion earlier in this project taught us to do.

Data in `full_schedule.csv`. Checkpoints `checkpoints/bdh_n2048_full4000.pt` and
`checkpoints/bdh_n8192_full4000.pt`. Reproduce with
`python sparsity_scan.py --embd 64 --mult 32 --budget 36000 --steps 4000 --stop-at 4000` and
`python sparsity_scan.py --embd 128 --mult 64 --budget 64800 --steps 4000 --stop-at 4000`.
