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

**The effect survives.** Layer 2 moves by 0.0085, which is 0.6% of the value and under two
half-widths of the five-sample spread (±0.0047). Every qualitative feature is preserved: layer 0
runs backwards, layers 1 and 2 are positive, layer 2 is the peak, layer 3 is flat.

Worth noting what the loss did: **0.1739 → 0.1705 across 1,691 additional steps.** The last 42% of
the schedule buys almost nothing on this task, which is the most likely reason the ratio barely
moves. That also means the "none of them finished" limitation was probably never costing much,
though we could not have known that without checking.

## What this does and does not license

**Licenses:** saying the effect is not an artefact of stopping early, at n=2,048.

**Does not license:** any claim about monotonicity, or about n=8,192 and n=16,384 until they are
measured the same way. A completed n=8,192 run is in progress; n=16,384 does not fit before the
deadline and is not attempted.

**The published headline stays the step-matched family.** These runs are an additional control,
kept in a separate file so nothing can mix them into `measured.csv`, exactly as the two-family
confusion earlier in this project taught us to do.

Data in `full_schedule.csv`. Checkpoint `checkpoints/bdh_n2048_full4000.pt`. Reproduce with
`python sparsity_scan.py --embd 64 --mult 32 --budget 36000 --steps 4000 --stop-at 4000`.
