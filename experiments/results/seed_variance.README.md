# A second seed, and what it costs us

Until 2026-09-08 every model in this project was one training run. One seed per size, three sizes,
and a scaling curve drawn through three points with no idea how much a point moves if you simply
train it again. The error bars we quoted were the spread across five pinned **evaluation** samples
of one trained model, which measures sampling noise in the measurement and says nothing at all
about variation between training runs.

This is the second seed. It was affordable: 56 minutes of laptop CPU at n=2,048.

## The run

Identical to the published n=2,048 model in every respect except `--seed 1338` instead of 1337:
same architecture, same 4,000-step OneCycle schedule stopped at the same 2,309 steps, same
optimiser, same task, same evaluation of five pinned samples over 1,280 sequences. A seed change
redraws the weight initialisation, the 13-letter warm-up sequence and the training words, which is
what an independent replication of this experiment actually is.

Both runs pass the precondition. `TASK_LEARNED=True`, repetition loss 0.0090 and 0.0037.

## The result, layer by layer, mem/rep on the xy tensor

| layer | seed 1337, published | seed 1338 | difference |
|---|---|---|---|
| 0 | 0.8028 [0.8003 .. 0.8044] | 0.8542 [0.8531 .. 0.8552] | +0.0514 |
| 1 | 1.2215 [1.2198 .. 1.2226] | 1.1930 [1.1917 .. 1.1952] | &minus;0.0285 |
| **2, the effect layer** | **1.4705** [1.4647 .. 1.4740] | **1.7219** [1.7124 .. 1.7294] | **+0.2514** |
| 3 | 0.9654 [0.9603 .. 0.9695] | 1.1847 [1.1791 .. 1.1899] | +0.2193 |

## What this costs us

**The scaling non-monotonicity is no longer established, and we are retracting the strength of
that claim.**

| quantity | value |
|---|---|
| the published scaling curve, one seed each | 1.4705, 2.1201, 1.8742 at n = 2k, 8k, 16k |
| the "dip", the largest gap between adjacent sizes (8k to 16k) | **0.2459** |
| the gap between two seeds at n=2,048 alone | **0.2514** |
| the within-seed five-sample half-width we had been quoting | 0.0047 |

**Changing the seed at one size moves layer 2 further than the entire dip between two sizes.**
The README said the dip was "40× the mean half-width". That is true and it was the wrong error
bar: it is the spread of five evaluation samples drawn from one trained model, so it measures how
precisely we measured that model, not how much another model of the same size would differ. Against
the only between-run number we now have, the dip is **1.0×**, not 40×.

So the honest statement about scaling is: with one seed per size, **no size-to-size difference in
this project is distinguishable from seed variation.** Not the dip, not the rise from 2k to 8k,
none of it. Three points drawn from three single runs do not make a curve.

## What survives

The claim itself does. Layer 2 is well above 1 in both seeds, 1.4705 and 1.7219, and the second
seed is the *stronger* of the two. Layer 0 runs backwards in both, 0.8028 and 0.8542. Layer 1 is
positive in both, 1.2215 and 1.1930. **The signature is not a property of one lucky initialisation**,
which is the thing a second seed was most likely to destroy and did not.

## What else it breaks

**"Layer 3 is flat at n=2,048" does not survive a seed change.** It reads 0.9654 in the published
model and 1.1847 in this one, crossing from below 1 to clearly above it. The limits table said layer
3 was flat at 2k and 8k and weakly positive at 16k, and presented that as a property of depth. On
two seeds at one size it is not stable enough to be called a property of anything.

## What would settle it

Three seeds at each of three sizes, nine runs. At the measured rates that is roughly 30 hours of
CPU for the two smaller sizes and considerably more for n=16,384, so it was never in scope for this
deadline.

**A second seed at n=16,384 was started and then killed at step 1,250 of 2,309.** It was running at
11.6 s/step and would have landed at 15:25 on the submission day, which left no time to review what
it changed. A run stopped early is not step-matched to anything, and this project has already
learned once that a partial run is worse than no run because it invites a comparison it cannot
support, so it was discarded rather than reported. Nothing from it appears in any CSV.

So the comparison in this file rests on **one** between-seed difference, at one size. That is enough to show
the five-sample half-width is the wrong ruler, which is what it is used for here. It is not a
variance estimate and this file does not present it as one.

**The right way to read the three published ratios is as three single draws, each measured
precisely, from a distribution whose width we have now sampled exactly once and found to be wide.**

Data in `seed_variance.csv` and `seed_variance_correlations.csv`. Checkpoint
`checkpoints/bdh_n2048_seed1338.pt`. Log `stepmatched/train_n2048_seed1338.log`. Reproduce with
`python sparsity_scan.py --embd 64 --mult 32 --steps 4000 --stop-at 2309 --seed 1338` then
`python measure.py --ckpt checkpoints/bdh_n2048_seed1338.pt --embd 64 --mult 32 --repeats 5`.
