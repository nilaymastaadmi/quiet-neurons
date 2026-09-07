# The provenance ladder: pre-registration

**Written and committed 2026-09-07 at ~14:35 IST, while the K=16 and K=256 runs were still
training and before any of their output existed.** Check the git history: this file is committed
in advance of `pool_ladder.csv`. That is the point of it. A prediction written after the numbers
are known is worth nothing, and the honest way to make one count is to make it falsifiable in
public first.

## Why a ladder rather than another single control

The `--fixed-word` control moved the 8-letter word from the context into the weights and the
warm-up-to-repetition gap collapsed from 2.354× to 1.035×. That is the predicted direction, but
it is one degenerate point: final loss 0.0004, `TASK_LEARNED=False`, and the activation level
collapsed to one constant across every layer, tensor and block. When a condition is that flat,
its ratios are ratios of numbers the manipulation forced equal, not evidence about the variable
under study. It cannot separate provenance from task difficulty, and no amount of argument about
the shape of the shift rescues it.

So graduate the intervention instead of defending it. Draw the word each sequence from a fixed
pool of **K** words:

- **K = 1** is `--fixed-word`: the whole word is in the weights, nothing to read from context.
- **K = 16** and **K = 256**: the letter content is in the weights, but the model must still read
  the context to know *which* of the K it is looking at. The copy task stays alive.
- **K = ∞** is the base task: every word is new, all of it comes from the context.

K = 256 requires the context to supply log2(256) = 8 bits of identity while the letters
themselves are already learned. That is the interesting middle.

## Prediction

**warm/rep at layer 2, n=2048, rises monotonically with K:**

> 1.035 (K=1)  <  r(16)  <  r(256)  <  2.354 (K=∞)

Final loss also rises with K, so **monotonicity on its own does not remove the difficulty
confound** and we are not claiming it would.

## The discriminating test

Plot warm/rep against final loss for the four points.

- **If task difficulty is the only driver**, all four fall on one line between (0.0004, 1.035)
  and (0.174, 2.354).
- **If provenance carries signal independent of difficulty**, **K=256 sits above that line**: the
  context still has to supply the identity while the letter content is already in the weights, so
  it should recover most of the gap at a fraction of the loss.

## Precondition

Every condition must pass the degeneracy check that `measure.py` now prints: **CV above 5%**
across the four layers, per tensor, per block. A condition flatter than that has collapsed to one
activation level and its ratio is uninformative, exactly as K=1 was. A degenerate point is
reported as degenerate and excluded from the trend, not quietly plotted.

## What we will say if the prediction fails

If the four points lie on the line, the write-up will say: **"provenance and difficulty are not
separable in this design, and we now have four points showing it rather than one."** That is a
registered negative and it stands as the result. It will not be re-framed, re-cut, or explained
away after the fact, and this paragraph exists so that promise is on the record before the data
is.

## Runs

```
python sparsity_scan.py --embd 64 --mult 32 --budget 21600 --stop-at 2309 --word-pool 16
python sparsity_scan.py --embd 64 --mult 32 --budget 21600 --stop-at 2309 --word-pool 256
```

Same seed, same 4,000-step OneCycle schedule, same 2,309-step stop as every published model.
Measured with `--repeats 5` into `results/pool_ladder.csv`, never into `measured.csv`. The pool
travels inside the checkpoint, so each model is evaluated on the distribution it was trained on
rather than on freshly drawn words it has never seen.
