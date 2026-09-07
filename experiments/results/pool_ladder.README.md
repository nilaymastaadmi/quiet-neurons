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

---

# THE RESULT

Measured 2026-09-07, after the pre-registration above was committed. Both intermediate models
trained 2,309 steps of the same 4,000-step schedule as every published model, and both pass the
precondition: `TASK_LEARNED=True`, and neither trips the degeneracy check.

| condition | letters in weights | final loss | warm% | rep% | **warm/rep** | CV (xy) |
|---|---|---|---|---|---|---|
| K=1, fixed word | 8 | 0.0004 | 21.71 | 20.98 | **1.035** | 2.08% **collapsed** |
| K=16 | 128 | 0.0188 | 12.93 | 14.01 | **0.923** | 15.58% |
| K=256 | 2,048 | 0.0384 | 10.47 | 4.13 | **2.536** | 27.16% |
| K=∞, base task | 0 | 0.1739 | 13.61 | 5.78 | **2.354** | 33.34% |

## 1. The monotonicity prediction FAILED

We predicted `1.035 < r(16) < r(256) < 2.354`. What we got is `1.035, 0.923, 2.536, 2.354`, which
is out of order at both intermediate points. K=16 falls *below* K=1, and K=256 rises *above* the
base task. **The registered prediction was wrong and we are reporting it as wrong.**

## 2. The discriminating test: positive at K=256, negative at K=16, and sensitive to the axis

This is the test the pre-registration said would separate provenance from task difficulty, and it
is the reason the ladder was run at all.

The registration drew a straight line in (final loss, warm/rep) through K=1 and the base task and
asked whether K=256 sits above it. It does: the line predicts **1.324** at K=256's loss and the
model measures **2.536**, a residual of **+1.212**. **K=16 sits below the same line**: 0.923
against a predicted 1.175.

The linear-in-loss axis was our choice and it matters. On a **log-loss** axis K=256's residual
falls to **+0.51** and K=16's grows to **−0.95**, which makes K=16 the larger anomaly rather than
the smaller one. A test whose verdict depends on the axis is not a clean separation.

So the defensible statement is that the effect is **not a monotone function of task difficulty**:
K=256 is 4.5× easier than the base task by final loss and shows a larger effect, which no
monotone dependence on difficulty produces. **Difficulty is not ruled out as a driver**, and no
argument about the geometry rescues K=16.

## 3. K=16 is unexplained, and we are not going to pretend otherwise

At K=16 every layer sits between 0.90 and 1.01, with warm ≈ mem ≈ rep at layer 2 (0.129 / 0.142 /
0.140). The effect is not merely weaker there, it is absent, and at layers 1, 2 and 3 it inverts
slightly. Its CV of 15.58% keeps it clear of the degeneracy threshold, so it is not the K=1
collapse in a milder form by that test, but it is the flattest non-degenerate condition we have.

The reading we find most plausible, and cannot demonstrate: 128 letters is little enough that the
model can serve the repeat block largely from weights without leaning on context, so neither block
is context-held and the distinction has nothing to separate. At 2,048 letters it must use the
context to identify which of 256 words it is seeing, and the distinction returns. **That is a
story, not a measurement.** Settling it needs a K sweep we do not have time for.

## What this does and does not license

**It licenses**: saying the effect survives an intervention that holds the task recognisably the
same while moving where the word's content lives, and that difficulty alone cannot generate it.

**It does not license**: calling the mechanism established, or calling the relationship monotone
in provenance. One of four points is unexplained and the ordering we predicted did not hold.

Data in `pool_ladder.csv`. Checkpoints `checkpoints/bdh_n2048_pool16.pt` and `checkpoints/bdh_n2048_pool256.pt`, each
carrying its own word pool so it can only be re-measured in distribution. Reproduce with
`python tools/ladder.py`.
