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

| condition | letters in weights | first-exposure surprise | share of word already in weights | final loss | **warm/rep** | **mem/rep** | CV (xy) |
|---|---|---|---|---|---|---|---|
| K=1, fixed word | 8 | 0.0004 | 99.99% | 0.0004 | **1.035** | **1.004** | 2.08% **collapsed** |
| K=16 | 128 | 0.3532 | **89.2%** | 0.0188 | **0.923** | **1.013** | 15.58% |
| K=256 | 2,048 | 0.7309 | 77.7% | 0.0384 | **2.536** | **1.547** | 27.16% |
| K=∞, base task | 0 | 3.2738 | 0% | 0.1739 | **2.354** | **1.470** | 33.34% |

First-exposure surprise is how many nats the model spends on the eight letters of the word the
first time it sees them in a sequence. It is the direct measure of how much the model must **read**
rather than **recall**, the random baseline is log 26 = 3.258, and it was recorded in
`pool_ladder.csv` from the day these runs were made.

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

## 3. K=16, revisited 2026-09-08: the story had a measurement all along

**What this section said until 2026-09-08.** That K=16 was unexplained; that the plausible reading
was that 128 letters is little enough to be served from the weights, leaving the distinction
nothing to separate; and that this was "a story, not a measurement" needing a K sweep we did not
have time for.

**It did not need a K sweep.** How much a model must read rather than recall is measured directly,
once per condition, by first-exposure surprise, and every value had been in `pool_ladder.csv` since
the runs were made. Read that column and the ladder stops being a failed monotone rise and becomes
a **step**:

- **Word in the weights.** K=1 at 0.0004 nats and K=16 at 0.3532, so 99.99% and 89.2% of the word
  already known. mem/rep **1.004** and **1.013**. No effect.
- **Word must be read.** K=256 at 0.7309 nats and the base task at 3.2738, so 77.7% and 0% already
  known. mem/rep **1.547** and **1.470**. Full effect.

The two no-effect conditions agree to 0.009 and the two full-effect conditions to 0.077. K=16 shows
nothing because there is almost nothing for it to show: with 89% of the word already in the
weights, the repeat block is barely more context-held than the first-sight block.

**And K=16 was never an inversion.** 0.923 against K=1's 1.035 is a gap between two points that
both mean "nothing here", and we had been reading it as signal for a day. On **mem/rep**, which is
the quantity every headline number in this project reports (1.4705, 2.1201, 1.8742), K=16 reads
**1.013**. There is no inversion to explain.

**What is still not settled.** The switch happens somewhere between first-exposure surprise 0.35
and 0.73 nats. Two points bracket it; none locates it. A K sweep would locate it, and that is still
not affordable. And none of this says *why* context-held knowledge needs more neurons, which is the
question the whole project keeps arriving at.

## What this does and does not license

**It licenses**: saying the effect survives an intervention that holds the task recognisably the
same while moving where the word's content lives; that it appears exactly when the model must read
the word rather than recall it, on four conditions spanning 0% to 100% context-dependence; and that
it is not a monotone function of task difficulty, since K=256 is 4.5× easier by final loss
than the base task and shows a slightly larger effect.

**It does not license**: calling the mechanism established, calling the relationship monotone in
provenance, or claiming to know where the switch sits. The registered prediction was a monotone
rise and what we got is a step; two points bracket the threshold and none locates it.

Data in `pool_ladder.csv`. Checkpoints `checkpoints/bdh_n2048_pool16.pt` and `checkpoints/bdh_n2048_pool256.pt`, each
carrying its own word pool so it can only be re-measured in distribution. Reproduce with
`python tools/ladder.py`.
