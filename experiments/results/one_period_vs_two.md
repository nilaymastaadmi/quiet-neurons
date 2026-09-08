# Why the live page reads 2.5 to 2.6 and the published figure is 2.35

A reader opened `web/explain.html` cold on 2026-09-08, ran its controls eleven times, and reported
that every live ratio they got was above the 2.35× the page quotes three lines below as the
1,280-sequence figure. They were right, and until they said so neither page explained it.

## The cause is the sequence length, and it is in the code

| | sequence | what happens to the warm-up |
|---|---|---|
| `web/index.html`, `web/explain.html` | **T = 77**, one period | appears once, at the very start |
| `experiments/measure.py` | **T = 154**, two periods, averaged position-wise | appears twice; the second time it has already been seen in this sequence |

`measure.py` line 60 sets `T = PERIOD * NPER` with `NPER = 2`, and line 154 averages the two
periods with `.view(NPER, PERIOD).mean(0)`. The pages run a single period.

That matters because the whole claim turns on where knowledge is held. In period 1 the warm-up is
purely weight-held: the model has not seen it yet in this sequence. **By period 2 the warm-up is
also context-held**, because it appeared 77 letters ago, so it goes quieter, and the ratio it
produces is smaller.

## Measured, in the browser, on the shipped weights

Three two-period sequences, layer 2, xy tensor, run in a tab against the live site:

| run | period 1 alone | period 2 alone | both averaged, as measure.py does |
|---|---|---|---|
| 1 | 2.481 | 2.157 | **2.316** |
| 2 | 2.580 | 2.269 | **2.424** |
| 3 | 2.618 | 2.228 | **2.421** |

Warm-up firing was **14.44%** in period 1 of all three runs, because the warm-up is the same 13
letters every time, and **12.93 / 12.69 / 12.62%** in period 2.

The averaged column brackets the published **2.354** from `measured.csv`. The single-period column
brackets what the pages show live.

## What follows

**The published number is the conservative one.** It includes a period in which the warm-up is no
longer purely weight-held, which dilutes exactly the contrast the claim is about. Quoting 2.354 as
the headline understates the clean case by roughly 0.2, and that is the right direction for a
number to be wrong in, but it should be stated rather than left for a reader to trip over.

Both pages now say so at the point where the two numbers sit next to each other.

**Three runs in a browser is not five pinned samples over 1,280 sequences.** This is enough to
explain a discrepancy, not enough to publish a new ratio, and no published figure was changed on
the strength of it. The headline family stays exactly as `measure.py` computed it.

**Why the protocol is two periods and not one.** It was not chosen to flatter or to deflate. Two
periods is what the first version of `sparsity_scan.py` measured and every published number has
used it since, so changing it now would break comparability with the wall-clock family, the
step-matched family, the ladder and the full-schedule runs, all of which were measured that way.
The honest fix is the sentence, not the protocol.

Measured 2026-09-08 against `checkpoints/bdh_n2048.pt`, the checkpoint the pages ship.
