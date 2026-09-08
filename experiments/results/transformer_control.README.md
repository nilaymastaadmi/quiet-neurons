# Is the signature a property of BDH, or of the task?

Every reader who has looked at this project asked the same question, and until 2026-09-08 the
answer was a table *describing* how BDH differs from a Transformer. A description is not a
control. This is the measurement.

## The design

`experiments/transformer_control.py`. A dense Transformer trained on the **identical** task, with
everything held fixed except the architecture:

| held fixed | value |
|---|---|
| seed | 1337, and the 13-letter warm-up is drawn at the same point in the RNG stream |
| task | same 13-letter warm-up, same 8-letter word redrawn per sequence, 8 repeats, T=154, B=16 |
| optimiser | AdamW, lr 3e-3, weight decay 0.1 |
| schedule | OneCycle, 4,000 steps, pct_start 0.1, **`--stop-at 2309`** |
| gradient clipping | 1.0 |
| evaluation | 5 pinned samples x 16 batches x 16 = **1,280 sequences**, same warm-up / first-sight / repeat blocks |

The architecture: 4 layers, d_model 64, 4 heads, pre-LN, causal attention with softmax, ReLU MLP
with hidden width 2,048.

**Two deliberate choices, both stated rather than buried.** The MLP is ReLU and not GELU, because
a GELU has no exact zeros and counting its "off" units would be a threshold choice rather than a
measurement. And the hidden width is 2,048 to match BDH's 2,048 countable units per layer, which
makes the Transformer **1,129,216 parameters against BDH's 397,312**. The Transformer is 2.8x the
larger model. If it fails to show the signature, it was not starved.

## Both models learn the task, to the same loss

| | BDH n=2,048 | Transformer |
|---|---|---|
| final training loss | 0.1739 | **0.1705** |
| first-exposure loss (random baseline is 3.258) | 3.2738 | 3.2674 |
| repetition loss | 0.0090 | **0.0007** |
| TASK_LEARNED | True | True |

This is what makes the comparison worth anything. The Transformer is not failing to copy in
context; it copies slightly better than BDH does.

## The result

Fraction of units active, layer by layer. BDH counts non-zeros in the elementwise product of two
rectified vectors; the Transformer counts non-zeros in its post-ReLU MLP hidden layer. Both are
"how many units are on", which is the quantity the paper's Section 6.4 is about.

| layer | BDH warm / rep | BDH mem / rep | Transformer warm / rep | Transformer mem / rep |
|---|---|---|---|---|
| 0 | 0.597 | 0.803 | 0.931 | 1.001 |
| 1 | 1.181 | 1.222 | 0.855 | 0.893 |
| **2** | **2.354** | **1.471** | **0.962** | **0.831** |
| 3 | 1.928 | 0.965 | 1.015 | 0.839 |

**The Transformer shows no provenance signature at any layer.** Every one of its eight ratios sits
between 0.83 and 1.02. BDH's layer 2 reads 2.354 on the same comparison. The five-sample spreads
on the Transformer are tiny, at most 0.002 wide, so this is not a sampling accident.

**What that licenses.** The effect is not simply a property of the task. A model trained on the
same sequences, to the same loss, with the same schedule and seed, does not do it. On this
comparison the signature is a property of the architecture.

**What it does not license.** One Transformer, one seed, one size, one task. It does not tell us
*why* BDH does it, which remains the open question this project keeps returning to.

## Two honest asymmetries

**The Transformer's activations are dense.** It runs at 42% to 49% of units on, where BDH's layer 2
runs at 5.8% to 13.6%. So this compares a ratio measured on a sparse quantity against a ratio
measured on a dense one, and a dense ReLU sitting near half-on is close to the point where it is
least free to move in either direction. A sceptic can fairly say the Transformer had less room to
show a ratio. The counter-argument is that room was available: three of its four layers do move,
just the wrong way, and by up to 17%.

**The mild opposite effect is real and is not being hidden.** Transformer layers 1, 2 and 3 have
mem/rep below 1 (0.893, 0.831, 0.839), meaning slightly *more* units fire while repeating than
while first seeing the word. That is the opposite direction to BDH's layer 2, and small. We are
not claiming it means anything.

## Reproduce

```bash
cd experiments
python transformer_control.py --stop-at 2309 --repeats 5
```

About 17 minutes of laptop CPU at 0.44 s per step. Log in
`stepmatched/transformer_control.log`, data in `results/transformer_control.csv`, checkpoint in
`checkpoints/transformer_control.pt`.

Measured 2026-09-08, after the n=8,192 full-schedule run and after the claim it tests was already
published, which is the wrong order for a pre-registration and the right order for a control that
could have embarrassed us. It did not, and it would have been reported here if it had.
