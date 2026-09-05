# Quiet Neurons

**A 397,000-parameter Dragon Hatchling runs in your browser. It quietens on text it just
learned, but not on text baked into its weights. Both are perfectly predictable.**

DataForge 2026, Pathway Track. Approved topic: **Sparse Non-Negative Activations**.

- **Live artifact:** https://nilaymastaadmi.github.io/quiet-neurons/
- **Reproduce every number:** two commands, see [Reproducing the results](#reproducing-the-results)
- **Check the browser model against PyTorch:** open `web/parity.html`

---

## The claim

> A trained BDH quietens down when it has just learned something from the text in front of
> it. Not when the text is merely predictable. Those two look identical from the outside,
> and they are not.

This starts from the Dragon Hatchling paper's Section 6.4 and Figure 14, which reports that
BDH's neuron activity "varies with predictability rather than following a fixed sparsity
budget". We reproduced that at three model sizes, then found the framing is too loose, and the
artifact teaches the corrected version.

**The counterexample, measured at n=8192 over 256 sequences:**

| what the model is reading | surprise (nats) | layer-2 neurons firing |
|---|---|---|
| warm-up letter 11, identical in every sequence | 0.000 | **13.59%** |
| repeated word, letter 40 | 0.003 | **3.50%** |

Both are predicted essentially perfectly. One uses roughly four times as many neurons. The
warm-up is memorised in the **weights** during training and keeps neurons busy; the repeated
word is learned from the **context** moments earlier, and that is what goes quiet. So the
variable is not predictability. It is where the knowledge lives.

**It is falsifiable, in the artifact, in under a minute.** Inject a letter the model cannot
predict into the middle of the repetition. If activity does not rise at that letter, the
claim is wrong. Measured on the shipped model: 5.03% → 5.96% at the same position, +18%.
The letter *before* the surprise is identical to nine decimal places, because the model only
reads leftwards.

---

## Who this is for

**Audience.** Anyone who has met a neural network once: undergraduates, engineers new to
post-Transformer architectures, and researchers who would rather have the BDH sparsity
result checked than described.

**Prerequisites.** That a model turns text into numbers and predicts what comes next.
Nothing about attention, state space models or Hebbian learning is assumed. Terms are
defined where they first appear.

**Learning objectives.** After ten minutes a reader can:

1. Say what a sparse non-negative activation is and why BDH's are both.
2. Explain why BDH quietens on context-learned text but not on weight-learned text.
3. Name the layers where the effect is absent, and say so out loud rather than hiding it.
4. Explain why BDH attention is a Gram matrix of neuron activations, and what follows from
   there being no softmax in it.
5. Reproduce every number here with two commands.

---

## What is live, what is precomputed, what is synthetic

The rubric asks for this explicitly, so it is near the top rather than buried.

| Element | Status |
|---|---|
| The model on the page (n=2048, 397,312 params) | **Live.** Real weights, real forward pass, computed in your browser on every interaction. Not a recording. |
| Neuron grid, sparsity trace, counterexample panel, surprise test | **Live.** All recomputed from that forward pass. |
| Attention heatmap and binding-by-lag chart | **Live**, from the same run. |
| Which individual dot lights up in the neuron grid | **Illustrative.** The *count* is real and stated; the scatter is a deterministic layout, because which particular neuron fires is not what the claim is about. The page says so in its caption. |
| Scaling chart points, and the larger-model curves the size switcher overlays | **Precomputed** by `experiments/measure.py`, shipped as `web/data/scaling.json` and `web/data/traces/`. Only n=2048 runs live; n=8192 and n=16384 are 8x and 16x the compute and cannot, and the page labels their curves "measured, not live". |
| The paper's band at n=65536 | **Not reproduced by us.** Read off Figure 14 of arXiv:2509.26507 and drawn as a band, because that is how it is reported. |
| Training data | **Synthetic**, exactly the paper's §6.4 protocol. No natural language, deliberately. |

Nothing on the page is animation standing in for computation.

---

## Architecture

```
experiments/                     PyTorch: train, measure, export
  bdh.py                         Pathway's reference implementation, UNMODIFIED (MIT)
  sparsity_scan.py               trains one model at a chosen size, then measures it
  measure.py                     measures a saved checkpoint on a PINNED word sample,
                                 with --repeats to report the spread. Source of every
                                 number quoted anywhere in this project.
  export_weights.py              dumps weights + a PyTorch reference trace for the browser
  make_scaling.py                builds web/data/scaling.json from measured.csv
  checkpoints/                   trained weights (n=2048, n=8192, n=16384)
  results/                       raw run logs and measured.csv

web/
  index.html                     the explainer
  bdh.js                         the forward pass ported to JavaScript, mirroring bdh.py
  worker.js                      runs it off the main thread so typing never freezes
  parity.html                    checks bdh.js against the PyTorch reference trace
  profile.html                   per-phase timings, used to stop guessing about speed
  data/weights.bin               397,824 float32, 1.55 MB
  data/reference.json            PyTorch's answers for a fixed input; the parity target
  data/scaling.json              the measured scaling points
```

**There is no backend.** The page is static files plus 1.55 MB of weights, and every
computation happens in the reader's browser. This is a deliberate choice: the most visible
prior BDH explainer currently returns HTTP 502 from its model endpoint while its animation
keeps playing. A page that cannot reach a server cannot be honest about what it is showing.

### Is the browser model really the same model?

`web/parity.html` answers this, and it is the first thing a sceptical reader should open.
It runs the JavaScript forward pass on a fixed input and diffs it against PyTorch's output.

| check | result |
|---|---|
| logits, 154 x 32 values | max difference **3.998e-5** (threshold 5e-3) |
| 11 of 12 sparsity series | **exactly zero** difference |
| layer 3 `y` | one neuron in 2048, at one position |

That last row is not waved through. Layer 3 contains a pre-ReLU value of magnitude
**1.10e-07**. Float32 carries about seven significant digits, so the two implementations
round to opposite sides of zero and one neuron flips. The threshold in the test permits
exactly one such neuron and no more, and the reason is written in the code.

---

## Reproducing the results

```bash
pip install torch                                   # CPU is fine; no GPU used anywhere

cd experiments
python sparsity_scan.py --embd 64 --mult 32 --budget 2700    # trains n=2048, ~30 min CPU
python measure.py --ckpt checkpoints/bdh_n2048.pt --embd 64 --mult 32 --repeats 5 --append
python make_scaling.py                                        # rebuilds web/data/scaling.json

cd ../web && python -m http.server 8123                       # then open localhost:8123
```

`measure.py` prints `TASK_LEARNED` before it prints anything else. If that is `False` the
model never learned to copy the in-context word, and the sparsity numbers are meaningless
rather than negative. That distinction is built into the script because the first version of
this experiment was undertrained and would have produced a clean-looking false negative.

### Measured results

Layer 2, `xy` product tensor, memorisation over repetition. Each row is 1,280 sequences
across 5 independent pinned samples.

| n | params | steps | MEM | REP | ratio | spread over 5 samples |
|---|---|---|---|---|---|---|
| 2,048 | 397,312 | 1,854 | 0.0824 | 0.0576 | **1.4318x** | 1.4299 – 1.4334 |
| 8,192 | 3,153,920 | 1,917 | 0.0718 | 0.0364 | **1.9731x** | 1.9695 – 1.9798 |
| 16,384 | 6,299,648 | 2,309 | 0.0820 | 0.0438 | **1.8742x** | 1.8699 – 1.8782 |
| 65,536 | Pathway's | — | 4.0–7.5% | ~2.5% | 1.6–3.0x | reported as a range, not reproduced here |

**The scaling is not monotonic, and we published the opposite before the third model
finished.** n=16384 comes in at 1.87, *below* n=8192's 1.97, and that 0.10 gap is roughly
twenty times the sampling spread, so it is not noise.

A confound we cannot rule out: all three models share one 4,000-step OneCycle schedule but
were each stopped by a wall-clock budget on a laptop CPU, at 1,854, 1,917 and 2,309 steps.
They finished at different points on that schedule. Final losses are close (0.176, 0.175,
0.172), so they are comparably trained on the task, but that is not a controlled comparison.

The defensible statement is therefore narrower than the one we started with: **the effect is
far stronger at 8k and 16k than at 2k, and both land inside the range the paper reports for
a model four to eight times larger again. Whether it grows monotonically, we do not know.**
Settling it needs three models trained for an identical number of steps, about thirteen hours
of CPU that did not fit before the deadline.

**A note on why `measure.py` exists.** `sparsity_scan.py` draws its evaluation words *after*
training has consumed the random number stream, so its sample depends on the entire training
history and cannot be regenerated from a checkpoint. Re-measuring n=8192 on a fresh sample
moved layer 2 from 1.983 to 1.971. Not a bug, just a different draw, but it means those
numbers were not checkable. Everything quoted in this project comes from `measure.py`.

---

## Where the effect is absent

| Limit | What we actually measure |
|---|---|
| Layer 0 runs **backwards** | ratio 0.82 at n=2048, 0.79 at n=8192 |
| Layer 3 shows **nothing** | 0.91 and 0.96, flat |
| Scaling is **not** monotonic | 1.43 → 1.97 → 1.87 at n = 2k, 8k, 16k. We expected monotone growth and said so publicly until the third model landed |
| The three models are not step-matched | 1,854 / 1,917 / 2,309 steps of one 4,000-step schedule, each cut by wall clock |
| A single sequence is noisy | one sequence gave 1.93 where 1,280 give 1.43 |
| Surprise and sparsity do **not** track per letter | layer 2 gives Pearson 0.35 but Spearman −0.05. Inside the first-exposure block surprise is flat at 3.27 while sparsity falls 12.9% → 6.3%. The relationship is between phases, not letters. This killed a stronger claim we wanted to make. |
| Toy model, not an official BDH checkpoint | architecture is Pathway's and unmodified; the weights are ours |
| Synthetic task, not natural language | so is the paper's §6.4 protocol, deliberately |

---

## Three claims we deliberately refuse

1. **"BDH is linear attention."** The public code materialises a full T×T score matrix. It
   is quadratic. We measured the crossover: the linear form only costs less past roughly
   4×D = 256 tokens (27.1M versus 40.4M multiply-adds per layer at T=154). Below that the
   quadratic form is genuinely cheaper, which is presumably why the reference ships it.
2. **97.4% on Sudoku Extreme.** Pathway's own README states this comes from their internal
   implementation and that the open repository does not reproduce it. Neither did we.
3. **Anything about BDH-CQ's internals.** Its technical report states that its dimensions and
   update rules are proprietary. It is a citation here, never a mechanism we model.

---

## Primary sources

1. A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.
   *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain.*
   arXiv:2509.26507 (2025). **§6.4 and Figure 14 are what this project reproduces.**
2. B. Engdahl, A. Kosowski, J. Chorowski, Z. Stamirowska, P. Uznański et al.
   *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning.* arXiv:2608.09888 (2026).
   Cited for the in-context-adaptation framing; its internals are proprietary and not modelled.
3. A. Vaswani et al. *Attention Is All You Need.* arXiv:1706.03762 — the softmax attention
   that BDH's Gram-matrix formulation departs from.
4. A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
   arXiv:2312.00752 (2023) — the state-space line BDH is explicitly *not* in, per the
   problem statement's own note.

---

## Credits, licences and AI assistance

**`experiments/bdh.py` is Pathway's reference implementation, used unmodified**, from
[pathwaycom/bdh](https://github.com/pathwaycom/bdh), MIT, Copyright 2025 Pathway
Technology, Inc. Full record in [`experiments/LICENSES.md`](experiments/LICENSES.md).

Everything else in this repository was written for this submission: the training and
measurement scripts, the JavaScript port, the parity test, and the explainer. The trained
checkpoints were trained here from scratch on synthetic data; they are not official BDH
weights and are not presented as such.

**AI assistance.** This project was built with AI assistance (Claude) throughout: code,
prose and analysis. Every claim, number and line of code was checked by running it. The
parity test, the pinned-sample measurement and the per-phase profiler all exist because
assumptions made during the build turned out to be wrong and needed to be caught by
measurement rather than by review. Specific corrections that came from running things
rather than reasoning about them are recorded in the git history.

Typefaces: IBM Plex Sans, Sans Condensed and Mono, served from Google Fonts, SIL Open Font
License 1.1. No other third-party code, data, weights or graphics are used.
