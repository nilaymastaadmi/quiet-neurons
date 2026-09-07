# Quiet Neurons

**A 397,000-parameter Dragon Hatchling runs in your browser. It quietens on text it just
learned, but not on text baked into its weights. Both are perfectly predictable.**

DataForge 2026, Pathway Track. Approved topic: **Sparse Non-Negative Activations**.

## Every link in this submission

| What | Where |
|---|---|
| **Live artifact**, opens without sign-in | https://nilaymastaadmi.github.io/quiet-neurons/ |
| **Public source repository** | https://github.com/nilaymastaadmi/quiet-neurons |
| **One-page concept summary (PDF)** | https://nilaymastaadmi.github.io/quiet-neurons/concept-summary.pdf &nbsp;&middot;&nbsp; `web/concept-summary.pdf` in this package |
| **Parity check**, browser model against PyTorch | https://nilaymastaadmi.github.io/quiet-neurons/parity.html |
| **Per-phase profiler** | https://nilaymastaadmi.github.io/quiet-neurons/profile.html |
| **AI assistance disclosure** | `AI_DISCLOSURE.md` |
| **Source and licence record** | `experiments/LICENSES.md` and `LICENSE` |
| **Trained checkpoints**, too large for the zip | `experiments/checkpoints/` in the repository above |

Reproduce every number: see [Reproducing the results](#reproducing-the-results).

---

## The claim

> A trained BDH quietens down when it has just learned something from the text in front of
> it. Not when the text is merely predictable. Those two look identical from the outside,
> and they are not.

This starts from the Dragon Hatchling paper's Section 6.4 and Figure 14, which reports that
"neuron activity correlates with signal predictability: fewer neurons are active ... for more
predictable input signals" (arXiv:2509.26507, section 6.4). We reproduced that at three model
sizes and it holds. Then we found a case it does not cover, and the artifact teaches the
sharpened version.

In aggregate the paper is right: predictable text is quieter. The counterexample below shows
that what actually predicts quietness is *where the knowledge came from*, not predictability
by itself.

**The counterexample.** Layer 2 of the n=8192 model over **1,280 sequences**, read from the
committed trace `web/data/traces/n8192.json`. Whole blocks, not hand-picked letters:

| what the model is reading | mean surprise | layer-2 neurons firing |
|---|---|---|
| the 13-letter warm-up, identical in every sequence, held in the **weights** | 0.0004 nats | **9.65%** |
| the 56 repeated letters, new every run, learned from the **context** | 0.0041 nats | **3.64%** |

Both blocks are predicted essentially perfectly, and one uses **2.65x** as many neurons. At the
extremes the gap is wider still: **letter 12** runs at 16.10% against **letter 41** at 3.52%,
4.6x apart, both under 0.003 nats of surprise. (Letters are numbered from 1, as the page's
scrub control labels them. In `traces/n8192.json` those are array indices 11 and 40.) So what separates them is not predictability but where the
knowledge came from: the activation level tracks parametric against in-context memory. We show
that it tracks, not why it does.

**It is falsifiable, in the artifact, in under a minute.** Inject a letter the model cannot
predict into the middle of the repetition. If activity does not rise at that letter, the
claim is wrong. On the page's default word `tmredfpf`: **4.79% → 6.01%** at the same position,
**+26%** in layer 2.

**Quote the scatter, not the single number.** We drove the shipped page across fifteen words and
the jump ranges from **−8% to +33%** (`experiments/results/word_scatter.csv`; median +9%, and 5 of
15 at or above +20%). One word is one sample. A reader who changes the word and lands near zero has
found the scatter, not a broken claim; the 1,280-sequence table below is what does not scatter.
The letter *before* the surprise is identical to nine decimal places either way, because the model
only reads leftwards.

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
| What you see in the first second | **Precomputed, and labelled on screen.** Rather than showing empty dashes while the browser works, the page paints PyTorch's own reference trace for the word it opens on (`web/data/reference.json`). The badge reads "PyTorch reference" until the live pass returns, then flips to "live". The page opens on exactly the word that trace was exported for, so the live run recomputes those same values and you can watch them agree. |
| The model on the page (n=2048, 397,312 params) | **Live.** Real weights, real forward pass, computed in your browser on every interaction. Not a recording. |
| Neuron grid, sparsity trace, counterexample panel, surprise test | **Live.** All recomputed from that forward pass. |
| Attention heatmap and binding-by-lag chart | **Live**, from the same run. |
| Which individual dot lights up in the neuron grid | **Count live, placement illustrative.** The number of lit cells is computed by the forward pass and printed under the grid. Where they sit is a fixed scatter keyed to the letter position, because which particular neuron fires is not what the claim is about. The grid's own caption says exactly this. |
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
  checkpoints/
    bdh_n2048.pt  n8192.pt  n16384.pt      THE PUBLISHED MODELS. Every number quoted
                                 anywhere in this project comes from these three, and the
                                 README's reproduction commands name exactly these files.
    bdh_n2048_wallclock.pt  n8192_wallclock.pt
                                 superseded: cut by wall clock at 1,854 and 1,917 steps
                                 instead of step-matched. Kept so the before-and-after in
                                 "We removed the confound" is checkable, never quoted as
                                 a headline number.
    bdh_n2048_fixedword.pt       the provenance control. Degenerate, see below.
  results/                       raw run logs, measured.csv, and a method note beside
                                 every measurement that was taken by driving the page
                                 rather than by measure.py

tools/sweep.py                   cross-surface consistency: every published number against
                                 measured.csv, every referenced path, disclosure coverage
verify.sh                        runs all of it and exits non-zero on any drift. Run this
                                 before any commit that touches a number.

web/
  index.html                     the explainer
  bdh.js                         the forward pass ported to JavaScript, mirroring bdh.py
  worker.js                      runs it off the main thread so typing never freezes
  parity.html                    checks bdh.js against the PyTorch reference trace
  profile.html                   per-phase timings, used to stop guessing about speed
  data/weights.bin               397,824 float32, 1,591,296 bytes (1.59 MB)
  data/reference.json            PyTorch's answers for a fixed input; the parity target
  data/scaling.json              the measured scaling points
```

### What each part of the page does

`index.html` above is one line in a file tree, so here is the artifact itself, in order. Every
section has one job and the sequence is the guided narrative: watch it, understand it, check it
is not a trick, compare it to the paper, then test yourself.

| Section | Its one job | Live? |
|---|---|---|
| 00 Who this is for | Audience, prerequisites, and the claim in one falsifiable sentence | text |
| 01 The instrument | The reader watches activity fall as the model learns the word. Opens already running, no Run button | **live** |
| 02 The mechanism | Why context-learned text should be cheaper. The neuron grid and the non-negativity panel | **live** |
| 03 The setup | What the model is actually reading, so the reader can tell signal from artefact | **live** |
| 04 Beside the paper | Our three ratios against Figure 14's band, and the non-monotonicity we did not expect | precomputed |
| 05 Where this lives in BDH | The BDH module: which tensor, which layer, and why this architecture and not another | text + live readout |
| 06 Check yourself | Three predict-before-you-touch questions, three misconceptions, and a say-it-back prompt with a marking key | interactive |
| 07 Limits | Every place the effect is absent, weak, or against us, including one confound we removed | text |

The four-step tour at the top drives the same controls the sandbox does, so the guided path and
the free exploration are the same instrument, not two modes.

**There is no backend.** The page is static files plus 1.59 MB of weights, and every
computation happens in the reader's browser. That is deliberate rather than a shortcut: any
page that depends on a server can stop working between deployment and judging, and this one
has no server to lose. Everything you see was computed in your own tab.

### Is the browser model really the same model?

`web/parity.html` answers this, and it is the first thing a sceptical reader should open.
It runs the JavaScript forward pass on a fixed input and diffs it against PyTorch's output.

| check | result |
|---|---|
| logits, 154 x 32 values | max difference **6.064e-5** (threshold 5e-3) |
| all 12 sparsity series | **exactly zero** difference |

Sparsity is a count of active neurons over a fixed denominator, so it has to match exactly,
and on the shipped checkpoint it does, on every layer and both tensors.

The test still tolerates **one** neuron of disagreement, and that allowance is not idle. The
earlier checkpoint had a layer-3 pre-ReLU value of magnitude 1.10e-07; float32 carries about
seven significant digits, so PyTorch and JavaScript rounded to opposite sides of zero and one
neuron in 2,048 flipped. It is a real property of the arithmetic rather than a bug, it can
recur on any checkpoint, and the threshold permits exactly one such neuron and no more. This
model happens not to have one.

---

## Reproducing the results

```bash
pip install torch                                   # CPU is fine; no GPU used anywhere

cd experiments
# the model that runs on the page: 2,309 steps at ~1.45 s/step, about an hour of laptop CPU.
# --stop-at is not optional. It halts at a fixed point of an unchanged 4,000-step schedule,
# which is what makes the three sizes comparable; cutting by --budget alone stops each model
# at a different learning rate and reproduces the confound this project removed.
python sparsity_scan.py --embd 64 --mult 32 --budget 7200 --steps 4000 --stop-at 2309
python measure.py --ckpt checkpoints/bdh_n2048.pt --embd 64 --mult 32 --repeats 5 \
                  --append --export-trace ../web/data/traces/n2048.json
python make_scaling.py                              # rebuilds web/data/scaling.json

cd ../web && python -m http.server 8123             # then open localhost:8123
```

That reproduces the n=2048 row and every number the live page prints. The other two rows, and
the counterexample quoted at the top, come from the larger models: the same two commands with
different sizes and a longer budget.

```bash
python sparsity_scan.py --embd 128 --mult 64  --budget 43200 --steps 4000 --stop-at 2309   # n=8192,  ~6.5 h CPU
python sparsity_scan.py --embd 128 --mult 128 --budget 86400 --steps 4000 --stop-at 2309   # n=16384, ~13 h CPU
python measure.py --ckpt checkpoints/bdh_n8192.pt  --embd 128 --mult 64  --repeats 5 \
                  --append --export-trace ../web/data/traces/n8192.json
python measure.py --ckpt checkpoints/bdh_n16384.pt --embd 128 --mult 128 --repeats 5 \
                  --append --export-trace ../web/data/traces/n16384.json
```

The trained checkpoints are in the repository, so `measure.py` on its own re-derives every
published number without retraining anything. They are left out of the submission zip only
because they are 37 MB.

**Or check the whole thing in one command:**

```bash
./verify.sh              # every published ratio, cross-surface agreement, the one-pager
./verify.sh --with-page  # and parity.html driven in headless Chrome
```

It exits non-zero on any drift. It exists because a defect reached a built package in which
this very section's command pointed at a stale checkpoint and printed
the superseded **1.4318** where the page published **1.4705**: the numbers were right, the instructions for regenerating them were
not, and nothing was comparing the two. It caught the one-pager at 973 words on its first run.

`measure.py` prints `TASK_LEARNED` before it prints anything else. If that is `False` the
model never learned to copy the in-context word, and the sparsity numbers are meaningless
rather than negative. That distinction is built into the script because the first version of
this experiment was undertrained and would have produced a clean-looking false negative.

### Measured results

Layer 2, `xy` product tensor, memorisation over repetition. Each row is 1,280 sequences
across 5 independent pinned samples.

**Which tensor `xy` is, and why it is the paper's `y`.** Figure 14 counts "the fraction of
neurons with non-zero entry `y_{t,l}`", so everything depends on which tensor that is. The paper
defines it three times and all three agree:

- **Eq. (8)**, the BDH-GPU state-space form: `y_{t,l} := ( D_y LN( ρ_{t−1,l} x_{t,l} ) )_+ ⊙ x_{t,l}`.
  The `⊙ x_{t,l}` is an elementwise product with `x`. Eq. (4) and Eq. (7) give the same thing in
  the other two representations.
- **The Figure 3 caption**: "Vector `y_{t,l} ∈ (R+)^n`, `y_{t,l}` is (typically) sparse in the
  sense of `‖y_{t,l}‖₀`" — the zero-count of that product is the sparsity measure.
- **The Appendix E PyTorch listing**: `y = F.relu(self.ln(a_ast) @ self.decoder_y) * x`.

That last line is `xy_sparse` in `bdh.py`, term for term. There is a naming collision worth
stating plainly, because it is the thing that makes this look ambiguous when it is not: the
released repository calls the rectified factor `y_sparse`, but **the repository's `y_sparse` is
the paper's `(...)_+` intermediate, not the paper's `y`.** The paper's `y` is the repository's
`xy_sparse`. We count `xy_sparse`, which is the paper's `y_{t,l}`.

The magnitudes corroborate it rather than establishing it: `xy` is the only quantity in the public
code that lands inside Figure 14's 4.0–7.5% band, while `x` and `y_sparse` alone run from 19% to
59% dense across the step-matched runs. And the direction of the result does not depend on the
choice at all: measured on all three tensors at layer 2, n=8,192, the ratios are 1.40 (`x`),
1.27 (`y_sparse`) and 2.12 (`xy`). Every one is above 1.0. Picking a different tensor changes the
size of the effect, never its sign.

| n | params | steps | MEM | REP | ratio | spread over 5 samples |
|---|---|---|---|---|---|---|
| 2,048 | 397,312 | 2,309 | 0.0850 | 0.0578 | **1.4705x** | 1.4647 – 1.4740 |
| 8,192 | 3,153,920 | 2,309 | 0.0772 | 0.0364 | **2.1201x** | 2.1139 – 2.1301 |
| 16,384 | 6,299,648 | 2,309 | 0.0820 | 0.0438 | **1.8742x** | 1.8699 – 1.8782 |
| 65,536 | Pathway's | — | 4.0–7.5% | ~2.5% | 1.6–3.0x | reported as a range, not reproduced here |

**The scaling is not monotonic, and we published the opposite before the third model
finished.** n=16384 comes in at 1.87, *below* n=8192's 2.12. The gap is **0.246**, and it is
**40×** the sampling spread, so it is not noise.

*"Spread" means one thing throughout this README: the half-width of the five-sample range.* The
tables quote the same quantity as a full min-to-max, which is twice as large, and mixing the two
is how this number becomes unreproducible. Explicitly: 8,192 ranges 2.1139–2.1301, half-width
0.0081; 16,384 ranges 1.8699–1.8782, half-width 0.0042; mean half-width **0.0061**; and
0.246 ÷ 0.0061 = **40.1**. Against the full widths instead it is 15× and 30×, which is the same
finding stated on a different denominator.

**We removed the confound rather than disclosing it.** The three models were originally each
cut by a wall-clock budget at 1,854, 1,917 and 2,309 steps of one 4,000-step OneCycle schedule,
so each stopped at a different learning rate. We retrained the two short ones to **2,309 steps**,
the point the largest already reached, leaving the schedule itself untouched: `--stop-at` caps
the loop without shortening the schedule, which is what step-matching actually requires. Both
rose. n=2,048 went from 1.4318 to **1.4705**; n=8,192 from 1.9731 to **2.1201**. The wall-clock
cut had been suppressing the two smaller models, which is exactly what a confound does when you
take it away.

**The non-monotonicity survives, and sharpens.** Controlled, the sequence reads 1.47, 2.12,
1.87, and the drop from 8k to 16k is 40× the mean half-width, as computed above. It is a property
of the models, not of where training stopped. Both families are in
`experiments/results/measured.csv`, told apart by a `steps_trained` column, so the before and
after are both checkable. Final losses are close (0.174, 0.173,
0.172), so they are comparably trained on the task, but that is not a controlled comparison.

The defensible statement is therefore narrower than the one we started with: **the effect is
far stronger at 8k and 16k than at 2k, and both land inside the range the paper reports for
a model four to eight times larger again. Whether it grows monotonically, we do not know.**
What is no longer in doubt is the step count: all three now stop at 2,309 steps of the same
schedule, so *where training stopped* cannot be what produces the dip. What is still
uncontrolled is that none of them **finished** that schedule. It is 4,000 steps long and all
three stop at 2,309, about 58% of the way through, at a learning rate that is still falling.
Settling monotonicity properly needs three fully-trained models, roughly thirteen hours of CPU
that did not fit before the deadline.

**A bug our own measurement found.** Until 2026-09-06 the per-position loss was paired with
the wrong position. `pl[t]` is the cost of *predicting* token `t+1`, so the surprise of
*reading* letter `t` is `pl[t-1]`. The first-exposure slice therefore dropped the first
genuinely new letter and pulled in one the model had already learned, and reported first-sight
loss as **2.88**. The random baseline for 26 letters is 3.258, and a first sight of a random
word cannot be easier than chance, so that number should have been impossible. Aligned, it
reads **3.27** on the shipped step-matched model, sitting on the baseline exactly as it should.
(It read 3.28 on the wall-clock-cut checkpoint this bug was found on; the step-matched retrain
moved it by 0.009.) Every sparsity number is
unaffected: the activation counts never depended on the loss array. The correlations above are
computed on the aligned curve, and `web/data/traces/*.json` now ship it as `surprise`.

The training logs in `experiments/results/run_*.log` were written **before** this fix and
still print the unaligned figure of about 2.87. They are kept as the historical record of
those training runs and cannot be regenerated without retraining. `measure.py` is the source
of every number quoted in this project, and it is aligned; `sparsity_scan.py` has been
corrected too, so a fresh training run prints the right value.

**A note on why `measure.py` exists.** `sparsity_scan.py` draws its evaluation words *after*
training has consumed the random number stream, so its sample depends on the entire training
history and cannot be regenerated from a checkpoint. Re-measuring n=8192 on a fresh sample
moved layer 2 from 1.983 to 1.971. Not a bug, just a different draw, but it means those
numbers were not checkable. Everything quoted in this project comes from `measure.py`.

---

## The provenance control: we moved the word into the weights

Everything above is a **correlation**. Activity is low on text the model learned from its
context and high on text it learned during training, but we never intervened on that variable,
so we could only say activity *tracks* provenance.

So we intervened. In the normal task the 13-letter warm-up is drawn once and shared by every
sequence, which puts it in the **weights**, while the 8-letter word is redrawn per sequence and
can only be known from the **context**. `sparsity_scan.py --fixed-word` draws the word once too.
Identical content, identical protocol, identical 2,309 steps of the identical schedule; the only
thing that changes is where the word's knowledge lives.

Prediction, written down before the run: the warm-up-over-repetition gap should collapse toward
1.0, and it should collapse *because repetition rises*.

| n=2048, layer 2 | warm-up | repetition | warm / rep |
|---|---|---|---|
| Word in the **context** | 13.61% | **5.78%** | **2.354×** |
| Word in the **weights** (`--fixed-word`) | 21.71% | **20.98%** | **1.035×** |

The gap collapsed, from the predicted direction. First-sight loss confirms the manipulation
took: 3.2738 nats in the context model against **0.0004** in the control, so the word genuinely
is not being learned from the text any more.

**This control is degenerate, and we are retracting the argument we first built on it.**

An earlier version of this section argued that the collapse was *asymmetric* and that the
asymmetry rescued it: the block whose provenance changed rose 3.63×, the block whose provenance
did not rose 1.60×, and a global shift would move both equally. **That reasoning is wrong.** It
holds only for a *multiplicative* shift. What this control actually produced is a shift to a
**common level**, which moves cells by different factors by construction. Computed from
`mechanism_control.csv` against `measured.csv`:

| | control | the same cells, real model |
|---|---|---|
| spread across the 12 cells, tensor `x` | **2.21%** | 15.13% |
| tensor `y` | **1.73%** | 17.74% |
| tensor `xy` | **2.08%** | 33.34% |

Every cell converged on one constant per tensor. The correlation between a cell's starting value
and its rise factor is **−0.86** across 36 cells: the rise is almost entirely explained by where
the cell began, not by whether its provenance changed. Concretely:

- **28 of 36 cells rose below the 1.60× "unchanged" baseline**, five of them by 1.10× or less.
- Layer 0 `xy` **repetition**, whose provenance *did* change: 20.70% → 20.46%, a rise of
  **0.99×**. It did not rise at all.
- Layer 0 `xy` **warm-up**, whose provenance did *not* change: 12.36% → 20.93%, a rise of
  **1.69×**, larger than the figure we had quoted as the unchanged reference.
- The 3.63× we cited is simply the cell that started lowest in the entire matrix, at 5.78%.

**What the control does and does not show.** It shows that removing the provenance distinction
removes the activation distinction. But it removes it by collapsing the whole model onto a single
activation level while solving a trivial task (final loss 0.0004 against 0.174), which is equally
consistent with provenance mattering and with difficulty mattering. **It cannot separate them, and
no argument about the shape of the collapse can rescue it.** `measure.py` now prints a degeneracy
check, coefficient of variation per tensor per block, flagging anything under 5%; it exists
because this condition collapsed and we argued around the collapse instead of detecting it.

The graduated version of this experiment, which keeps the copy task alive at intermediate pool
sizes, is pre-registered in `experiments/results/pool_ladder.README.md`.

Data in `experiments/results/mechanism_control.csv`, checkpoint in
`experiments/checkpoints/bdh_n2048_fixedword.pt`, and the flag is off by default so every
published run is bit-identical to before it existed.

---

## Where the effect is absent

| Limit | What we actually measure |
|---|---|
| Layer 0 runs **backwards** at every size | ratio 0.80, 0.80, 0.86 at n = 2k, 8k, 16k |
| Layer 3 is flat at 2k and 8k, weakly positive at 16k | 0.97, 0.95, **1.11**. "No effect" is true of the two smaller models only |
| Scaling is **not** monotonic, and now controlled | 1.47 → 2.12 → 1.87 at n = 2k, 8k, 16k, all trained for the same 2,309 steps. We expected monotone growth and said so publicly until the third model landed |
| The three models **are** step-matched, as of 2026-09-06 | all three stop at 2,309 steps of the same 4,000-step schedule. The earlier wall-clock-cut numbers are kept in `measured.csv` for comparison |
| A single sequence is noisy | the surprise-injection jump ranges −8% to +33% across 15 words, median +9%, all in `experiments/results/word_scatter.csv`. The 1,280-sequence ratios do not scatter: every five-sample spread is under 0.02 |
| Surprise and sparsity do **not** track per letter | at n=8192, layer 2: **Pearson 0.30, Spearman −0.05**. Across the eight letters of first sight, surprise is flat at 3.27 to 3.29 nats while sparsity falls 13.3% to 4.5%. The relationship is between phases, not letters, and a near-zero Spearman is what says so. This killed a stronger claim we wanted to make. Printed by `measure.py`, recorded in `experiments/results/correlations.csv` |
| We show the signature, not its cause | we ran the intervention: moving the word from context into the weights collapses the gap 2.354× → 1.035×. But that control is **degenerate**, every cell converging on one activation level (spread 1.7–2.2% against 15–33% in the real model, and rise factor correlating −0.86 with starting value), so it cannot separate provenance from task difficulty and an asymmetry argument we briefly published on it was wrong and is retracted above. Nothing here explains *why* weight-held knowledge needs more neurons. An independent reader of the one-page summary raised exactly this, and they were right |
| Toy model, not an official BDH checkpoint | architecture is Pathway's and unmodified; the weights are ours |
| Synthetic task, not natural language | so is the paper's §6.4 protocol, deliberately |

---

## Accessibility and robustness, measured rather than asserted

Text contrast passes **WCAG AA in both themes**, measured on 2026-09-07 by walking every
rendered element on the running page, resolving each one's actual computed foreground against
its nearest opaque ancestor background, and applying the large-text exemption (3:1 at ≥24 px, or
≥18.66 px bold) only where it genuinely applies. **305 text elements per theme, zero failures.**
The floor is **5.12:1** in dark and **4.71:1** in light, against a 4.5:1 threshold.

An earlier version of this section claimed 4.69:1 and was **wrong**. Two things actually failed:
white text on the orange accent measured **2.06:1** in dark mode, which included the "Inject a
surprise" button, the most important control on the page; and the small stat labels sitting on
the accent-tinted panel measured 4.34:1 in dark and 4.26:1 in light. Both are fixed rather than
re-asserted. Buttons on the accent now use a near-black foreground (9.17:1), the light accent was
darkened from `#B45911` to `#9A4A0E`, and `--muted` moved in both themes. The reproduction is in
`experiments/results/contrast.md`, and re-running it is the only thing that should ever be allowed
to change the numbers in this paragraph.

At a **375 px**
viewport the document has **zero horizontal overflow** (`scrollWidth == clientWidth`); wide
tables scroll inside their own container rather than pushing the page sideways. Every control
is a native input, button or `<details>`, so the whole page is keyboard navigable, and focus
rings are explicit rather than suppressed. Charts are SVG with `role="img"` and text
alternatives; the neuron grid and attention canvases carry `aria-label`. Animation respects
`prefers-reduced-motion`. There is no build step, no framework and no backend: one HTML file,
one JavaScript module, a worker, and 1.59 MB of weights.

---

## Three claims we deliberately refuse

1. **"BDH is linear attention."** The public code materialises a full T×T score matrix. It
   is quadratic. We *counted operations* rather than timing them: at T=154, N=512, D=64 and
   4 heads, the shipped quadratic form costs 27.1M multiply-adds per layer against 40.4M for
   the recurrent form, and the two cross at T = 4ND/(N+D) + 1 ≈ **230 tokens**. Below that
   the quadratic form is genuinely the cheaper one, which is presumably why the reference
   ships it. This is an operation count, not a benchmark: no linear implementation was
   written and nothing was timed.
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
3. Z. Li, C. You, S. Bhojanapalli et al. *The Lazy Neuron Phenomenon: On Emergence of
   Activation Sparsity in Transformers.* arXiv:2210.06313 (2022); short version at ICLR 2023.
   **Activation sparsity is not designed in, it emerges during training**, in ordinary
   Transformer feed-forward blocks too. This is the control our claim needs: sparsity by
   itself is not what makes BDH unusual.
4. I. Mirzadeh, K. Alizadeh, S. Mehta et al. *ReLU Strikes Back: Exploiting Activation
   Sparsity in Large Language Models.* arXiv:2310.04564 (2023). The choice of ReLU over a
   smooth activation is what produces exact zeros, and therefore sparsity you can count —
   the same design decision `bdh.py` makes twice per layer.
5. Z. Liu, J. Wang, T. Dao et al. *Deja Vu: Contextual Sparsity for Efficient LLMs at
   Inference Time.* arXiv:2310.17157 (2023); ICML 2023. **Which** units fire depends on the
   input, not merely how many. That input-dependence is what our counterexample probes one
   step further, by asking *which property* of the input it depends on.
6. A. Vaswani et al. *Attention Is All You Need.* arXiv:1706.03762 — the softmax attention
   that BDH's Gram-matrix formulation departs from.
7. A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
   arXiv:2312.00752 (2023) — the state-space line BDH is explicitly *not* in, per the
   problem statement's own note.

---

## Credits, licences and AI assistance

**`experiments/bdh.py` is Pathway's reference implementation, used unmodified**, from
[pathwaycom/bdh](https://github.com/pathwaycom/bdh), MIT, Copyright 2025 Pathway
Technology, Inc. It is the **BDH-GPU** formulation, which is what that repository ships.
Full record in [`experiments/LICENSES.md`](experiments/LICENSES.md); this repository's own
code is MIT, see [`LICENSE`](LICENSE).

Everything else in this repository was written for this submission: the training and
measurement scripts, the JavaScript port, the parity test, and the explainer. The trained
checkpoints were trained here from scratch on synthetic data; they are not official BDH
weights and are not presented as such.

**AI assistance.** Summarised here, recorded in full in
[`AI_DISCLOSURE.md`](AI_DISCLOSURE.md). This project was built with AI assistance (Claude)
throughout: code, prose and analysis. Every claim, number and line of code was checked by running it. The
parity test, the pinned-sample measurement and the per-phase profiler all exist because
assumptions made during the build turned out to be wrong and needed to be caught by
measurement rather than by review. Specific corrections that came from running things
rather than reasoning about them are recorded in the git history.

Typefaces: IBM Plex Sans, Sans Condensed and Mono, served from Google Fonts, SIL Open Font
License 1.1. No other third-party code, data, weights or graphics are used.
