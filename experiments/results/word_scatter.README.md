# How `word_scatter.csv` was produced

These fifteen rows are **not** produced by `measure.py`. They are a page-level measurement, because
the surprise injection is a page behaviour: the injected letter is a fixed rot-13 shift placed at
`WARM + WORD*4 + 3` = position 48, which lives in `web/index.html`, not in the PyTorch code.

They are recorded here because two published statements depend on them: the README's falsification
paragraph, and the caveat in step 4 of the guided tour. Neither should rest on a number nothing in
the repository produces.

## To reproduce

Serve `web/` and open `index.html`, then in the browser console:

```js
const $ = s => document.querySelector(s);
const settle = async () => { for (let i = 0; i < 200; i++) {
  await new Promise(r => setTimeout(r, 100));
  if ($('#runbadge').textContent.trim() === 'live') return; } };

async function one(w) {
  const inp = $('#word');
  inp.value = w; inp.dispatchEvent(new Event('input', { bubbles: true }));
  await new Promise(r => setTimeout(r, 150)); await settle();
  $('#surprise').click();
  await new Promise(r => setTimeout(r, 150)); await settle();
  return { word: w, base: $('#boBase').textContent, at: $('#boAt').textContent,
           jump: $('#boJump').textContent };
}
await one('tmredfpf');
```

The layer selector must be on **2**, which is the page default. `base_pct` and `injected_pct` are
`#boBase` and `#boAt`; `jump_pct` is `#boJump`, which the page computes as
`(injected / base - 1) * 100` at the same position with and without the injection, so it is a
controlled comparison rather than a difference against the repetition average.

## What it shows

Range **−8% to +33%**, median **+9%**, five of fifteen at or above +20%. One word is one sequence
and it scatters hard. The 1,280-sequence ratios in `measured.csv` do not: their five-sample spreads
are all under 0.02.

The negative row (`vbnmasdf`, −8%) is kept deliberately. Dropping it would make the scatter look
one-sided when it is not.

Measured 2026-09-07 against `checkpoints/bdh_n2048.pt`, the checkpoint the page ships.

## The eight-word batch control, run in the browser

The scatter above is why the page grew a **Run 8 random words** button: one word can read as a
refutation, so the falsification path had to offer a sample rather than a single draw. The button
generates eight words in the browser, runs a full forward pass on each, and reports layer 2's
warm-up and repeat firing rates with both mean surprises.

Runs are recorded here as they are made, on `checkpoints/bdh_n2048.pt`, the checkpoint the page
ships.

| run | date | median warm/rep | rows above 1.3x | worst warm-up surprise |
|---|---|---|---|---|
| 1 | 2026-09-07 21:0x | **2.48x** | 8 of 8 | 0.001 |
| 2 | not yet run | | | |
| 3 | not yet run | | | |

Run 1: warm-up firing 14.4% on every row, repeats 5.4-6.2%, ratios 2.33x to 2.70x, warm-up
surprise 0.001 throughout and repeat surprise 0.001 to 0.005. The acceptance condition set in
advance was every row under 0.005 warm-up surprise and most rows above 1.3x; run 1 meets it on
every row.

**Runs 2 and 3 are outstanding, and this is not yet the three-run record the plan asks for.** They
could not be taken on the night of 2026-09-07: the browser tab was not visible, and Windows runs a
hidden tab's renderer at background quality-of-service, so with the n=8,192 training job saturating
the CPU the page's worker returned nothing at all. A probe worker posted a forward pass and had no
reply after 20 seconds, while main-thread timers in the same tab fired normally, which is what
starvation looks like rather than a page fault. In a visible tab on a quiet machine the same eight
words complete in about twenty seconds. Runs 2 and 3 belong with the other measurements that need a
quiet machine and a visible tab.
