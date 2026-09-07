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

| run | date | median warm/rep | rows above 1.3x | worst warm-up surprise | words |
|---|---|---|---|---|---|
| 1 | 2026-09-07 21:0x | 2.48x | 8 of 8 | 0.001 | warm 14.4%, rep 5.4-6.2% |
| 2 | 2026-09-08 04:2x | 2.54x | 8 of 8 | 0.001 | warm 14.4%, rep 5.5-6.1% |
| 3 | 2026-09-08 04:3x | 2.55x | 8 of 8 | 0.001 | warm 14.4%, rep 5.3-5.9% |

**24 words, 24 above 1.3x, no exceptions.** The acceptance condition was set in advance: every row
under 0.005 warm-up surprise and most rows above 1.3x. All three runs meet it on every row. The
three medians span 2.48x to 2.55x and sit above the 1,280-sequence population value of 2.35x, which
is what the eight-word sample should do given the warm-up block is identical every time and only
the repeated word is drawn fresh.

Note the contrast with the fifteen single-word runs above, which scatter from -8% to +33%. The
per-word quantity that scatters is the *surprise-injection jump*, a two-letter event. The
warm-up-versus-repeat firing ratio measured here is a phase average over 13 and 56 letters and is
far steadier: the widest row in 24 is 2.38x and the narrowest 2.72x. Both numbers are on the page,
and a reader who conflates them will think one of them is unstable when it is not.

One row worth keeping: run 2's `spprhibm` shows a repeat surprise of 0.157, roughly a hundred times
its neighbours, and still reads 2.60x. A word the model finds harder to repeat did not weaken the
effect. That is one observation, not a result.

**Measurement conditions.** Runs 2 and 3 were taken on 2026-09-08 after the n=8,192 training job
finished, on a quiet machine. Run 1's numbers stand as recorded. The browser tab was not visible
during any of them, which costs speed and nothing else: the first forward pass after a reload took
3,009 ms against roughly 200 ms in a visible tab, because Windows runs a hidden tab's renderer at
background quality-of-service. Chrome also clamps a hidden tab's timers hard after five minutes,
which stalled one attempt at word 7 of 8; a reload resets that. Neither affects the arithmetic, and
the ratios are the same to two decimals across the three runs.
