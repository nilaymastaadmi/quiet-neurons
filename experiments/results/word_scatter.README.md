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

Measured 2026-09-07 against `checkpoints/bdh_n2048_s2309.pt`, the checkpoint the page ships.
