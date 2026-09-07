# Interaction latency, measured 2026-09-07

The problem statement's design standards say "Fast feedback. Controls should respond in under a
second." This is what the page actually does, measured on a quiet machine rather than asserted.

## Result

Eight word changes, input event to the readout going live, one 2,048-neuron model, T=77:

| | ms |
|---|---|
| **median total, input to readout** | **1,010** |
| of which the forward pass | 868 |
| of which worker round-trip and transfer | 19 to 43 |
| of which render | ~97 typical |
| **first change after page load** | **1,793** |

Individual totals: 1793, 1011, 989, 1010, and separately 1670, 989.

So the steady-state interaction is **about one second**, and the first one is roughly 1.8 s
because it also pays for layout and JIT.

## Rerun, 2026-09-08 04:5x, on the final build

Same script, quiet machine, n=8,192 training job finished. Four word changes:

| | ms |
|---|---|
| **median total after discarding the warm-up** | **1,004** |
| individual totals | 1577 (warm-up, discarded), 1004, 1068, 930 |
| forward pass, all four | 853, 877, 873, 851 |

The steady state reproduces the 2026-09-07 record to within 1%: 1,004 against 1,010 ms total, and
853-877 against 868 ms of forward pass. The batch panel and the standfirst counter added since then
cost nothing measurable, which is what you would expect: neither runs during a word change.

**One condition of the recorded method was not met.** The browser tab was not visible during this
rerun, because the machine was unattended at the time. That matters for exactly one number: the
very first pass after a page load measured **3,009 ms** here against the 1,793 ms in the table
above, since Windows runs a hidden tab's renderer at background quality-of-service and the first
pass is the one that pays for JIT compilation. It did not affect the steady state, which is the
number the design standard is about and which is within 6 ms of the record. **A strict rerun with
the tab visible is still worth thirty seconds** before quoting the first-pass figure to anyone.

## Why this is worth recording

An external audit measured 2,606 / 2,002 / 1,002 ms and concluded the page runs at 1.7 to 2.6 s,
which would clearly fail the standard. That measurement was not wrong: their third sample, 1,002
ms, is exactly the steady state, and the first two were warm-up. Quoting the first runs of a
warm-dependent measurement is the same mistake that produced this project's earlier "the loop
unroll does nothing" reading. The honest summary is one second steady state, not two.

It also means the obvious optimisation targets are not where they looked. **Serialisation is not
the problem**: moving the result back from the worker costs 19 to 43 ms, because the score
matrices are transferred as typed arrays rather than copied. **The forward pass is 86% of the
cost.** Getting under a second reliably needs the arithmetic to get faster (WASM or SIMD), not
plumbing changes, and that was out of scope this close to the deadline.

## To reproduce

Serve `web/`, open `index.html`, let it settle, then in the console:

```js
const $ = s => document.querySelector(s);
const settle = async () => { for (let i = 0; i < 400; i++) {
  await new Promise(r => setTimeout(r, 15));
  if ($('#runbadge').textContent.trim() === 'live') return performance.now(); } };

const inp = $('#word'), tot = [], fwd = [];
for (const w of ['aaaabbbb','ccccdddd','eeeeffff','gggghhhh']) {
  inp.value = w;
  const t0 = performance.now();
  inp.dispatchEvent(new Event('input', { bubbles: true }));
  const t2 = await settle();
  tot.push(Math.round(t2 - t0));
  fwd.push(+($('#perf').textContent.match(/: (\d+) ms/) || [])[1]);
}
console.log({ totals: tot, forward: fwd });
```

Discard the first sample; it is the warm-up. `render()` runs synchronously on the main thread, so
the polling loop cannot tick during it, which is what makes this loop measure render time rather
than miss it.

## Two optimisations that were proposed, checked, and rejected

An external audit proposed getting the median under 800 ms two ways. Both were investigated
against the code rather than assumed, and neither survives.

**"Cache the RoPE tables across word changes (82 ms)."** Already done. `bdh.js` `_rope(T)` opens
with `if (this._ropeT === T) return;`, and T is a constant 77 across word changes, so the cos/sin
tables are built once at first run and reused for the life of the page. The 82 ms the profiler
bills to `rope` is not table construction; it is *applying* the rotation to `x_sparse`, which is
different on every pass by definition. There is nothing there to cache.

**"Count sparsity only for the displayed layer while typing."** This would save about 50 ms of
1,010 ms, roughly 5%, because `count` is ~67 ms spread across four layers. It would also make the
layer selector, which is currently **instant** because switching layers is a pure re-render of
already-computed numbers, cost a full ~870 ms recompute. Moving one control from 0 ms to 870 ms in
order to move another from 1,010 ms to 960 ms is a bad trade on any page, and a worse one on a
page judged against "controls should respond in under a second".

**What would actually work**, and what we are not doing before the deadline: the forward pass is
**86% of the interaction** and `xs` + `ys` + `scores` are ~85% of that. Those are dense
float32 matmuls in scalar JavaScript. Getting materially under 800 ms means WASM or SIMD, which is
a rewrite of the file that `parity.html` exists to guard, plus a full re-verification of every
number the page prints. That is the right change and it is the wrong week.

So the honest position stands: **median 1.01 s, which is at the standard rather than under it.**
The cost is stated at the control, and the reason it is not lower is stated here.
