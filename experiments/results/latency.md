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
