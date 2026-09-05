/* Runs the BDH forward pass off the main thread so editing never freezes the page.
 * The model is small (397k parameters) but the attention is quadratic in sequence
 * length, so a run takes a noticeable fraction of a second and must not block input. */

import { loadBDH } from "./bdh.js";

let model = null;
let pending = null;     // only the most recent request matters; drop stale ones
let running = false;

async function ensure() {
  if (!model) model = await loadBDH("data");
  return model;
}

async function drain() {
  if (running || !pending) return;
  running = true;
  const job = pending;
  pending = null;
  try {
    const m = await ensure();
    const t0 = performance.now();
    const { logits, sparsity, T } = m.forward(job.tokens);
    const ms = performance.now() - t0;
    // Float32Arrays transfer by copy through structuredClone; convert the small
    // per-position series to plain arrays and drop the large logits tensor, which
    // the page does not draw.
    self.postMessage({
      id: job.id,
      ms,
      T,
      sparsity: sparsity.map(s => ({
        layer: s.layer,
        x: Array.from(s.x), y: Array.from(s.y), xy: Array.from(s.xy),
      })),
      nNeurons: m.m.n_neurons,
      // top prediction at each position, so the page can show whether the model
      // is actually predicting the repeated word correctly
      argmax: (() => {
        const V = m.m.vocab_size, out = new Array(T);
        for (let t = 0; t < T; t++) {
          let best = 0, bv = -Infinity;
          for (let v = 0; v < V; v++) {
            const val = logits[t * V + v];
            if (val > bv) { bv = val; best = v; }
          }
          out[t] = best;
        }
        return out;
      })(),
    });
  } catch (e) {
    self.postMessage({ id: job.id, error: String(e && e.stack || e) });
  } finally {
    running = false;
    if (pending) drain();
  }
}

self.onmessage = (ev) => {
  pending = ev.data;      // supersede any queued-but-unstarted request
  drain();
};
