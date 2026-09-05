/* Runs the BDH forward pass off the main thread so editing never freezes the page.
 * The model is small (397k parameters) but the attention is quadratic in sequence
 * length, so a run takes a noticeable fraction of a second and must not block input. */

// Import lazily inside ensure(), never at top level. A top-level await delays the
// self.onmessage assignment below, and any message the page posts before that lands
// has no handler and is silently dropped, so the page hangs on "computing" forever.
const BUILD = new URL(self.location.href).searchParams.get("v") || "0";
let loadBDH = null;

let model = null;
let pending = null;     // only the most recent request matters; drop stale ones
let running = false;

async function ensure() {
  if (!loadBDH) ({ loadBDH } = await import(`./bdh.js?v=${BUILD}`));
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
    const { logits, sparsity, scores, T } = m.forward(job.tokens, null, job.wantScores ?? null);
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
      scores: scores ? Array.from(scores) : null,
      // Per-letter surprise: the cross-entropy of the true next letter, in nats.
      // The page needs this to show that surprise and sparsity are NOT the same
      // signal. The fixed warm-up is predicted perfectly and still keeps neurons
      // busy; only context-learned text goes quiet.
      surprise: (() => {
        const V = m.m.vocab_size, out = new Array(T).fill(0);
        for (let t = 0; t < T - 1; t++) {
          const o = t * V;
          let mx = -Infinity;
          for (let v = 0; v < V; v++) if (logits[o + v] > mx) mx = logits[o + v];
          let sum = 0;
          for (let v = 0; v < V; v++) sum += Math.exp(logits[o + v] - mx);
          const target = job.tokens[t + 1];
          out[t] = -(logits[o + target] - mx - Math.log(sum));   // nats
        }
        out[T - 1] = NaN;                                        // no next letter to predict
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
