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
    const { logits, sparsity, scores, T } = m.forward(job.tokens, null, job.wantScores === true);
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
      scores: scores ? scores.map(a => Array.from(a)) : null,
      // Per-letter surprise: the cross-entropy of the true next letter, in nats.
      // The page needs this to show that surprise and sparsity are NOT the same
      // signal. The fixed warm-up is predicted perfectly and still keeps neurons
      // busy; only context-learned text goes quiet.
      surprise: (() => {
        const V = m.m.vocab_size;
        // raw[t] is what the model paid to PREDICT the letter at t+1.
        const raw = new Array(T).fill(NaN);
        for (let t = 0; t < T - 1; t++) {
          const o = t * V;
          let mx = -Infinity;
          for (let v = 0; v < V; v++) if (logits[o + v] > mx) mx = logits[o + v];
          let sum = 0;
          for (let v = 0; v < V; v++) sum += Math.exp(logits[o + v] - mx);
          const target = job.tokens[t + 1];
          raw[t] = -(logits[o + target] - mx - Math.log(sum));   // nats
        }
        // The surprise of READING letter t is raw[t-1], so shift by one. Without this
        // the page pairs a letter's activation count with the NEXT letter's surprise,
        // which is the same off-by-one that made the measured first-exposure loss come
        // out below the random baseline. Position 0 has no predecessor in this buffer.
        const out = new Array(T).fill(NaN);
        for (let t = 1; t < T; t++) out[t] = raw[t - 1];
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
