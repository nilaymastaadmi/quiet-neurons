/* BDH forward pass in the browser, batch size 1.
 *
 * Mirrors pathwaycom/bdh bdh.py (MIT, Copyright 2025 Pathway Technology, Inc.)
 * line for line. Every step below names the PyTorch line it reproduces, because
 * the whole page is worthless if this drifts from the reference implementation.
 *
 * Verified against experiments/export_weights.py output by parity.html.
 */

const EPS = 1e-5;

/* torch.nn.LayerNorm(D, elementwise_affine=False, bias=False), applied per row */
function layerNorm(buf, rows, D) {
  for (let r = 0; r < rows; r++) {
    const o = r * D;
    let mean = 0;
    for (let d = 0; d < D; d++) mean += buf[o + d];
    mean /= D;
    let varsum = 0;
    for (let d = 0; d < D; d++) { const v = buf[o + d] - mean; varsum += v * v; }
    const inv = 1 / Math.sqrt(varsum / D + EPS);
    for (let d = 0; d < D; d++) buf[o + d] = (buf[o + d] - mean) * inv;
  }
}

export class BDH {
  /* manifest: parsed manifest.json,  blob: ArrayBuffer of weights.bin */
  constructor(manifest, blob) {
    const all = new Float32Array(blob);
    this.m = manifest;
    this.w = {};
    for (const t of manifest.tensors) {
      this.w[t.name] = all.subarray(t.offset, t.offset + t.count);
    }
    const { N, n_head } = manifest;
    // Attention.phases_cos_sin precomputes per (position, neuron); cache lazily
    // because T changes when the learner edits the text.
    this._ropeT = -1;
    this._cos = null;
    this._sin = null;
    this._N = N;
    this._nh = n_head;
  }

  /* r_phases = arange(T) * freqs ; phases = (phases % 1) * 2pi  -> cos, sin
     Matches Attention.forward + phases_cos_sin. freqs are pair-quantized in
     get_freqs (floor(i/2)*2), so cos/sin repeat across each rotation pair. */
  _rope(T) {
    if (this._ropeT === T) return;
    const N = this._N, freqs = this.w.freqs;
    const cos = new Float32Array(T * N), sin = new Float32Array(T * N);
    for (let t = 0; t < T; t++) {
      for (let n = 0; n < N; n++) {
        let p = (t * freqs[n]) % 1;
        if (p < 0) p += 1;                       // JS % keeps sign, Python's does not
        const ang = p * 2 * Math.PI;
        cos[t * N + n] = Math.cos(ang);
        sin[t * N + n] = Math.sin(ang);
      }
    }
    this._cos = cos; this._sin = sin; this._ropeT = T;
  }

  /* Returns {logits, sparsity} where sparsity[layer] = {x,y,xy} arrays of
     length T giving the fraction of neurons active at each position. */
  forward(tokens) {
    const { D, N, n_head: nh, n_layer: L, vocab_size: V } = this.m;
    const T = tokens.length;
    const { embed, encoder, encoder_v, decoder, lm_head } = this.w;
    this._rope(T);
    const cos = this._cos, sin = this._sin;

    // x = self.ln(self.embed(idx).unsqueeze(1))
    const x = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const src = tokens[t] * D, dst = t * D;
      for (let d = 0; d < D; d++) x[dst + d] = embed[src + d];
    }
    layerNorm(x, T, D);

    const xs = new Float32Array(nh * T * N);   // x_sparse
    const qr = new Float32Array(nh * T * N);   // QR (K is Q, so KR === QR)
    const ys = new Float32Array(nh * T * N);   // y_sparse
    const ykv = new Float32Array(nh * T * D);
    const ymlp = new Float32Array(T * D);
    const sparsity = [];

    for (let l = 0; l < L; l++) {
      // x_sparse = relu(x @ self.encoder)
      xs.fill(0);
      for (let h = 0; h < nh; h++) {
        const eBase = h * D * N, oBase = h * T * N;
        for (let t = 0; t < T; t++) {
          const xo = t * D, so = oBase + t * N;
          for (let d = 0; d < D; d++) {
            const xv = x[xo + d];
            if (xv === 0) continue;
            const eo = eBase + d * N;
            for (let n = 0; n < N; n++) xs[so + n] += xv * encoder[eo + n];
          }
          for (let n = 0; n < N; n++) if (xs[so + n] < 0) xs[so + n] = 0;
        }
      }

      // QR = rope(r_phases, Q) with v_rot[2k] = -v[2k+1], v_rot[2k+1] = v[2k]
      for (let h = 0; h < nh; h++) {
        const b = h * T * N;
        for (let t = 0; t < T; t++) {
          const o = b + t * N, po = t * N;
          for (let n = 0; n < N; n += 2) {
            const a = xs[o + n], c = xs[o + n + 1];
            qr[o + n]     = a * cos[po + n]         + (-c) * sin[po + n];
            qr[o + n + 1] = c * cos[po + n + 1]     + ( a) * sin[po + n + 1];
          }
        }
      }

      // scores = (QR @ KR.mT).tril(diagonal=-1) ; yKV = scores @ V, V = x
      ykv.fill(0);
      for (let h = 0; h < nh; h++) {
        const qb = h * T * N, kb = h * T * D;
        for (let t = 1; t < T; t++) {           // row 0 is all-zero after tril(-1)
          const qo = qb + t * N, yo = kb + t * D;
          for (let s = 0; s < t; s++) {         // strictly lower triangular
            const so = qb + s * N;
            let dot = 0;
            for (let n = 0; n < N; n++) dot += qr[qo + n] * qr[so + n];
            if (dot === 0) continue;
            const xo = s * D;
            for (let d = 0; d < D; d++) ykv[yo + d] += dot * x[xo + d];
          }
        }
      }
      layerNorm(ykv, nh * T, D);                // yKV = self.ln(yKV)

      // y_sparse = relu(yKV @ self.encoder_v) ; xy_sparse = x_sparse * y_sparse
      ys.fill(0);
      for (let h = 0; h < nh; h++) {
        const eBase = h * D * N, oBase = h * T * N, kb = h * T * D;
        for (let t = 0; t < T; t++) {
          const yo = kb + t * D, so = oBase + t * N;
          for (let d = 0; d < D; d++) {
            const yv = ykv[yo + d];
            if (yv === 0) continue;
            const eo = eBase + d * N;
            for (let n = 0; n < N; n++) ys[so + n] += yv * encoder_v[eo + n];
          }
          for (let n = 0; n < N; n++) ys[so + n] = ys[so + n] > 0 ? ys[so + n] : 0;
        }
      }

      // record what the page draws, before xy is consumed
      const sx = new Float32Array(T), sy = new Float32Array(T), sxy = new Float32Array(T);
      for (let t = 0; t < T; t++) {
        let cx = 0, cy = 0, cxy = 0;
        for (let h = 0; h < nh; h++) {
          const o = h * T * N + t * N;
          for (let n = 0; n < N; n++) {
            if (xs[o + n] > 0) cx++;
            if (ys[o + n] > 0) cy++;
            if (xs[o + n] > 0 && ys[o + n] > 0) cxy++;
          }
        }
        const tot = nh * N;
        sx[t] = cx / tot; sy[t] = cy / tot; sxy[t] = cxy / tot;
      }
      sparsity.push({ layer: l, x: sx, y: sy, xy: sxy });

      // yMLP = xy_sparse.transpose(1,2).reshape(B,1,T,N*nh) @ self.decoder
      // head-major: decoder row index is h*N + n
      ymlp.fill(0);
      for (let t = 0; t < T; t++) {
        const mo = t * D;
        for (let h = 0; h < nh; h++) {
          const so = h * T * N + t * N, rowBase = h * N;
          for (let n = 0; n < N; n++) {
            const v = xs[so + n] * ys[so + n];
            if (v === 0) continue;
            const dr = (rowBase + n) * D;
            for (let d = 0; d < D; d++) ymlp[mo + d] += v * decoder[dr + d];
          }
        }
      }
      layerNorm(ymlp, T, D);                    // y = self.ln(yMLP)
      for (let i = 0; i < T * D; i++) x[i] += ymlp[i];
      layerNorm(x, T, D);                       // x = self.ln(x + y)
    }

    // logits = x @ self.lm_head
    const logits = new Float32Array(T * V);
    for (let t = 0; t < T; t++) {
      const xo = t * D, lo = t * V;
      for (let d = 0; d < D; d++) {
        const xv = x[xo + d];
        if (xv === 0) continue;
        const wo = d * V;
        for (let v = 0; v < V; v++) logits[lo + v] += xv * lm_head[wo + v];
      }
    }
    return { logits, sparsity, T };
  }
}

/* Build the paper's Section 6.4 input: a fixed warm-up, then one word repeated. */
export function buildSequence(warmup, word, reps, periods) {
  const block = warmup.concat(...Array(reps).fill(word));
  const out = [];
  while (out.length < block.length * periods) out.push(...block);
  return out.slice(0, block.length * periods);
}

export async function loadBDH(base = "data") {
  const [manifest, blob] = await Promise.all([
    fetch(`${base}/manifest.json`).then(r => r.json()),
    fetch(`${base}/weights.bin`).then(r => r.arrayBuffer()),
  ]);
  return new BDH(manifest, blob);
}
