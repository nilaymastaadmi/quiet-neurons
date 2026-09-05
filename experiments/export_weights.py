"""Export a trained BDH checkpoint for the browser, plus a reference trace the
JavaScript port must reproduce.

Produces three files in web/data/:
  weights.bin    float32, concatenated in the order listed in manifest.json
  manifest.json  config, tensor shapes and byte offsets
  reference.json a fixed input and the activations PyTorch produces for it

reference.json is the parity test. If the JS forward pass does not reproduce
these numbers, every figure the page shows is wrong.

Usage:
  python export_weights.py --ckpt checkpoints/bdh_n2048.pt --embd 64 --mult 32
"""
import argparse, json, os, struct
import torch
import torch.nn.functional as F
import bdh

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--embd", type=int, required=True)
ap.add_argument("--mult", type=int, required=True)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--vocab", type=int, default=32)
ap.add_argument("--out", default="../web/data")
a = ap.parse_args()

VOCAB, WARM, WORD, REPS = a.vocab, 13, 8, 8
PERIOD = WARM + WORD * REPS          # 77
NPER = 2
T = PERIOD * NPER                    # 154

cfg = bdh.BDHConfig(n_layer=a.layers, n_embd=a.embd, n_head=a.heads,
                    mlp_internal_dim_multiplier=a.mult, vocab_size=VOCAB, dropout=0.0)
N = a.mult * a.embd // a.heads
n_neurons = a.heads * N

model = bdh.BDH(cfg)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
warmup = ck["warmup"]

os.makedirs(a.out, exist_ok=True)

# ---------------------------------------------------------------- weights ---
# Order matters: the JS reader walks this list in sequence.
tensors = [
    ("embed",     model.embed.weight.detach()),         # (vocab, D)
    ("encoder",   model.encoder.detach()),              # (nh, D, N)
    ("encoder_v", model.encoder_v.detach()),            # (nh, D, N)
    ("decoder",   model.decoder.detach()),              # (nh*N, D)
    ("lm_head",   model.lm_head.detach()),              # (D, vocab)
    ("freqs",     model.attn.freqs.detach().view(-1)),  # (N,)
]
blob, entries, off = bytearray(), [], 0
for name, t in tensors:
    vals = t.contiguous().view(-1).to(torch.float32).tolist()
    entries.append({"name": name, "shape": list(t.shape),
                    "offset": off, "count": t.numel()})
    blob += struct.pack("<%df" % len(vals), *vals)   # little-endian float32
    off += t.numel()

with open(os.path.join(a.out, "weights.bin"), "wb") as f:
    f.write(blob)

manifest = {
    "n_neurons": n_neurons, "N": N, "D": a.embd, "n_head": a.heads,
    "n_layer": a.layers, "vocab_size": VOCAB,
    "period": PERIOD, "warm": WARM, "word": WORD, "reps": REPS,
    "dtype": "float32", "tensors": entries,
    "source_checkpoint": os.path.basename(a.ckpt),
}
with open(os.path.join(a.out, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# -------------------------------------------------------------- reference ---
torch.manual_seed(7)
word = torch.randint(0, 26, (WORD,))
block = torch.cat([warmup, word.repeat(REPS)])          # 77
idx = block.repeat(NPER + 1)[:T].unsqueeze(0)           # 1, T

acts = []
def fwd(m, ids):
    C = m.config
    Bs, Ts = ids.size()
    Nn = C.n_embd * C.mlp_internal_dim_multiplier // C.n_head
    x = m.ln(m.embed(ids).unsqueeze(1))
    for lev in range(C.n_layer):
        xs = F.relu(x @ m.encoder)
        yKV = m.ln(m.attn(Q=xs, K=xs, V=x))
        ys = F.relu(yKV @ m.encoder_v)
        xy = xs * ys
        acts.append({
            "layer": lev,
            "x":  (xs > 0).float().mean(dim=(0, 1, 3)).tolist(),
            "y":  (ys > 0).float().mean(dim=(0, 1, 3)).tolist(),
            "xy": (xy > 0).float().mean(dim=(0, 1, 3)).tolist(),
        })
        x = m.ln(x + m.ln(xy.transpose(1, 2).reshape(Bs, 1, Ts, Nn * C.n_head) @ m.decoder))
    return x.view(Bs, Ts, C.n_embd) @ m.lm_head

with torch.no_grad():
    logits = fwd(model, idx)

ref = {
    "tokens": idx.view(-1).tolist(),
    "word": word.tolist(),
    "sparsity": acts,
    # full logits are the strictest check available
    "logits": [round(v, 6) for v in logits.view(-1).tolist()],
    "logits_shape": [T, VOCAB],
}
with open(os.path.join(a.out, "reference.json"), "w") as f:
    json.dump(ref, f)

l2 = acts[2]
mem = sum(l2["xy"][WARM:WARM + WORD]) / WORD
rep = sum(l2["xy"][WARM + WORD:PERIOD]) / (PERIOD - WARM - WORD)
print(f"exported n={n_neurons} D={a.embd} N={N} -> {a.out}")
print(f"  weights.bin   {len(blob)/1024:.0f} KB ({off:,} float32)")
print(f"  reference.json {os.path.getsize(os.path.join(a.out,'reference.json'))/1024:.0f} KB")
print(f"  layer2 xy on this fixed input: MEM {mem:.4f} REP {rep:.4f} ratio {mem/rep:.2f}")
