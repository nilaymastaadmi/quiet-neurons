"""Measure the Figure 14 sparsity effect on an already-trained checkpoint.

sparsity_scan.py trains and measures in one process. This script measures only, so a
checkpoint can be re-measured without retraining, and so every model is compared on the
SAME pinned sample of random words.

That pinning matters. sparsity_scan.py draws its evaluation words after a full training
run has consumed the RNG stream, so its sample depends on training history and cannot be
reproduced from a checkpoint. Re-measuring the n=8192 checkpoint with a fresh sample moved
the layer-2 ratio from 1.983 to 1.971: not a bug, just a different draw. --repeats
quantifies exactly that, by re-measuring across several pinned samples and reporting the
spread. Numbers quoted anywhere else in this project come from this script, not from
sparsity_scan.py's inline measurement.

  python measure.py --ckpt checkpoints/bdh_n2048.pt --embd 64  --mult 32 --repeats 5 --append
  python measure.py --ckpt checkpoints/bdh_n8192.pt --embd 128 --mult 64 --repeats 5 --append
"""
import argparse, csv, os
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
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--eval-seed", type=int, default=20260908,
                help="pins the evaluation words. sparsity_scan.py draws its eval words "
                     "after training has consumed the RNG stream, so its sample is not "
                     "reproducible from a checkpoint alone; this is.")
ap.add_argument("--repeats", type=int, default=1,
                help="re-measure with eval_seed, eval_seed+1, ... and report the spread. "
                     "This is the error bar on the ratio.")
ap.add_argument("--csv", default="results/scaling_results.csv")
ap.add_argument("--append", action="store_true", help="append measured rows to --csv")
ap.add_argument("--out", default="results/measured.csv")
a = ap.parse_args()

# --- protocol, identical to sparsity_scan.py -------------------------------
VOCAB, WARM, WORD, REPS = a.vocab, 13, 8, 8
PERIOD = WARM + WORD * REPS          # 77
NPER, B = 2, 16
T = PERIOD * NPER                    # 154
EVAL_BATCHES = 16

torch.manual_seed(a.seed)
cfg = bdh.BDHConfig(n_layer=a.layers, n_embd=a.embd, n_head=a.heads,
                    mlp_internal_dim_multiplier=a.mult, vocab_size=VOCAB, dropout=0.0)
N = a.mult * a.embd // a.heads
n_neurons = a.heads * N

model = bdh.BDH(cfg)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
WARMUP_SEQ = ck["warmup"]
nparams = sum(p.numel() for p in model.parameters())

def make_batch(b=B):
    words = torch.randint(0, 26, (b, WORD))
    block = torch.cat([WARMUP_SEQ.expand(b, WARM), words.repeat(1, REPS)], dim=1)
    seq = block.repeat(1, NPER + 1)
    return seq[:, :T].contiguous(), seq[:, 1:T + 1].contiguous()

def fwd(m, idx, acts):
    C = m.config
    Bs, Ts = idx.size()
    Nn = C.n_embd * C.mlp_internal_dim_multiplier // C.n_head
    x = m.ln(m.embed(idx).unsqueeze(1))
    for lev in range(C.n_layer):
        xs = F.relu(x @ m.encoder)
        ys = F.relu(m.ln(m.attn(Q=xs, K=xs, V=x)) @ m.encoder_v)
        xy = xs * ys
        acts.setdefault(lev, []).append({"x": (xs > 0).float().mean(dim=(0, 1, 3)),
                                         "y": (ys > 0).float().mean(dim=(0, 1, 3)),
                                         "xy": (xy > 0).float().mean(dim=(0, 1, 3))})
        x = m.ln(x + m.ln(xy.transpose(1, 2).reshape(Bs, 1, Ts, Nn * C.n_head) @ m.decoder))

def one_pass(eval_seed):
    """One full measurement on a pinned sample of random words."""
    torch.manual_seed(eval_seed)
    pl = torch.zeros(T)
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = make_batch()
            logits, _ = model(x, y)
            pl += F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                                  reduction="none").view(B, T).mean(0)
    pl /= EVAL_BATCHES
    p = pl[:PERIOD]
    fe = p[WARM:WARM + WORD].mean().item()
    rp = p[WARM + WORD:].mean().item()

    torch.manual_seed(eval_seed)          # same words for loss and activations
    acts = {}
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            xb, _ = make_batch()
            fwd(model, xb, acts)
    out = {}
    for lev in sorted(acts):
        for key in ("x", "y", "xy"):
            v = torch.stack([d[key] for d in acts[lev]]).mean(0)[:PERIOD * NPER] \
                     .view(NPER, PERIOD).mean(0)
            out[(lev, key)] = (v[:WARM].mean().item(),
                               v[WARM:WARM + WORD].mean().item(),
                               v[WARM + WORD:].mean().item())
    return fe, rp, out

passes = [one_pass(a.eval_seed + i) for i in range(a.repeats)]
first_expo = sum(p[0] for p in passes) / len(passes)
repeats_loss = sum(p[1] for p in passes) / len(passes)
learned = bool(repeats_loss < 0.5 * first_expo and repeats_loss < 1.5)

rows = []
for key in passes[0][2]:
    lev, tname = key
    wms = [p[2][key][0] for p in passes]
    mms = [p[2][key][1] for p in passes]
    rrs = [p[2][key][2] for p in passes]
    ratios = [m / r for m, r in zip(mms, rrs) if r > 0]
    mean = lambda xs: sum(xs) / len(xs)
    rows.append({"n": n_neurons, "d": a.embd, "layers": a.layers, "params": nparams,
                 "layer": lev, "tensor": tname,
                 "warmup": round(mean(wms), 5), "mem": round(mean(mms), 5),
                 "rep": round(mean(rrs), 5),
                 "mem_over_rep": round(mean(ratios), 4) if ratios else "",
                 "ratio_min": round(min(ratios), 4) if ratios else "",
                 "ratio_max": round(max(ratios), 4) if ratios else "",
                 "eval_passes": a.repeats})

print(f"n={n_neurons} params={nparams:,}  TASK_LEARNED={learned}  "
      f"first_expo={first_expo:.4f} repetition={repeats_loss:.4f}  "
      f"({a.repeats} eval pass{'es' if a.repeats > 1 else ''}, "
      f"{a.repeats * EVAL_BATCHES * B} sequences)")
for r in rows:
    if r["tensor"] == "xy":
        spread = (f"  [{r['ratio_min']} .. {r['ratio_max']}]" if a.repeats > 1 else "")
        print(f"  layer {r['layer']} xy: MEM {r['mem']:.4f} REP {r['rep']:.4f} "
              f"ratio {r['mem_over_rep']}{spread}")

if a.append:
    fields = ["n", "d", "layers", "params", "task_learned", "first_expo_loss",
              "repetition_loss", "layer", "tensor", "warmup", "mem", "rep",
              "mem_over_rep", "ratio_min", "ratio_max", "eval_passes"]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            r = dict(r)
            r.update({"task_learned": learned,
                      "first_expo_loss": round(first_expo, 4),
                      "repetition_loss": round(repeats_loss, 4)})
            w.writerow(r)
    print(f"appended {len(rows)} rows to {a.out}")
