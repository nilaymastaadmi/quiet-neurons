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
ap.add_argument("--steps-trained", type=int, default=None,
                help="how many optimiser steps this checkpoint actually saw. Recorded in the "
                     "CSV so wall-clock-cut and step-matched runs of the same size cannot be "
                     "confused for each other. Supply it; the checkpoint does not carry it.")
ap.add_argument("--repeats", type=int, default=1,
                help="re-measure with eval_seed, eval_seed+1, ... and report the spread. "
                     "This is the error bar on the ratio.")
ap.add_argument("--csv", default="results/scaling_results.csv")
ap.add_argument("--append", action="store_true", help="append measured rows to --csv")
ap.add_argument("--out", default="results/measured.csv")
ap.add_argument("--export-trace", default=None,
                help="also write the per-letter sparsity curve to this JSON path, so the "
                     "explainer can overlay models too big to run in a browser")
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
    # pl[t] is the loss of PREDICTING token t+1, so the surprise of READING token t is
    # pl[t-1]. The sequence is exactly one period long and periodic, so index -1 wraps
    # to the last position of the previous period, which is the same letter.
    pv = p.tolist()
    surprise = [pv[(i - 1) % PERIOD] for i in range(PERIOD)]
    # Sliced on the aligned curve, the first-exposure block is the eight letters the
    # model is genuinely seeing for the first time, and lands on the random baseline
    # (log 26 = 3.258) as it should.
    fe = sum(surprise[WARM:WARM + WORD]) / WORD
    rp = sum(surprise[WARM + WORD:PERIOD]) / (PERIOD - WARM - WORD)

    torch.manual_seed(eval_seed)          # same words for loss and activations
    acts = {}
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            xb, _ = make_batch()
            fwd(model, xb, acts)
    out = {}
    curves = {}
    for lev in sorted(acts):
        for key in ("x", "y", "xy"):
            v = torch.stack([d[key] for d in acts[lev]]).mean(0)[:PERIOD * NPER] \
                     .view(NPER, PERIOD).mean(0)
            out[(lev, key)] = (v[:WARM].mean().item(),
                               v[WARM:WARM + WORD].mean().item(),
                               v[WARM + WORD:].mean().item())
            if key == "xy":
                curves[lev] = v.tolist()
    # p is the per-position loss over one period, in nats. It was already computed
    # for the precondition; returning it is what makes the correlation below
    # reproducible instead of a number from a script nobody kept.
    return fe, rp, out, curves, surprise

def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((p - mx) * (q - my) for p, q in zip(xs, ys))
    dx = sum((p - mx) ** 2 for p in xs) ** 0.5
    dy = sum((q - my) ** 2 for q in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _ranks(v):
    """Average ranks, so ties do not bias Spearman."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys))


passes = [one_pass(a.eval_seed + i) for i in range(a.repeats)]

# The claim the artifact teaches is that surprise is NOT the variable. That is a
# statement about a correlation, so the correlation has to be measured here rather
# than asserted in prose. Pearson picks up the coarse difference between phases;
# Spearman asks whether the two actually track position by position.
surprise_curve = [sum(p[4][i] for p in passes) / len(passes) for i in range(PERIOD)]
corr = []
for lev in sorted(passes[0][3]):
    spars = [sum(p[3][lev][i] for p in passes) / len(passes) for i in range(PERIOD)]
    fe_loss = surprise_curve[WARM:WARM + WORD]
    fe_spars = spars[WARM:WARM + WORD]
    corr.append({
        "n": n_neurons, "steps_trained": a.steps_trained if a.steps_trained is not None else "",
        "layer": lev, "tensor": "xy",
        "pearson": round(_pearson(surprise_curve, spars), 4),
        "spearman": round(_spearman(surprise_curve, spars), 4),
        "first_expo_loss_min": round(min(fe_loss), 4),
        "first_expo_loss_max": round(max(fe_loss), 4),
        "first_expo_sparsity_first": round(fe_spars[0], 5),
        "first_expo_sparsity_last": round(fe_spars[-1], 5),
        "positions": PERIOD, "eval_passes": a.repeats,
    })
if a.export_trace:
    import json
    # The per-letter curve, averaged over the same pinned samples the ratios come from.
    # The explainer overlays these so a reader can see models that are far too big to
    # run in a browser, clearly labelled as measured rather than live.
    layers = sorted(passes[0][3])
    json.dump({"n": n_neurons, "params": nparams, "period": PERIOD,
               "warm": WARM, "word": WORD,
               "eval_sequences": a.repeats * EVAL_BATCHES * B,
               # Surprise of READING the letter at each position, in nats: the
               # cross-entropy the model paid to predict that letter, aligned to the
               # same index as the activation curves below. Shipped so the surprise
               # numbers quoted in the README and the one-page summary are readable
               # out of a committed file rather than taken on trust.
               "surprise": [round(v, 6) for v in surprise_curve],
               "layers": {str(l): [round(sum(p[3][l][i] for p in passes) / len(passes), 6)
                                   for i in range(PERIOD)] for l in layers}},
              open(a.export_trace, "w"))
    print(f"wrote per-letter curves to {a.export_trace}")
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
                 "steps_trained": a.steps_trained if a.steps_trained is not None else "",
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

print("\n--- surprise vs sparsity, per position, one period, xy tensor ---")
print("    Pearson picks up the phase difference; Spearman asks whether they track letter by letter.")
for c in corr:
    print(f"  layer {c['layer']}: Pearson {c['pearson']:+.3f}  Spearman {c['spearman']:+.3f}"
          f"   first sight: surprise {c['first_expo_loss_min']:.2f} to "
          f"{c['first_expo_loss_max']:.2f} nats, sparsity "
          f"{c['first_expo_sparsity_first'] * 100:.1f}% to {c['first_expo_sparsity_last'] * 100:.1f}%")

if a.append:
    cfields = list(corr[0].keys())
    cpath = "results/correlations.csv"
    cnew = not os.path.exists(cpath)
    with open(cpath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cfields)
        if cnew:
            w.writeheader()
        for c in corr:
            w.writerow(c)
    print(f"appended {len(corr)} rows to {cpath}")

if a.append:
    fields = ["n", "d", "layers", "params", "steps_trained", "task_learned", "first_expo_loss",
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
