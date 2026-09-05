"""Reproduce BDH paper Fig 14 (Sec 6.4) at a chosen neuron count n, and record
how strongly activation sparsity tracks input predictability.

Paper @ n=65536, d=256, L=4: layer 2 has 4.0-7.5% non-zero while memorizing a
new in-context word, ~2.5% while repeating it (ratio ~1.6-3.0x).

Usage:  python sparsity_scan.py --embd 128 --mult 64 --budget 10800
Appends one row per (layer, tensor) to scaling_results.csv
"""
import argparse, csv, os, time, torch, torch.nn.functional as F, bdh

ap = argparse.ArgumentParser()
ap.add_argument("--embd", type=int, default=64)
ap.add_argument("--mult", type=int, default=32)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--budget", type=int, default=2700, help="training seconds")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--seed", type=int, default=1337)
a = ap.parse_args()

torch.manual_seed(a.seed)
VOCAB, WARM, WORD, REPS = 32, 13, 8, 8
PERIOD = WARM + WORD * REPS          # 77
NPER, B = 2, 16
T = PERIOD * NPER                    # 154
WARMUP_SEQ = torch.randint(0, 26, (WARM,))

def make_batch(b=B):
    words = torch.randint(0, 26, (b, WORD))
    block = torch.cat([WARMUP_SEQ.expand(b, WARM), words.repeat(1, REPS)], dim=1)
    return (lambda s: (s[:, :T].contiguous(), s[:, 1:T+1].contiguous()))(
        block.repeat(1, NPER + 1))

cfg = bdh.BDHConfig(n_layer=a.layers, n_embd=a.embd, n_head=a.heads,
                    mlp_internal_dim_multiplier=a.mult, vocab_size=VOCAB, dropout=0.0)
N = a.mult * a.embd // a.heads
n_neurons = a.heads * N
model = bdh.BDH(cfg)
nparams = sum(p.numel() for p in model.parameters())
print(f"n={n_neurons} d={a.embd} L={a.layers} params={nparams:,} T={T} B={B} "
      f"budget={a.budget}s", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=a.steps, pct_start=0.1)
t0, step, losses = time.time(), 0, []
while time.time() - t0 < a.budget and step < a.steps:
    x, y = make_batch()
    _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad()
    losses.append(loss.item()); step += 1
    if step % 50 == 0:
        print(f"step {step:5d} loss {sum(losses[-50:])/50:.4f} "
              f"elapsed {time.time()-t0:6.0f}s", flush=True)
final_loss = sum(losses[-50:]) / max(1, len(losses[-50:]))
print(f"TRAINED n={n_neurons} steps={step} final_loss={final_loss:.4f}", flush=True)
torch.save({"model": model.state_dict(), "warmup": WARMUP_SEQ, "cfg": vars(a)},
           f"bdh_n{n_neurons}.pt")

model.eval()
# precondition: per-position loss -> did it learn in-context copying?
pl = torch.zeros(T)
with torch.no_grad():
    for _ in range(16):
        x, y = make_batch(16)
        logits, _ = model(x, y)
        pl += F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                              reduction="none").view(16, T).mean(0)
pl /= 16
p = pl[:PERIOD]
first_expo, repeats = p[WARM:WARM+WORD].mean().item(), p[WARM+WORD:].mean().item()
learned = bool(repeats < 0.5 * first_expo and repeats < 1.5)
print(f"PRECONDITION first_exposure_loss={first_expo:.4f} repetition_loss={repeats:.4f} "
      f"TASK_LEARNED={learned}", flush=True)

acts = {}
def fwd(m, idx):
    C = m.config; Bs, Ts = idx.size()
    Nn = C.n_embd * C.mlp_internal_dim_multiplier // C.n_head
    x = m.ln(m.embed(idx).unsqueeze(1))
    for lev in range(C.n_layer):
        xs = F.relu(x @ m.encoder)
        ys = F.relu(m.ln(m.attn(Q=xs, K=xs, V=x)) @ m.encoder_v)
        xy = xs * ys
        acts.setdefault(lev, []).append({"x": (xs > 0).float().mean(dim=(0,1,3)),
                                         "y": (ys > 0).float().mean(dim=(0,1,3)),
                                         "xy": (xy > 0).float().mean(dim=(0,1,3))})
        x = m.ln(x + m.ln(xy.transpose(1,2).reshape(Bs,1,Ts,Nn*C.n_head) @ m.decoder))

with torch.no_grad():
    for _ in range(16):
        xb, _ = make_batch(16); fwd(model, xb)

new = not os.path.exists("scaling_results.csv")
with open("scaling_results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["n","d","layers","params","steps","final_loss","task_learned",
                    "first_expo_loss","repetition_loss","layer","tensor",
                    "warmup","mem","rep","mem_over_rep"])
    for lev in sorted(acts):
        for key in ("x","y","xy"):
            v = torch.stack([d[key] for d in acts[lev]]).mean(0)[:PERIOD*NPER].view(NPER,PERIOD).mean(0)
            wm = v[:WARM].mean().item(); mm = v[WARM:WARM+WORD].mean().item(); rr = v[WARM+WORD:].mean().item()
            w.writerow([n_neurons,a.embd,a.layers,nparams,step,round(final_loss,4),learned,
                        round(first_expo,4),round(repeats,4),lev,key,
                        round(wm,5),round(mm,5),round(rr,5),
                        round(mm/rr,4) if rr>0 else ""])
            if key=="xy":
                print(f"  layer {lev} xy: MEM {mm:.4f} REP {rr:.4f} ratio {mm/rr if rr>0 else 0:.2f}", flush=True)
print("WROTE scaling_results.csv", flush=True)
