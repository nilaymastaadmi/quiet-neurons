"""Reproduce BDH paper Fig 14 (Sec 6.4) at a chosen neuron count n, and record
how strongly activation sparsity tracks input predictability.

Paper @ n=65536, d=256, L=4: layer 2 has 4.0-7.5% non-zero while memorizing a
new in-context word, ~2.5% while repeating it (ratio ~1.6-3.0x).

Usage:  python sparsity_scan.py --embd 128 --mult 64 --budget 10800
Appends one row per (layer, tensor) to scaling_results.csv

scaling_results.csv is a SINGLE-SAMPLE training-time scan and is not shipped. Every
published number comes from results/measured.csv, which measure.py writes by averaging
5 independently pinned eval samples. Expect the two to differ in the third decimal
(1.4729 here against a published 1.4705, for instance); that spread is the error bar,
and results/measured.csv is the source of record.
"""
import argparse, csv, os, time, torch, torch.nn.functional as F, bdh

ap = argparse.ArgumentParser()
ap.add_argument("--embd", type=int, default=64)
ap.add_argument("--mult", type=int, default=32)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--budget", type=int, default=2700, help="training seconds")
ap.add_argument("--steps", type=int, default=4000,
                help="length of the OneCycle schedule. Changing it changes the schedule, so "
                     "models trained with different values are NOT comparable.")
ap.add_argument("--word-pool", type=int, default=None,
                help="draw the 8-letter word each sequence from a fixed pool of K words. K=1 is "
                     "equivalent to --fixed-word; omit for the base task, where the word is "
                     "redrawn freely every sequence. Intermediate K keeps the in-context copy "
                     "task alive (the model must still read the context to know WHICH word) "
                     "while more of the word's letter content sits in the weights, which is the "
                     "point: it graduates provenance instead of switching it off.")
ap.add_argument("--fixed-word", action="store_true",
                help="provenance control: draw the 8-letter word ONCE and share it across every "
                     "sequence, so it lives in the weights like the warm-up instead of in the "
                     "context. Off by default and consumes no RNG when off.")
ap.add_argument("--stop-at", type=int, default=None,
                help="stop after this many steps WITHOUT shortening the schedule. This is how "
                     "you step-match models of different sizes: same --steps, same --stop-at, "
                     "so every model halts at the identical point on the identical schedule. "
                     "Cutting by wall clock instead leaves each model at a different learning "
                     "rate, which is the confound this exists to remove.")
ap.add_argument("--seed", type=int, default=1337)
a = ap.parse_args()

torch.manual_seed(a.seed)
VOCAB, WARM, WORD, REPS = 32, 13, 8, 8
PERIOD = WARM + WORD * REPS          # 77
NPER, B = 2, 16
T = PERIOD * NPER                    # 154
WARMUP_SEQ = torch.randint(0, 26, (WARM,))

# --fixed-word is the provenance control. Normally the warm-up is drawn once and shared by
# every sequence, so it ends up in the WEIGHTS, while the 8-letter word is redrawn per
# sequence and can only be known from the CONTEXT. That difference is the whole claim.
# With this flag the word is drawn once too, so identical content moves from context into
# weights and nothing else about the task changes. Note the draw is inside the branch: with
# the flag off no extra number is consumed from the RNG, so training is bit-identical to
# every run that produced the published numbers.
FIXED_WORD = torch.randint(0, 26, (WORD,)) if a.fixed_word else None
if FIXED_WORD is not None:
    print(f"FIXED WORD control: word={FIXED_WORD.tolist()} shared by every sequence", flush=True)
POOL = torch.randint(0, 26, (a.word_pool, WORD)) if a.word_pool else None
if POOL is not None:
    print(f"WORD POOL control: K={a.word_pool} words, {a.word_pool * WORD} letters in the "
          f"weights; identity still comes from the context", flush=True)

def make_batch(b=B):
    words = (POOL[torch.randint(0, a.word_pool, (b,))] if POOL is not None
             else FIXED_WORD.expand(b, WORD) if FIXED_WORD is not None
             else torch.randint(0, 26, (b, WORD)))
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
stop_at = a.stop_at if a.stop_at is not None else a.steps
while time.time() - t0 < a.budget and step < stop_at:
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
print(f"TRAINED n={n_neurons} steps={step} of a {a.steps}-step schedule "
      f"(stop_at={stop_at}) final_loss={final_loss:.4f}", flush=True)
torch.save({"model": model.state_dict(), "warmup": WARMUP_SEQ, "cfg": vars(a),
            # saved so a --fixed-word control can be re-measured on the word it was
            # actually trained on, without re-deriving it from the RNG stream
            "fixed_word": (FIXED_WORD.tolist() if FIXED_WORD is not None else None),
            # the pool a --word-pool model was trained on: measuring it on freshly drawn
            # random words would be testing it out of distribution
            "word_pool": (POOL.tolist() if POOL is not None else None),
            # D10: the ACTUAL step count reached, so a CSV row can never again depend on
            # the operator remembering to pass --steps-trained by hand
            "steps_trained": step},
           f"bdh_n{n_neurons}"
           f"{'_fixedword' if a.fixed_word else ''}"
           f"{'_pool%d' % a.word_pool if a.word_pool else ''}"
           f"{'_full%d' % a.steps if a.stop_at == a.steps else ''}.pt")

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
# pl[t] is the loss of PREDICTING token t+1, so the surprise of READING token t is pl[t-1].
# Without this shift the first-exposure slice drops the first genuinely new letter and picks
# up one the model has already learned, which reports first-sight loss below the random
# baseline. measure.py, which is the source of every published number, does the same.
pv = p.tolist()
surprise = [pv[(i - 1) % PERIOD] for i in range(PERIOD)]
first_expo = sum(surprise[WARM:WARM+WORD]) / WORD
repeats = sum(surprise[WARM+WORD:PERIOD]) / (PERIOD - WARM - WORD)
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
