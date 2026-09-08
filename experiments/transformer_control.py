# -*- coding: utf-8 -*-
"""Is the provenance signature a property of BDH, or of the task?

Every reader who has looked at this project has asked the same question, and until now the
answer was a table describing how BDH differs from a Transformer rather than a measurement.
This trains a dense Transformer on the identical task and measures the identical thing.

What is held fixed against sparsity_scan.py: the seed (1337), the warm-up sequence drawn from
it, the task construction, batch size, sequence length, AdamW at lr 3e-3 with weight decay 0.1,
a 4,000-step OneCycle schedule with pct_start 0.1, gradient clipping at 1.0, and --stop-at 2309.
The evaluation matches measure.py: five pinned samples of 16 batches of 16, so 1,280 sequences,
sliced into the same warm-up / first-sight / repeat blocks.

What changes is the architecture, which is the point. BDH-GPU's counted quantity is the
elementwise product of two rectified vectors. A dense Transformer has one rectifier per layer,
in the feed-forward hidden layer, so that is what is counted: the fraction of post-ReLU MLP
units strictly above zero.

One asymmetry to state rather than hide. Matching BDH's 2,048 countable units per layer at
d_model 64 gives an MLP hidden width of 2,048, which is about 1.1M parameters against BDH's
397k. The Transformer is the larger model here. If it fails to show the signature, it is not
because it was starved.

  python transformer_control.py --stop-at 2309 --budget 14400
"""
import argparse, csv, io, os, time, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--embd", type=int, default=64)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--hidden", type=int, default=2048, help="MLP hidden width = countable ReLU units")
ap.add_argument("--steps", type=int, default=4000, help="length of the OneCycle schedule")
ap.add_argument("--stop-at", type=int, default=2309,
                help="halt here without shortening the schedule, exactly as the BDH runs did")
ap.add_argument("--budget", type=int, default=14400)
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--eval-seed", type=int, default=4242)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--out", default="results/transformer_control.csv")
a = ap.parse_args()

torch.manual_seed(a.seed)
VOCAB, WARM, WORD, REPS = 32, 13, 8, 8
PERIOD = WARM + WORD * REPS          # 77
NPER, B = 2, 16
T = PERIOD * NPER                    # 154
EVAL_BATCHES = 16
WARMUP_SEQ = torch.randint(0, 26, (WARM,))   # drawn exactly where sparsity_scan.py draws it


def make_batch(b=B):
    words = torch.randint(0, 26, (b, WORD))
    block = torch.cat([WARMUP_SEQ.expand(b, WARM), words.repeat(1, REPS)], dim=1)
    seq = block.repeat(1, NPER + 1)
    return seq[:, :T].contiguous(), seq[:, 1:T + 1].contiguous()


class Block(nn.Module):
    """Pre-LN, causal self-attention, ReLU MLP.

    The ReLU is deliberate. A GELU has no exact zeros, so counting its "off" units would be a
    choice about a threshold rather than a measurement, and the whole comparison depends on
    counting the same kind of thing on both sides.
    """

    def __init__(self, d, h, hidden):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc_in = nn.Linear(d, hidden, bias=False)
        self.fc_out = nn.Linear(hidden, d, bias=False)

    def forward(self, x, sink=None):
        Bs, Ts, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (t.view(Bs, Ts, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(Bs, Ts, d))
        hidden = F.relu(self.fc_in(self.ln2(x)))
        if sink is not None:
            # fraction of hidden units strictly above zero, per position
            sink.append((hidden > 0).float().mean(dim=(0, 2)))
        return x + self.fc_out(hidden)


class Tf(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, a.embd)
        self.pos = nn.Embedding(T, a.embd)
        self.blocks = nn.ModuleList([Block(a.embd, a.heads, a.hidden) for _ in range(a.layers)])
        self.lnf = nn.LayerNorm(a.embd)
        self.head = nn.Linear(a.embd, VOCAB, bias=False)

    def forward(self, idx, targets=None, sinks=None):
        Bs, Ts = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(Ts, device=idx.device))
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if sinks is None else sinks.setdefault(i, []))
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        return logits, loss


model = Tf()
nparams = sum(p.numel() for p in model.parameters())
print("TRANSFORMER CONTROL d=%d L=%d heads=%d hidden=%d countable_units_per_layer=%d "
      "params=%s T=%d B=%d" % (a.embd, a.layers, a.heads, a.hidden, a.hidden,
                               format(nparams, ","), T, B), flush=True)
print("  BDH n=2048 for comparison: 397k params, 2048 countable units per layer", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=a.steps, pct_start=0.1)
t0, step, losses = time.time(), 0, []
while time.time() - t0 < a.budget and step < a.stop_at:
    x, y = make_batch()
    _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad()
    losses.append(loss.item()); step += 1
    if step % 50 == 0:
        print("step %5d loss %.4f elapsed %6.0fs"
              % (step, sum(losses[-50:]) / 50, time.time() - t0), flush=True)
final_loss = sum(losses[-50:]) / max(1, len(losses[-50:]))
print("TRAINED transformer steps=%d of a %d-step schedule (stop_at=%d) final_loss=%.4f"
      % (step, a.steps, a.stop_at, final_loss), flush=True)
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model": model.state_dict(), "warmup": WARMUP_SEQ,
            "cfg": vars(a), "steps_trained": step}, "checkpoints/transformer_control.pt")

model.eval()


def one_pass(eval_seed):
    torch.manual_seed(eval_seed)
    pl = torch.zeros(T)
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = make_batch()
            logits, _ = model(x, y)
            pl += F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                                  reduction="none").view(B, T).mean(0)
    pl /= EVAL_BATCHES
    pv = pl[:PERIOD].tolist()
    # pl[t] is the cost of PREDICTING token t+1, so the surprise of READING token t is pl[t-1].
    surprise = [pv[(i - 1) % PERIOD] for i in range(PERIOD)]
    fe = sum(surprise[WARM:WARM + WORD]) / WORD
    rp = sum(surprise[WARM + WORD:PERIOD]) / (PERIOD - WARM - WORD)

    torch.manual_seed(eval_seed)          # same words for the loss and the activations
    sinks = {}
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            xb, _ = make_batch()
            model(xb, None, sinks)
    out = {}
    for lev in sorted(sinks):
        v = torch.stack(sinks[lev]).mean(0)[:PERIOD * NPER].view(NPER, PERIOD).mean(0)
        out[lev] = (v[:WARM].mean().item(),
                    v[WARM:WARM + WORD].mean().item(),
                    v[WARM + WORD:].mean().item())
    return fe, rp, out


passes = [one_pass(a.eval_seed + i) for i in range(a.repeats)]
first_expo = sum(p[0] for p in passes) / len(passes)
repeats_loss = sum(p[1] for p in passes) / len(passes)
learned = bool(repeats_loss < 0.5 * first_expo and repeats_loss < 1.5)
print("PRECONDITION first_expo=%.4f repetition=%.4f TASK_LEARNED=%s  (%d eval passes, "
      "%d sequences)" % (first_expo, repeats_loss, learned, a.repeats,
                         a.repeats * EVAL_BATCHES * B), flush=True)


def mean(v):
    return sum(v) / len(v)


rows = []
for lev in sorted(passes[0][2]):
    wms = [p[2][lev][0] for p in passes]
    mms = [p[2][lev][1] for p in passes]
    rrs = [p[2][lev][2] for p in passes]
    ratios = [m / r for m, r in zip(mms, rrs) if r]
    warm_ratios = [w / r for w, r in zip(wms, rrs) if r]
    rows.append({"model": "transformer", "d": a.embd, "layers": a.layers,
                 "hidden_units": a.hidden, "params": nparams, "steps_trained": step,
                 "task_learned": learned, "first_expo_loss": round(first_expo, 4),
                 "repetition_loss": round(repeats_loss, 4), "layer": lev,
                 "tensor": "mlp_relu", "warmup": round(mean(wms), 5),
                 "mem": round(mean(mms), 5), "rep": round(mean(rrs), 5),
                 "mem_over_rep": round(mean(ratios), 4) if ratios else "",
                 "warm_over_rep": round(mean(warm_ratios), 4) if warm_ratios else "",
                 "ratio_min": round(min(ratios), 4) if ratios else "",
                 "ratio_max": round(max(ratios), 4) if ratios else "",
                 "eval_passes": a.repeats})
    r = rows[-1]
    print("  layer %d mlp_relu: WARM %.4f MEM %.4f REP %.4f  mem/rep %s  warm/rep %s"
          % (lev, r["warmup"], r["mem"], r["rep"], r["mem_over_rep"], r["warm_over_rep"]),
          flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
new = not os.path.exists(a.out)
with io.open(a.out, "a", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if new:
        w.writeheader()
    for r in rows:
        w.writerow(r)
print("appended %d rows to %s" % (len(rows), a.out), flush=True)
