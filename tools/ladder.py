# -*- coding: utf-8 -*-
"""Apply the pre-registered discriminating test to the provenance ladder.

Reads the four conditions, prints the trend, and answers the one question the
pre-registration committed to in advance:

  If task difficulty is the only driver, all four points lie on one line between
  (loss 0.0004, ratio 1.035) and (loss 0.174, ratio 2.354). If provenance carries signal
  independent of difficulty, K=256 sits ABOVE that line.

Run:  python tools/ladder.py
"""
import csv, io, os, re, sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
R = "experiments/results"
LOGS = "experiments/stepmatched"


def rows(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(io.open(path, encoding="utf-8")))


def layer2(rs, want_pool=None):
    """warm/rep and mem/rep at layer 2, xy, for the matching row."""
    for r in rs:
        if r["layer"] != "2" or r["tensor"] != "xy":
            continue
        if want_pool is not None and str(r.get("word_pool", "")).strip() != str(want_pool):
            continue
        w, rep = float(r["warmup"]), float(r["rep"])
        return {"warm": w, "rep": rep, "warm_over_rep": w / rep,
                "mem_over_rep": float(r["mem_over_rep"]),
                # how much of the word the model must READ rather than RECALL. This is the
                # variable the ladder is actually ordered by, and it was in the CSV all along.
                "first_expo": float(r["first_expo_loss"])}
    return None


def cv_by_tensor(rs, want_pool=None):
    """Degeneracy: coefficient of variation across the 12 cells of each tensor."""
    out = {}
    for t in ("x", "y", "xy"):
        vals = []
        for r in rs:
            if r["tensor"] != t:
                continue
            if want_pool is not None and str(r.get("word_pool", "")).strip() != str(want_pool):
                continue
            vals += [float(r[c]) for c in ("warmup", "mem", "rep")]
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            out[t] = 100.0 * sd / m if m else 0.0
    return out


def final_loss_from_log(pattern):
    for name in sorted(os.listdir(LOGS)):
        if not re.search(pattern, name):
            continue
        txt = io.open(os.path.join(LOGS, name), encoding="utf-8", errors="replace").read()
        m = re.findall(r"final_loss=([0-9.]+)", txt)
        if m:
            return float(m[-1])
    return None


base = [r for r in rows(f"{R}/measured.csv")
        if r["n"] == "2048" and r["steps_trained"] == "2309"]
ctrl = rows(f"{R}/mechanism_control.csv")
ladder = rows(f"{R}/pool_ladder.csv")

CONDS = [
    ("K=1   (fixed word)", ctrl, None, 0.0004),
    ("K=16", ladder, 16, final_loss_from_log(r"pool16")),
    ("K=256", ladder, 256, final_loss_from_log(r"pool256")),
    ("K=inf (base task)", base, None, 0.1739),
]

print("=" * 74)
print("THE PROVENANCE LADDER")
print("prediction, committed in advance: warm/rep rises monotonically with K")
print("=" * 74)
print()
BASELINE = 3.258          # log 26, the surprise of a letter the model cannot know
print("  condition             final loss  1st-expo   in weights   warm/rep   mem/rep")
pts = []
missing = []
for label, rs, pool, loss in CONDS:
    d = layer2(rs, pool)
    if d is None:
        missing.append(label)
        print(f"  {label:<20}  (not measured yet)")
        continue
    ls = f"{loss:.4f}" if loss is not None else "  ?   "
    held = max(0.0, 100.0 * (1.0 - d["first_expo"] / BASELINE))
    print(f"  {label:<20}  {ls:>10}   {d['first_expo']:7.4f}   {held:8.1f}%"
          f"    {d['warm_over_rep']:6.3f}    {d['mem_over_rep']:6.3f}")
    if loss is not None:
        pts.append((label, loss, d["warm_over_rep"]))

if missing:
    print()
    print("  waiting on: " + ", ".join(missing))
    sys.exit(0)

print()
print("=" * 74)
print("DEGENERACY CHECK  (CV across the 12 cells per tensor; under 5% = collapsed)")
print("=" * 74)
degenerate = set()
for label, rs, pool, _ in CONDS:
    c = cv_by_tensor(rs, pool)
    if not c:
        continue
    flat = all(v < 5.0 for v in c.values())
    if flat:
        degenerate.add(label)
    print(f"  {label:<20} " + "  ".join(f"{t} {v:5.2f}%" for t, v in c.items())
          + ("   <-- COLLAPSED, ratio uninformative" if flat else ""))

print()
print("=" * 74)
print("MONOTONICITY")
print("=" * 74)
ratios = [r for _, _, r in pts]
mono = all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))
print(f"  {' < '.join(f'{r:.3f}' for r in ratios)}")
print(f"  monotonic in K: {'YES, as predicted' if mono else 'NO -- prediction FAILED'}")

print()
print("=" * 74)
print("THE DISCRIMINATING TEST")
print("=" * 74)
lo = min(pts, key=lambda p: p[1])
hi = max(pts, key=lambda p: p[1])
print(f"  line through {lo[0].strip()} (loss {lo[1]:.4f}, ratio {lo[2]:.3f})")
print(f"           and {hi[0].strip()} (loss {hi[1]:.4f}, ratio {hi[2]:.3f})")
print()
slope = (hi[2] - lo[2]) / (hi[1] - lo[1])
verdict_any = False
for label, loss, ratio in pts:
    if label in (lo[0], hi[0]):
        continue
    pred = lo[2] + slope * (loss - lo[1])
    resid = ratio - pred
    above = resid > 0
    print(f"  {label:<20} loss {loss:.4f}  ratio {ratio:.3f}  "
          f"line predicts {pred:.3f}  residual {resid:+.3f}"
          f"  {'ABOVE' if above else 'on/below'}")
    if label in degenerate:
        print(f"  {'':<20} ^ but this condition is DEGENERATE; its ratio is uninformative")
    elif above and abs(resid) > 0.05:
        verdict_any = True

print()
if verdict_any:
    print("  READING: at least one intermediate point sits above the line and at least one")
    print("  below it. The effect is not a monotone function of difficulty. State both")
    print("  residuals; do not call difficulty ruled out.")
else:
    print("  READING: the points lie on the difficulty line. Say exactly that:")
    print("  provenance and difficulty are not separable in this design, and we now have")
    print("  four points showing it rather than one. This is the registered negative.")

print()
print("=" * 74)
print("THE STEP  (added 2026-09-08: rank by what the model must READ, not by K)")
print("=" * 74)
# The registered prediction was a monotone rise in K and it failed. Ordered by first-exposure
# surprise, which measures how much of the word is NOT already in the weights, the same four
# points fall into two flat regimes. That is a different shape from a failed rise, and it is
# the shape the data has.
ranked = sorted(((layer2(rs, pool)["first_expo"], label, layer2(rs, pool))
                 for label, rs, pool, _ in CONDS if layer2(rs, pool)), key=lambda t: t[0])
LO, HI = [], []
for fe, label, d in ranked:
    (LO if d["mem_over_rep"] < 1.25 else HI).append((fe, label, d))
for name, group in (("word is in the WEIGHTS, no effect", LO),
                    ("word must be READ from context, full effect", HI)):
    print(f"  {name}:")
    for fe, label, d in group:
        print(f"    {label:<20} first-expo {fe:6.4f} nats"
              f"   {max(0.0, 100.0*(1.0-fe/BASELINE)):5.1f}% already known"
              f"   mem/rep {d['mem_over_rep']:6.3f}")
if LO and HI:
    print()
    print(f"  Within each regime the ratios agree to "
          f"{max(d['mem_over_rep'] for _, _, d in LO) - min(d['mem_over_rep'] for _, _, d in LO):.3f}"
          f" and "
          f"{max(d['mem_over_rep'] for _, _, d in HI) - min(d['mem_over_rep'] for _, _, d in HI):.3f}.")
    print(f"  The switch lies between first-exposure surprise "
          f"{max(fe for fe, _, _ in LO):.4f} and {min(fe for fe, _, _ in HI):.4f} nats.")
    print("  Two points bracket it. None locates it. Do not claim a threshold value.")
print("=" * 74)
