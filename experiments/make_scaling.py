"""Build web/data/scaling.json from results/measured.csv.

Only rows written by measure.py are used, because only those come from a pinned
evaluation sample with a recorded spread. sparsity_scan.py's inline numbers are
excluded on purpose: its eval words depend on training history and cannot be
reproduced from a checkpoint.

The paper's entry is a BAND, not a point, because Figure 14 reports a range
(4.0-7.5% while memorizing, ~2.5% while repeating) rather than a single value.
"""
import csv, json, os

SRC = "results/measured.csv"
OUT = "../web/data/scaling.json"
LAYER, TENSOR = 2, "xy"
# measured.csv now holds two families: the original runs cut by a wall-clock budget at
# 1,854 / 1,917 / 2,309 steps, and the step-matched family in which all three stop at 2,309
# steps of the same 4,000-step schedule. Only the step-matched family is a controlled
# comparison, so it is selected explicitly. The old "later rows win" rule silently picked
# whichever family happened to be appended last, which is exactly the kind of positional
# assumption that publishes the wrong number without anyone noticing.
STEPS = 2309

points = []
if os.path.exists(SRC):
    seen = {}
    for r in csv.DictReader(open(SRC)):
        if (int(r["layer"]) == LAYER and r["tensor"] == TENSOR
                and str(r.get("steps_trained", "")).strip() == str(STEPS)):
            seen[int(r["n"])] = r
    for n in sorted(seen):
        r = seen[n]
        lo, hi = float(r["ratio_min"]), float(r["ratio_max"])
        points.append({
            "label": f"ours, {int(r['params']):,} params",
            "n": n,
            "ratio": round(float(r["mem_over_rep"]), 4),
            "spread": f"{lo:.3f} to {hi:.3f}",
            "peak": False,
            "note": f"{int(r['eval_passes']) * 256} sequences, {STEPS} steps",
            "tag": "tag-ok",
        })

points.append({
    "label": "Pathway, paper Fig. 14",
    "n": 65536,
    "ratio": None,
    "band": "1.6× to 3.0×",
    "lo": 1.6, "hi": 3.0,
    "spread": "reported as a range",
    "peak": True,
    "note": "not reproduced here",
    "tag": "tag-ok",
})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({
    "layer": LAYER,
    "tensor": TENSOR,
    "points": points,
    "note": ("Layer 2, xy product tensor. Ours are measured on models trained here; the "
             "paper's is read off Figure 14 of arXiv:2509.26507 and is not reproduced."),
}, open(OUT, "w"), indent=2)
print(f"wrote {OUT} with {len(points)} points: " +
      ", ".join(f"n={p['n']}" + (f" {p['ratio']}x" if p["ratio"] else " band") for p in points))
