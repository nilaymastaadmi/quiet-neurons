# -*- coding: utf-8 -*-
"""Build dist/quiet-neurons-dataforge2026-nilay-toshniwal.zip from the working tree.

Copy to tools/build_zip.py and run from anywhere:  python tools/build_zip.py

The zip was previously assembled by hand and shipped without verify.sh and tools/, which
its own README tells the judge to run. Building it from a list makes the omission
impossible to repeat and lets tools/gate_checks.py zip verify the contents.
"""
import os, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "dist", "quiet-neurons-dataforge2026-nilay-toshniwal.zip")

FILES = ["START_HERE.txt", "README.md", "AI_DISCLOSURE.md", "LICENSE", "concept-summary.html",
         "verify.sh", "tools/sweep.py", "tools/ladder.py", "tools/gate_checks.py", "tools/build_zip.py",
         "experiments/LICENSES.md", "experiments/bdh.py", "experiments/sparsity_scan.py",
         "experiments/measure.py", "experiments/export_weights.py", "experiments/make_scaling.py",
         "experiments/scaling_results.csv"]
DIRS = ["experiments/results", "experiments/stepmatched", "web"]
EXCLUDE_EXT = (".pyc",)
EXCLUDE_DIRS = ("__pycache__",)

def walk(d):
    for root, dirs, names in os.walk(os.path.join(ROOT, d)):
        dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS]
        for n in sorted(names):
            if n.endswith(EXCLUDE_EXT):
                continue
            yield os.path.relpath(os.path.join(root, n), ROOT).replace("\\", "/")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
names = list(FILES)
for d in DIRS:
    names += list(walk(d))
missing = [n for n in names if not os.path.exists(os.path.join(ROOT, n))]
if missing:
    raise SystemExit("refusing to build: missing " + ", ".join(missing))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for n in names:
        z.write(os.path.join(ROOT, n), n)
print("wrote %s with %d files, %.2f MB" % (OUT, len(names), os.path.getsize(OUT) / 1e6))
