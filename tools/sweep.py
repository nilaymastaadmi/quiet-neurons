# -*- coding: utf-8 -*-
"""Mechanical sweep of the submission surfaces. Run from anywhere: python tools/sweep.py

Called by verify.sh. Checks that every number on every surface still traces to
experiments/results/measured.csv, that every referenced path exists, that the
disclosure covers every shipped file, and that no superseded value crept back in.

The previous version read measured.csv by column POSITION. Adding `steps_trained`
at index 4 shifted every field after it, so its filters silently matched nothing
and it would have reported a clean run over an empty set. Everything below reads
by column NAME and filters the published family explicitly, and the loader fails
loudly if the filter selects zero rows.
"""
import csv, io, json, os, re, sys, zipfile

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
STEPS = "2309"          # the published, step-matched family
problems = []


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


def head(n, t):
    print()
    print("=" * 74)
    print("%d. %s" % (n, t))
    print("=" * 74)


rows = list(csv.DictReader(io.open("experiments/results/measured.csv", encoding="utf-8")))
pub = [r for r in rows if r["steps_trained"] == STEPS]
if not pub:
    print("FATAL: no rows at steps_trained=%s -- the filter is broken, not the data" % STEPS)
    sys.exit(2)
R = {(int(r["n"]), int(r["layer"]), r["tensor"]): r for r in pub}

head(1, "RELATIVE PATHS REFERENCED IN PROSE THAT DO NOT EXIST")
# Resolve by basename anywhere in the tree. The previous version searched only the
# repo root, experiments/ and web/, so it called six files missing that all exist
# one level deeper (results/, web/data/, checkpoints/).
ondisk = {}
for root, dirs, names in os.walk("."):
    dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", "dist", "bdhvenv")]
    for nm in names:
        ondisk.setdefault(nm, os.path.join(root, nm).replace("\\", "/").lstrip("./"))
docs = ["README.md", "START_HERE.txt", "AI_DISCLOSURE.md", "experiments/LICENSES.md",
        "concept-summary.html"]
pat = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|js|html|json|csv|md|bin|pt|log|pdf|txt|yml))`")
seen = set()
for d in docs:
    for m in sorted(set(pat.findall(read(d)))):
        cand = m.lstrip("./")
        if cand.startswith("../"):
            cand = cand[3:]
        hit = (os.path.exists(cand) or os.path.basename(cand) in ondisk or "*" in cand)
        if not hit and (d, m) not in seen:
            seen.add((d, m))
            problems.append("missing path %r referenced in %s" % (m, d))
            print("  MISSING  %-42s referenced in %s" % (m, d))
print("  (nothing listed above = every referenced file exists)")

head(2, "TRACE FILES: SHAPE AND KEYS")
for n in (2048, 8192, 16384):
    d = json.load(open("web/data/traces/n%d.json" % n))
    ok = "surprise" in d and "layers" in d and "loss" not in d
    print("  n=%-6d keys=%s" % (n, ",".join(sorted(d.keys()))))
    if not ok:
        problems.append("trace n=%d has wrong keys" % n)
        print("     ^^ PROBLEM")
    if len(d["surprise"]) != d["period"] or len(d["layers"]["2"]) != d["period"]:
        problems.append("trace n=%d length mismatch" % n)
        print("     ^^ LENGTH MISMATCH")

head(3, "WHO READS THE MEASUREMENT CSVs, AND HOW")
# Only consumers matter. A script that merely names the CSV as an argparse default is
# a producer; flagging it as a positional reader was a false positive last run.
READ = re.compile(r"DictReader|csv\.reader|open\((?![^)]*['\"][aw])[^)]*(?:measured|correlations)")
for root, dirs, names in os.walk("."):
    dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", "dist", "checkpoints",
                                            "bdhvenv", "stepmatched")]
    for nm in sorted(names):
        if not nm.endswith(".py"):
            continue
        p = os.path.join(root, nm).replace("\\", "/").lstrip("./")
        src = read(p)
        if "measured.csv" not in src and "correlations.csv" not in src:
            continue
        if not READ.search(src):
            print("  %-34s writes only, never reads  ok" % p)
            continue
        byname = "DictReader" in src
        print("  %-34s %s" % (p, "reads by NAME (DictReader)" if byname
                              else "reads by POSITION -- breaks when a column is inserted"))
        if not byname:
            problems.append("%s reads the CSV by position" % p)
        if "steps_trained" in src:
            print("  %-34s   filters on steps_trained  ok" % "")
        else:
            problems.append("%s does not filter the training family" % p)
            print("  %-34s   NO family filter -- may mix 1854/1917 with 2309" % "")

head(4, "CROSS-SURFACE NUMBER AGREEMENT (step-matched family only)")
rd, ix, cs = read("README.md"), read("web/index.html"), read("concept-summary.html")
sj = json.load(open("web/data/scaling.json"))
sjr = {p["n"]: p.get("ratio") for p in sj["points"] if p.get("ratio") is not None}

for n, token in ((2048, "1.47"), (8192, "2.12"), (16384, "1.87")):
    m = float(R[(n, 2, "xy")]["mem_over_rep"])
    where = [w for w, t in (("README", rd), ("page", ix), ("PDF", cs)) if token in t]
    agree_prose = abs(m - float(token)) < 0.005
    agree_json = abs(sjr[n] - m) < 1e-9
    ok = agree_prose and agree_json and len(where) == 3
    print("  layer-2 xy n=%-6d csv %.4f  scaling.json %.4f  token %s in %-22s %s"
          % (n, m, sjr[n], token, "+".join(where) or "NOWHERE", "ok" if ok else "MISMATCH"))
    if not ok:
        problems.append("layer-2 n=%d disagreement" % n)

for n, layer, claim in ((2048, 0, 0.80), (8192, 0, 0.80), (16384, 0, 0.86),
                        (2048, 3, 0.97), (8192, 3, 0.95), (16384, 3, 1.11)):
    m = float(R[(n, layer, "xy")]["mem_over_rep"])
    ok = abs(m - claim) < 0.006
    print("  layer-%d n=%-6d csv %.4f  documented %.2f  %s"
          % (layer, n, m, claim, "ok" if ok else "MISMATCH"))
    if not ok:
        problems.append("layer-%d n=%d value" % (layer, n))

for t, claim in (("x", 1.40), ("y", 1.27), ("xy", 2.12)):
    m = float(R[(8192, 2, t)]["mem_over_rep"])
    ok = abs(m - claim) < 0.006
    print("  tensor %-2s n=8192 L2  csv %.4f  defense sheet %.2f  %s"
          % (t, m, claim, "ok" if ok else "MISMATCH"))
    if not ok:
        problems.append("tensor %s row" % t)

corr = {}
for r in csv.DictReader(io.open("experiments/results/correlations.csv", encoding="utf-8")):
    if r["steps_trained"] != STEPS:      # correlations.csv carries both families too.
        continue                          # Without this the 1917 rows, which come last,
    corr[(int(r["n"]), int(r["layer"]))] = (float(r["pearson"]), float(r["spearman"]))
if not corr:
    print("FATAL: correlations filter selected zero rows")
    sys.exit(2)
p8 = corr[(8192, 2)]
print("  correlations n=8192 L2: Pearson %.4f Spearman %.4f" % p8)
for nm, txt in (("README", rd), ("PDF", cs)):
    m = re.search(r"Pearson ([0-9.]+),? Spearman (?:&minus;|-|\u2212)?([0-9.]+)", txt)
    if m:
        pe, sp = float(m.group(1)), -float(m.group(2))
        ok = abs(pe - p8[0]) < 0.006 and abs(sp - p8[1]) < 0.006
        print("     %-7s says Pearson %.2f Spearman %.2f  %s"
              % (nm, pe, sp, "ok" if ok else "MISMATCH"))
        if not ok:
            problems.append("correlation mismatch in " + nm)
    else:
        print("     %-7s no correlation sentence found" % nm)

head(5, "STALE NUMBERS FROM THE WALL-CLOCK FAMILY STILL ON A SURFACE")
old = {"1.4318": "old n=2048 ratio, 4dp",
       "1.9731": "old n=8192 ratio, 4dp",
       "eleven of twelve": "old parity wording",
       "11 of 12": "old parity wording",
       "confound we cannot rule out": "old confound wording",
       "cannot rule out": "old confound wording"}
# The README states both old 4-dp ratios once, inside the deliberate before-and-after
# sentence ("went from 1.4318 to 1.4705"). That is the point of the sentence, so it is
# allowed exactly there and nowhere else, and the guard checks the context, not just
# the count -- a second stray mention would still be caught.
# A superseded number may be cited deliberately, to show a before-and-after or to explain why
# a guard exists. The marker for "this mention is on purpose" is the word `superseded` on the
# same line, or the explicit before/after sentence. Anything else is drift.
def _deliberate(txt, tok):
    return sum(1 for line in txt.splitlines()
               if tok in line and ("superseded" in line or "went from" in line or "from 1.9731" in line))
for tok, why in sorted(old.items()):
    flagged = False
    for nm, txt in (("README", rd), ("page", ix), ("PDF", cs)):
        c = txt.count(tok)
        budget = _deliberate(txt, tok) if tok in ("1.4318", "1.9731") else 0
        if c > budget:
            print("  %-30s %dx in %-7s (budget %d) (%s)" % (tok, c, nm, budget, why))
            problems.append("stale token %r in %s" % (tok, nm))
            flagged = True
        elif c:
            print("  %-30s %dx in %-7s -- deliberate before/after sentence  ok" % (tok, c, nm))
            flagged = True
    if not flagged:
        print("  %-30s absent everywhere  ok" % tok)
for nm, txt in (("README", rd), ("page", ix), ("PDF", cs)):
    stated = "2,309" in txt or "2309" in txt
    print("  %-7s states the matched step count: %s" % (nm, "yes" if stated else "NO"))
    if not stated:
        problems.append("%s never states the step count" % nm)

head(6, "CHECK-YOURSELF ANSWERS AGAINST THE DATA")
m0 = re.search(r"Layer 0 sits at about ([0-9.]+) on this model", ix)
if m0:
    v, actual = float(m0.group(1)), float(R[(2048, 0, "xy")]["mem_over_rep"])
    ok = abs(v - actual) < 0.01
    print("  page says layer 0 = %.2f ; measured %.4f  %s" % (v, actual, "ok" if ok else "MISMATCH"))
    if not ok:
        problems.append("check-yourself layer-0 value")
else:
    problems.append("layer-0 self-check claim not found in page")
    print("  could not find the layer-0 claim in the page")

zp = "dist/quiet-neurons-dataforge2026-nilay-toshniwal.zip"
if not os.path.exists(zp):
    print()
    print("  zip not built yet -- sections 7 and 8 skipped")
else:
    z = zipfile.ZipFile(zp)
    head(7, "AI_DISCLOSURE / LICENSES COVER EVERY SHIPPED FILE")
    disc = read("AI_DISCLOSURE.md") + read("experiments/LICENSES.md")
    # The disclosure covers whole classes with globs (`experiments/results/*.csv`).
    # Matching only literal basenames called ten glob-covered files undisclosed last run.
    globs = re.findall(r"`([A-Za-z0-9_./*-]*\*[A-Za-z0-9_./*-]*)`", disc)
    import fnmatch
    uncovered = []
    for n in z.namelist():
        base = os.path.basename(n)
        if base in ("START_HERE.txt", "README.md", "LICENSE", "AI_DISCLOSURE.md", "LICENSES.md"):
            continue
        stem = base.rsplit(".", 1)[0]
        if base in disc or stem in disc or n in disc:
            continue
        if any(fnmatch.fnmatch(n, g) for g in globs):
            continue
        uncovered.append(n)
    for u in uncovered:
        print("  NOT MENTIONED  " + u)
        problems.append("undisclosed file " + u)
    if not uncovered:
        print("  every shipped file is named in the disclosure or licence record")

    head(8, "START_HERE MATCHES WHAT IS ACTUALLY IN THE ZIP")
    sh = read("START_HERE.txt")
    for claim in ["web/concept-summary.pdf", "concept-summary.html", "README.md",
                  "AI_DISCLOSURE.md", "LICENSE", "experiments/LICENSES.md"]:
        inzip = claim in z.namelist()
        named = claim in sh or os.path.basename(claim) in sh
        flag = "ok" if (inzip and named) else "PROBLEM"
        if flag != "ok":
            problems.append("START_HERE vs zip: " + claim)
        print("  %-28s in zip=%-5s named in START_HERE=%-5s %s" % (claim, inzip, named, flag))

print()
print("=" * 74)
print("VERDICT: %d problem(s)   [%d CSV rows, %d in the published family]"
      % (len(problems), len(rows), len(pub)))
for p in problems:
    print("  - " + p)
print("=" * 74)
sys.exit(1 if problems else 0)
