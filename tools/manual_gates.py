# -*- coding: utf-8 -*-
"""Machine-checkable evidence for the four gates that were judged by hand.

The gate linter is right that a gate with no CHECK is only as good as whoever read it, and this
project has already shipped one checker that could not fail. These do not re-perform the
measurements: a contrast sweep needs a browser and an acceptance test needs a fresh model. What
they check is that the required record exists, is dated to the final build, and states the
outcome the gate demands. That is strictly weaker than re-measuring and strictly stronger than a
tick in a box, and the difference is stated here rather than hidden behind a green line.

G22 is the exception and is checked for real: the row it requires is in the CSV, and the sentence
it requires gone is gone.

Run:  python tools/manual_gates.py <g19|g20|g21|g22|all>
"""
import csv, io, os, re, sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FAILED = []


def fail(g, msg):
    FAILED.append("%s: %s" % (g, msg))


def read(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


def g19():
    """Contrast: a dated rerun carrying both floors and zero failures."""
    t = read("experiments/results/contrast.md")
    if "Rerun, 2026-09-08" not in t:
        fail("G19", "no rerun dated 2026-09-08 in contrast.md")
        return
    blk = t.split("Rerun, 2026-09-08", 1)[1][:1400]
    for want in ("5.12:1", "4.71:1"):
        if want not in blk:
            fail("G19", "rerun does not carry the %s floor" % want)
    if len(re.findall(r"\|\s*\*\*0\*\*\s*\|", blk)) < 2:
        fail("G19", "rerun does not show zero failures in both themes")


def g20():
    """Latency: G20's stated condition includes a VISIBLE tab, and the agent could not produce
    one. The author ran the recorded script in a foreground tab on 2026-09-08 and got 358 ms
    median against the 1,010 ms on record. The check requires that rerun, its median, and an
    acknowledgement that the gap to the earlier number is not fully explained."""
    t = read("experiments/results/latency.md")
    if "Rerun, 2026-09-08" not in t:
        fail("G20", "no rerun dated 2026-09-08 in latency.md")
        return
    blk = t.split("Rerun, 2026-09-08", 1)[1][:2000]
    if not re.search(r"\*\*1,0\d\d\*\*", blk):
        fail("G20", "rerun states no median near the recorded 1,010 ms")
    if "Visible-tab rerun, 2026-09-08" not in t:
        fail("G20", "no visible-tab rerun recorded; the hidden-tab one does not meet G20")
        return
    vis = t.split("Visible-tab rerun, 2026-09-08", 1)[1][:2200]
    if "**358**" not in vis:
        fail("G20", "visible-tab rerun states no median")
    if "cannot fully account" not in vis:
        fail("G20", "the 2.8x gap against the earlier record is not acknowledged")


def g21():
    """Acceptance test. G21's condition is ZERO major inaccuracies, so checking only that a run
    was recorded would be a check that cannot fail on the thing the gate is about. The record
    must contain a run that came back clean against the shipped PDF, and it must still carry the
    earlier runs' raw output and dispositions, so a clean result cannot be produced by deleting
    the runs that were not."""
    t = read("experiments/results/onepager_test.md")
    for want, why in (("Run 2, 2026-09-08", "run 2 is missing"),
                      ("Run 3, 2026-09-08", "run 3 is missing"),
                      ("Raw output", "no raw output recorded"),
                      ("What was changed", "no record of what the findings changed"),
                      ("Not fixed", "no record of what was left unfixed")):
        if want not in t:
            fail("G21", why)
    if "MAJOR INACCURACIES: NONE" not in t:
        fail("G21", "no run has come back with zero major inaccuracies against the shipped PDF")


def g22():
    """n=8,192 full schedule: the row exists, and the stale sentence is gone. Checked for real."""
    rows = list(csv.DictReader(io.open("experiments/results/full_schedule.csv", encoding="utf-8")))
    hit = [r for r in rows if r["n"] == "8192" and r["steps_trained"] == "4000"
           and r["tensor"] == "xy" and r["layer"] == "2"]
    if not hit:
        fail("G22", "no n=8192 steps_trained=4000 layer-2 xy row in full_schedule.csv")
    else:
        ratio = float(hit[0]["mem_over_rep"])
        for name in ("README.md", "experiments/results/full_schedule.README.md",
                     "web/index.html", "concept-summary.html"):
            if "%.4f" % ratio not in read(name):
                fail("G22", "%s does not carry the measured %.4f" % (name, ratio))
    for name in ("README.md", "web/index.html", "concept-summary.html"):
        t = read(name)
        for stale in ("is training to completion now", "training to completion now"):
            if stale in t:
                fail("G22", "%s still says the n=8192 run is training" % name)
                break
    if "n=8,192, complete schedule" not in read("experiments/results/full_schedule.README.md"):
        fail("G22", "full_schedule.README.md has no n=8,192 section")


GATES = {"g19": g19, "g20": g20, "g21": g21, "g22": g22}
which = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
for name in (sorted(GATES) if which == "all" else [which]):
    if name not in GATES:
        print("unknown gate %r" % name)
        sys.exit(2)
    GATES[name]()

if FAILED:
    for f in FAILED:
        print("MANUAL GATE FAIL " + f)
    sys.exit(1)
print("MANUAL GATE PASS " + which)
