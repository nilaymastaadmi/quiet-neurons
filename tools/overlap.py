# -*- coding: utf-8 -*-
"""Factor-overlap statistics quoted in README "Which tensor xy is" and page section 05.

The counted tensor xy is the elementwise product of two rectified factors, x and y. If the
two fired independently, the density of the product would be x * y. This prints the measured
density over that independence baseline, xy / (x * y), for each block at layer 2, plus the
mem/rep ratios of x, y, x * y and xy, all from the step-matched rows of measured.csv.

An observation on shipped numbers, not a mechanism: it says where in the product the drop
sits, not why.

Run:  python tools/overlap.py
"""
import csv, io, os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CSV = "experiments/results/measured.csv"
STEPS = "2309"
SIZES = (2048, 8192, 16384)
BLOCKS = ("warmup", "mem", "rep")


def rows():
    out = {}
    for r in csv.DictReader(io.open(CSV, encoding="utf-8")):
        if r["steps_trained"] == STEPS and r["layer"] == "2" and r["tensor"] in ("x", "y", "xy"):
            out[(int(r["n"]), r["tensor"])] = {b: float(r[b]) for b in BLOCKS}
    return out


def overlap(rs, n):
    """xy / (x * y) per block, and mem/rep for x, y, x*y, xy. Every value from the CSV."""
    x, y, xy = rs[(n, "x")], rs[(n, "y")], rs[(n, "xy")]
    ov = {b: xy[b] / (x[b] * y[b]) for b in BLOCKS}
    rat = {"x": x["mem"] / x["rep"], "y": y["mem"] / y["rep"],
           "x*y": (x["mem"] * y["mem"]) / (x["rep"] * y["rep"]), "xy": xy["mem"] / xy["rep"]}
    return ov, rat


def main():
    rs = rows()
    for n in SIZES:
        ov, rat = overlap(rs, n)
        print("n=%-6d overlap warm %.3f mem %.3f rep %.3f;  ratios x %.3f y %.3f x*y %.3f xy %.3f"
              % (n, ov["warmup"], ov["mem"], ov["rep"], rat["x"], rat["y"], rat["x*y"], rat["xy"]))


if __name__ == "__main__":
    main()
