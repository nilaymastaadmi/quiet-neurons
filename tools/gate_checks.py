# -*- coding: utf-8 -*-
"""Gate checks for the final submission. Copy to tools/gate_checks.py.

Each subcommand prints exactly one line starting with "GATE PASS <name>" on success and
exits 0, or prints "GATE FAIL <name>: <reason>" lines and exits 1. Written so that the
unlazy ledger (GATES.md) can use `python tools/gate_checks.py <name>` as CHECK and
`GATE PASS <name>` as EXPECT. No shell tools are used, so it runs the same under cmd.exe,
PowerShell and Git Bash.

    python tools/gate_checks.py all            run every gate that needs no torch/browser
    python tools/gate_checks.py <name>         run one gate
    python tools/gate_checks.py --root <dir> <name>   run against another checkout (used
                                               by the negative control, which plants a
                                               wrong number in a copy and must FAIL)

Why this file exists: on 2026-09-07 an external examiner planted seven wrong numbers and a
wrong reproduction checkpoint in a copy of this repository and verify.sh + sweep.py passed
all of them. These gates read the numbers off the surfaces and compare them to the data.
"""
import csv, io, json, os, re, shutil, subprocess, sys, tempfile, zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if len(sys.argv) > 2 and sys.argv[1] == "--root":
    ROOT = sys.argv[2]
    sys.argv = [sys.argv[0]] + sys.argv[3:]
ROOT = os.path.abspath(ROOT)

def P(*parts):
    return os.path.join(ROOT, *parts)

def read(*parts):
    with io.open(P(*parts), encoding="utf-8", errors="replace") as f:
        return f.read()

def norm(s):
    s = s.replace("&middot;", "·").replace("&mdash;", "—").replace("&times;", "×")
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()

FAILS = []
def fail(msg):
    FAILS.append(msg)

# ---------------------------------------------------------------- data loaders
def measured():
    rows = list(csv.DictReader(io.open(P("experiments/results/measured.csv"), encoding="utf-8")))
    pub = {(int(r["n"]), int(r["layer"]), r["tensor"]): r
           for r in rows if r["steps_trained"] == "2309"}
    if not pub:
        raise SystemExit("GATE FAIL loader: measured.csv has no steps_trained=2309 rows")
    return pub

def ratio(pub, n, layer, tensor="xy"):
    return float(pub[(n, layer, tensor)]["mem_over_rep"])

def warm_over_rep(row):
    return float(row["warmup"]) / float(row["rep"])

def trace(n):
    return json.load(io.open(P("web/data/traces/n%d.json" % n), encoding="utf-8"))

def ladder_rows():
    ctrl = list(csv.DictReader(io.open(P("experiments/results/mechanism_control.csv"), encoding="utf-8")))
    pool = list(csv.DictReader(io.open(P("experiments/results/pool_ladder.csv"), encoding="utf-8")))
    pub = measured()
    def l2(rows, k=None):
        for r in rows:
            if r["layer"] == "2" and r["tensor"] == "xy" and (k is None or r.get("word_pool", "") == str(k)):
                return warm_over_rep(r)
        return None
    def l2m(rows, k=None):
        for r in rows:
            if r["layer"] == "2" and r["tensor"] == "xy" and (k is None or r.get("word_pool", "") == str(k)):
                return float(r["mem_over_rep"])
        return None
    return {"K1": l2(ctrl), "K16": l2(pool, 16), "K256": l2(pool, 256),
            "base": warm_over_rep(pub[(2048, 2, "xy")]),
            # mem/rep is the column every published headline uses, and since 2026-09-08 it is
            # the column the one-pager reports the ladder on.
            "mK1": l2m(ctrl), "mK16": l2m(pool, 16), "mK256": l2m(pool, 256),
            "mbase": float(pub[(2048, 2, "xy")]["mem_over_rep"])}

# ---------------------------------------------------------------- the gates
CANON_CLAIM = ("goes quiet on words it has just picked up from the text in front of it")
CANON_AUDIENCE_MARK = "no machine-learning background is assumed"

def g_claim_consistent():
    """The claim sentence is the same on page, README, START_HERE and one-pager."""
    for f in ("web/index.html", "README.md", "START_HERE.txt", "concept-summary.html"):
        if CANON_CLAIM not in norm(read(f)):
            fail("claim sentence missing or different in %s" % f)
    for f in ("README.md", "START_HERE.txt", "concept-summary.html"):
        if "quietens down when it has just learned" in norm(read(f)):
            fail("old claim wording still present in %s" % f)

def g_audience_consistent():
    for f in ("web/index.html", "README.md"):
        if CANON_AUDIENCE_MARK not in norm(read(f)):
            fail("audience sentence missing in %s" % f)
    if "anyone who has met a neural network once" in norm(read("README.md")):
        fail("old audience wording still in README")

FORBIDDEN = {
    "difficulty ruled out": ("web/index.html", "README.md", "concept-summary.html",
                             "experiments/results/pool_ladder.README.md"),
    "passed decisively": ("README.md", "experiments/results/pool_ladder.README.md"),
    "cannot be what generates": ("README.md", "experiments/results/pool_ladder.README.md"),
    "difficulty alone cannot generate": ("README.md",),
    "strongest evidence in this project": ("experiments/results/pool_ladder.README.md", "README.md"),
    "so it is not noise": ("README.md", "web/index.html"),
    "exits non-zero on any drift": ("README.md",),
    "if the line does not jump, the claim above is wrong": ("web/index.html",),
    "about 30 minutes": ("START_HERE.txt",),
    "95 of every 100": ("web/index.html",),
    "guessing the next word": ("web/index.html",),
}
def g_no_overclaim():
    for tok, files in FORBIDDEN.items():
        for f in files:
            if tok in norm(read(f)):
                fail("forbidden phrase %r still in %s" % (tok, f))

def g_readme_numbers():
    """Every number in the README's limits table and headline traces to the data."""
    pub = measured(); rd = read("README.md")
    def want(pat, vals, what):
        m = re.search(pat, rd)
        if not m:
            fail("README: could not find %s" % what); return
        got = [float(x) for x in m.groups()]
        for g, v in zip(got, vals):
            if abs(g - v) > 0.006:
                fail("README %s: says %s, data %.4f" % (what, g, v))
    want(r"ratio ([0-9.]+), ([0-9.]+), ([0-9.]+) at n = 2k, 8k, 16k",
         [ratio(pub, n, 0) for n in (2048, 8192, 16384)], "layer-0 row")
    want(r"\| Layer 3 is flat[^|]*\| ([0-9.]+), ([0-9.]+), \*\*([0-9.]+)\*\*",
         [ratio(pub, n, 3) for n in (2048, 8192, 16384)], "layer-3 row")
    want(r"\*\*([0-9.]+)x\*\* \| [0-9.]+ – [0-9.]+ \|\n\| 8,192", [ratio(pub, 2048, 2)], "n=2048 headline")
    want(r"\| 8,192 \|[^\n]*\*\*([0-9.]+)x\*\*", [ratio(pub, 8192, 2)], "n=8192 headline")
    want(r"\| 16,384 \|[^\n]*\*\*([0-9.]+)x\*\*", [ratio(pub, 16384, 2)], "n=16384 headline")
    t = trace(8192); W, WD = 13, 8; L2 = t["layers"]["2"]; s = t["surprise"]
    warm = 100 * sum(L2[:W]) / W; rep = 100 * sum(L2[W + WD:]) / (77 - W - WD)
    want(r"\*\*([0-9.]+)%\*\* \|\n\| the 56 repeated", [warm], "counterexample warm-up %")
    want(r"\| the 56 repeated[^\n]*\*\*([0-9.]+)%\*\*", [rep], "counterexample repeats %")
    want(r"one uses \*\*([0-9.]+)x\*\* as many", [warm / rep], "counterexample factor")
    want(r"\*\*letter 12\*\* runs at ([0-9.]+)% against \*\*letter 41\*\* at ([0-9.]+)%",
         [100 * L2[11], 100 * L2[40]], "letter 12 / 41")
    lad = ladder_rows()
    want(r"\| K=16 \|[^\n]*\*\*([0-9.]+)\*\* \|", [lad["K16"]], "ladder K=16")
    want(r"\| K=256 \|[^\n]*\*\*([0-9.]+)\*\* \|", [lad["K256"]], "ladder K=256")
    want(r"\| K=∞, base task \|[^\n]*\| ([0-9.]+) \|", [lad["base"]], "ladder base")
    for tok in ("4.0–7.5%", "1.6–3.0x"):
        if tok not in rd:
            fail("README: Figure 14 band token %r missing" % tok)

def g_page_numbers():
    pub = measured(); ix = read("web/index.html")
    def want(pat, vals, what):
        m = re.search(pat, ix)
        if not m:
            fail("page: could not find %s" % what); return
        for g, v in zip([float(x) for x in m.groups()], vals):
            if abs(g - v) > 0.006:
                fail("page %s: says %s, data %.4f" % (what, g, v))
    want(r"([0-9.]+), ([0-9.]+), ([0-9.]+) at 2k, 8k, 16k <span", [ratio(pub, n, 0) for n in (2048, 8192, 16384)], "layer-0 row")
    want(r"([0-9.]+), ([0-9.]+), ([0-9.]+) <span class=\"tag tag-no\">against us", None or [ratio(pub, n, 3) for n in (2048, 8192, 16384)], "layer-3 row")
    want(r"const POP2048 = ([0-9.]+);", [warm_over_rep(pub[(2048, 2, "xy")])], "POP2048 constant")
    lad = ladder_rows()
    want(r"<strong>([0-9.]+)</strong> \(K=16\), <strong>([0-9.]+)</strong> \(K=256\), <strong>([0-9.]+)</strong>",
         [lad["K16"], lad["K256"], lad["base"]], "ladder row")
    for n, tok in ((2048, "1.47"), (8192, "2.12"), (16384, "1.87")):
        if abs(ratio(pub, n, 2) - float(tok)) > 0.005:
            fail("page/scaling token %s no longer matches csv" % tok)
    sj = json.load(io.open(P("web/data/scaling.json"), encoding="utf-8"))
    for p in sj["points"]:
        if p.get("ratio") is not None and abs(p["ratio"] - ratio(pub, p["n"], 2)) > 1e-9:
            fail("scaling.json n=%d ratio %s != csv" % (p["n"], p["ratio"]))

def g_onepager_numbers():
    cs = norm(read("concept-summary.html")); pub = measured()
    for n, tok in ((2048, "1.47×"), (8192, "2.12×"), (16384, "1.87×")):
        if tok not in cs:
            fail("one-pager missing %s" % tok)
    lad = ladder_rows()
    # All four rungs, on the column the one-pager states them in. Every value has to appear.
    for key, label in (("mK1", "K=1"), ("mK16", "K=16"), ("mK256", "K=256"), ("mbase", "base task")):
        tok = "%.2f" % round(lad[key], 2)
        if tok not in cs:
            fail("one-pager ladder %s mem/rep %s not present" % (label, tok))
    # and the difficulty argument's two losses, which the same sentence depends on
    for tok in ("0.0384", "0.1739"):
        if tok not in cs:
            fail("one-pager missing ladder loss %s" % tok)

def g_repro_command():
    """The README's own measure.py command names the checkpoint verify.sh tests."""
    rd = read("README.md"); vs = read("verify.sh")
    m = re.findall(r"measure\.py --ckpt (checkpoints/[A-Za-z0-9_]+\.pt) --embd 64 --mult 32", rd)
    if not m:
        fail("README has no n=2048 measure.py command")
    for ck in m:
        if ck != "checkpoints/bdh_n2048.pt":
            fail("README n=2048 command names %s" % ck)
        if ck not in vs:
            fail("verify.sh does not test %s" % ck)
    if "wallclock.pt --embd" in rd:
        fail("README reproduction command points at a wall-clock checkpoint")
    man = json.load(io.open(P("web/data/manifest.json"), encoding="utf-8"))
    if man.get("source_checkpoint") != "bdh_n2048.pt":
        fail("manifest source_checkpoint is %r" % man.get("source_checkpoint"))

def g_size_switch_label():
    ix = read("web/index.html")
    if 'id="statsScope"' not in ix:
        fail("page has no #statsScope label for the four live stats")
    if "n=2048, live" not in ix and "n=2,048, live" not in ix:
        fail("page stats scope label does not say n=2048, live")

def g_tour_disclosure():
    ix = norm(read("web/index.html"))
    if "chose this word because" not in ix:
        fail("tour step 2 does not disclose that the word was chosen for a clear jump")
    if "4 of 15" not in ix and "four of fifteen" not in ix:
        fail("tour/break-it copy does not state 4 of 15 random words show under +5%")

def g_verify_honest():
    vs = read("verify.sh")
    if 'echo "ALL CHECKS PASSED"' in vs and "SKIPPED" not in vs.split('echo "ALL CHECKS PASSED"')[0][-400:]:
        fail("verify.sh still prints ALL CHECKS PASSED without saying section 1 was skipped")
    if "section 1 SKIPPED" not in vs and "SECTION 1 SKIPPED" not in vs:
        fail("verify.sh has no honest summary line for the skipped section")

def g_zip():
    zp = P("dist/quiet-neurons-dataforge2026-nilay-toshniwal.zip")
    if not os.path.exists(zp):
        fail("zip not built"); return
    z = zipfile.ZipFile(zp); names = set(z.namelist())
    for need in ("verify.sh", "tools/sweep.py", "tools/ladder.py", "tools/gate_checks.py",
                 "START_HERE.txt", "README.md", "web/concept-summary.pdf", "web/index.html",
                 "web/data/weights.bin", "AI_DISCLOSURE.md", "experiments/LICENSES.md"):
        if need not in names:
            fail("zip missing %s" % need)
    for n in names:
        if n.endswith("/"):
            continue
        if n.startswith("web/") or n in ("README.md", "START_HERE.txt", "verify.sh") or n.startswith("tools/"):
            local = P(*n.split("/"))
            if os.path.exists(local) and z.read(n) != io.open(local, "rb").read():
                fail("zip %s differs from working tree" % n)

def g_onepager_length():
    try:
        from pypdf import PdfReader
    except ImportError:
        fail("pypdf not installed"); return
    r = PdfReader(P("web/concept-summary.pdf"))
    t = "\n".join((p.extract_text() or "") for p in r.pages)
    body = len(t.split("[1] Kosowski")[0].split())
    if len(r.pages) != 1:
        fail("one-pager has %d pages" % len(r.pages))
    if not 500 <= body <= 950:
        fail("one-pager body is %d words" % body)

def g_negative_control():
    """Plant a wrong number in a copy and require g_readme_numbers to FAIL there."""
    tmp = tempfile.mkdtemp(prefix="gatectl_")
    try:
        for d in ("experiments/results", "web/data", "web/data/traces"):
            os.makedirs(os.path.join(tmp, d), exist_ok=True)
        for f in ("README.md", "experiments/results/measured.csv",
                  "experiments/results/mechanism_control.csv", "experiments/results/pool_ladder.csv",
                  "web/data/traces/n8192.json", "web/data/scaling.json"):
            shutil.copy(P(*f.split("/")), os.path.join(tmp, *f.split("/")))
        rp = os.path.join(tmp, "README.md")
        s = io.open(rp, encoding="utf-8").read()
        s2 = re.sub(r"ratio ([0-9.]+), ([0-9.]+), ([0-9.]+) at n = 2k, 8k, 16k",
                    "ratio 0.90, 0.90, 0.96 at n = 2k, 8k, 16k", s)
        if s2 == s:
            fail("negative control could not plant a wrong number (layer-0 row not found)"); return
        io.open(rp, "w", encoding="utf-8").write(s2)
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--root", tmp, "readme_numbers"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            fail("negative control PASSED on a planted wrong number; the checker cannot fail")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def g_repro():
    """Needs torch: BDH_PYTHON must point at the venv. Reproduces the n=2048 headline."""
    py = os.environ.get("BDH_PYTHON")
    if not py:
        fail("set BDH_PYTHON to the venv python that has torch"); return
    out = subprocess.run([py, "-u", "measure.py", "--ckpt", "checkpoints/bdh_n2048.pt",
                          "--embd", "64", "--mult", "32", "--repeats", "5", "--out", os.devnull],
                         cwd=P("experiments"), capture_output=True, text=True)
    m = re.search(r"layer 2 xy: .*ratio ([0-9.]+)", out.stdout)
    pub = measured()
    if not m:
        fail("measure.py printed no layer 2 xy ratio: %s" % out.stderr[-300:]); return
    if abs(float(m.group(1)) - ratio(pub, 2048, 2)) > 1e-9:
        fail("measure.py printed %s, csv says %.4f" % (m.group(1), ratio(pub, 2048, 2)))

def g_live_matches_head():
    """The deployed page and PDF are byte-identical to the working tree."""
    import urllib.request
    base = "https://nilaymastaadmi.github.io/quiet-neurons/"
    for url, local in (("", "web/index.html"), ("concept-summary.pdf", "web/concept-summary.pdf"),
                       ("explain.html", "web/explain.html"),
                       ("parity.html", "web/parity.html"), ("bdh.js", "web/bdh.js")):
        try:
            remote = urllib.request.urlopen(base + url, timeout=30).read()
        except Exception as e:
            fail("could not fetch %s: %s" % (url or "index", e)); continue
        if remote != io.open(P(*local.split("/")), "rb").read():
            fail("live %s differs from %s" % (url or "index", local))

def g_git_pushed():
    def git(*a):
        return subprocess.run(["git"] + list(a), cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if not l.endswith("train_n8192_full4000.log")]
    if dirty:
        fail("working tree not clean: %s" % "; ".join(dirty[:5]))
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=ROOT)
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/main"):
        fail("HEAD is not pushed to origin/main")

GATES = {
    "live_matches_head": g_live_matches_head,
    "git_pushed": g_git_pushed,
    "claim_consistent": g_claim_consistent,
    "audience_consistent": g_audience_consistent,
    "no_overclaim": g_no_overclaim,
    "readme_numbers": g_readme_numbers,
    "page_numbers": g_page_numbers,
    "onepager_numbers": g_onepager_numbers,
    "repro_command": g_repro_command,
    "size_switch_label": g_size_switch_label,
    "tour_disclosure": g_tour_disclosure,
    "verify_honest": g_verify_honest,
    "zip": g_zip,
    "onepager_length": g_onepager_length,
    "negative_control": g_negative_control,
    "repro": g_repro,           # needs BDH_PYTHON; excluded from "all"
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); print("gates:", ", ".join(GATES)); return 0
    name = sys.argv[1]
    names = [n for n in GATES if n != "repro"] if name == "all" else [name]
    rc = 0
    for n in names:
        if n not in GATES:
            print("GATE FAIL %s: unknown gate" % n); rc = 1; continue
        del FAILS[:]
        try:
            GATES[n]()
        except Exception as e:  # a crash is a failure, never a pass
            FAILS.append("exception %s: %s" % (type(e).__name__, e))
        if FAILS:
            rc = 1
            for f in FAILS:
                print("GATE FAIL %s: %s" % (n, f))
        else:
            print("GATE PASS %s" % n)
    return rc

if __name__ == "__main__":
    sys.exit(main())
