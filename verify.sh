#!/usr/bin/env bash
# Regenerate every published number and fail loudly if any of them has drifted.
#
# This exists because a defect got all the way to a submitted build in which the README's own
# reproduction command pointed at a superseded checkpoint and printed 1.4318 where the page
# published 1.4705. Nothing caught it, because nothing ever ran the documented command and
# compared its output to the documented number. That is what this does.
#
#   ./verify.sh              numeric checks only (no browser needed)
#   ./verify.sh --with-page  also drives parity.html in headless Chrome
#
# CHAIN IT WITH && , NOT A NEWLINE. On 2026-09-07 this script reported a failure and the
# commit went out anyway, because it was separated from `git commit` by a newline instead
# of &&. A gate you can walk past is not a gate:
#
#   ./verify.sh && git commit -am "..." && git push
#
# Exit code is non-zero if ANY check fails. Intended to be the last thing run before a commit
# that touches numbers, and the first thing run when picking the project back up.

set -uo pipefail
cd "$(dirname "$0")"

PY="${BDH_PYTHON:-python}"
FAILED=0

# Section 1 needs a Python with torch installed. If the default one does not have it, say so
# once and clearly, rather than letting three checks fail in a way that looks like number drift.
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "NOTE: '$PY' has no torch, so section 1 cannot run."
  echo "      Point BDH_PYTHON at the environment you trained in, e.g."
  echo "        BDH_PYTHON=/path/to/venv/Scripts/python.exe ./verify.sh"
  echo "      or set VERIFY_SKIP_MEASURE=1 to run everything else knowingly."
  echo
  VERIFY_SKIP_MEASURE=1
fi
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAILED=1; }

echo "=============================================================="
echo " 1. measure.py reproduces every published ratio"
echo "=============================================================="
# Each line: label | checkpoint | embd | mult | expected layer-2 xy ratio
if [ "${VERIFY_SKIP_MEASURE:-0}" = "1" ]; then
  echo "  SKIPPED (VERIFY_SKIP_MEASURE=1). Never skip this before a commit that touches numbers."
fi
while IFS='|' read -r label ckpt embd mult expect; do
  [ "${VERIFY_SKIP_MEASURE:-0}" = "1" ] && continue
  [ -z "${label// }" ] && continue
  if [ ! -f "experiments/$ckpt" ]; then
    fail "$label: experiments/$ckpt is missing"
    continue
  fi
  got=$(cd experiments && "$PY" -u measure.py --ckpt "$ckpt" --embd "$embd" --mult "$mult" \
          --repeats 5 --out /dev/null 2>/dev/null \
        | awk '/layer 2 xy/ {print $6}')
  if [ "$got" = "$expect" ]; then
    pass "$label -> $got"
  else
    fail "$label -> got '${got:-<nothing>}', published $expect"
  fi
done <<'ROWS'
n=2048  | checkpoints/bdh_n2048.pt  | 64  | 32  | 1.4705
n=8192  | checkpoints/bdh_n8192.pt  | 128 | 64  | 2.1201
n=16384 | checkpoints/bdh_n16384.pt | 128 | 128 | 1.8742
ROWS

echo
echo "=============================================================="
echo " 2. cross-surface agreement, paths, disclosure, stale numbers"
echo "=============================================================="
if "$PY" tools/sweep.py; then pass "sweep: 0 problems"; else fail "sweep reported problems"; fi

echo
echo "=============================================================="
echo " 3. documented commands are the ones that actually reproduce"
echo "=============================================================="
if grep -rn "sparsity_scan.py --embd" README.md web/index.html | grep -qv -- "--stop-at"; then
  fail "a documented training command omits --stop-at (reproduces the wall-clock confound)"
else
  pass "every documented training command carries --stop-at"
fi
if grep -q '"source_checkpoint": "bdh_n2048.pt"' web/data/manifest.json; then
  pass "manifest names a checkpoint the documented command reproduces"
else
  fail "manifest source_checkpoint does not name checkpoints/bdh_n2048.pt"
fi

echo
echo "=============================================================="
echo " 4. no retracted or overclaiming language"
echo "=============================================================="
for pat in "Causal, not correlational" "1.4318" "1.9731" "eleven of twelve"; do
  hits=$(grep -rn "$pat" README.md web/index.html concept-summary.html 2>/dev/null \
         | grep -v "went from 1.4318" | grep -v "from 1.9731" | grep -v "retract"          | grep -v "superseded" || true)
  if [ -z "$hits" ]; then pass "absent: $pat"; else fail "present: $pat"; echo "$hits" | head -3; fi
done

echo
echo "=============================================================="
echo " 5. one-pager: one page, inside the recommended word count"
echo "=============================================================="
"$PY" - <<'PYEOF'
import sys
try:
    import pymupdf
except ImportError:
    print("  SKIP  pymupdf not installed"); sys.exit(0)
d = pymupdf.open("web/concept-summary.pdf")
t = "".join(p.get_text() for p in d)
body = len(t.split("[1] Kosowski")[0].split())
ok = d.page_count == 1 and 500 <= body <= 950
print(("  PASS  " if ok else "  FAIL  ") + f"one-pager: {d.page_count} page(s), {body} body words")
sys.exit(0 if ok else 1)
PYEOF
[ $? -eq 0 ] || FAILED=1

if [ "${1:-}" = "--with-page" ]; then
  echo
  echo "=============================================================="
  echo " 6. parity, in a real browser"
  echo "=============================================================="
  CHROME="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"
  "$PY" -m http.server 8199 --bind 127.0.0.1 --directory web >/dev/null 2>&1 &
  SRV=$!
  sleep 3
  DOM=$("$CHROME" --headless=new --disable-gpu --virtual-time-budget=25000 \
        --dump-dom http://127.0.0.1:8199/parity.html 2>/dev/null)
  kill $SRV 2>/dev/null
  if echo "$DOM" | grep -q "PARITY PASS"; then pass "parity.html: PARITY PASS"; else fail "parity.html did not report PASS"; fi
  zeros=$(echo "$DOM" | grep -o "0\.000e+0" | wc -l)
  if [ "$zeros" -ge 24 ]; then pass "twelve of twelve sparsity series exact ($zeros zero cells)"
  else fail "expected >=24 exact-zero cells, saw $zeros"; fi
fi

echo
if [ "$FAILED" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
else
  echo "SOME CHECKS FAILED -- do not commit numbers until this is green"
fi
exit $FAILED
