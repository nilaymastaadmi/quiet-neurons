#!/usr/bin/env bash
# G16 on its own. verify.sh --with-page also re-derives all three headline ratios first, which
# takes minutes and made the gate checker time out at 120 s on a check that itself takes seconds.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${BDH_PYTHON:-python}"
CHROME="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"
"$PY" -m http.server 8197 --bind 127.0.0.1 --directory web >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 3
DOM=$("$CHROME" --headless=new --disable-gpu --virtual-time-budget=25000 \
      --dump-dom http://127.0.0.1:8197/parity.html 2>/dev/null)
if echo "$DOM" | grep -q "PARITY PASS"; then
  echo "parity.html: PARITY PASS"
else
  echo "parity.html did NOT report PASS"
  exit 1
fi
