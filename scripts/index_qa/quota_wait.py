#!/usr/bin/env python3
"""Sleep until the 5-hour quota window resets if its last logged utilization is >= the threshold (argv[1]);
otherwise sleep 10 minutes. Reads the newest rate-limit event in the reader event logs
(results/index_qa/claude/*__events.jsonl). Prints what it does. Used by run_c11.sh between quota stops.

    .venv/bin/python scripts/index_qa/quota_wait.py 0.65
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "results/index_qa/claude"
thr = float(sys.argv[1])
best = None
for f in ROOT.glob("*__events.jsonl"):
    with f.open("rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 400_000))
        lines = fh.read().decode(errors="ignore").splitlines()[1:]
    for line in reversed(lines):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = next((x for x in e.get("events", []) if x.get("type") == "rate_limit_event"), None)
        if ev:
            w = ev["rate_limit_info"].get("unifiedWindows", {}).get("five_hour", {})
            if best is None or e["time"] > best[0]:
                best = (e["time"], w.get("utilization"), w.get("resetsAt"))
            break
if best and best[1] is not None and best[1] >= thr and best[2]:
    wait = best[2] - time.time() + 90
    print(f"5h utilization {best[1]} >= {thr}; sleeping {wait / 60:.0f} min until the reset", flush=True)
    time.sleep(max(0, wait))
else:
    print(f"5h utilization {best[1] if best else None} < {thr}; sleeping 10 min", flush=True)
    time.sleep(600)
