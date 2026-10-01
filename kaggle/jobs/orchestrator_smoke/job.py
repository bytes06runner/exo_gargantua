"""Tiny CPU job used to test the GitHub Actions orchestrator (no science)."""
import datetime as dt
import json
import os
import time

COMMIT = "{{COMMIT}}"
os.makedirs("/kaggle/working/out", exist_ok=True)
json.dump({"commit": COMMIT, "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "launched_by": "github-actions orchestrator test"}, open("/kaggle/working/out/hello.json", "w"))
print("orchestrator smoke ok", COMMIT, flush=True)
time.sleep(900)  # stay RUNNING for 15 min so a second orchestrator run can be shown to skip it
print("done", flush=True)
