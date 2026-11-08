"""Local queue driver for when the GitHub Actions orchestrator cannot reach Kaggle: pushes the jobs of a list file as
Kaggle CPU slots open (one `kaggle kernels push` per absent job, oldest first; a "Maximum batch CPU session" reply ends
the round). Works on a private copy of the job directory, so editing the working tree does not disturb it. A job that
exists on Kaggle is never re-pushed. Exits when all are COMPLETE or any job fails. Run with caffeinate on macOS.
Usage: python kaggle/drain_local.py <repo path> <snapshot commit> <job> [<job> ...]"""
import datetime as dt
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT, COMMIT, JOBS = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
KG = str(ROOT / ".venv/bin/kaggle")
STAGE = Path(tempfile.mkdtemp(prefix="exog-drain-"))
for j in JOBS:  # frozen copy of each job at start-up
    shutil.copytree(ROOT / "kaggle/jobs" / j, STAGE / j)


def say(m):
    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M} {m}", flush=True)


def status(j):
    try:
        r = subprocess.run([KG, "kernels", "status", "srijeetbanerjee/exog-" + j.replace("_", "-")], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "unknown"
    o = r.stdout + r.stderr
    if "KernelWorkerStatus." in o:
        return o.split("KernelWorkerStatus.")[-1].split()[0].strip('"')
    return "absent" if "was denied" in o or "404" in o else "unknown"


def push(j):
    d = STAGE / j
    f = d / "job.py"
    f.write_text(f.read_text().replace("{{COMMIT}}", COMMIT))
    m = json.loads((d / "kernel-metadata.json").read_text())
    m.pop("compute", None), m.pop("after", None), m.pop("est_quota_h", None)
    (d / "kernel-metadata.json").write_text(json.dumps(m, indent=2))
    try:
        r = subprocess.run([KG, "kernels", "push", "-p", str(d)], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "timeout"  # may have reached Kaggle: status decides on the next round
    return r.stdout + r.stderr


prev = None
while True:
    s = {j: status(j) for j in JOBS}
    done = sum(v == "COMPLETE" for v in s.values())
    bad = {j: v for j, v in s.items() if v in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL")}
    line = f"{done}/{len(JOBS)} complete; running {sum(v == 'RUNNING' for v in s.values())}; queued {sum(v == 'QUEUED' for v in s.values())}; absent {sum(v == 'absent' for v in s.values())}"
    if line != prev:
        say(line)
        prev = line
    if bad:
        say(f"STOP {bad}")
        sys.exit(1)
    if done == len(JOBS):
        say("ALL COMPLETE")
        break
    for j in [k for k, v in s.items() if v == "absent"]:
        out = push(j)
        if "Maximum batch" in out:
            break
        say(f"pushed {j}: {out.strip()[:70]}")
    time.sleep(300)
