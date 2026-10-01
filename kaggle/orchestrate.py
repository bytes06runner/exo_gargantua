"""Kaggle job orchestrator, run by GitHub Actions every 15 minutes (.github/workflows/kaggle-orchestrator.yml).

Reads kaggle/queue.json (an ordered list of job directories under kaggle/jobs/), asks Kaggle for the
status of each job's kernel, and pushes the next pending jobs while capacity allows.

Guards (same as the local launchers):
  * no double launch: a job whose kernel already exists on Kaggle in any state (queued, running,
    complete, error, cancelled) is never pushed again by the orchestrator; re-running a job is a
    manual decision;
  * capacity: at most 5 concurrent CPU sessions (measured, G2-m) and at most 1 GPU session;
  * push timeout 300 s, and after a timeout the kernel status is checked before anything is retried;
  * compute tag enforcement (G2) and commit pinning come from kaggle/push_job.py.
Credentials: KAGGLE_USERNAME / KAGGLE_KEY environment variables (GitHub Secrets), never files in git.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = {"cpu": 5, "gpu": 1}
ACTIVE = ("QUEUED", "RUNNING", "NEW", "CANCEL_REQUESTED")


def kaggle(*args, timeout=90):
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle.cli", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, "timeout"


def status(kernel_id: str) -> str:
    """'absent', 'unknown' (API trouble), or the KernelWorkerStatus name."""
    code, out = kaggle("kernels", "status", kernel_id)
    if "KernelWorkerStatus." in out:
        return out.split("KernelWorkerStatus.")[-1].strip().strip('"').split()[0].strip('"')
    if "Permission 'kernels.get' was denied" in out or "404" in out or "Not Found" in out:
        return "absent"
    return "unknown"


def note(msg: str, level: str = "notice") -> None:
    """Print, and also emit a GitHub Actions annotation + step-summary line (readable without signing in)."""
    print(msg, flush=True)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        safe = msg.replace("\n", " ")[:900]
        print(f"::{level} title=orchestrator::{safe}", flush=True)
        summ = os.environ.get("GITHUB_STEP_SUMMARY")
        if summ:
            with open(summ, "a") as fh:
                fh.write(f"- {safe}\n")


def main():
    dry = "--dry-run" in sys.argv
    queue = json.loads((ROOT / "kaggle" / "queue.json").read_text())
    jobs = []
    for name in queue["jobs"]:
        meta = json.loads((ROOT / "kaggle" / "jobs" / name / "kernel-metadata.json").read_text())
        jobs.append({"name": name, "id": meta["id"], "compute": meta["compute"]})
    for j in jobs:
        j["status"] = status(j["id"])
        note(f"status {j['name']} ({j['compute']}): {j['status']}")
    if any(j["status"] == "unknown" for j in jobs):
        note("Kaggle API did not answer for some jobs; pushing nothing this round (no-double-launch guard).", "warning")
        return 1
    running = {c: sum(1 for j in jobs if j["compute"] == c and j["status"] in ACTIVE) for c in CAP}
    pushed = []
    for j in jobs:
        if j["status"] != "absent":
            note(f"skip {j['name']}: already on Kaggle ({j['status']}); never re-pushed")
            continue
        if running[j["compute"]] >= CAP[j["compute"]]:
            note(f"skip {j['name']}: {j['compute']} capacity full {running}")
            continue
        if dry:
            print(f"DRY-RUN would push {j['name']}")
            running[j["compute"]] += 1
            continue
        r = subprocess.run([sys.executable, str(ROOT / "kaggle" / "push_job.py"), j["name"]],
                           capture_output=True, text=True, timeout=400)
        out = (r.stdout + r.stderr).strip()
        if "pushed" in out:
            running[j["compute"]] += 1
            pushed.append(j["name"])
            note(f"PUSHED {j['name']}: {out.splitlines()[-1] if out else ''}")
        elif "timed out" in out:
            st = status(j["id"])
            note(f"push of {j['name']} timed out; Kaggle status now: {st}", "warning")
            if st in ACTIVE:
                running[j["compute"]] += 1
            break  # do not push anything else this round
        elif "Maximum batch" in out:
            note(f"Kaggle session limit reached at {j['name']}; waiting for the next round")
            running[j["compute"]] = CAP[j["compute"]]
        else:
            note(f"push of {j['name']} refused/failed: {out[-400:]}", "error")
            return 1
    note(f"round done: pushed={pushed} running={running}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
