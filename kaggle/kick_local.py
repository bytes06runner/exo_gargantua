"""Local backstop for the GitHub Actions orchestrator (its cron fires irregularly): when a queued job finishes and
others are still absent (or hourly while jobs are absent), bump kaggle/queue.json's "kick" field and push, which
triggers the orchestrator workflow. Never pushes Kaggle jobs itself. Exits when every queued job is COMPLETE or any
job fails. Usage: python kaggle/kick_local.py <repo path>"""
import datetime as dt
import json
import subprocess
import sys
import time

REPO = sys.argv[1]
KG = REPO + "/.venv/bin/kaggle"


def jobs():
    q = json.load(open(f"{REPO}/kaggle/queue.json"))
    return ["exog-" + j.replace("_", "-") for j in q["jobs"]]


def st(j):
    try:
        r = subprocess.run([KG, "kernels", "status", f"srijeetbanerjee/{j}"], capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return "unknown"
    out = r.stdout + r.stderr
    if "KernelWorkerStatus." in out:
        return out.split("KernelWorkerStatus.")[-1].strip().strip('"').split()[0].strip('"')
    if "Permission 'kernels.get' was denied" in out or "404" in out:
        return "absent"
    return "unknown"


def kick(why):
    subprocess.run(["git", "-C", REPO, "pull", "-q", "--ff-only", "origin", "v2-alias-resolver"], capture_output=True)
    q = json.load(open(f"{REPO}/kaggle/queue.json"))
    q["kick"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds") + " " + why
    open(f"{REPO}/kaggle/queue.json", "w").write(json.dumps(q, indent=2) + "\n")
    r = subprocess.run(["git", "-C", REPO, "commit", "-q", "-m", f"Orchestrator kick: {why}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
                        "--", "kaggle/queue.json"], capture_output=True, text=True)
    if r.returncode == 0:
        subprocess.run(["git", "-C", REPO, "push", "-q", "origin", "v2-alias-resolver"], timeout=300)
    print(f"{dt.datetime.now(dt.timezone.utc):%H:%M} kick ({why}) rc={r.returncode}", flush=True)


prev_done, last = -1, time.time()
while True:
    J = jobs()
    s = {j: st(j) for j in J}
    if list(s.values()).count("unknown") > 2:
        time.sleep(300)
        continue
    bad = {j: v for j, v in s.items() if v in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL")}
    done = sum(v == "COMPLETE" for v in s.values())
    absent = [j for j, v in s.items() if v == "absent"]
    if done != prev_done:
        print(f"{dt.datetime.now(dt.timezone.utc):%H:%M} {done}/{len(J)} complete; absent {len(absent)}", flush=True)
    if bad:
        print("STOP", bad, flush=True)
        break
    if done == len(J):
        print("ALL COMPLETE", flush=True)
        break
    if absent and (done > prev_done >= 0 or time.time() - last > 3600):
        kick(f"{done}/{len(J)} complete")
        last = time.time()
    prev_done = done
    time.sleep(600)
