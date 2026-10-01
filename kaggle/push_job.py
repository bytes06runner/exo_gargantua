"""Push a Kaggle job pinned to the current (pushed) git commit.

Usage: python kaggle/push_job.py <job_name>
Copies kaggle/jobs/<job_name>/ to kaggle/build/<job_name>/ (gitignored), replaces {{COMMIT}}
with HEAD, refuses if the tree is dirty or HEAD is not on origin, then `kaggle kernels push`.
The push record (commit, kernel id, UTC time) is appended to results/kaggle/<job_name>/pushes.jsonl.
"""
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()


def check_compute(meta):
    """Decision G2 (docs/decisions.md): GPU sessions only for GPU work, CPU sessions for CPU-bound work.

    Returns the extra `kaggle kernels push` arguments; exits on any mismatch.
    """
    compute = meta.get("compute")
    if compute == "gpu":
        if not meta.get("enable_gpu") or meta.get("machine_shape") != "NvidiaTeslaT4":
            sys.exit("refusing: a GPU job must set enable_gpu=true and machine_shape=NvidiaTeslaT4 (G2)")
        return ["--accelerator", "NvidiaTeslaT4"]
    if compute == "cpu":
        if meta.get("enable_gpu") or meta.get("enable_tpu") or meta.get("machine_shape"):
            sys.exit("refusing: a CPU-bound job may not request a GPU/TPU session (G2)")
        return []
    sys.exit('refusing: kernel-metadata.json must declare "compute": "cpu" or "gpu" (G2)')


def main(job):
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=no").splitlines()
             if not l[3:].startswith("results/kaggle/")]  # push logs written by this script are allowed
    if dirty:
        sys.exit("refusing: working tree has uncommitted changes")
    head = git("rev-parse", "HEAD")
    if not git("branch", "-r", "--contains", head):
        sys.exit("refusing: HEAD is not pushed to origin")
    src, build = ROOT / "kaggle" / "jobs" / job, ROOT / "kaggle" / "build" / job
    shutil.rmtree(build, ignore_errors=True)
    shutil.copytree(src, build)
    for f in build.glob("*.py"):
        f.write_text(f.read_text().replace("{{COMMIT}}", head))
    meta = json.loads((build / "kernel-metadata.json").read_text())
    args = check_compute(meta)
    compute = meta.pop("compute")
    meta.pop("est_quota_h", None)  # orchestrator-only field
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    kaggle = str(Path(sys.executable).with_name("kaggle"))
    try:
        res = subprocess.run([kaggle, "kernels", "push", "-p", str(build), *args], capture_output=True, text=True,
                             timeout=300)
    except subprocess.TimeoutExpired:
        # the upload may still have reached Kaggle (seen 2026-09-30); callers must check status before retrying
        print(f"push timed out: verify with `kaggle kernels status {meta['id']}` before retrying")
        return 3
    print(res.stdout, res.stderr)
    rec = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "commit": head,
           "kernel": meta["id"], "enable_gpu": meta["enable_gpu"], "compute": compute, "accelerator": meta.get("machine_shape"), "push_output": res.stdout.strip()}
    out = ROOT / "results" / "kaggle" / job
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pushes.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return res.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
