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


def main(job):
    if git("status", "--porcelain", "--untracked-files=no"):
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
    kaggle = str(Path(sys.executable).with_name("kaggle"))
    res = subprocess.run([kaggle, "kernels", "push", "-p", str(build)], capture_output=True, text=True)
    print(res.stdout, res.stderr)
    rec = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "commit": head,
           "kernel": meta["id"], "enable_gpu": meta["enable_gpu"], "push_output": res.stdout.strip()}
    out = ROOT / "results" / "kaggle" / job
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pushes.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return res.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
