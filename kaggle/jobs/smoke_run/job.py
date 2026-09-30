"""Kaggle job: Phase 2 smoke run: 20 targets, cache + BLS/TLS (CPU, internet on). Filled in by kaggle/push_job.py."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"

subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/smoke_run.py", "--out", "/kaggle/working/smoke",
                "--commit", COMMIT], check=True)
