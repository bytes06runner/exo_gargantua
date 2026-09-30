"""Kaggle job: freeze the Phase 2 sample (CPU, internet on). Filled in by kaggle/push_job.py."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"

subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, f"{SRC}/scripts/freeze_sample.py", "--out", "/kaggle/working/freeze",
                "--commit", COMMIT], check=True)
