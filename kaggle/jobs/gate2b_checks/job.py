"""Kaggle job (GPU T4): Gate 2b checks -- limits probe, parallel BLS identity (A1), GPU BLS test (A2 iv)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"

subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/gate2b_checks.py",
                "--smoke", f"{SRC}/results/kaggle/smoke_run/v1_20260930T1241/smoke/smoke_results.json",
                "--out", "/kaggle/working/gate2b", "--commit", COMMIT], check=True)
