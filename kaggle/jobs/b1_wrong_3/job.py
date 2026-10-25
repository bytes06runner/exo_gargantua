"""Kaggle job (CPU, G2): B1(i) wrong-seed resolver runs, shard 3 of 5."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b1_run.py", "--mode", "wrong", "--shard", "3", "--nshards", "5",
                "--out", "/kaggle/working/b1", "--commit", COMMIT], cwd=SRC, check=True)
