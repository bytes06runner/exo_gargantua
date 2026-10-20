"""Kaggle job (CPU, G2): B5 TLS seed + frozen resolver runs, shard 4 of 5."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b5_run.py", "--engine", "tls", "--shard", "4", "--nshards", "5",
                "--out", "/kaggle/working/b5", "--commit", COMMIT], cwd=SRC, check=True)
