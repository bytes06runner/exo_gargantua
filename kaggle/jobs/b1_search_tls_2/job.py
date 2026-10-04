"""Kaggle job (CPU, G2): B1(ii) TLS seed search + resolver on the 2,000-injection subsample, shard 2 of 5."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b1_run.py", "--mode", "search", "--engine", "tls", "--shard", "2",
                "--nshards", "5", "--out", "/kaggle/working/b1", "--commit", COMMIT], cwd=SRC, check=True)
