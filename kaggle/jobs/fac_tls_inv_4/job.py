"""Kaggle job (CPU, G2): POST HOC false-alarm context, TLS SDE on the inverted FA1 light curves, shard 4 of 5."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/fa_context_run.py", "--engine", "tls", "--variant", "inverted", "--shard", "4", "--nshards", "5",
                "--out", "/kaggle/working/fac", "--commit", COMMIT], cwd=SRC, check=True)
