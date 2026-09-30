"""Kaggle job (CPU, G2): B2 holdout TLS seeds, shard 9 of 10 (sealed, A4: no truth comparison)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"

subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b2_seeds.py", "--mode", "tls", "--shard", "9", "--nshards", "10",
                "--out", "/kaggle/working/b2", "--commit", COMMIT], check=True)
