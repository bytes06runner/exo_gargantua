"""Kaggle job (CPU, G2): frozen resolver on the B2 holdout TOIs from every seed source, shard 4 of 5 (no truth read)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b2_resolve.py", "--shard", "4", "--nshards", "5",
                "--out", "/kaggle/working/b2res", "--commit", COMMIT], cwd=SRC, check=True)
