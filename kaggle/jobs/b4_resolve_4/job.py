"""Kaggle job (CPU, G2): B4 runtime, frozen resolver per light curve from the BLS seed, cache chunk 4."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b4_run.py", "--stage", "resolver", "--shard", "1", "--nshards", "1",
                "--out", "/kaggle/working/b4", "--commit", COMMIT], cwd=SRC, check=True)
