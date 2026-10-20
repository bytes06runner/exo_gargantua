"""Kaggle job (CPU, G2): B4 compact cache, Sector 49, chunk 3 of 4."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/build_cache.py", "--pins", "data/b4_pins.csv", "--n-chunks", "4", "--chunk", "3",
                "--no-prev", "--out", "/kaggle/working/cache", "--commit", COMMIT], cwd=SRC, check=True)
