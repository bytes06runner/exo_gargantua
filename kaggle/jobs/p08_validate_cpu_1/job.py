"""Kaggle job (CPU, G2): P08 SDE validation -- astropy BLS, half 1 of 2 of the 50 pre-drawn stars (decision V1)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/p08_screen.py", "--engine", "astropy", "--tics-file",
                "data/p08_validation_stars.csv", "--shard", "1", "--nshards", "2", "--workers", "4",
                "--out", "/kaggle/working/p08val", "--commit", COMMIT], cwd=SRC, check=True)
