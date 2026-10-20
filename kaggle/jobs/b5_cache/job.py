"""Kaggle job (CPU, G2): B5 compact cache of the 3,443 pinned early-sector files (docs/b5_definition.md)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/build_cache.py", "--pins", "data/b5_pins_early.csv", "--n-chunks", "1", "--chunk", "1",
                "--no-prev", "--out", "/kaggle/working/cache", "--commit", COMMIT], cwd=SRC, check=True)
