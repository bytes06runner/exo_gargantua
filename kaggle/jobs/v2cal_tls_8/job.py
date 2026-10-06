"""Kaggle job (CPU, G2): resolver v2 calibration data (V2-b), TLS seed search + resolver, shard 8 of 10."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b1_run.py", "--mode", "search", "--engine", "tls", "--shard", "8", "--nshards", "10",
                "--inj-file", "data/v2_calib_injections.csv", "--subsample-file", "", "--prefix", "v2cal", "--out", "/kaggle/working/v2cal", "--commit", COMMIT], cwd=SRC, check=True)
