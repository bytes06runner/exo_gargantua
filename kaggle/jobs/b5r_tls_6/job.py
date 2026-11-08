"""Kaggle job (CPU, G2): B5 rerun with TIC stellar parameters (decision B5-r1): TLS seed and resolver, shard 6 of 10."""
import glob
import subprocess
import sys

COMMIT = "{{COMMIT}}"
SRC = "/tmp/exog"
snap_bin = glob.glob("/kaggle/input/**/exog_src.bin", recursive=True)
assert snap_bin, "code snapshot exog_src.bin not found under /kaggle/input"
subprocess.run(["mkdir", "-p", SRC], check=True)
subprocess.run(["tar", "-xzf", snap_bin[0], "-C", SRC], check=True)
snapshot = open(f"{SRC}/SNAPSHOT_COMMIT").read().strip()
print("snapshot commit:", snapshot, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b5_run.py", "--engine", "tls", "--shard", "6", "--nshards", "10", "--stars", "data/b5_stellar_params.csv",
                "--out", "/kaggle/working/b5", "--commit", snapshot], cwd=SRC, check=True)
