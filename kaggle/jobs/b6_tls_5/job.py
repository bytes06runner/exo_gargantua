"""Kaggle job (CPU): B6 TLS seed and frozen v2 resolver, shard 5 of 12 (docs/b6_definition.md)."""
import glob
import subprocess
import sys

SRC = "/tmp/exog"
snap_bin = glob.glob("/kaggle/input/**/exog_src.bin", recursive=True)
assert snap_bin, "code snapshot exog_src.bin not found under /kaggle/input"
subprocess.run(["mkdir", "-p", SRC], check=True)
subprocess.run(["tar", "-xzf", snap_bin[0], "-C", SRC], check=True)
snapshot = open(f"{SRC}/SNAPSHOT_COMMIT").read().strip()
print("snapshot commit:", snapshot, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b5_run.py", "--engine", "tls", "--shard", "5", "--nshards", "12", "--log", "data/b6_sample_log.csv",
                "--stars", "data/b6_stellar_params.csv", "--out", "/kaggle/working/b6", "--commit", snapshot], cwd=SRC, check=True)
