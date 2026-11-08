"""Kaggle job (CPU, G2): B5 rerun with TIC stellar parameters (decision B5-r1): TLS seed and resolver, shard 2 of 10."""
import glob
import shutil
import subprocess
import sys

COMMIT = "{{COMMIT}}"
SRC = "/tmp/exog"
tars = glob.glob("/kaggle/input/**/exog_src.tar.gz", recursive=True)
if tars:
    subprocess.run(["mkdir", "-p", SRC], check=True)
    subprocess.run(["tar", "-xzf", tars[0], "-C", SRC], check=True)
else:
    snap = glob.glob("/kaggle/input/**/SNAPSHOT_COMMIT", recursive=True)
    assert snap, "code snapshot not found under /kaggle/input"
    shutil.copytree(snap[0].rsplit("/", 1)[0], SRC)
snapshot = open(f"{SRC}/SNAPSHOT_COMMIT").read().strip()
print("snapshot commit:", snapshot, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b5_run.py", "--engine", "tls", "--shard", "2", "--nshards", "10", "--stars", "data/b5_stellar_params.csv",
                "--out", "/kaggle/working/b5", "--commit", snapshot], cwd=SRC, check=True)
