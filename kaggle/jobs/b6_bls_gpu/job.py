"""Kaggle job (GPU T4 x2): B6 GPU BLS seed and frozen v2 resolver from that seed (docs/b6_definition.md)."""
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
procs = [subprocess.Popen([sys.executable, "-u", f"{SRC}/scripts/b5_run.py", "--engine", "bls_gpu", "--shard", str(i + 1), "--nshards", "2", "--device", f"cuda:{i}",
                           "--log", "data/b6_sample_log.csv", "--stars", "data/b6_stellar_params.csv", "--out", "/kaggle/working/b6", "--commit", snapshot], cwd=SRC) for i in range(2)]
codes = [p.wait() for p in procs]
if any(codes):
    raise SystemExit(f"failed: {codes}")
