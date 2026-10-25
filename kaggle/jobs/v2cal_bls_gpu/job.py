"""Kaggle job (GPU T4 x2, G2): resolver v2 calibration data (V2-b), GPU BLS seed search + resolver (two device-shards)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
procs = [subprocess.Popen([sys.executable, "-u", f"{SRC}/scripts/b1_run.py", "--mode", "search", "--engine", "bls_gpu", "--shard", str(i + 1),
                           "--nshards", "2", "--device", f"cuda:{i}", "--inj-file", "data/v2_calib_injections.csv", "--subsample-file", "", "--prefix", "v2cal", "--out", "/kaggle/working/v2cal", "--commit", COMMIT], cwd=SRC) for i in range(2)]
codes = [p.wait() for p in procs]
if any(codes):
    raise SystemExit(f"failed: {codes}")
