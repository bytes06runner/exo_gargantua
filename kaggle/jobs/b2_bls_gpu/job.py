"""Kaggle job (GPU T4 x2, G2): B2 holdout GPU BLS seeds, one shard per T4 (sealed, A4)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"

subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
procs = [subprocess.Popen([sys.executable, "-u", f"{SRC}/scripts/b2_seeds.py", "--mode", "bls_gpu", "--shard", str(i + 1),
                           "--nshards", "2", "--device", f"cuda:{i}", "--out", "/kaggle/working/b2", "--commit", COMMIT])
         for i in range(2)]
codes = [p.wait() for p in procs]
if any(codes):
    raise SystemExit(f"GPU BLS shard failed: {codes}")
