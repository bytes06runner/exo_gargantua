"""Kaggle job (GPU T4 x2, G2): B4 runtime, load + prepare + GPU BLS per light curve, cache chunks (1, 2)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
procs = [subprocess.Popen([sys.executable, "-u", f"{SRC}/scripts/b4_run.py", "--stage", "bls", "--shard", str(i + 1), "--nshards", "2",
                           "--device", f"cuda:{i}", "--out", "/kaggle/working/b4", "--commit", COMMIT], cwd=SRC) for i in range(2)]
codes = [p.wait() for p in procs]
if any(codes):
    raise SystemExit(f"failed: {codes}")
