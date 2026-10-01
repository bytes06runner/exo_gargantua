"""Kaggle job (GPU T4 x2, G2): P08 SDE validation -- GPU BLS on the 50 pre-drawn stars (decision V1)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
procs = [subprocess.Popen([sys.executable, "-u", f"{SRC}/scripts/p08_screen.py", "--engine", "gpu", "--tics-file",
                           "data/p08_validation_stars.csv", "--shard", str(i + 1), "--nshards", "2", "--device", f"cuda:{i}",
                           "--out", "/kaggle/working/p08val", "--commit", COMMIT], cwd=SRC) for i in range(2)]
codes = [p.wait() for p in procs]
if any(codes):
    raise SystemExit(f"failed: {codes}")
