"""Kaggle job (GPU T4 x2, G2): full P08 pool screen, session 4 of 7 (amendments A5, A6).
Device-shards 7 and 8 of 14 (scripts/p08_screen.py cost-balanced sharding); raw and A6 SDE recorded per star.
Plan: results/p08_screen_plan.json."""
import subprocess
import sys

COMMIT = "2197afa0a0ba12b1a2834ae1cddfef0f60d96baa"  # pinned: every screen session runs the code of session 1
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
procs = [subprocess.Popen([sys.executable, "-u", f"{SRC}/scripts/p08_screen.py", "--engine", "gpu",
                           "--shard", str(s), "--nshards", "14", "--device", f"cuda:{i}",
                           "--out", "/kaggle/working/p08screen", "--commit", COMMIT], cwd=SRC)
         for i, s in enumerate((7, 8))]
codes = [p.wait() for p in procs]
if any(codes):
    raise SystemExit(f"failed: {codes}")
