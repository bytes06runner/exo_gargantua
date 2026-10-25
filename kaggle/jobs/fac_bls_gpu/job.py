"""Kaggle job (GPU T4 x2, G2): POST HOC false-alarm context, A6 and raw BLS SDE on the FA1 light curves (inverted on cuda:0, scrambled on cuda:1)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
procs = [subprocess.Popen([sys.executable, "-u", f"{SRC}/scripts/fa_context_run.py", "--engine", "bls_gpu", "--variant", v, "--shard", "1", "--nshards", "1",
                           "--device", f"cuda:{i}", "--out", "/kaggle/working/fac", "--commit", COMMIT], cwd=SRC) for i, v in enumerate(("inverted", "scrambled"))]
codes = [p.wait() for p in procs]
if any(codes):
    raise SystemExit(f"failed: {codes}")
