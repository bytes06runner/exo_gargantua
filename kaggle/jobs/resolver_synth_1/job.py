"""Kaggle job (CPU, G2): Phase 3 synthetic check of the resolver, shard 1 of 2 (fully synthetic light curves)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/resolver_synth_run.py", "--n", "3000", "--shard", "1", "--nshards", "2",
                "--out", "/kaggle/working/synth", "--commit", COMMIT], cwd=SRC, check=True)
