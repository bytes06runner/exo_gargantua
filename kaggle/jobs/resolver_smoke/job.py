"""Kaggle job (CPU, G2): Phase 3 execution/runtime test of the resolver on the 20 smoke targets (sealed B2 members; no truth comparison)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/resolver_smoke.py", "--out", "/kaggle/working/resolver_smoke", "--commit", COMMIT],
               cwd=SRC, check=True)
