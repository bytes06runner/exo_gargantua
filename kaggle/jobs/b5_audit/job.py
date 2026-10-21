"""Kaggle job (CPU, G2): B5 catalog-truth audit -- cache the later sectors of the 319 audit targets, then fit (docs/b5_definition.md §6)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/build_cache.py", "--pins", "data/b5_pins_audit.csv", "--n-chunks", "1", "--chunk", "1",
                "--no-prev", "--out", "/tmp/auditcache", "--commit", COMMIT], cwd=SRC, check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b5_audit.py", "--cache", "/tmp/auditcache", "--out", "/kaggle/working/b5audit", "--commit", COMMIT], cwd=SRC, check=True)
