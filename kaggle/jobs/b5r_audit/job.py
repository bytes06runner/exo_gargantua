"""Kaggle job (CPU, G2): B5 catalog-truth audit of the rerun (decision B5-r1): cache the later sectors of the audit targets, then fit (docs/b5_definition.md §6)."""
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
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/build_cache.py", "--pins", "data/b5_pins_audit.csv", "--n-chunks", "1", "--chunk", "1",
                "--no-prev", "--out", "/tmp/auditcache", "--commit", snapshot], cwd=SRC, check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b5_audit.py", "--cache", "/tmp/auditcache", "--out", "/kaggle/working/b5audit", "--commit", snapshot], cwd=SRC, check=True)
