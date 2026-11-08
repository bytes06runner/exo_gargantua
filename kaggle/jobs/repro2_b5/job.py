"""Kaggle job (CPU): reproducibility check on B5 (decision Repro-1)."""
import glob
import subprocess
import sys

COMMIT = "{{COMMIT}}"
SRC = "/tmp/exog"
snap_bin = glob.glob("/kaggle/input/**/exog_src.bin", recursive=True)
assert snap_bin, "code snapshot exog_src.bin not found under /kaggle/input"
subprocess.run(["mkdir", "-p", SRC], check=True)
subprocess.run(["tar", "-xzf", snap_bin[0], "-C", SRC], check=True)
snapshot = open(f"{SRC}/SNAPSHOT_COMMIT").read().strip()
print("snapshot commit:", snapshot, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/repro_check.py", '--sample', 'b5', '--out', '/kaggle/working/repro', "--commit", snapshot], cwd=SRC, check=True)
