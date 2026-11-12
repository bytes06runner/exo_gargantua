"""Kaggle job (CPU): Tschudi (2026a) correction on the B5 TLS results (decision B5-tsch), shard 2 of 10."""
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
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b5_tschudi.py", "--shard", "2", "--nshards", "10",
                "--out", "/kaggle/working/b5tsch", "--commit", snapshot], cwd=SRC, check=True)
