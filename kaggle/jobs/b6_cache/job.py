"""Kaggle job (CPU): B6 compact cache of the 4,054 pinned early-window files (docs/b6_definition.md)."""
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
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/build_cache.py", "--pins", "data/b6_pins_early.csv", "--n-chunks", "1", "--chunk", "1",
                "--no-prev", "--out", "/kaggle/working/cache", "--commit", snapshot], cwd=SRC, check=True)
