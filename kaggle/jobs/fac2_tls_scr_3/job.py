"""Kaggle job (CPU, G2): POST HOC false-alarm context (decision FA-ctx.2), TLS SDE on the scrambled FA1 light curves,
shard 3 of 10 (rerun of the shards that failed when the repository became private). The code comes from a private
Kaggle dataset holding a `git archive` snapshot of the repository instead of a git clone."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
SRC = "/tmp/exog"
import glob
import shutil
tars = glob.glob("/kaggle/input/**/exog_src.tar.gz", recursive=True)
if tars:  # dataset kept as an archive
    subprocess.run(["mkdir", "-p", SRC], check=True)
    subprocess.run(["tar", "-xzf", tars[0], "-C", SRC], check=True)
else:  # Kaggle extracted it on upload
    snap = glob.glob("/kaggle/input/**/SNAPSHOT_COMMIT", recursive=True)
    assert snap, "code snapshot not found under /kaggle/input"
    shutil.copytree(snap[0].rsplit("/", 1)[0], SRC)
print("snapshot commit:", open(f"{SRC}/SNAPSHOT_COMMIT").read().strip(), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/fa_context_run.py", "--engine", "tls", "--variant", "scrambled", "--shard", "3",
                "--nshards", "10", "--out", "/kaggle/working/fac", "--commit", open(f"{SRC}/SNAPSHOT_COMMIT").read().strip()],
               cwd=SRC, check=True)
