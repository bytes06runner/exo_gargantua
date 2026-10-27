"""Kaggle job (CPU, G2): POST HOC false-alarm context (decision FA-ctx.2), TLS SDE on the scrambled FA1 light curves,
shard 1 of 10 (rerun of the shards that failed when the repository became private). The code comes from a private
Kaggle dataset holding a `git archive` snapshot of the repository instead of a git clone."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
SRC = "/tmp/exog"
subprocess.run(["mkdir", "-p", SRC], check=True)
subprocess.run(["tar", "-xzf", "/kaggle/input/exog-src/exog_src.tar.gz", "-C", SRC], check=True)
print("snapshot commit:", open(f"{SRC}/SNAPSHOT_COMMIT").read().strip(), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/fa_context_run.py", "--engine", "tls", "--variant", "scrambled", "--shard", "1",
                "--nshards", "10", "--out", "/kaggle/working/fac", "--commit", open(f"{SRC}/SNAPSHOT_COMMIT").read().strip()],
               cwd=SRC, check=True)
