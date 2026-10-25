"""Kaggle job (CPU, G2): train the learned combiner and choose abstain thresholds on the B1 training split (F1)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/b1_train.py"], cwd=SRC, check=True)
import shutil
shutil.copytree(f"{SRC}/models", "/kaggle/working/models")
shutil.copy(f"{SRC}/results/b1_training_report.json", "/kaggle/working/")
