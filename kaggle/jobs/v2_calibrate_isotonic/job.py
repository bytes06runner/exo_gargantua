"""Kaggle job (CPU, G2): fit the v2 isotonic calibration map and thresholds on all v2 calibration runs (V2-f)."""
import subprocess
import sys

COMMIT = "{{COMMIT}}"
REPO = "https://github.com/bytes06runner/exo_gargantua.git"
SRC = "/tmp/exog"
subprocess.run(["git", "clone", "--quiet", REPO, SRC], check=True)
subprocess.run(["git", "-C", SRC, "checkout", "--quiet", COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", SRC], check=True)
subprocess.run([sys.executable, "-u", f"{SRC}/scripts/v2_calibrate_isotonic.py"], cwd=SRC, check=True)
import shutil
shutil.copytree(f"{SRC}/models/resolver_v2", "/kaggle/working/models/resolver_v2")
shutil.copy(f"{SRC}/results/v2_calibration_final.json", "/kaggle/working/")
