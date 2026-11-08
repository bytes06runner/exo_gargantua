"""TIC v8.2 stellar parameters for every included B5 star (decision B5-r1), with the same query as the Gate 2 freeze
(scripts/freeze_sample.py::tic_params). Output: data/b5_stellar_params.csv and results/b5_stellar_params_run.json."""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_sample import tic_params  # noqa: E402


def main():
    log = pd.read_csv(ROOT / "data" / "b5_sample_log.csv")
    tics = sorted(int(t) for t in log.loc[log["decision"] == "include", "tic"])
    df = tic_params(tics).drop_duplicates("tic").sort_values("tic")
    assert set(df["tic"]) <= set(tics)
    df.to_csv(ROOT / "data" / "b5_stellar_params.csv", index=False)
    info = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
            "query_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "source": "MAST TIC v8.2 via astroquery Catalogs",
            "stars_requested": len(tics), "stars_returned": int(len(df)), "with_rho": int(np.isfinite(df["rho"]).sum()),
            "with_rad_and_mass": int((np.isfinite(df["rad"]) & np.isfinite(df["mass"])).sum())}
    (ROOT / "results" / "b5_stellar_params_run.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
