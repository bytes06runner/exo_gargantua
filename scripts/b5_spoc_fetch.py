"""B5 SPOC baseline, step 1 (decision B5-spoc; Kaggle CPU): download the frozen list of MAST dvr-tcestats files (freeze
manifest of 2026-09-30) and keep the TCEs on the 1,500 included B5 stars. Writes periods only, no comparison with truth.
Output: spoc_tces_b5_hosts.csv.gz, done_b5spoc.json."""

import argparse
import glob
import io
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://archive.stsci.edu/missions/tess/catalogs/tce/"
COLS = ["ticid", "tce_plnt_num", "sectors", "tce_period", "tce_period_err", "tce_time0bt", "tce_time0bt_err", "tce_duration",
        "tce_depth", "tce_model_snr", "tce_max_mult_ev", "tce_num_transits", "tce_impact"]


def fetch(name, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + name, headers={"User-Agent": "exog-b5spoc"}), timeout=300) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    man = json.loads(Path(glob.glob(str(ROOT / "results/kaggle/freeze_sample/v2_*/freeze/freeze_manifest.json"))[0]).read_text())
    files = man["spoc_tce"]["tce_files"]
    log = pd.read_csv(ROOT / "data" / "b5_sample_log.csv")
    tics = set(log.loc[log["decision"] == "include", "tic"])

    def one(name):
        s0, s1 = re.search(r"-s(\d{4})-s(\d{4})_dvr", name).groups()
        df = pd.read_csv(io.BytesIO(fetch(name)), comment="#", low_memory=False)
        df = df[df["ticid"].isin(tics)][[c for c in COLS if c in df.columns]].copy()
        df["run"], df["run_s0"], df["run_s1"] = name, int(s0), int(s1)
        return df

    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        tces = pd.concat(list(ex.map(one, files)), ignore_index=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tces.to_csv(out / "spoc_tces_b5_hosts.csv.gz", index=False)
    (out / "done_b5spoc.json").write_text(json.dumps({"git_commit": a.commit, "tce_files": len(files), "rows": int(len(tces)),
                                                      "stars_with_tce": int(tces["ticid"].nunique()), "wall_s": time.time() - t0}))
    print(len(files), "files;", len(tces), "TCE rows;", tces["ticid"].nunique(), "stars", flush=True)


if __name__ == "__main__":
    sys.exit(main())
