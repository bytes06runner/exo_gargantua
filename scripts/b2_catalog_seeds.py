"""B2 catalog-based seeds (docs/baselines.md §4-5): SPOC-1, SPOC-E, QLP-hist, SPOC-hist.

Local, light (table lookups). Sealed (A4): writes seed periods only, never compares with truth.
TCE <-> TOI matching uses the TOI catalog ephemeris (same TIC; |P_TCE/(r P_TOI) - 1| < 0.01 for r in
the alias set; TCE epoch within max(0.5 x TOI duration, 0.1 d) of a TOI-predicted transit; highest
tce_model_snr if several). Output: results/b2/seeds_catalog.csv + run info JSON.
"""

from __future__ import annotations

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
from exogargantua import sample as S  # noqa: E402
import b2_seeds as B2  # noqa: E402


def match_tce(tces: pd.DataFrame, toi) -> pd.Series | None:
    if tces.empty:
        return None
    win = max(0.5 * toi["duration_h"] / 24.0 if np.isfinite(toi["duration_h"]) else 0.0, 0.1)
    ok = []
    for _, c in tces.iterrows():
        if S.alias_ratio_match(c["tce_period"], toi["period"]) is None:
            continue
        if S.epoch_coincides(c["tce_time0bt"], toi["epoch_btjd"], toi["period"], win):
            ok.append(c)
    if not ok:
        return None
    return max(ok, key=lambda c: c.get("tce_model_snr", -np.inf) if np.isfinite(c.get("tce_model_snr", np.nan)) else -np.inf)


def guerrero_table() -> pd.DataFrame:
    f = sorted((ROOT / "data" / "raw").glob("guerrero2021_toi_table2_*.dat"))[0]
    cols = [(0, 5, "pipeline"), (15, 25, "tic"), (26, 33, "toi"), (127, 137, "per"), (196, 234, "sectors")]
    rows = [{n: line[a:b].strip() for a, b, n in cols} for line in f.read_text().splitlines() if line.strip()]
    g = pd.DataFrame(rows)
    g["toi"] = g["toi"].map(lambda x: f"{float(x):.2f}")
    g["per"] = pd.to_numeric(g["per"], errors="coerce")
    return g.set_index("toi")


def main():
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    toi_table = S.normalise_toi(pd.read_csv(ROOT / snaps["S-TOI"]["file"], comment="#"))
    toi_table["key"] = toi_table["toi"].map(lambda x: f"{x:.2f}")
    toi_table = toi_table.set_index("key")
    tces = pd.read_csv(ROOT / "data" / "spoc_tces_toi_hosts.csv.gz")
    tces_by_tic = {int(k): g for k, g in tces.groupby("ticid")}
    g21 = guerrero_table()
    out = []
    for _, r in B2.b2_targets().iterrows():
        toi, tic = toi_table.loc[r["toi"]], int(r["tic"])
        early = [int(x) for x in str(r["sectors_early"]).split(";")]
        t = tces_by_tic.get(tic, pd.DataFrame(columns=tces.columns))
        row = {"toi": r["toi"], "tic": tic}
        # SPOC-1: single-sector run of the first early sector
        s1 = t[(t["run_s0"] == early[0]) & (t["run_s1"] == early[0])]
        m1 = match_tce(s1, toi)
        row["spoc1_period"] = float(m1["tce_period"]) if m1 is not None else np.nan
        row["spoc1_run"] = m1["run"] if m1 is not None else ""
        # SPOC-E: longest run fully inside the early window (sector-number span within [min E, max E])
        inside = t[(t["run_s0"] >= min(early)) & (t["run_s1"] <= max(early))].copy()
        mE = None
        if not inside.empty:
            inside["span"] = inside["run_s1"] - inside["run_s0"]
            for span in sorted(inside["span"].unique(), reverse=True):
                cand = inside[inside["span"] == span].sort_values("run", ascending=False)  # later processing first
                for run in cand["run"].unique():
                    mE = match_tce(cand[cand["run"] == run], toi)
                    if mE is not None:
                        break
                if mE is not None:
                    break
        row["spocE_period"] = float(mE["tce_period"]) if mE is not None else np.nan
        row["spocE_run"] = mE["run"] if mE is not None else ""
        # QLP-hist / SPOC-hist: Guerrero+2021 table period if all its sectors are in the early window
        row["qlp_hist_period"] = row["spoc_hist_period"] = np.nan
        if r["toi"] in g21.index:
            gr = g21.loc[r["toi"]]
            secs = [int(x) for x in str(gr["sectors"]).split()]
            if secs and set(secs) <= set(early) and np.isfinite(gr["per"]):
                row[f"{gr['pipeline']}_hist_period"] = float(gr["per"])
        out.append(row)
    df = pd.DataFrame(out)
    dest = ROOT / "results" / "b2"
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "seeds_catalog.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    info = {"git_commit": commit, "run_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n_toi": int(len(df)), "coverage": {c: int(df[c].notna().sum()) for c in df.columns if c.endswith("_period")},
            "sealed": "A4: no truth comparison"}
    (dest / "seeds_catalog_run.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
