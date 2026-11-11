"""Post hoc (decision Indep-1): B5 periods against Gaia DR3 eclipsing-binary periods (gaiadr3.vari_eclipsing_binary), which are
derived from Gaia photometry only, independently of TESS and of the Prsa et al. (2022) catalog.
data/b5_stellar_params.csv holds Gaia DR2 identifiers from the TIC; they are matched to DR3 through gaiadr3.dr2_neighbourhood.
Writes data/b5_gaia_dr3_eb.csv (the fetched table) and results/b5_gaia_check.json."""

import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b2_score import wilson, mcnemar  # noqa: E402

TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
TOL = 1e-3


def query(ids):
    adql = ("SELECT n.dr2_source_id, n.dr3_source_id, n.angular_distance, e.frequency, e.frequency_error, e.global_ranking "
            "FROM gaiadr3.dr2_neighbourhood AS n JOIN gaiadr3.vari_eclipsing_binary AS e ON n.dr3_source_id = e.source_id "
            f"WHERE n.dr2_source_id IN ({','.join(str(i) for i in ids)})")
    for k in range(6):
        r = requests.post(TAP, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}, timeout=300)
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text))
        print("HTTP", r.status_code, "retrying", flush=True)
        time.sleep(60 * (k + 1))
    raise SystemExit(f"Gaia TAP unavailable: HTTP {r.status_code}")


def rate(x):
    x = np.asarray(x, bool)
    return {"count": int(x.sum()), "n": int(x.size), "fraction": float(x.mean()) if x.size else None, "ci95": wilson(int(x.sum()), int(x.size))}


def main():
    st = pd.read_csv(ROOT / "data/b5_stellar_params.csv")
    st = st[st["GAIA"].notna()].copy()
    st["GAIA"] = st["GAIA"].astype("int64")
    out = ROOT / "data/b5_gaia_dr3_eb.csv"
    if not out.exists():
        ids = st["GAIA"].tolist()
        parts = [query(ids[i:i + 500]) for i in range(0, len(ids), 500)]
        pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    g = pd.read_csv(out).sort_values("angular_distance").drop_duplicates("dr2_source_id")
    g["P_gaia"] = 1.0 / g["frequency"]
    m = st[["tic", "GAIA"]].merge(g, left_on="GAIA", right_on="dr2_source_id")
    d = pd.read_csv(ROOT / "results/b5_per_target.csv").merge(m[["tic", "P_gaia", "global_ranking"]], on="tic")
    p = d[d["primary_sample"]].copy()
    # Gaia reports the orbital frequency for most systems but can report twice it; accept P_gaia and 2 P_gaia as the same system and say so
    def ok(col, mult=1.0):
        return np.abs(p[col] / (mult * p["P_gaia"]) - 1) < TOL
    rep = {"label": "POST HOC (decision Indep-1)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "fetched_rows": int(len(g)), "primary_with_gaia": int(len(p)), "catalog_equals_gaia": rate(ok("P_catalog")), "catalog_equals_2x_gaia": rate(ok("P_catalog", 2.0)),
           "catalog_equals_half_gaia": rate(ok("P_catalog", 0.5)),
           "vs_gaia": {k: rate(ok(c)) for k, c in (("raw_tls", "P_tls"), ("raw_bls", "P_bls"), ("resolver", "P_resolver"))},
           "mcnemar_vs_tls": mcnemar(ok("P_tls"), ok("P_resolver")), "mcnemar_vs_bls": mcnemar(ok("P_bls"), ok("P_resolver"))}
    (ROOT / "results/b5_gaia_check.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
