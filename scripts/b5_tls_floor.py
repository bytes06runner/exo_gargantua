"""Post hoc (decision B5-floor): why raw TLS changes between the two B5 runs, and what the density term F4 does on binaries.

TLS builds its trial-period grid for a single star of the given radius and mass, with a shortest period at an orbit of
3 stellar radii (transitleastsquares.grid.period_grid, f_max). With TIC parameters (the protocol, run B5-r1) many binaries
have a true period below that floor; with TLS's default solar values (original run) few do.
Reads the rerun (results/kaggle/b5r) and the original run (results/kaggle/b5). Output: results/b5_tls_floor.json.
"""

import glob
import json
import subprocess
import sys
from math import pi
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402
from b2_score import wilson, mcnemar  # noqa: E402
import resolver_models as RM  # noqa: E402

TOL = 1e-3
G, MSUN, RSUN = 6.674e-11, 1.989e30, 6.957e8  # as in tls_constants, to the precision needed here


def floor_days(rad, mass):
    return 2 * pi * np.sqrt((3 * rad * RSUN) ** 3 / (G * mass * MSUN)) / 86400


def load(run):
    out = {}
    for p in sorted(glob.glob(str(ROOT / f"results/kaggle/{run}/*/b5/b5_*.jsonl.gz"))):
        for r in read_records(p):
            out[(r["tic"], r["engine"])] = r
    return out


def rate(x):
    x = np.asarray(x, bool)
    return {"n": int(x.size), "accuracy": float(x.mean()) if x.size else None, "ci95": wilson(int(x.sum()), int(x.size))}


def score(r, Pt, model):
    """(seed correct, resolver correct, F4 active on the chosen alias) for one run record."""
    if r is None or "error" in r:
        return False, False, False
    ok = [abs(x["P"] / Pt - 1) < TOL for x in r["rows"]]
    i, _ = RM.decide(model, [(x["r"], x["P"], x["x"]) for x in r["rows"]])
    f4 = any(abs(x["x"].get("d_F4_density", 0.0) or 0.0) > 0 for x in r["rows"])
    return bool(abs(r["seed"]["P_seed"] / Pt - 1) < TOL), bool(ok[i]), f4


def main():
    v2 = RM.load("v2")[0]
    new, old = load("b5r"), load("b5")
    log = pd.read_csv(ROOT / "data/b5_sample_log.csv")
    log = log[(log["decision"] == "include") & (log["tier"] == "B")].set_index("tic")
    st = pd.read_csv(ROOT / "data/b5_stellar_params.csv").set_index("tic")
    rows = []
    for tic, L in log.iterrows():
        t = int(tic)
        rn = new[(t, "tls")]
        if rn.get("h3_eclipses", 0) < 2:
            continue
        Pt = float(L["period"])
        a = score(rn, Pt, v2)
        b = score(old.get((t, "tls")), Pt, v2)
        c = score(new[(t, "bls_gpu")], Pt, v2)
        d = score(old.get((t, "bls_gpu")), Pt, v2)
        rad, mass = (st.loc[t, "rad"], st.loc[t, "mass"]) if t in st.index else (np.nan, np.nan)
        Pn = np.nan if "error" in rn else rn["seed"]["P_seed"]
        rows.append({"tic": t, "P": Pt, "rad": rad, "mass": mass, "P_tls_new": Pn, "tls_error": "error" in rn,
                     "tls_new": a[0], "v2_new": a[1], "f4_tls": a[2], "tls_old": b[0], "v2_old": b[1], "v2_bls_new": c[1], "f4_bls": c[2], "bls": c[0], "v2_bls_old": d[1]})
    d = pd.DataFrame(rows)
    d["floor"] = floor_days(d["rad"], d["mass"])
    known = d[d["floor"].notna()]
    below = known[known["P"] < known["floor"]]
    above = known[known["P"] >= known["floor"]]
    lost = d[d["tls_old"] & ~d["tls_new"]]
    ratio = lost["P_tls_new"] / lost["P"]
    rep = {"label": "POST HOC (decision B5-floor)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "primary_n": int(len(d)), "with_radius_and_mass": int(len(known)), "tls_errors_primary": int(d["tls_error"].sum()),
           "true_period_below_floor": int(len(below)),
           "below_floor": {"raw_tls_rerun": rate(below["tls_new"]), "raw_tls_original": rate(below["tls_old"]), "v2_rerun": rate(below["v2_new"]), "raw_bls": rate(below["bls"])},
           "above_floor": {"raw_tls_rerun": rate(above["tls_new"]), "raw_tls_original": rate(above["tls_old"]), "v2_rerun": rate(above["v2_new"]), "raw_bls": rate(above["bls"]),
                           "v2_vs_raw_tls": mcnemar(above["tls_new"], above["v2_new"]), "v2_vs_raw_bls": mcnemar(above["bls"], above["v2_new"])},
           "raw_tls_correct_only_in_original": int(len(lost)), "raw_tls_correct_only_in_rerun": int((~d["tls_old"] & d["tls_new"]).sum()),
           "lost_at_integer_multiple": int((np.abs(ratio - ratio.round()) < 0.01)[ratio.round() >= 2].sum()),
           "lost_within_1pct_of_truth": int((np.abs(ratio - 1) < 0.01).sum()),
           "f4_active_tls_seeded": int(d["f4_tls"].sum()), "f4_active_bls_seeded": int(d["f4_bls"].sum()),
           "f4_effect_same_bls_seeds": {"v2_with_f4": rate(d["v2_bls_new"]), "v2_without_f4": rate(d["v2_bls_old"]), "mcnemar_without_vs_with": mcnemar(d["v2_bls_new"], d["v2_bls_old"])},
           "original_vs_rerun_v2_tls": mcnemar(d["v2_new"], d["v2_old"])}
    # the same floor on the other benchmarks
    sp = pd.read_csv(ROOT / "data/stellar_params.csv").drop_duplicates("tic").set_index("tic")
    b2 = pd.read_csv(ROOT / "results/b2_per_target.csv")
    b2f = floor_days(b2["tic"].map(sp["rad"]), b2["tic"].map(sp["mass"]))
    inj = pd.read_csv(ROOT / "data/b1_injections.csv")
    sub = inj[inj["inj_id"].isin(pd.read_csv(ROOT / "data/b1_search_subsample.csv")["inj_id"])]
    b1f = floor_days(sub["r_star"], sub["m_star"])
    rep["other_benchmarks"] = {"b2": {"n_with_floor": int(b2f.notna().sum()), "below_floor": int((b2["P_true"] < b2f).sum())},
                               "b1_search_subsample": {"n_with_floor": int(b1f.notna().sum()), "below_floor": int((sub["P_true"] < b1f).sum())}}
    (ROOT / "results/b5_tls_floor.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
