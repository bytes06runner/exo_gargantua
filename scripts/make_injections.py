"""B1 injection parameters, host split and full-search subsample (docs/injection_design.md §1-4, §6; A7).

Deterministic; reads only frozen/committed files:
  data/injection_pool_log.csv        host order
  data/injection_pool_screened.csv   in_pool (P07 + P08/A6), stratum, A5 window sectors (P07-eligible)
  data/stellar_params.csv            TIC v8.2 R*, M*, Teff, logg, rho (solar units)
  data/raw/tess_orbit_times_*.csv    sector start times (epoch draw)
Writes:
  data/b1_split.csv                  tic, stratum, split (fit / calib / test)
  data/b1_injections.csv             one row per injection: host, split, sectors, class, true parameters, seed
  data/b1_search_subsample.csv       the 2,000 B1(ii) injections (from test only)
  results/b1_generation.json         counts and checks
Nothing here runs a search or the resolver. The light curves are built at run time (scripts/b1_run.py) by
multiplying these signals into the cached PDCSAP flux of the listed sectors.

Random streams (fixed before generation):
  split:      numpy.random.default_rng(20261004)  per stratum in order 2-3, 4-6, 7-12, 13-inf; hosts sorted
              by TIC then permuted; first round(0.6 n) training (first round(0.75 x training) fit, rest calib),
              remaining test.
  injections: numpy.random.default_rng(20260930)  hosts in pool-log order, 10 injections each, a fixed
              number of draws per injection (k, class, P, depth, b, epoch phase, then all EB extras);
              after all injections the same generator permutes the stratified seed-factor list.
  subsample:  a fresh numpy.random.default_rng(20260930), choice of 2,000 test injections without replacement.
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from exogargantua import sample as S  # noqa: E402

SEED_INJ, SEED_SPLIT = 20260930, 20261004
PER_STAR = 10
STRATA = ("2-3", "4-6", "7-12", "13-inf")
TRAIN_FRAC, FIT_FRAC = 0.6, 0.75
CLASSES = ("planet", "eb_twin", "eb_unequal", "eb_ecc")
CLASS_P = (0.60, 0.15, 0.15, 0.10)
ALIAS = ("1/5", "1/4", "1/3", "1/2", "2/3", "1", "3/2", "2", "3", "4", "5")
N_SUBSAMPLE = 2000


def split_hosts(scr):
    rng = np.random.default_rng(SEED_SPLIT)
    rows = []
    for st in STRATA:
        hosts = np.sort(scr.loc[scr["in_pool"] & (scr["stratum"] == st), "tic"].to_numpy())
        hosts = hosts[rng.permutation(hosts.size)]
        n_tr = int(round(TRAIN_FRAC * hosts.size))
        n_fit = int(round(FIT_FRAC * n_tr))
        for i, tic in enumerate(hosts):
            rows.append({"tic": int(tic), "stratum": st, "split": "fit" if i < n_fit else ("calib" if i < n_tr else "test")})
    return pd.DataFrame(rows)


def main():
    log = pd.read_csv(ROOT / "data" / "injection_pool_log.csv")
    scr = pd.read_csv(ROOT / "data" / "injection_pool_screened.csv")
    stars = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")
    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    split = split_hosts(scr)
    split_of = dict(zip(split["tic"], split["split"]))
    scr = scr.set_index("tic")
    hosts = [int(t) for t in log.loc[log["decision"] == "include", "tic"] if bool(scr.loc[int(t), "in_pool"])]

    rng = np.random.default_rng(SEED_INJ)
    rows = []
    for tic in hosts:
        win = [int(x) for x in str(scr.loc[tic, "window_sectors"]).split(";")]
        st = stars.loc[tic]
        for j in range(PER_STAR):
            k_draw = int(rng.integers(1, 11))
            cls = CLASSES[int(rng.choice(len(CLASSES), p=CLASS_P))]
            P = float(np.exp(rng.uniform(np.log(0.5), np.log(30.0))))
            depth = float(np.exp(rng.uniform(np.log(200e-6), np.log(0.05))))
            b = float(rng.uniform(0, 0.95))
            phase = float(rng.uniform(0, 1))
            ratio_unequal = float(rng.uniform(0.05, 0.8))
            ecc = float(rng.uniform(0.05, 0.5))
            omega = float(rng.uniform(0, 2 * np.pi))
            ratio_ecc = float(rng.uniform(0.3, 1.0))
            secs = win[:min(k_draw, len(win))]
            depth2 = {"planet": 0.0, "eb_twin": depth, "eb_unequal": depth * ratio_unequal, "eb_ecc": depth * ratio_ecc}[cls]
            rows.append({"inj_id": f"{tic}_{j}", "tic": tic, "split": split_of[tic], "stratum": scr.loc[tic, "stratum"],
                         "k_draw": k_draw, "n_sectors": len(secs), "sectors": ";".join(map(str, secs)), "cls": cls,
                         "P_true": P, "t0_true": float(starts[secs[0]]) + phase * P, "depth": depth, "b": b,
                         "depth2": depth2, "ecc": ecc if cls == "eb_ecc" else 0.0, "omega": omega if cls == "eb_ecc" else 0.0,
                         "r_star": float(st["rad"]), "m_star": float(st["mass"]), "teff": float(st["Teff"]),
                         "logg": float(st["logg"]), "rho_solar": float(st["rho"]), "e_rho_solar": float(st["e_rho"])})
    inj = pd.DataFrame(rows)
    n = len(inj)
    r_list = np.array([ALIAS[i % len(ALIAS)] for i in range(n)])  # stratified: n/11 +- 1 per factor
    inj["seed_r"] = r_list[rng.permutation(n)]
    inj["P_seed"] = [float(Fraction(r)) * P for r, P in zip(inj["seed_r"], inj["P_true"])]

    test_ids = inj.loc[inj["split"] == "test", "inj_id"].to_numpy()
    sub = np.sort(np.random.default_rng(SEED_INJ).choice(test_ids, N_SUBSAMPLE, replace=False))

    split.sort_values("tic").to_csv(ROOT / "data" / "b1_split.csv", index=False)
    inj.to_csv(ROOT / "data" / "b1_injections.csv", index=False)
    pd.DataFrame({"inj_id": sub}).to_csv(ROOT / "data" / "b1_search_subsample.csv", index=False)

    hosts_by_split = split.groupby("split")["tic"].apply(set)
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "seeds": {"injections": SEED_INJ, "split": SEED_SPLIT, "subsample": SEED_INJ},
           "hosts": len(hosts), "injections": n,
           "hosts_by_split": split["split"].value_counts().to_dict(),
           "hosts_by_split_and_stratum": split.groupby(["stratum", "split"]).size().unstack().to_dict(orient="index"),
           "injections_by_split": inj["split"].value_counts().to_dict(),
           "injections_by_class": inj["cls"].value_counts().to_dict(),
           "injections_by_seed_r": inj["seed_r"].value_counts().to_dict(),
           "injections_by_n_sectors": inj["n_sectors"].value_counts().sort_index().to_dict(),
           "no_host_in_two_splits": bool(all(len(hosts_by_split[a] & hosts_by_split[b]) == 0
                                             for a in hosts_by_split.index for b in hosts_by_split.index if a < b)),
           "subsample": {"n": int(sub.size), "all_test": bool(inj.set_index("inj_id").loc[sub, "split"].eq("test").all())}}
    (ROOT / "results" / "b1_generation.json").write_text(json.dumps(rep, indent=1, default=int))
    print(json.dumps(rep, indent=1, default=int))


if __name__ == "__main__":
    main()
