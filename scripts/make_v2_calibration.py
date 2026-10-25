"""Resolver v2 calibration injections (decision V2-b, docs/decisions.md).

2,000 fresh injections on TRAINING-split hosts only (fit + calib in data/b1_split.csv; never a test host).
numpy.random.default_rng(20261006): for each injection, a host uniform over the training hosts (with replacement),
then exactly the per-injection draws of scripts/make_injections.py (k, class, P, depth, b, epoch phase, EB extras).
No wrong seed: each injection is searched with GPU BLS and with TLS (scripts/b1_run.py --mode search) and the
resolver runs from both search seeds.
Writes data/v2_calib_injections.csv and results/v2_calib_generation.json.
"""

from __future__ import annotations

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
from make_injections import CLASSES, CLASS_P  # noqa: E402

SEED = 20261006
N = 2000


def main():
    split = pd.read_csv(ROOT / "data" / "b1_split.csv")
    train_hosts = np.sort(split.loc[split["split"].isin(["fit", "calib"]), "tic"].to_numpy())
    test_hosts = set(split.loc[split["split"] == "test", "tic"])
    scr = pd.read_csv(ROOT / "data" / "injection_pool_screened.csv").set_index("tic")
    stars = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")
    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    strat = dict(zip(split["tic"], split["stratum"]))
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(N):
        tic = int(train_hosts[int(rng.integers(train_hosts.size))])
        win = [int(x) for x in str(scr.loc[tic, "window_sectors"]).split(";")]
        st = stars.loc[tic]
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
        rows.append({"inj_id": f"v2c_{i:04d}", "tic": tic, "split": "v2cal", "stratum": strat[tic], "k_draw": k_draw,
                     "n_sectors": len(secs), "sectors": ";".join(map(str, secs)), "cls": cls, "P_true": P,
                     "t0_true": float(starts[secs[0]]) + phase * P, "depth": depth, "b": b, "depth2": depth2,
                     "ecc": ecc if cls == "eb_ecc" else 0.0, "omega": omega if cls == "eb_ecc" else 0.0,
                     "r_star": float(st["rad"]), "m_star": float(st["mass"]), "teff": float(st["Teff"]), "logg": float(st["logg"]),
                     "rho_solar": float(st["rho"]), "e_rho_solar": float(st["e_rho"]), "seed_r": "", "P_seed": np.nan})
    inj = pd.DataFrame(rows)
    assert not set(inj["tic"]) & test_hosts, "a test host was drawn"
    inj.to_csv(ROOT / "data" / "v2_calib_injections.csv", index=False)
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "seed": SEED, "n": N, "distinct_hosts": int(inj["tic"].nunique()), "training_hosts_available": int(train_hosts.size),
           "test_hosts_used": 0, "by_class": inj["cls"].value_counts().to_dict(), "by_stratum": inj["stratum"].value_counts().to_dict(),
           "by_n_sectors": inj["n_sectors"].value_counts().sort_index().to_dict()}
    (ROOT / "results" / "v2_calib_generation.json").write_text(json.dumps(rep, indent=1, default=int))
    print(json.dumps(rep, indent=1, default=int))


if __name__ == "__main__":
    main()
