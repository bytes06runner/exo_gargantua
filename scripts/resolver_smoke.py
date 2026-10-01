"""Phase 3 execution test of the resolver on the 20 Phase-2 smoke targets (Kaggle CPU).

These 20 TOIs are members of the sealed B2 holdout (amendment A4). This script is an EXECUTION and RUNTIME
test only: it loads each target's early-sector light curve exactly as the B2 seed runs did, seeds the resolver
with the committed (sealed) TLS and BLS seed periods, and records runtime and resolver outputs. It reads no
truth column and computes no comparison with any catalog or truth period. The outputs are stored sealed;
no script may compare them with truth before the tag `resolver-frozen-v1` exists
(exogargantua.scoring.require_frozen_resolver).
Output: resolver_smoke.csv (runtime and resolver outputs only).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import sample as S  # noqa: E402
from exogargantua.resolver import Star, resolve  # noqa: E402
import b2_seeds as B2  # noqa: E402

RHO_SUN = 1.41  # g cm^-3; TIC v8.2 'rho' is in solar units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    smoke = json.loads(Path(glob.glob(str(ROOT / "results/kaggle/smoke_run/*/smoke/smoke_results.json"))[0]).read_text())
    tois = [str(r["toi"]) for r in smoke["targets"]]
    targets = B2.b2_targets()
    targets = targets[targets["toi"].isin(tois)]
    tls = pd.concat(pd.read_csv(f, dtype={"toi": str}) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/seeds_tls_*.csv")))
    bls = pd.concat(pd.read_csv(f, dtype={"toi": str}) for f in glob.glob(str(ROOT / "results/kaggle/b2/*/b2/seeds_bls_gpu_*.csv")))
    cache = B2.find_cache(a.cache)
    man = pd.read_csv(cache / "manifest.csv")
    excluded = set(zip(man.loc[man["excluded"].notna(), "tic"], man.loc[man["excluded"].notna(), "sector"]))
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    toi_table = S.normalise_toi(pd.read_csv(ROOT / snaps["S-TOI"]["file"], comment="#"))
    stars = pd.read_csv(ROOT / "data" / "stellar_params.csv").drop_duplicates("tic").set_index("tic")
    rows = []
    for _, r in targets.iterrows():
        tic = int(r["tic"])
        t0 = time.time()
        t, f, m = B2.early_lc(cache, excluded, r, toi_table)
        load_s = time.time() - t0
        st = stars.loc[tic] if tic in stars.index else None
        star = Star(float(st["rho"]) * RHO_SUN, float(st["e_rho"]) * RHO_SUN) if st is not None else Star()
        for seed_name, tab, col in (("tls", tls, "tls_period"), ("bls", bls, "bls_period")):
            s_row = tab[tab["toi"] == r["toi"]]
            if s_row.empty or not np.isfinite(s_row[col].iloc[0]):
                rows.append({"toi": r["toi"], "seed": seed_name, "status": "no seed"})
                continue
            try:
                s, hyps = resolve(t[m], f[m] / np.nanmedian(f[m]), float(s_row[col].iloc[0]), star=star)
                rows.append({"toi": r["toi"], "tic": tic, "seed": seed_name, "status": "ok", "n_points": s["n_points"],
                             "load_s": load_s, "runtime_s": s["runtime_s"], "n_hypotheses": len(hyps), "beta": s["beta"],
                             "p_max": s["p_max"], "abstain": s["abstain"], "r_map": s["r_map"], "P_map": s["P_map"]})
            except Exception as e:
                rows.append({"toi": r["toi"], "seed": seed_name, "status": f"error: {e!r}"[:300]})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "resolver_smoke.csv", index=False)
    (out / "done.json").write_text(json.dumps({"git_commit": a.commit, "n_targets": int(len(targets)),
                                               "sealed": "no truth comparison (A4)"}))


if __name__ == "__main__":
    main()
