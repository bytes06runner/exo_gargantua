"""B1 sensitivity to the host-screening rule (promised in amendment A6: "the paper reports pool size and injection
results under both the raw rule (V1) and A6"). The injections were generated on the A6 pool only, so the raw-rule
pool cannot be injected afresh; instead the frozen v1 B1 test results are re-scored on the test-split hosts that ALSO
pass the raw rule (A6 and raw), and on the A6-only hosts that the raw rule would have excluded. Same scoring code as
scripts/b1_score.py. Output: results/b1_raw_rule_sensitivity.json."""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import sample as S, scoring  # noqa: E402
import b1_score as B  # noqa: E402

PATTERNS = {"B1_i_wrong_seed": "results/kaggle/b1/*/b1/b1_wrong_*.jsonl.gz",
            "B1_ii_tls_seed": "results/kaggle/b1/*/b1/b1_search_tls_*.jsonl.gz",
            "B1_ii_bls_seed": "results/kaggle/b1/*/b1/b1_search_bls_gpu_*.jsonl.gz"}


def main():
    tag = scoring.require_frozen_resolver()
    inj0 = pd.read_csv(ROOT / "data" / "b1_injections.csv").set_index("inj_id")
    pool = pd.read_csv(ROOT / "data" / "injection_pool_screened.csv").set_index("tic")
    raw_ok = set(pool.index[pool["in_pool_raw_rule"].astype(bool)])
    with open(ROOT / "models" / "resolver" / "learned.pkl", "rb") as fh:
        lc = pickle.load(fh)
    thr = json.loads((ROOT / "models" / "resolver" / "thresholds.json").read_text())["abstain_threshold"]
    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    test_hosts = set(inj0.loc[inj0["split"] == "test", "tic"])
    rep = {"label": "Sensitivity analysis promised in amendment A6 (resolver frozen at resolver-frozen-v1)",
           "resolver_frozen_v1": tag,
           "test_hosts": {"all": len(test_hosts), "pass_raw_rule": len(test_hosts & raw_ok), "a6_only": len(test_hosts - raw_ok)}}
    for subset, keep in (("pass_raw_rule", lambda t: t in raw_ok), ("a6_only", lambda t: t not in raw_ok)):
        inj = inj0.copy()
        inj.loc[~inj["tic"].map(keep), "split"] = "excluded"  # score() keeps split == "test" only
        rep[subset] = {}
        for key, pat in PATTERNS.items():
            out = B.score(pat, inj, lc, thr, starts)
            rep[subset][key] = {"n": out["n"], "seed_alone_accuracy": out["seed_alone_accuracy"],
                                "truth_in_candidates": out["truth_in_candidates"], "learned": out["learned"]["all"]}
    (ROOT / "results" / "b1_raw_rule_sensitivity.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
