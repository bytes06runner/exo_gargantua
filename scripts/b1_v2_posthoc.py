"""POST HOC (decision V2-c): resolver v2 on the B1 test split, next to v1. The B1(ii) v1 results had been seen when
v2 was decided, so these v2 numbers are post hoc. v2 changes only probabilities/thresholds, never the chosen alias,
so accuracy is identical to v1; calibration (ECE), accuracy at 90 % coverage and abstention change.
Guarded by both freeze tags. Output: results/b1_v2_posthoc_report.json."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from exogargantua import sample as S, scoring  # noqa: E402
import b1_score as B  # noqa: E402
import resolver_models as RM  # noqa: E402


def main():
    t1 = scoring.require_frozen_resolver()
    t2 = scoring.require_frozen_resolver(tag_name=scoring.FROZEN_TAG_V2, paths=scoring.FROZEN_PATHS_V2)
    inj = pd.read_csv(ROOT / "data" / "b1_injections.csv").set_index("inj_id")
    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    rep = {"label": "POST HOC (V2-c): v2 was decided after v1's B1(ii) test results were seen",
           "resolver_frozen_v1": t1, "resolver_frozen_v2": t2}
    for v in ("v1", "v2"):
        lc, thr = RM.load(v)
        rep[v] = {"abstain_threshold": thr}
        for key, pat in (("B1_i_wrong_seed", "results/kaggle/b1/*/b1/b1_wrong_*.jsonl.gz"),
                         ("B1_ii_bls_seed", "results/kaggle/b1/*/b1/b1_search_bls_gpu_*.jsonl.gz"),
                         ("B1_ii_tls_seed", "results/kaggle/b1/*/b1/b1_search_tls_*.jsonl.gz")):
            s = B.score(pat, inj, lc, thr, starts)
            rep[v][key] = {"all": s["learned"]["all"], "by_stratum": s["learned"]["by_stratum"], "by_seed_r": s["learned"]["by_seed_r"],
                           "n": s["n"], "errors_counted_wrong": s["errors_counted_wrong"]}
    (ROOT / "results" / "b1_v2_posthoc_report.json").write_text(json.dumps(rep, indent=1, default=float))
    for v in ("v1", "v2"):
        for k in ("B1_i_wrong_seed", "B1_ii_bls_seed", "B1_ii_tls_seed"):
            a = rep[v][k]["all"]
            print(v, k, {x: (round(a[x], 3) if isinstance(a[x], float) else a[x]) for x in
                         ("n", "accuracy", "ece", "accuracy_at_90pct_coverage", "abstain_rate", "accuracy_non_abstained")})


if __name__ == "__main__":
    main()
