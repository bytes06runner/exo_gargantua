"""Cost of the P08 pool screen (docs/sample_definition.md §5), literal vs amendment A5.

(a) literal: BLS (pre-registered §6 configuration) on each pool star's FULL pinned light curve;
(b) A5: the same search on the data the star's injections use (its first <= 10 pinned sectors that
    start within 365.25 d of the first one; docs/injection_design.md §1).
Costs use the measured constants in results/cost_estimate_v2.json (GPU BLS on one T4; astropy BLS
on one CPU core). Output: results/p08_cost.json.
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
import estimate_costs as E1  # noqa: E402

SESSION_H = 12.0


def main():
    c = json.loads((ROOT / "results" / "cost_estimate_v2.json").read_text())["constants"]
    starts = S.sector_start_btjd(pd.read_csv(sorted((ROOT / "data" / "raw").glob("tess_orbit_times_*.csv"))[0]))
    pool = pd.read_csv(ROOT / "data" / "injection_pool_log.csv")
    pool = pool[pool["decision"] == "include"]
    pins = pd.read_csv(ROOT / "data" / "pinned_products.csv", usecols=["tic", "sector", "role"])
    secs = pins[pins["role"] == "pool"].groupby("tic")["sector"].apply(sorted)
    out = {}
    for label, pick in (("a_literal_full_pinned", lambda s: s),
                        ("b_injection_window", lambda s: [x for x in s if starts[x] < starts[s[0]] + 365.25][:10])):
        gpu, cpu, span = [], [], []
        for tic in pool["tic"]:
            s = pick(secs[tic])
            sp = E1.window_span(s, starts)
            work = len(s) * c["bins_per_sector"] * E1.n_bls_periods(sp)
            gpu.append(c["c_gpu_bls"] * work)
            cpu.append(c["c_bls"] * work)
            span.append(sp)
        gpu, cpu, span = map(np.asarray, (gpu, cpu, span))
        out[label] = {"stars": int(len(gpu)), "median_span_d": float(np.median(span)), "max_span_d": float(span.max()),
                      "gpu_single_t4_h": float(gpu.sum() / 3600), "gpu_quota_h_dual_t4": float(gpu.sum() / 7200),
                      "max_single_star_t4_h": float(gpu.max() / 3600),
                      "stars_over_12h_session_on_one_t4": int((gpu > SESSION_H * 3600).sum()),
                      "cpu_core_h": float(cpu.sum() / 3600), "cpu_wall_h_5_sessions": float(cpu.sum() / 3600 / 5)}
    out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (ROOT / "results" / "p08_cost.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
