"""Score the POST HOC Tschudi (2026a) baseline on B5 (decision B5-tsch). Primary sample, 0.1 % tolerance; a star with no
Tschudi period (TLS failure) counts as wrong; exact paired McNemar against the v2 TLS-seeded resolver and against raw TLS.
Checks that the TLS period of the Tschudi run equals the stored rerun period.
Output: results/b5_tschudi_report.json."""

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402
from b2_score import wilson, mcnemar  # noqa: E402

TOL = 1e-3


def acc(x):
    x = np.asarray(x, bool)
    return {"n": int(x.size), "accuracy": float(x.mean()), "ci95": wilson(int(x.sum()), int(x.size))}


def main():
    rows = [r for p in sorted(glob.glob(str(ROOT / "results/kaggle/b5tsch/*/b5tsch/b5_tschudi_*.jsonl.gz"))) for r in read_records(p)]
    t = pd.DataFrame(rows).drop_duplicates("tic").set_index("tic")
    d = pd.read_csv(ROOT / "results/b5_per_target.csv").set_index("tic")
    assert set(d.index) == set(t.index), f"Tschudi run covers {len(t)} of {len(d)} B5 stars"
    d = d.join(t[["tls_period", "tschudi_period", "tschudi_resolved", "tschudi_harmonic_ratio"] + (["error"] if "error" in t else [])])
    both = d["tls_period"].notna() & d["P_tls"].notna()
    max_diff = float((d.loc[both, "tls_period"] / d.loc[both, "P_tls"] - 1).abs().max())
    assert max_diff < 1e-6, max_diff
    p = d[d["primary_sample"]]
    ok_t = (p["tschudi_period"].notna() & (np.abs(p["tschudi_period"] / p["P_catalog"] - 1) < TOL)).to_numpy()
    changed = p["tschudi_resolved"].fillna(False).astype(bool)
    rep = {"label": "POST HOC (decision B5-tsch)", "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "run_commits": sorted({json.loads(Path(f).read_text())["git_commit"] for f in glob.glob(str(ROOT / "results/kaggle/b5tsch/*/b5tsch/done_*.json"))}),
           "tschudi_commit": sorted({json.loads(Path(f).read_text())["tschudi_commit"] for f in glob.glob(str(ROOT / "results/kaggle/b5tsch/*/b5tsch/done_*.json"))}),
           "stars": int(len(t)), "max_frac_tls_period_difference_from_rerun": max_diff,
           "primary": {"n": int(len(p)), "no_period": int(p["tschudi_period"].isna().sum()), "tschudi": acc(ok_t), "raw_tls": acc(p["ok_tls"]), "resolver": acc(p["ok_resolver"]),
                       "period_changed_by_tschudi": int(changed.sum()),
                       "changed_to_correct": int((changed & ok_t & ~p["ok_tls"]).sum()), "changed_to_wrong": int((changed & ~ok_t & p["ok_tls"]).sum()),
                       "changed_by_ratio": {str(int(k)): int(v) for k, v in p.loc[changed, "tschudi_harmonic_ratio"].value_counts().items()},
                       "tschudi_vs_raw_tls": mcnemar(p["ok_tls"].to_numpy(), ok_t), "resolver_vs_tschudi": mcnemar(ok_t, p["ok_resolver"].to_numpy())}}
    (ROOT / "results/b5_tschudi_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
