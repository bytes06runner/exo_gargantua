"""False-alarm test FA1 (docs/decisions.md FA1; criterion pre-registered in docs/success_criteria.md: confident
resolution on scrambled data < 5 %). 'Confident' = the chosen alias's probability >= the version's abstain threshold
(v2 primary, v1 and the principled combiner also reported). A failed run counts as not confident (reported).
Output: results/false_alarm_report.json."""

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from b1_run import read_records  # noqa: E402
from b2_score import wilson  # noqa: E402
import resolver_models as RM  # noqa: E402

CRITERION = 0.05


def main():
    models = {v: RM.load(v) for v in ("v1", "v2")}
    rep = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "criterion": "confident resolution on scrambled light curves < 5 % (pre-registered)", "variants": {}}
    for variant in ("scrambled", "inverted"):
        recs = [r for p in sorted(glob.glob(str(ROOT / f"results/kaggle/fa/*/fa/fa_{variant}_*.jsonl.gz"))) for r in read_records(p)]
        ok = [r for r in recs if "error" not in r]
        res = {"n": len(recs), "errors": len(recs) - len(ok)}
        for name in ("v2", "v1", "principled"):
            k = 0
            for r in ok:
                g = [(x["r"], x["P"], x["x"]) for x in r["rows"]]
                if name == "principled":
                    c, thr = max(r["summary"]["p_alias"].values()), models["v2"][1]["principled"]
                else:
                    _, c = RM.decide(models[name][0], g)
                    thr = models[name][1]["learned"]
                k += c >= thr
            n = len(recs)
            res[name] = {"confident": int(k), "rate": k / n if n else None, "ci95": wilson(int(k), n),
                         "meets_criterion": bool(n and k / n < CRITERION) if variant == "scrambled" else None}
        res["seed_periods_median_d"] = float(np.median([r["seed"]["P_seed"] for r in ok])) if ok else None
        rep["variants"][variant] = res
    (ROOT / "results" / "false_alarm_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
