"""Decision R1 before/after: the same 3,000 synthetic targets (same seed) resolved before
(results/kaggle/resolver_synth/, commit of the Phase 3 check) and after (results/kaggle/resolver_synth_v2/)
the refinement fix. Counts detectable signals (SNR >= 7.1) with an alias candidate within 0.1 % of the true
period, and the principled combiner's top-1 accuracy, in all splits and in the test split.
Output: results/resolver_refine_fix_report.json."""

from __future__ import annotations

import glob
import gzip
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETECTABLE_SNR = 7.1


def load(pattern):
    recs = {}
    for p in glob.glob(str(ROOT / pattern)):
        with gzip.open(p, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                recs[r["i"]] = r
    return recs


def stats(recs, ids):
    det = [i for i in ids if "error" not in recs[i] and recs[i]["truth"]["cls"] != "none" and recs[i]["truth"]["snr_true"] >= DETECTABLE_SNR]
    has = [i for i in det if any(x["correct"] for x in recs[i]["rows"])]
    top = [i for i in det if [x for x in recs[i]["rows"] if x["r"] == recs[i]["summary"]["r_map"]][0]["correct"]]
    return {"detectable": len(det), "truth_in_candidates": len(has), "missing": len(det) - len(has),
            "principled_top1_correct": len(top), "principled_top1_accuracy": len(top) / len(det) if det else None}, set(det) - set(has)


def main():
    old = load("results/kaggle/resolver_synth/*/synth/synth_*.jsonl.gz")
    new = load("results/kaggle/resolver_synth_v2/*/synth/synth_*.jsonl.gz")
    assert set(old) == set(new), "different target sets"
    for i in old:  # same synthetic data: identical truth
        assert old[i].get("truth") == new[i].get("truth") or "error" in old[i] or "error" in new[i]
    out = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "n_targets": len(old), "errors_before": sum("error" in r for r in old.values()), "errors_after": sum("error" in r for r in new.values())}
    for name, ids in (("all_splits", sorted(old)), ("test_split", sorted(i for i in old if i % 10 >= 8))):
        b, miss_b = stats(old, ids)
        a, miss_a = stats(new, ids)
        out[name] = {"before": b, "after": a, "fixed": len(miss_b - miss_a), "newly_missing": sorted(miss_a - miss_b),
                     "still_missing": sorted(miss_a & miss_b)}
    still = out["all_splits"]["still_missing"] + out["all_splits"]["newly_missing"]
    out["remaining_misses"] = [{"i": i, "cls": new[i]["truth"]["cls"], "snr": round(new[i]["truth"]["snr_true"], 1),
                                "n_sectors": new[i]["truth"]["n_sectors"], "P_true": round(new[i]["truth"]["P_true"], 4)} for i in still]
    (ROOT / "results" / "resolver_refine_fix_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "remaining_misses"}, indent=1))


if __name__ == "__main__":
    main()
