"""Amendment A3: re-freeze ONLY the truth labels of data/sample_log.csv.

Recomputes truth_tier / truth_period / truth_source (and the H1-dependent b2_eligible /
b2_reason) with `assign_truth(..., amendment_a3=True)` from the same committed snapshots.
Asserts that every other column is byte-identical, writes the per-TOI change log to
data/truth_amendment_log.csv, and records old and new checksums in data/FROZEN.json.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exogargantua import sample as S  # noqa: E402

TRUTH_COLS = ["truth_tier", "truth_period", "truth_source", "b2_eligible", "b2_reason"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    log_path = ROOT / "data" / "sample_log.csv"
    old = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    snaps = json.loads((ROOT / "data" / "raw" / "snapshots.json").read_text())
    toi = S.normalise_toi(pd.read_csv(ROOT / snaps["S-TOI"]["file"], comment="#"))
    toi["key"] = toi["toi"].map(lambda x: f"{x:.2f}")
    toi = toi.set_index("key")
    ps = pd.read_csv(ROOT / snaps["S-PS"]["file"], low_memory=False)
    ps["tic"] = pd.to_numeric(ps["tic_id"].astype(str).str.replace("TIC", "").str.strip(), errors="coerce")
    ps_by_tic = {int(k): g for k, g in ps.dropna(subset=["tic"]).groupby("tic")}
    eb = pd.read_csv(ROOT / snaps["S-EB"]["file"])
    eb_by_tic = {int(k): g for k, g in eb.groupby("tess_id")}

    new = old.copy()
    changes = []
    for i, r in old[old["decision"] == "include"].iterrows():
        t = toi.loc[r["toi"]]
        tr = S.assign_truth(t, ps_by_tic.get(int(r["tic"]), pd.DataFrame()), eb_by_tic.get(int(r["tic"]), pd.DataFrame()),
                            amendment_a3=True)
        h1, h2 = tr.tier in ("A", "B"), r["sectors_later"] != ""
        vals = {"truth_tier": tr.tier, "truth_period": "" if not np.isfinite(tr.period) else repr(tr.period),
                "truth_source": tr.source or tr.note,
                "b2_eligible": "pending_H3" if (h1 and h2) else "no",
                "b2_reason": ("H3 evaluated at download" if (h1 and h2) else
                              ";".join(c for c, ok in (("H1: no Tier A/B truth", h1), ("H2: no later sectors", h2)) if not ok))}
        # keep the frozen text when periods agree to 1e-12 (CSV float parsing differs by 1 ulp across platforms)
        if vals["truth_period"] and r["truth_period"] and \
                abs(float(vals["truth_period"]) / float(r["truth_period"]) - 1) < 1e-12:
            vals["truth_period"] = r["truth_period"]
        if any(vals[c] != r[c] for c in TRUTH_COLS):
            changes.append({"toi": r["toi"], "tic": r["tic"], "disposition": t["disposition"],
                            **{f"old_{c}": r[c] for c in TRUTH_COLS}, **{f"new_{c}": vals[c] for c in TRUTH_COLS}})
            for c in TRUTH_COLS:
                new.at[i, c] = vals[c]

    other = [c for c in old.columns if c not in TRUTH_COLS]
    assert (new[other] == old[other]).all().all(), "A3 must not change any non-truth column"
    before = sha(log_path)
    new.to_csv(log_path, index=False)
    ch = pd.DataFrame(changes)
    ch.to_csv(ROOT / "data" / "truth_amendment_log.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    frozen = json.loads((ROOT / "data" / "FROZEN.json").read_text())
    frozen.setdefault("amendments", []).append({
        "id": "A3", "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": commit, "file": "data/sample_log.csv", "sha256_before": before, "sha256_after": sha(log_path),
        "changed_rows": int(len(ch)), "change_log": "data/truth_amendment_log.csv",
        "columns_changed": TRUTH_COLS,
    })
    frozen["files"]["data/sample_log.csv"] = sha(log_path)
    frozen["files"]["data/truth_amendment_log.csv"] = sha(ROOT / "data" / "truth_amendment_log.csv")
    (ROOT / "data" / "FROZEN.json").write_text(json.dumps(frozen, indent=1))
    print(ch[["toi", "disposition", "old_truth_tier", "new_truth_tier", "old_truth_period", "new_truth_source",
              "old_b2_eligible", "new_b2_eligible"]].to_string() if len(ch) else "no changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
