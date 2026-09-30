"""CLAUDE.md rule 3: the frozen sample may not change after the freeze."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_files_unchanged():
    rec = json.loads((ROOT / "data" / "FROZEN.json").read_text())
    for path, sha in rec["files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, f"{path} changed after the freeze"


def test_every_sample_log_row_has_decision_reason_and_commit():
    import pandas as pd
    log = pd.read_csv(ROOT / "data" / "sample_log.csv", dtype=str)
    assert log["decision"].isin(["include", "exclude"]).all()
    assert log.loc[log["decision"] == "exclude", "reason"].str.match(r"X0[1-3]:").all()
    assert log["git_commit"].str.fullmatch(r"[0-9a-f]{40}").all()
    inc = log[log["decision"] == "include"]
    assert (inc["sectors_all"].str.count(";") >= 1).all()  # >= 2 pinned sectors
    assert inc["product_files"].notna().all()
