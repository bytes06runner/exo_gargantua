"""CLAUDE.md rule 2: summary numbers in the manuscript must be recomputable from the released per-target tables, and
every macro the manuscript uses must be generated."""

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"

pytestmark = pytest.mark.skipif(not (P / "generated_macros.tex").exists(), reason="paper assets not generated")


def macros():
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", (P / "generated_macros.tex").read_text()))


def num(s):
    return float(s.replace("{,}", ""))


def test_b2_accuracies_recomputable_from_per_target_table():
    m, d = macros(), pd.read_csv(P / "data" / "b2_per_target.csv")
    assert len(d) == num(m["bTwoN"])
    for col, key in (("ok_tls", "bTwoRawTls"), ("ok_bls", "bTwoRawBls"), ("ok_spoc1", "bTwoRawSpocOne"), ("ok_ours", "bTwoAcc")):
        assert f"{d[col].mean():.3f}" == m[key], (col, key)
    assert f"{100 * d['P_spoc1'].notna().mean():.1f}" == m["bTwoCovSpocOne"]
    assert (d["tier"] == "A").sum() == num(m["bTwoNA"]) and (d["tier"] == "B").sum() == num(m["bTwoNB"])


def test_audit_counts_recomputable_from_released_table():
    m, d = macros(), pd.read_csv(P / "data" / "b5_candidate_period_revisions.csv")
    assert d["status"].str.startswith("candidate revision").sum() == num(m["auditVtwo"])
    assert d["status"].str.startswith("catalog period").sum() == num(m["auditCat"])
    assert ((d["delta_chi2"] > 10) == d["status"].str.startswith("candidate revision")).all()


def test_every_manuscript_macro_is_defined():
    m = macros()
    used = set()
    for f in [P / "ms.tex", *sorted((P / "tables").glob("*.tex"))]:
        used |= set(re.findall(r"\\((?:bOne|bTwo|bFive|bFour|audit|fa|ctx|sel|abst|rc|auc|cmp|nogo|pool|cal|cv|temp|thr|vtwo)[A-Za-z]*)", f.read_text()))
    used -= {"facilities"}
    assert used, "no macros found"
    assert not (used - set(m)), sorted(used - set(m))
