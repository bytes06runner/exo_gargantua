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
    m, a = macros(), pd.read_csv(P / "data" / "eb_candidate_period_revisions.csv")
    d, e = a[a["sample"] == "B5"], a[a["sample"] == "B6"]
    assert d["status"].str.startswith("candidate revision").sum() == num(m["auditVtwo"])
    assert d["status"].str.startswith("catalog period").sum() == num(m["auditCat"])
    assert e["status"].str.startswith("candidate revision").sum() == num(m["bSixAuditRes"])
    assert e["status"].str.startswith("catalog period").sum() == num(m["bSixAuditCat"])
    d = a
    assert ((d["delta_chi2"] > 10) == d["status"].str.startswith("candidate revision")).all()


def test_every_manuscript_macro_is_defined():
    m = macros()
    used = set()
    for f in [P / "ms.tex", *sorted((P / "tables").glob("*.tex"))]:
        used |= set(re.findall(r"\\((?:bOne|bTwo|bFive|bFour|audit|fa|ctx|sel|abst|rc|auc|cmp|nogo|pool|cal|cv|temp|thr|vtwo|lc|mast|raw|flag|lit|rev|occ|tol|abl|bSix|pool)[A-Za-z]*)", f.read_text()))
    used -= {"facilities"}
    assert used, "no macros found"
    assert not (used - set(m)), sorted(used - set(m))


TESS_ACK = ("This paper made use of data collected by the TESS mission and are publicly available from the Mikulski Archive for Space "
            "Telescopes (MAST) operated by the Space Telescope Science Institute (STScI). Funding for the TESS mission is provided by "
            "NASA's Science Mission Directorate. We acknowledge the use of public TESS data from pipelines at the TESS Science Office and "
            "at the TESS Science Processing Operations Center. Resources supporting this work were provided by the NASA High-End Computing "
            "(HEC) Program through the NASA Advanced Supercomputing (NAS) Division at Ames Research Center for the production of the SPOC "
            "data products.")
FORBIDDEN = ("fourier", "ttv smearing", "black-box", "black box", "heavy institutional", "brute force", "democratiz", "perfectly", "definitively",
             "pristine", "periodogram", "runge", "first principles", "state of the art", "state-of-the-art", "outperform", "convexity", "thermal")


def test_reviewer_wording_and_required_text():
    s = (P / "ms.tex").read_text()
    low = s.lower()
    assert not [w for w in FORBIDDEN if w in low], [w for w in FORBIDDEN if w in low]
    assert " ".join(TESS_ACK.split()) in " ".join(s.split()), "the standard TESS acknowledgement must appear verbatim"
    for needle in ("10.17909/t9-nmc8-f686", "\\dataset[", "10.5281/zenodo.", "AstroNet", "TESS Exoplanet Classifier", "transit-timing variations are out of scope",
                   "centroid", "not a false-alarm filter"):
        assert needle in s, needle


def test_every_table_and_figure_is_called_out():
    s = (P / "ms.tex").read_text() + "".join(f.read_text() for f in sorted((P / "tables").glob("*.tex"))) + (P / "amendments_table.tex").read_text()
    labels = set(re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", s))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
    assert labels and not (labels - refs), sorted(labels - refs)
    inputs = set(re.findall(r"\\input\{tables/([a-z0-9_]+)\}", s))
    assert inputs == {f.stem for f in (P / "tables").glob("*.tex")}, "every generated table is included, and nothing else"


def test_released_tables_carry_sectors_and_snr():
    for name in ("b2_per_target.csv", "b5_per_target.csv", "b6_per_target.csv", "eb_candidate_period_revisions.csv"):
        cols = set(pd.read_csv(P / "data" / name, nrows=1).columns)
        assert {"sectors_early", "snr_seed"} <= cols, (name, sorted(cols))


def test_abstract_within_aas_limit():
    s = (P / "ms.tex").read_text()
    a = s[s.index("\\begin{abstract}") + len("\\begin{abstract}"):s.index("\\end{abstract}")]
    assert len(a.split()) <= 250, len(a.split())
