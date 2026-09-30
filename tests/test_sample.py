import numpy as np
import pandas as pd

from exogargantua import sample as S


def test_parse_bulk_script_extracts_tic_sector_filename():
    text = ("#!/bin/sh\ncurl -C - -L -o tess2018206045859-s0001-0000000278660115-0120-s_lc.fits "
            "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/"
            "tess2018206045859-s0001-0000000278660115-0120-s_lc.fits\n")
    df = S.parse_bulk_script(text)
    assert df.to_dict("records") == [{"tic": 278660115, "sector": 1,
                                      "filename": "tess2018206045859-s0001-0000000278660115-0120-s_lc.fits"}]


def test_parse_bulk_script_any_configuration_id():
    df = S.parse_bulk_script("tess2018234235059-s0002-0000000270622780-0121-s_lc.fits")
    assert df["sector"].tolist() == [2] and df["tic"].tolist() == [270622780]


def test_fast_cadence_files_are_ignored():
    text = "tess2020186164531-s0027-0000000038846515-0120-a_fast-lc.fits"
    assert S.parse_bulk_script(text).empty


def test_early_later_split_uses_first_year():
    starts = {1: 1325.0, 13: 1653.0, 14: 1683.0, 27: 2036.0, 40: 2390.0}
    early, later = S.early_later_split([27, 1, 14, 13, 40], starts)
    # sector 14 starts 358 d after sector 1 (< 365.25 d), so it is early; 27 starts 711 d after
    assert early == [1, 13, 14] and later == [27, 40]


def test_early_later_split_boundary():
    starts = {1: 0.0, 2: 365.0, 3: 365.3}
    early, later = S.early_later_split([1, 2, 3], starts)
    assert early == [1, 2] and later == [3]


def test_alias_ratio_match():
    assert S.alias_ratio_match(10.0, 10.0) == 1.0
    assert S.alias_ratio_match(5.0, 10.0) == 0.5
    assert S.alias_ratio_match(30.0, 10.0) == 3.0
    assert S.alias_ratio_match(10.0 * 2.9, 10.0) is None  # non-integer factor, as in the v1 errors


def test_epoch_coincides():
    assert S.epoch_coincides(1000.0 + 7 * 3.0 + 0.01, 1000.0, 3.0, 0.05)
    assert not S.epoch_coincides(1000.0 + 7 * 3.0 + 0.5, 1000.0, 3.0, 0.05)


def test_eb_ambiguity_proxy():
    twin = pd.Series({"period": 2.0, "prim_depth_pf": 0.10, "sec_depth_pf": 0.095, "prim_pos_pf": 0.0, "sec_pos_pf": 0.505})
    unequal = pd.Series({"period": 2.0, "prim_depth_pf": 0.10, "sec_depth_pf": 0.03, "prim_pos_pf": 0.0, "sec_pos_pf": 0.5})
    eccentric = pd.Series({"period": 2.0, "prim_depth_pf": 0.10, "sec_depth_pf": 0.098, "prim_pos_pf": 0.0, "sec_pos_pf": 0.42})
    nop = pd.Series({"period": np.nan})
    assert S.eb_period_ambiguous(twin) and S.eb_period_ambiguous(nop)
    assert not S.eb_period_ambiguous(unequal) and not S.eb_period_ambiguous(eccentric)


def _toi(period, epoch, disp="PC"):
    return pd.Series({"period": period, "epoch_btjd": epoch, "duration_h": 3.0, "disposition": disp})


def test_truth_tier_a_allows_alias_catalog_period():
    ps = pd.DataFrame([{"pl_name": "X b", "pl_orbper": 10.0, "pl_orbpererr1": 1e-5,
                        "pl_tranmid": 2459000.0, "pl_tranmiderr1": 1e-3}])
    t = S.assign_truth(_toi(5.0, 2000.0 + 10.0 * 3), ps, pd.DataFrame())  # TOI at P/2, epoch on a real transit
    assert t.tier == "A" and t.period == 10.0


def test_truth_ambiguous_when_two_planets_match():
    ps = pd.DataFrame([{"pl_name": n, "pl_orbper": 10.0, "pl_orbpererr1": 1e-5, "pl_tranmid": 2459000.0,
                        "pl_tranmiderr1": 1e-3} for n in ("X b", "X c")])
    t = S.assign_truth(_toi(10.0, 2000.0), ps, pd.DataFrame())
    assert t.tier == "" and t.note.startswith("T-AMB")


def test_truth_tier_c_only_for_planet_dispositions():
    assert S.assign_truth(_toi(3.0, 2000.0, "PC"), pd.DataFrame(), pd.DataFrame()).tier == "C"
    assert S.assign_truth(_toi(3.0, 2000.0, "FP"), pd.DataFrame(), pd.DataFrame()).tier == ""


def test_n_covered_transits_counts_gapped_data():
    t = np.arange(-1, 21, 2 / 1440)
    t = t[(t < 8) | (t > 12)]  # the gap swallows the transit at t = 10
    assert S.n_covered_transits(t, 5.0, 0.0, 0.1) == 4  # 0, 5, 15, 20 fully covered
    t_half = t[~((t > 14.99) & (t < 15.05))]  # keep < 50 % of the t = 15 transit
    assert S.n_covered_transits(t_half, 5.0, 0.0, 0.1) == 3


def test_amendment_a3_confirmed_planet_never_gets_eb_truth():
    eb = pd.DataFrame([{"tess_id": 1, "signal_id": 1, "period": 5.179905, "bjd0": 2000.0, "bjd0_uncert": 1e-4,
                        "period_uncert": 1e-5, "prim_depth_pf": 0.001, "sec_depth_pf": np.nan,
                        "prim_pos_pf": 0.0, "sec_pos_pf": np.nan}])
    cp = _toi(5.18, 2000.0, "CP")
    t = S.assign_truth(cp, pd.DataFrame(), eb)
    assert t.tier == "C" and not t.source.startswith("tess-ebs")  # catalog reference only, never EB truth
    assert S.assign_truth(cp, pd.DataFrame(), eb, amendment_a3=False).tier == "B"
    assert S.assign_truth(_toi(5.18, 2000.0, "FP"), pd.DataFrame(), eb).tier == "B"
