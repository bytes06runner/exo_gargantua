"""Pure functions implementing docs/sample_definition.md (no network, no file I/O).

Each rule carries the code used in the pre-registration so logs can be traced back to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Alias set R (sample_definition.md, Section 3)
ALIAS_RATIOS = (1 / 5, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 1.0, 3 / 2, 2.0, 3.0, 4.0, 5.0)
SECTOR_RANGE = (1, 106)  # S-LC scope
P_MIN, P_MAX = 0.5, 30.0  # X02
BTJD_OFFSET = 2457000.0
YEAR_DAYS = 365.25

# the 4-digit field after the TIC is the spacecraft configuration ID and varies by sector
_LC_NAME = re.compile(r"(tess\d{13}-s(\d{4})-(\d{16})-\d{4}-s_lc\.fits)")


# ---------------------------------------------------------------- MAST product lists (S-LC)
def parse_bulk_script(text: str) -> pd.DataFrame:
    """Parse a MAST `tesscurl_sector_<N>_lc.sh` script into (tic, sector, filename)."""
    rows = []
    for m in _LC_NAME.finditer(text):
        fname, sector, tic = m.group(1), int(m.group(2)), int(m.group(3))
        rows.append((tic, sector, fname))
    df = pd.DataFrame(rows, columns=["tic", "sector", "filename"]).drop_duplicates()
    return df


def product_url(filename: str) -> str:
    return f"https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/{filename}"


# ---------------------------------------------------------------- sector timing (S-ORB)
def sector_start_btjd(orbit_times: pd.DataFrame) -> dict[int, float]:
    """Start of each sector's first orbit, in BTJD (UTC treated as TDB; the ~69 s offset is irrelevant)."""
    df = orbit_times.copy()
    df = df[pd.to_numeric(df["Sector"], errors="coerce").notna()]  # drops the trailing "# Created ..." line
    df["Sector"] = df["Sector"].astype(int)
    df["start"] = pd.to_datetime(df["Start of Orbit"], utc=True)
    first = df.groupby("Sector")["start"].min()
    jd = first.map(lambda t: t.to_julian_date())
    return {int(s): float(v - BTJD_OFFSET) for s, v in jd.items()}


def early_later_split(sectors: Iterable[int], starts: dict[int, float]) -> tuple[list[int], list[int]]:
    """Section 4: early = sectors starting < t_first + 365.25 d; later = the rest."""
    secs = sorted(set(int(s) for s in sectors))
    t_first = starts[secs[0]]
    early = [s for s in secs if starts[s] < t_first + YEAR_DAYS]
    later = [s for s in secs if s not in early]
    return early, later


# ---------------------------------------------------------------- ephemeris matching (Section 3)
def alias_ratio_match(p_toi: float, p_ref: float, tol: float = 0.01) -> Optional[float]:
    """Return r in R with |P_toi/(r P_ref) - 1| < tol, else None (closest r if several)."""
    if not (np.isfinite(p_toi) and np.isfinite(p_ref)) or p_ref <= 0:
        return None
    best = None
    for r in ALIAS_RATIOS:
        err = abs(p_toi / (r * p_ref) - 1.0)
        if err < tol and (best is None or err < best[1]):
            best = (r, err)
    return None if best is None else best[0]


def epoch_coincides(t_toi: float, t_ref: float, p_ref: float, window: float) -> bool:
    """True if t_toi lies within `window` days of a transit predicted by (t_ref, p_ref)."""
    n = np.round((t_toi - t_ref) / p_ref)
    return bool(abs(t_toi - (t_ref + n * p_ref)) < window)


def propagated_epoch_sigma(t_toi: float, t_ref: float, p_ref: float, sig_t: float, sig_p: float) -> float:
    n = abs(np.round((t_toi - t_ref) / p_ref))
    return float(np.hypot(sig_t if np.isfinite(sig_t) else 0.0, n * (sig_p if np.isfinite(sig_p) else 0.0)))


@dataclass
class Truth:
    tier: str  # "A", "B", "B-amb", "C", or "" (none)
    period: float = np.nan
    source: str = ""
    note: str = ""


def eb_period_ambiguous(row: pd.Series) -> bool:
    """Section 3.2 proxy: missing period, or equal-depth secondary at phase 0.5."""
    if not np.isfinite(row.get("period", np.nan)):
        return True
    dp, ds = row.get("prim_depth_pf", np.nan), row.get("sec_depth_pf", np.nan)
    pp, ps = row.get("prim_pos_pf", np.nan), row.get("sec_pos_pf", np.nan)
    if not all(np.isfinite([dp, ds, pp, ps])) or dp <= 0:
        return False
    sep = abs(((ps - pp) % 1.0))
    return bool(abs(dp - ds) / dp < 0.10 and abs(sep - 0.5) < 0.02)


def assign_truth(toi: pd.Series, ps_rows: pd.DataFrame, eb_rows: pd.DataFrame) -> Truth:
    """Tier A (pscomppars) > Tier B (Prsa EB) > Tier C (catalog reference), per Section 3."""
    p_toi, t_toi = float(toi["period"]), float(toi["epoch_btjd"])
    dur_d = float(toi["duration_h"]) / 24.0 if np.isfinite(toi["duration_h"]) else 0.1

    # Tier A
    hits = []
    for _, r in ps_rows.iterrows():
        p_ps = r.get("pl_orbper", np.nan)
        t_ps = r.get("pl_tranmid", np.nan) - BTJD_OFFSET
        if alias_ratio_match(p_toi, p_ps) is None or not np.isfinite(t_ps):
            continue
        sig = propagated_epoch_sigma(t_toi, t_ps, p_ps, r.get("pl_tranmiderr1", np.nan), r.get("pl_orbpererr1", np.nan))
        if epoch_coincides(t_toi, t_ps, p_ps, max(0.5 * dur_d, 3 * sig)):
            hits.append(r)
    if len(hits) > 1:
        tier_a = Truth("", note="T-AMB: >1 pscomppars planet matches")
    elif len(hits) == 1:
        r = hits[0]
        rel = r.get("pl_orbpererr1", np.nan) / r["pl_orbper"]
        if np.isfinite(rel) and rel < 1e-3:
            return Truth("A", float(r["pl_orbper"]), f"pscomppars:{r['pl_name']}")
        tier_a = Truth("", note="T-PREC: pscomppars period error >= 1e-3 or missing")
    else:
        tier_a = Truth("")

    # Tier B
    b_hits = []
    for _, r in eb_rows.iterrows():
        p_eb, t_eb = r.get("period", np.nan), r.get("bjd0", np.nan)
        if not np.isfinite(t_eb):
            continue
        if not np.isfinite(p_eb):
            b_hits.append((r, True))  # EB on this TIC with no period: ambiguous by the proxy
            continue
        if alias_ratio_match(p_toi, p_eb) is None:
            continue
        sig = propagated_epoch_sigma(t_toi, t_eb, p_eb, r.get("bjd0_uncert", np.nan), r.get("period_uncert", np.nan))
        if epoch_coincides(t_toi, t_eb, p_eb, max(0.5 * dur_d, 3 * sig)):
            b_hits.append((r, eb_period_ambiguous(r)))
    if len(b_hits) == 1:
        r, amb = b_hits[0]
        src = f"tess-ebs:{int(r['tess_id'])}-{int(r['signal_id'])}"
        return Truth("B-amb" if amb else "B", float(r["period"]) if np.isfinite(r["period"]) else np.nan, src)
    if len(b_hits) > 1:
        return Truth("", note="T-AMB: >1 EB signal matches")

    if tier_a.note:
        return tier_a
    if toi["disposition"] in ("PC", "APC", "CP", "KP"):
        return Truth("C", p_toi, "exofop_toi_catalog")
    return Truth("")


# ---------------------------------------------------------------- transit coverage (H3)
def n_covered_transits(time: np.ndarray, period: float, t0: float, duration_d: float, min_frac: float = 0.5,
                       cadence_d: float = 2.0 / 1440.0) -> int:
    """Number of predicted transits with >= min_frac of their in-transit cadences present."""
    if time.size == 0:
        return 0
    n0 = int(np.ceil((time.min() - t0 - duration_d / 2) / period))
    n1 = int(np.floor((time.max() - t0 + duration_d / 2) / period))
    expected = max(1, int(round(duration_d / cadence_d)))
    count = 0
    for n in range(n0, n1 + 1):
        tc = t0 + n * period
        got = np.count_nonzero(np.abs(time - tc) < duration_d / 2)
        if got >= min_frac * expected:
            count += 1
    return count


# ---------------------------------------------------------------- TOI table normalisation
def normalise_toi(toi_raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "toi": toi_raw["TOI"].astype(float),
        "tic": toi_raw["TIC ID"].astype("int64"),
        "disposition": toi_raw["TFOPWG Disposition"].fillna(""),
        "period": pd.to_numeric(toi_raw["Period (days)"], errors="coerce"),
        "epoch_btjd": pd.to_numeric(toi_raw["Epoch (BJD)"], errors="coerce") - BTJD_OFFSET,
        "duration_h": pd.to_numeric(toi_raw["Duration (hours)"], errors="coerce"),
        "depth_ppm": pd.to_numeric(toi_raw["Depth (ppm)"], errors="coerce"),
    })
    return df


def duplicate_tois(df: pd.DataFrame) -> set[float]:
    """X03: TOIs on the same TIC with period within 0.1 % and epoch within 0.1 d; lower TOI number kept."""
    drop = set()
    for _, g in df.sort_values("toi").groupby("tic"):
        rows = g.to_dict("records")
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if (np.isfinite(a["period"]) and np.isfinite(b["period"]) and a["period"] > 0
                        and abs(b["period"] / a["period"] - 1) < 1e-3
                        and epoch_coincides(b["epoch_btjd"], a["epoch_btjd"], a["period"], 0.1)):
                    drop.add(b["toi"])
    return drop
