"""Reading and writing P08 screen output rows.

scripts/p08_screen.py up to commit d6a21b9 appended one row at a time and took the CSV header from whichever
row came first; when that was a P07-excluded star the header has 3 fields while searched stars have the full
set. Rows are still written in a fixed key order, so they are recovered by field count.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

P07_COLS = ["tic", "p07", "p07_reason"]
WINDOW_COLS = ["window_sectors", "span_d", "n_binned"]
RAW_COLS = ["n_periods", "peak_period", "power_max", "power_mean", "power_std", "sde", "p08"]
V1_COLS = P07_COLS + WINDOW_COLS + RAW_COLS + ["search_s", "engine"]
A6_COLS = ["n_tls_periods", "sde_a6", "peak_period_a6", "p08_a6", "sde_a6_cellmax"]
DENSE_COLS = ["sde_dense_tlspts", "sde_dense_scaled"]
A6_FULL_COLS = P07_COLS + WINDOW_COLS + RAW_COLS + A6_COLS + DENSE_COLS + ["search_s", "a6_s", "engine"]
A6_NODENSE_COLS = P07_COLS + WINDOW_COLS + RAW_COLS + A6_COLS + ["search_s", "a6_s", "engine"]
SCHEMAS = {len(c): c for c in (P07_COLS, V1_COLS, A6_FULL_COLS, A6_NODENSE_COLS)}


def write_row(path: Path, row: dict, cols: list[str]) -> None:
    """Append one row in a fixed column order; the header is always the full column list."""
    pd.DataFrame([row]).reindex(columns=cols).to_csv(path, mode="a", header=not path.exists(), index=False)


def read_rows(path) -> pd.DataFrame:
    with open(path, newline="") as fh:
        lines = list(csv.reader(fh))
    header, body = lines[0], lines[1:]
    full_header = len(header) > len(P07_COLS)
    recs = []
    for r in body:
        if full_header and len(r) <= len(header):
            cols = header
        else:
            cols = SCHEMAS.get(len(r))
            if cols is None:
                raise ValueError(f"{path}: row with {len(r)} fields matches no known schema")
        recs.append(dict(zip(cols, r)))
    df = pd.DataFrame(recs).replace("", None)
    for c in df.columns:
        if c not in ("p07", "p07_reason", "window_sectors", "p08", "p08_a6", "engine"):
            df[c] = pd.to_numeric(df[c])
    df["tic"] = df["tic"].astype(int)
    return df
