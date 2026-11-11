"""Small counts the manuscript discloses: how often the Tschudi (2026a) correction changes the TLS period in B2, and how many
B4 stars had stellar parameters in the table the run used. Output: results/paper_disclosures.json."""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    s = pd.concat([pd.read_csv(p) for p in glob.glob(str(ROOT / "results/kaggle/b2*/**/seeds_tls_*.csv"), recursive=True)]).drop_duplicates("toi")
    ok = s["tls_period"].notna() & s["tschudi_period"].notna()
    differ = int((np.abs(s.loc[ok, "tschudi_period"] / s.loc[ok, "tls_period"] - 1) > 1e-6).sum())
    b4 = set(pd.read_csv(ROOT / "data/b4_pins.csv")["tic"])
    st = pd.read_csv(ROOT / "data/stellar_params.csv")
    have = set(st.loc[st["rho"].notna(), "tic"])  # F4 needs the density
    rep = {"tschudi_compared": int(ok.sum()), "tschudi_differs_from_tls": differ, "b4_stars": len(b4), "b4_with_stellar_density": len(b4 & have)}
    (ROOT / "results/paper_disclosures.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
