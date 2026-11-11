"""Kaggle job (CPU): fetch Gaia DR3 eclipsing-binary periods for the B5 stars (decision Indep-1). Scored locally by scripts/b5_gaia_check.py."""
import glob
import io
import subprocess
import time

import pandas as pd
import requests

SRC = "/tmp/exog"
snap_bin = glob.glob("/kaggle/input/**/exog_src.bin", recursive=True)
assert snap_bin, "code snapshot exog_src.bin not found under /kaggle/input"
subprocess.run(["mkdir", "-p", SRC], check=True)
subprocess.run(["tar", "-xzf", snap_bin[0], "-C", SRC], check=True)
st = pd.read_csv(f"{SRC}/data/b5_stellar_params.csv")
ids = st.loc[st["GAIA"].notna(), "GAIA"].astype("int64").tolist()
TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def query(chunk):
    adql = ("SELECT n.dr2_source_id, n.dr3_source_id, n.angular_distance, e.frequency, e.frequency_error, e.global_ranking "
            "FROM gaiadr3.dr2_neighbourhood AS n JOIN gaiadr3.vari_eclipsing_binary AS e ON n.dr3_source_id = e.source_id "
            f"WHERE n.dr2_source_id IN ({','.join(str(i) for i in chunk)})")
    for k in range(8):
        try:
            r = requests.post(TAP, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql}, timeout=600)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
            print("HTTP", r.status_code, flush=True)
        except requests.RequestException as e:
            print("error", e, flush=True)
        time.sleep(120 * (k + 1))
    raise SystemExit("Gaia TAP unavailable")


out = pd.concat([query(ids[i:i + 300]) for i in range(0, len(ids), 300)], ignore_index=True)
out.to_csv("/kaggle/working/b5_gaia_dr3_eb.csv", index=False)
print("rows", len(out), "of", len(ids), flush=True)
