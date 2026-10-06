"""Load the frozen combiners: v1 (models/resolver) and v2 (= v1 classifier, v2 temperature and thresholds from
models/resolver_v2; decision V2-a)."""

import copy
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(version: str):
    with open(ROOT / "models" / "resolver" / "learned.pkl", "rb") as fh:
        lc = pickle.load(fh)
    meta = json.loads((ROOT / "models" / "resolver" / "thresholds.json").read_text())
    if version == "v2":
        meta = json.loads((ROOT / "models" / "resolver_v2" / "calibration.json").read_text())
        lc = copy.deepcopy(lc)
        lc.temperature = meta["temperature"]
    return lc, meta["abstain_threshold"]
