"""Load the frozen combiners: v1 (models/resolver) and v2 (= v1 classifier and temperature + the isotonic top-
probability map and thresholds in models/resolver_v2; decision V2-f)."""

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(version: str):
    with open(ROOT / "models" / "resolver" / "learned.pkl", "rb") as fh:
        lc = pickle.load(fh)
    if version == "v1":
        return lc, json.loads((ROOT / "models" / "resolver" / "thresholds.json").read_text())["abstain_threshold"]
    from exogargantua.calibration_v2 import IsotonicTopCalibrated
    with open(ROOT / "models" / "resolver_v2" / "isotonic.pkl", "rb") as fh:
        iso = pickle.load(fh)
    meta = json.loads((ROOT / "models" / "resolver_v2" / "calibration.json").read_text())
    return IsotonicTopCalibrated(lc, iso), meta["abstain_threshold"]
