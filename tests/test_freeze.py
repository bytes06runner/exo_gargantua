import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freeze_sample as F  # noqa: E402


def test_gaia_id_is_exact_for_19_digit_ids():
    big = "4923860051276772608"  # float64 would turn this into ...772352
    assert F.gaia_id(big) == big
    assert F.gaia_id(4923860051276772608) == big
    assert F.gaia_id(np.int64(4923860051276772608)) == big
    assert F.gaia_id(None) == "" and F.gaia_id(float("nan")) == "" and F.gaia_id("") == ""
