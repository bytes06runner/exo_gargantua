"""Gate-0 root-cause check: run the LEGACY harmonic validator on synthetic
box transits whose true period is known, to show which code path moves a
correct seed period to a wrong alias. Prints one line per case; writes
results/legacy_audit/validator_regression.json.
"""
import json, subprocess, sys, contextlib, io
from pathlib import Path
import numpy as np
import lightkurve as lk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy"))
from exoplanet_pipeline.detection import resolve_fundamental_period  # noqa: E402

SEED = 20260930
CASES = [  # (true period [d], depth [ppm]) -- 13.0792 and 6.8660 are the Table-2 catalog periods
    (13.0792, 1500), (6.8660, 1500), (5.2000, 1500), (3.3000, 1500), (11.1453, 1500),
]

def synth(period, depth_ppm, rng):
    # two consecutive TESS-like sectors at 10-min cadence, 1-day mid-sector gaps
    t = np.arange(1325.0, 1379.0, 10 / 1440.0)
    gaps = ((t > 1338.5) & (t < 1339.5)) | ((t > 1352.0) & (t < 1353.0)) | ((t > 1365.5) & (t < 1366.5))
    t = t[~gaps]
    f = 1.0 + rng.normal(0, 400e-6, t.size)
    t0, dur = 1326.3, 0.15
    ph = (t - t0 + 0.5 * period) % period - 0.5 * period
    f[np.abs(ph) < dur / 2] -= depth_ppm * 1e-6
    return lk.LightCurve(time=t, flux=f)

rng = np.random.default_rng(SEED)
rows = []
for P, d in CASES:
    lc = synth(P, d, rng)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        pg, mult = resolve_fundamental_period(lc, P, max_period=20, min_period=0.5)
    out = float(pg.period_at_max_power.value) if pg is not None else None
    penalised = "Perigee Veto" in buf.getvalue()
    rows.append(dict(true_period=P, seed_period=P, depth_ppm=d, output_period=out,
                     multiplier=mult, ratio=(out / P if out else None), perigee_penalty_fired=penalised))
    print(f"true={P:8.4f} seed=true -> legacy output={out:8.4f}  ratio={out/P:6.3f}  mult={mult:.3f}  perigee_penalty={penalised}")

commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
outp = ROOT / "results" / "legacy_audit" / "validator_regression.json"
outp.parent.mkdir(parents=True, exist_ok=True)
outp.write_text(json.dumps({"seed": SEED, "git_commit": commit, "cases": rows}, indent=1))
