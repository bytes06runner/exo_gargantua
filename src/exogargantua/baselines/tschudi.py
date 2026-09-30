"""Adapter for the Tschudi (2026a) rule-based harmonic correction (docs/baselines.md §3).

Code of record: gitlab.com/yohanntschudi/mdwarf-transit-survey @ ff62f3643ed12f36f32244165fae7242a0a47307
(MIT), `pipeline/03_search_tls.py::resolve_harmonic_alias`. It is applied to the SAME TLS result as
our TLS seed (one TLS run per TOI), with the configuration their pipeline passes to it
(TLSConfig defaults for oversampling_factor, grid_baseline, n_transits_min; tls_threads = cores;
harmonic SDE thresholds at their in-function defaults), found_periods = [] (single-signal
evaluation) and their `KNOWN_PLANETS` lookup emptied (truth-leakage guard; that lookup is not used
by the resolver, emptied anyway).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://gitlab.com/yohanntschudi/mdwarf-transit-survey.git"
COMMIT = "ff62f3643ed12f36f32244165fae7242a0a47307"


def fetch(dest: Path) -> Path:
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == COMMIT, head
    return dest


def load(repo: Path):
    pipe = repo / "pipeline"
    sys.path.insert(0, str(pipe))
    spec = importlib.util.spec_from_file_location("tschudi_search_tls", pipe / "03_search_tls.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.KNOWN_PLANETS.clear()
    return mod


def config(mod, n_threads: int) -> dict:
    cfg = mod.TLSConfig()
    return {"oversampling_factor": cfg.oversampling_factor, "grid_baseline": cfg.grid_baseline,
            "n_transits_min": cfg.n_transits_min, "tls_threads": n_threads}


def correct(mod, cfg: dict, tls_results, time_, flux, r_star, m_star) -> dict:
    t0 = time.time()
    res = mod.resolve_harmonic_alias(model=None, tls_results=tls_results, time=time_, flux=flux, flux_err=None,
                                     r_star=r_star, m_star=m_star, config=cfg, verbose=False, found_periods=[])
    return {"tschudi_period": float(res["resolved_period"]), "tschudi_resolved": bool(res["resolved"]),
            "tschudi_harmonic_ratio": int(res.get("harmonic_ratio", 1)), "tschudi_trigger": str(res.get("trigger", "")),
            "tschudi_s": time.time() - t0}
