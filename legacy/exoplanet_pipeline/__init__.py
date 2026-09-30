"""
TESS Exoplanet Transit-Detection Pipeline
==========================================

A modular pipeline for detecting and vetting exoplanet transit candidates
from TESS (Transiting Exoplanet Survey Satellite) light curve data.

Phases:
    1. Data ingestion, cleaning, and stitching (ingestion.py)
    2. Noise decomposition and filtering (denoise.py)
    3. BLS transit search with FAP-based confidence (detection.py)
    4. Multi-test vetting: odd/even, secondary eclipse, centroid (vetting.py)
    5. MCMC parameter estimation via batman + emcee (estimation.py)
    6. Validation harness against known targets (validate.py)
"""

__version__ = "1.0.0"
