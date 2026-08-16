# 🪐 Exo Gargantua: TESS Exoplanet ML Vetting & Detection Pipeline

<div align="center">

[![PyPI version](https://badge.fury.io/py/exo-gargantua.svg)](https://pypi.org/project/exo-gargantua/)
[![Python Tests](https://github.com/bytes06runner/exo_gargantua/actions/workflows/test.yml/badge.svg)](https://github.com/bytes06runner/exo_gargantua/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-v0.1.3%20Production-success.svg)]()

**An end-to-end Machine Learning and Bayesian astrophysical framework for automated detection, pixel-level Difference Image Analysis (DIA) vetting, and MCMC parameter estimation of transiting exoplanets from TESS observations.**

---

<img src="output/TIC_25155310_vetting_summary_panel.png" alt="Exoplanet Vetting Summary" width="100%">

</div>

---

## 📖 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features & Major Upgrades (v2.0)](#-key-features--major-upgrades-v20)
- [Astrophysical & ML Architecture](#-astrophysical--ml-architecture)
- [Vetting Engine & Difference Image Analysis (DIA)](#-vetting-engine--difference-image-analysis-dia)
- [Machine Learning Vetting Architecture](#-machine-learning-vetting-architecture)
- [Synthetic Injection-Recovery Framework](#-synthetic-injection-recovery-framework)
- [Benchmark Results: TIC 25155310 (WASP-126b)](#-benchmark-results-tic-25155310-wasp-126b)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Command-Line Interface (CLI) Usage](#-command-line-interface-cli-usage)
- [Testing & Continuous Integration (CI)](#-testing--continuous-integration-ci)
- [Authors & Citation](#-authors--citation)

---

## 🔭 Executive Overview

**Exo Gargantua** is a publishable, research-grade pipeline engineered to process millions of TESS cadences. It transitions exoplanet discovery from brittle heuristics into a unified statistical and Machine Learning engine.

Starting from raw SPOC (Science Processing Operations Center) calibrated Target Pixel Files (TPFs) and Light Curves downloaded programmatically from the Mikulski Archive for Space Telescopes (MAST), the pipeline performs:
1. **Multi-Sector Ingestion & Stitching** with crowding metric (`CROWDSAP`) corrections.
2. **Continuum-Referenced Transit-Masked Detrending** to preserve true transit depths.
3. **Decimated Box Least Squares (BLS)** periodic transit searches with bootstrap False Alarm Probability (FAP) estimation.
4. **Dynamic Stellar Limb-Darkening Estimation** via `ldtk` and the TESS Input Catalog (TIC v8.2).
5. **Pixel-Level Difference Image Analysis (DIA)** to detect background blended contaminants down to sub-pixel accuracy.
6. **Supervised ML Vetting** via a calibrated Random Forest classifier trained on spatial and photometric false-positive models.
7. **Bayesian MCMC Posterior Sampling** using `emcee` and `batman` with full Monte Carlo physical parameter propagation (Rp, a, Teq, Seff).

---

## 🚀 Key Features & Major Upgrades (v2.0)

| Capability | Pipeline v1 (Baseline) | Pipeline v2 (Current) |
| :--- | :--- | :--- |
| **Vetting Paradigm** | Hard heuristic thresholds | **Supervised Random Forest Classifier + Physical DIA Vetoes** |
| **Centroid Analysis** | Simple flux-weighted 1st moments | **True Difference Image Analysis (DIA) Out-of-Transit vs In-Transit** |
| **Limb Darkening** | Static quadratic constants (u1=0.3, u2=0.1) | **Dynamic q1, q2 profiles computed via `ldtk` from TIC (Teff, log(g), [Fe/H])** |
| **Validation Suite** | Manual targets / Error-prone harness | **Full Synthetic Transit Injection-Recovery & Multi-Modal Dataset Generator** |
| **Packaging & CI** | Script collection (`requirements.txt`) | **Modern `pyproject.toml` PEP 518/621 standard + GitHub Actions CI** |
| **Reporting & Telemetry** | Basic terminal printouts | **Structured JSON Reports, Corner Plots, and 4-Panel Publication Figures** |

---

## 🏗️ Astrophysical & ML Architecture

```mermaid
flowchart TD
    A[MAST API / SPOC Lightcurves & TPFs] --> B[Phase 1: Ingestion, Stitching & Crowding Correction]
    B --> C[Phase 2: Continuum-Referenced Transit-Masked Detrending]
    C --> D[Phase 3: Fast Decimated BLS Search & Bootstrap FAP]
    
    D --> E[Phase 3b: Physical Vetting Engine]
    E --> F[Odd/Even Transit Depth Delta Check]
    E --> G[Rolling-Mean Red Noise Secondary Eclipse Test]
    E --> H[Difference Image Analysis DIA Centroid Shift]
    
    F --> I[Feature Extraction Vector]
    G --> I
    H --> I
    D --> I
    
    I --> J[Phase 3c: Calibrated ML Vetter Random Forest]
    J -->|Disposition & Planet Probability| K{Pass Physical & ML Vetting?}
    
    K -->|Yes / Validated Candidate| L[Phase 4: Dynamic Limb Darkening LDTk + MCMC emcee]
    K -->|No / Blended False Positive| L
    
    L --> M[Phase 4b: Monte Carlo Physical Parameter Derivation]
    M --> N[Phase 4c: Publication Figures & JSON Summary Report]
```

---

## 🔬 Vetting Engine & Difference Image Analysis (DIA)

The vetting suite isolates genuine planetary transits from astrophysical false positives (Eclipsing Binaries, Grazing Binaries, and Background Eclipsing Binaries):

### 1. Difference Image Analysis (DIA) Centroid Vetting
To verify that the transit signal originates from the target star rather than a nearby background star within the large TESS pixel aperture (~21 arcsec / pixel):
- Construct the reference out-of-transit average frame `I_out` and in-transit average frame `I_in`.
- Compute the differential flux image:
  ```
  ΔI = I_out - I_in
  ```
- Calculate the flux-weighted centroid of the differential dip `(x_res, y_res)` and the reference star `(x_ref, y_ref)`.
- Calculate the Euclidean centroid shift:
  ```
  Δr = √((x_res - x_ref)² + (y_res - y_ref)²)
  ```
- **Veto Threshold**: A shift `Δr ≥ 0.333 pixels` (~7 arcsec) triggers an immediate `FAILED_CENTROID_DIA` flag.

### 2. Odd/Even Depth Consistency
Calculates the absolute depth difference `Δδ = |δ_odd - δ_even|`. Large differences (`Δδ > 0.005`) indicate alternating primary and secondary eclipses characteristic of eclipsing binary systems.

### 3. Integrated Red-Noise Secondary Eclipse Check
Computes the rolling-mean standard deviation in out-of-transit chunks matched to the transit duration, scoring secondary eclipse significance:
```
σ_sec = δ_sec / σ_rolling
```
Any secondary eclipse with `|σ_sec| > 3.0σ` flags the candidate as an occulting stellar companion.

---

## 🤖 Machine Learning Vetting Architecture

Rather than relying purely on heuristic threshold cuts, `exoplanet_pipeline/ml_vetting.py` implements a calibrated **Random Forest Classifier** trained on multi-modal feature vectors:

```
Feature Vector = [ BLS_Power, Odd_Even_Depth_Diff, Secondary_Eclipse_Sigma, Centroid_Shift, SNR, Centroid_Failed_Flag ]
```

```
Feature Importances in Trained Model:
  centroid_shift           : 29.65%
  depth_diff               : 27.82%
  centroid_fail (DIA Veto) : 22.54%
  secondary_eclipse_sigma  : 17.76%
  snr                      :  1.31%
  bls_power                :  0.92%
```

### Spatial Veto & Consistency Guarantee
Because 1D transit injection cannot synthesize 2D pixel-level centroid shifts into Target Pixel Files, the training pipeline couples synthetic transit datasets with multi-modal physical false-positive generators (Background Eclipsing Binaries with centroid offsets 0.35 to 3.0 pixels, and EBs with high secondary eclipses). 

Furthermore, `MLVetter.predict()` enforces **hard physical gating**: a candidate failing the spatial DIA test (`Δr ≥ 0.333 pixels`) has its planet probability suppressed to `< 0.5%`, preventing high 1D SNR from overriding spatial contamination.

---

## 🧪 Synthetic Injection-Recovery Framework

The pipeline includes `exoplanet_pipeline/validate_injection.py` to evaluate pipeline recovery completeness across the parameter space (Period: 1–15 days, Rp/Rs: 0.01–0.15):
- Ingests real quiet TESS light curves.
- Injects non-linear Mandel & Agol (2002) transit models using `batman`.
- Runs blind detrending and BLS detection to evaluate recovery fraction and signal preservation.

---

## 📊 Benchmark Validation & Performance

### 1. Confirmed Exoplanet Benchmark: TIC 25155310 (WASP-126b)

The complete end-to-end pipeline was validated against published discovery literature (Maxted et al. 2016) on benchmark hot Jupiter **TIC 25155310** (WASP-126b) across 42 stitched TESS sectors (955,000+ cadences).

<div align="center">
  <img src="output/TIC_25155310_corner.png" alt="MCMC Corner Plot" width="75%">
</div>

#### Extracted Parameters (`output/TIC_25155310_summary.json`)

| Parameter | Literature (Maxted et al. 2016) | Pipeline MCMC Posterior | Literature Agreement | Physical Unit |
| :--- | :--- | :--- | :--- | :--- |
| **Orbital Period ($P$)** | $3.288790 \pm 0.000008$ | **`3.288787`** | $\pm 0.0000004$ ($<0.3\,$s diff) | days |
| **Transit Epoch ($t_0$)** | $1327.5204 \pm 0.0004$ | **`1327.520756`** | $\pm 0.0002$ ($<30\,$s diff) | BTJD |
| **Radius Ratio ($R_p / R_\star$)** | $0.0780 \pm 0.0009$ | **`0.077176`** | $\pm 0.00018$ ($1.0\%$ relative) | — |
| **Semi-Major Axis ($a / R_\star$)** | $7.90 \pm 0.15$ | **`7.917236`** | $\pm 0.04$ ($<0.3\%$ relative) | — |
| **Orbital Inclination ($i$)** | $87.9^\circ \text{ -- } 89.5^\circ$ | **`89.47`** | $\pm 0.4^\circ$ | degrees |
| **Planetary Radius ($R_p$)** | $10.65 \pm 0.34$ ($0.95\,R_{\text{Jup}}$) | **`10.70`** | $\pm 0.54$ ($0.953\,R_{\text{Jup}}$, **$<0.5\%$ error**) | $R_\oplus$ |
| **Semi-Major Axis ($a$)** | $0.0449 \pm 0.0009$ | **`0.0449`** | $\pm 0.0007$ (exact match) | AU |
| **Equilibrium Temp ($T_{\text{eq}}$)** | $1360 \pm 20$ | **`1359`** | $\pm 43$ (exact match) | K |
| **Centroid Shift ($\Delta r$)** | $<0.01$ | **`0.0043`** | `passed: true` (threshold $< 0.333$) | pixels |
| **ML Vetting Score** | Validated Planet | **`100.00%`** | **`CANDIDATE`** | Disposition |

---

### 2. Certified False Positive Benchmark: TIC 279999655 (TOI 959.01)

To demonstrate rigorous rejection of astrophysical false positives, the pipeline was benchmarked against **TIC 279999655 (TOI 959.01)**, an eclipsing binary certified as a False Positive (`tfopwg_disp = 'FP'`) in the NASA Exoplanet Archive TOI Catalog:

- **Orbital Period ($P$)**: `1.540972 ± 0.000048` days
- **Transit / Eclipse Depth ($\delta$)**: `4.95%` ($R_p/R_\star = 0.2226$)
- **Derived Companion Radius ($R_p$)**: `59.58 ± 7.4` $R_\oplus$ ($5.31\,R_{\text{Jup}}$)
- **Vetting Veto Triggered**:
  ```text
  [VETTING VETO] Derived physical radius Rp = 59.6 R_earth exceeds planetary limit (25 R_earth).
  -> Reclassified disposition: FALSE_POSITIVE (Eclipsing Binary / Stellar Companion)
  ```
- **Final Disposition**: **`FALSE_POSITIVE`** (`0.00%` planet probability) | Summary: [`output/TIC_279999655_summary.json`](output/TIC_279999655_summary.json)

> [!NOTE]
> **Known Vetting Limitations & Astrophysical Caveats:**
> - **Cool / Low-Mass Grazing Companions**: Single-bandpass photometric vetting cannot detect cool/low-mass grazing companions via odd-even or secondary eclipse when the correct period is found; radius ceiling is the only test sensitive to this failure mode.
> - **Unconstrained Grazing Geometries**: Candidates with near-grazing impact parameters ($b \gtrsim 0.95$) or BLS/MCMC depth disagreements exceeding $1.5\times$ where the $1\sigma$ posterior spans the $25.0\,R_\oplus$ boundary are explicitly classified as **`AMBIGUOUS`** requiring high-resolution spectroscopic follow-up.

---

## 💻 Installation & Environment Setup

### Prerequisites
- Python >= 3.9, <= 3.11
- Git

### Quick Install via PyPI
```bash
pip install exo-gargantua
```

### Development Installation from Source
```bash
# 1. Clone the repository
git clone https://github.com/bytes06runner/exo_gargantua.git
cd exo_gargantua

# 2. Create and activate a virtual environment
python3 -m venv exo-env
source exo-env/bin/activate

# 3. Install in editable mode with development & testing dependencies
pip install -e ".[dev]"
```

---

## 🛠️ Command-Line Interface (CLI) Usage

### 1. Run the Full End-to-End Pipeline
Process a TESS target through ingestion, detrending, BLS search, vetting, DIA, and MCMC parameter estimation:
```bash
python run_pipeline.py "TIC 25155310" --report
```

**Options:**
- `--report`: Generates summary JSON in `output/` and publication 4-panel diagnostic plots.
- `--no-mcmc`: Skips MCMC posterior estimation for fast screening.
- `--no-centroid`: Skips Target Pixel File (TPF) downloads.

### 2. Generate Synthetic Injection-Recovery Datasets
```bash
python exoplanet_pipeline/validate_injection.py
```

### 3. Train the ML Vetting Model
```bash
python exoplanet_pipeline/ml_vetting.py
```

### 4. Run Astrophysical Ground-Truth Validation Harness
```bash
python run_pipeline.py --validate
```

---

## 🧪 Testing & Continuous Integration (CI)

The repository includes a comprehensive `pytest` test suite configured with GitHub Actions CI on Ubuntu runners with Python 3.10 and 3.11:

```bash
# Execute local unit tests
pytest tests/ -v
```

**Test Coverage Includes:**
- Kepler's Third Law astronomical scaling and error boundaries (`tests/test_math.py`).
- Rolling-mean red-noise boxcar standard deviations (`tests/test_math.py`).
- Odd/Even transit depth deltas (`tests/test_math.py`).
- ML Vetter model training, prediction, and spatial DIA centroid veto consistency (`tests/test_ml_vetting.py`).

---

## 👤 Authors & Citation

**Srijeet Prasad Banerjee** ([@bytes06runner](https://github.com/bytes06runner))  
*Contact:* [cybrobasics@gmail.com](mailto:cybrobasics@gmail.com)  

### License
This project is open-source and licensed under the [MIT License](LICENSE).
