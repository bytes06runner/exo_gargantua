# B4: runtime on a sector-scale sample (protocol, 2026-10-20, owner-approved; no pass/fail criterion)

- **Sample.** All 19,999 SPOC 120-s light curves of TESS Sector 49 in the frozen listing
  (`data/b4_pins.csv`; Sector 49 is one of the sectors with ~20,000 light curves). They are downloaded and
  compacted exactly as the other caches (X10/X11 logged; `scripts/build_cache.py`).
- **Pipeline per light curve:**
  1. load;
  2. prepare (wotan biweight 0.75 d, 10-min bins);
  3. GPU BLS on the pre-registered grid; the seed is the BLS peak;
  4. the frozen v2 resolver from that seed.
  - Stages 1–3 run on a Kaggle GPU session, one process per Tesla T4.
  - Stage 4 runs on Kaggle CPU sessions, 4 worker processes with 1 BLAS thread each.
  - Exact hardware (CPU model, core count, GPU name) is recorded in every output.
- **Reported:** median and p95 time per light curve for each stage and for the total
  (load + prepare + BLS + resolver), the number of light curves processed, and failures.
- **TLS** is not run on this sample (owner decision). Its per-light-curve time is quoted from the measured B2
  and B1(ii) TLS runs, with their hardware.
- **No comparison** with SPOC/HECC compute is made.
- **Scripts:** `scripts/b4_run.py` (stages); summary `scripts/b4_summary.py`.
