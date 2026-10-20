# B5: eclipsing-binary alias benchmark (pre-registered)

Committed on 2026-10-20 **before any B5 light curve is downloaded or any B5 search or resolver run**.
Owner instruction 2026-10-20. B5 was added after the B1 and B2 results were known (Gate 4). It is a new,
separately pre-registered experiment and will be reported as such.

## 1. Sample

Source catalogs: the frozen 2026-09-30 snapshots in `data/raw/snapshots.json`.
- TESS EB catalog (Prša et al. 2022): S-EB.
- SPOC 120-s light-curve listing: `mast_lc_products_20260930.csv.gz`.
- TESS orbit times: S-ORB.

Selection script: `scripts/b5_select.py`. It writes `data/b5_sample_log.csv` (every catalog row, with
decision and reason), `data/b5_pinned_products.csv` and `results/b5_selection.json`.

| Code | Rule |
|---|---|
| Y01 | The TIC has SPOC 120-s light-curve products in ≥ 2 sectors. 20-s products are not used. |
| Y02 | Catalog period finite and 0.5 ≤ P ≤ 30 d (the B2 range), and `bjd0` finite. |
| Y03 | Exactly one catalog signal on the TIC. Several signals: excluded, because the truth is ambiguous. |
| Y04 | The TIC is not a B2 host and not an injection-pool star. |
| Y05 | Early window E and later sectors L both non-empty, as in B2 H2. E is the pinned sectors starting before t_first + 365.25 d (the B2 rule). |
| S01 | If more than 1,500 EBs pass Y01–Y05, a random subset of **1,500** is drawn with `numpy.random.default_rng(20261021)` (cost cap fixed here). |

**Counts** (from `results/b5_selection.json`):

| | Count |
|---|---|
| Catalog rows | 4,584 |
| Pass Y01–Y05 | **3,282** |
| Included after S01 | **1,500** |
| Tier B | 1,368 |
| Tier B-amb | 132 |

Exclusions by rule:

| Rule | Excluded |
|---|---|
| Y02 | 677 |
| Y01 | 372 |
| Y04 | 191 |
| Y05 | 54 |
| Y03 | 8 |
| S01 (not in the subset) | 1,782 |

**Pinning.** The included EBs' pinned SPOC 120-s files are recorded with exact filenames. B5 runs load only
the **early-sector** files (3,443 files), the same input rule as B2. The later sectors only define Y05.

**Download-time rules:** X10–X13 of `docs/sample_definition.md` §2 apply unchanged (logged per sector).

**Eligibility H3, after download, as in B2:**
- The catalog ephemeris (`period`, `bjd0`) predicts ≥ 2 primary eclipses in E, with ≥ 50 % of their
  in-eclipse cadences present.
- Eclipse duration for this check: `prim_width_pf × period`, else `prim_width_2g × period`, else 0.1 d.

## 2. Truth

P_true = the Prša et al. (2022) catalog `period`.
- Tier B-amb (the pre-registered equal-depth proxy of `docs/sample_definition.md` §3.2) is reported
  separately.
- B-amb never enters the primary metric.

Correct iff |P / P_true − 1| < 0.001.

## 3. Methods

- **Seeds:** GPU BLS peak and TLS peak on the early-sector light curve, exactly as in B2 (`search.prepare`, the
  pre-registered BLS grid, TLS 1.32 with TIC stellar parameters). **TLS on all included EBs**; no subset
  is needed at this size.
- **Resolver:** frozen **v2** exactly (`resolver-frozen-v2`, 88c4838), no changes. It runs from the TLS seed
  and from the BLS seed. v1 is reported as secondary.
- **Primary "ours":** v2 seeded with the TLS peak, top-1 alias (as in B2, decision V2-e).

## 4. Success criterion (written before any data)

**v2 (TLS-seeded) beats raw TLS AND raw BLS on the primary sample, each by an exact paired McNemar test with
p < 0.01.**
- Primary sample: Tier B, H3 pass.
- A method that gives no period counts as wrong.
- The result is reported whatever it is.

## 5. Also reported (secondary)

- Accuracy with Wilson 95 % intervals.
- ECE and abstention at the v2 threshold.
- The truth-among-candidates ceiling per seed source.
- v2 seeded by BLS against raw BLS.
- v1.
- Tier B-amb separately.
- Breakdown by early-sector stratum and by period range (< 1, 1–3, 3–10, > 10 d).

## 6. Secondary: catalog-truth audit (added 2026-10-20, owner instruction, before any B5 download)

Reported separately as **"catalog-truth audit"**. It never changes the primary result: the success criterion
(§4) stays exactly as committed and is scored against the catalog period.

- **Targets.** Every scored B5 target (H3 pass; Tier B and Tier B-amb, reported separately) where v2's answer
  differs from the catalog period by more than 0.1 %. v2's answer is the primary run: TLS-seeded, MAP alias,
  whatever its confidence.
- **Data.** The target's pinned **later** sectors L (outside the early window). They are downloaded only for these
  targets, after the primary runs, with the same X10/X11 rules and the same compact cache format.
- **Model.** Fitted on the later light curve, detrended with wotan biweight, window max(0.75 d, 3 T), where T is
  v2's fitted duration. The same eclipsing-binary model is fitted separately for each candidate period
  P ∈ {P_catalog, P_v2}:
  - a primary trapezoid eclipse (free depth, duration, ingress, epoch; period refined within ±T·P/span);
  - plus a secondary trapezoid at a free phase (grid over all phases outside the primary, then refined;
    depth ≥ 0; same duration and ingress as the primary).
  - Both models have the same number of parameters.
- **Statistic.** Δχ² = χ²(P_catalog model) − χ²(P_v2 model), with errors from the robust white-noise scatter
  times the red-noise factor, as in the resolver.
- **Verdict per target:**
  - "later data support v2" if Δχ² > 10;
  - "later data support the catalog" if Δχ² < −10;
  - "inconclusive" otherwise;
  - "insufficient" if either period predicts fewer than 2 primary eclipses with ≥ 50 % coverage in L.
- **Reported:** counts and fractions of each verdict (Wilson 95 %), by tier and by period ratio P_v2 / P_catalog.
  Equal-depth twins can make P and P/2 indistinguishable; those cases are expected to be inconclusive and are
  reported as such.
- **Script:** `scripts/b5_audit.py`. Its fitting uses the resolver's model functions as tools only. It does not
  run the resolver's decision.
