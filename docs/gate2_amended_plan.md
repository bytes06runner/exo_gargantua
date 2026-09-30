# Gate 2, amended plan: cost, download chunks, timeline (2026-09-30)

Numbers are copied from `results/cost_estimate_v2.json` (script `scripts/estimate_costs_v2.py`),
`results/kaggle/gate2b/v1_*/gate2b/gate2b_results.json` and `kaggle quota`. Amendments A1-A3 and
decision G1 are in `docs/decisions.md`.

## 1. Checks run on a Kaggle T4 session (`exog-gate2b-checks`, commit 9200933)

| Check | Result |
|---|---|
| A1: parallel BLS (4 at a time) vs serial, 20 smoke targets | **20 / 20 identical peak periods**; wall 2,765 s vs 6,068 s serial (bounded here by one 2,384 s target) |
| A2(iv): GPU BLS (PyTorch port of astropy `bls.c`) vs astropy | **Accepted.** 19 / 20 peaks at the same grid index, 1 (TOI 1190.01) one grid step away (allowed). One T4 is ~10-13x faster on large grids (TOI 1453.01: 178 s vs 2,384 s). GPU part took 538 s of the 2 h cap |
| Session limits (probe) | `/kaggle/working` 21.0 GB (saved output); scratch `/` and `/tmp` 1.1 TB free (shared overlay); 4 CPU cores; 33.7 GB RAM; 2x Tesla T4 (15 GB each) |
| Documented limits (kaggle.com/docs) | 12 h per GPU session; 20 GB saved output; datasets 200 GB each and 200 GB private total; a dataset is versioned from exactly one source; weekly GPU quota "30 hours or sometimes higher". **No global concurrent-session limit is documented**, and the API does not report one. |
| GPU quota now | 2.53 h used, **27.47 h remaining**, refresh 2026-10-03 00:00 UTC |

## 2. Cost of the amended design (GPU-quota hours; G1 makes every Kaggle hour a GPU hour)

| Work | BLS | TLS | Session hours |
|---|---|---|---|
| Download + compact 63,559 files (123 GB raw) | — | — | ~1.5 h download + ~2.5 h read/compact (~3 h with overlap) |
| B2 seed searches, 1,198 TOIs | 12.0 h on GPU (34.3 h if CPU-parallel) | 70.7 h | **~71 h** (GPU BLS runs concurrently with CPU TLS in the same session) |
| B1(ii) full searches, 2,000 injections | 20.6 h on GPU | 141.3 h | **~141 h** |
| B1(i) 30,000 wrong-seed resolver runs | none | none | 13.8 h detrending + resolver time (unknown until Phase 3) |
| **Total known** | | | **~229 h + resolver** |

TLS is now the bottleneck: it is CPU-bound (3.8 of 4 cores) and pre-registered, and under G1 its
~212 h are drawn from the GPU quota although it uses no GPU.

Compact cache size: 11.6 GB with float32 flux (as in the smoke cache), ~14.5 GB estimated with
float64 flux. I recommend float64: the A1/A2 identity checks were run on float64 inputs, and it
still fits one 20 GB session output.

## 3. Download chunk plan (from the measured limits)

- **4 chunks** of ~15,900 files (~31 GB raw, ~3.6 GB compact each), ~45-60 min per chunk session.
  One 12 h session and the 20 GB output would technically hold everything; chunks exist so a failure
  loses at most one ~1 h chunk.
- Inside a chunk: batches of 500 files (~1 GB raw on scratch) → compact in 4 worker processes →
  verify MD5 and `DATA_REL` against `data/pinned_products.csv` → delete raw → append batch entries to
  a manifest in `/kaggle/working` (checkpoint after every batch). A rerun skips files already in the
  manifest.
- Append: chunk k attaches chunk k-1's output (`kernel_sources`), copies it through, and adds its own
  files, so the final output (≤ ~14.5 GB < 20.9 GB) is the complete cache. Kaggle versions a dataset
  from exactly one source, and the API cannot create a dataset from a notebook output; so either
  (a) later jobs attach the final chunk output directly (fully automatic), or (b) you click
  "New Dataset" on that output once in the Kaggle UI to make it the private dataset. Your choice.
- Nothing is written to this Mac except manifests and checksums.

## 4. Wall-clock timeline at the current quota (30 GPU h/week)

| Week (quota window) | Work |
|---|---|
| now → 2026-10-02 (27.5 h left) | download + compact (~3 h); B2 seeds start (~24 h) |
| 2026-10-03 → 10-09 (30 h) | B2 seeds continue (~30 h) |
| 2026-10-10 → 10-16 (30 h) | B2 seeds finish (~17 h); B1(ii) searches start (~13 h) |
| 2026-10-17 → ~11-15 (~4.3 weeks) | B1(ii) searches (~128 h) |
| after Phase 3 | B1(i) resolver runs (~14 h + resolver) |

Seed searches finish around **2026-11-15** (~6.5 weeks from today, ~215 quota hours), before any resolver run.

## 5. Decisions for you (nothing launched)

1. Approve the download (4 chunks, ~3 GPU h) and float64 flux in the cache.
2. Choose dataset handling: (a) chained kernel outputs, or (b) one UI click to make a Dataset.
3. The timeline is set by TLS on CPU billed as GPU quota. Options, all yours:
   (a) keep G1 and accept ~6.5 weeks for the seed searches plus the resolver runs; (b) link a Colab Pro/Pro+ account for +15/+30 GPU h per week
   (Kaggle docs, experimental); (c) allow CPU sessions for TLS-only jobs, which have no weekly quota
   (this reverses G1, so only on your instruction).
4. Approve starting the B2 seed searches (and, separately, B1(ii)).
