#!/bin/bash
echo "Downloading light curves..."
python download_lcs.py

echo "Attempting to download CBVs (may fail due to MAST timeout)..."
python cbv_downloader.py || true

echo "Running benchmark..."
rm -f benchmark_results.jsonl benchmark_analysis_summary.json
PYTHONUNBUFFERED=1 python run_benchmark.py
