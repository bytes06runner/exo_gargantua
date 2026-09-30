#!/usr/bin/env python3
"""
run_pipeline.py — Thin CLI Entry Point
=======================================

Usage:
    python run_pipeline.py "TIC 25155310"
    python run_pipeline.py "TIC 25155310" --no-mcmc --no-centroid
    python run_pipeline.py --validate
"""

import sys
import argparse
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def main():
    parser = argparse.ArgumentParser(
        description="TESS Exoplanet Transit-Detection Pipeline"
    )
    parser.add_argument(
        'target', nargs='?', default=None,
        help='Target star ID (e.g., "TIC 25155310")'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Run validation harness against known targets'
    )
    parser.add_argument(
        '--no-mcmc', action='store_true',
        help='Skip MCMC parameter estimation'
    )
    parser.add_argument(
        '--no-centroid', action='store_true',
        help='Skip centroid vetting (faster)'
    )
    parser.add_argument(
        '--no-fap', action='store_true',
        help='Skip FAP bootstrap (faster)'
    )
    parser.add_argument(
        '--fap-trials', type=int, default=200,
        help='Number of FAP bootstrap trials (default: 200)'
    )
    parser.add_argument(
        '--mcmc-steps', type=int, default=2000,
        help='Number of MCMC steps (default: 2000)'
    )
    parser.add_argument(
        '--report', action='store_true',
        help='Generate candidate summary JSON and vetting panel (Phase 4)'
    )

    args = parser.parse_args()

    if args.validate:
        from exoplanet_pipeline.validate import run_validation
        run_validation()
        return

    if args.target is None:
        parser.print_help()
        sys.exit(1)

    from exoplanet_pipeline.pipeline import run_full_pipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_full_pipeline(
            args.target,
            run_mcmc=not args.no_mcmc,
            run_centroid=not args.no_centroid,
            run_fap=not args.no_fap,
            n_fap_trials=args.fap_trials,
            n_mcmc_steps=args.mcmc_steps,
            run_report=args.report,
        )

    if results is None:
        print(f"\nPipeline returned no results for {args.target}")
        sys.exit(1)

    print("\nPipeline completed successfully.")


if __name__ == '__main__':
    main()
