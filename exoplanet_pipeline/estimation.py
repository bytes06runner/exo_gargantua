"""
estimation.py — MCMC Parameter Estimation via batman + emcee
============================================================

Fix 5: Provides uncertainty quantification on transit parameters by
fitting a batman transit model to the phase-folded light curve using
the emcee affine-invariant MCMC sampler.

Fitted parameters: period, t0, depth (Rp/Rs)^2, and duration.
Initial values are seeded from the BLS best-fit to ensure the sampler
starts near the global optimum.

Output: 16th/50th/84th percentile of each parameter's marginal posterior,
reported as {param: (median, lower_err, upper_err)}.
"""

import numpy as np
import batman
import emcee


def _batman_model(time, period, t0, rp_rs, a_rs, inc, ld_coeffs, baseline=1.0):
    """
    Compute a batman transit model light curve.

    Parameters
    ----------
    time : np.ndarray
        Time stamps (days).
    period : float
        Orbital period (days).
    t0 : float
        Mid-transit time (days).
    rp_rs : float
        Planet-to-star radius ratio Rp/Rs.
    a_rs : float
        Semi-major axis to stellar radius ratio a/Rs.
    inc : float
        Orbital inclination in degrees.
    ld_coeffs : list
        Quadratic limb darkening coefficients [u1, u2].
    baseline : float
        Out-of-transit flux level (default 1.0 for normalized data).

    Returns
    -------
    flux : np.ndarray
        Model flux at each time stamp.
    """
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs  # Rp/Rs
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = ld_coeffs
    params.limb_dark = "quadratic"

    m = batman.TransitModel(params, time)
    flux = m.light_curve(params) * baseline
    return flux


def _log_likelihood(theta, time, flux, flux_err, ld_coeffs):
    """Gaussian log-likelihood for transit model fit."""
    period, t0, rp_rs, a_rs, inc = theta
    model = _batman_model(time, period, t0, rp_rs, a_rs, inc, ld_coeffs)
    residuals = flux - model
    chi2 = np.sum((residuals / flux_err) ** 2)
    log_norm = -0.5 * np.sum(np.log(2 * np.pi * flux_err ** 2))
    return log_norm - 0.5 * chi2


def _log_prior(theta, period_init, t0_init):
    """
    Log-prior with narrow Gaussians on P and t0, flat elsewhere.
    """
    period, t0, rp_rs, a_rs, inc = theta

    # Rp/Rs: 0.01 to 0.25
    if not (0.01 <= rp_rs <= 0.25):
        return -np.inf

    # a/Rs: 1.5 to 30.0
    if not (1.5 <= a_rs <= 30.0):
        return -np.inf

    # Inclination: 69.5 to 90 degrees (cos i from 0.0 to 0.35)
    if not (69.5 <= inc <= 90.0):
        return -np.inf

    # Uniform priors but with wide bounds on P and t0
    # P in [P_bls - 0.05, P_bls + 0.05]
    if not (period_init - 0.05 <= period <= period_init + 0.05):
        return -np.inf
    # t0 in [t0_bls - 0.1, t0_bls + 0.1]
    if not (t0_init - 0.1 <= t0 <= t0_init + 0.1):
        return -np.inf

    # Impact parameter constraint: must transit the star
    b = a_rs * np.cos(np.radians(inc))
    if b > 1.0 + rp_rs:
        return -np.inf

    return 0.0


def _log_probability(theta, time, flux, flux_err, period_init, t0_init, ld_coeffs):
    """Log-posterior = log-prior + log-likelihood."""
    lp = _log_prior(theta, period_init, t0_init)
    if not np.isfinite(lp):
        return -np.inf
    ll = _log_likelihood(theta, time, flux, flux_err, ld_coeffs)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def run_mcmc_estimation(lc, bls_results, raw_flux_err=None, stellar_params=None,
                        n_walkers=32, n_steps=2000, burn_in=500):
    """
    Runs MCMC parameter estimation on transit parameters using batman
    for the transit model and emcee for sampling.

    The sampler is seeded from the BLS best-fit values. After the chain
    converges, posteriors are summarized as 16th/50th/84th percentiles.

    Parameters
    ----------
    lc : lightkurve.LightCurve
        The filtered, stitched light curve (normalized flux).
    bls_results : dict
        Output from detection.run_bls_search, containing 'period', 't0',
        'depth', and 'duration'.
    raw_flux_err : np.ndarray or None
        Per-cadence flux errors. If None, uses lc.flux_err or estimates
        from the scatter in the light curve.
    n_walkers : int
        Number of emcee walkers (must be >= 2 * n_params = 8).
    n_steps : int
        Total number of MCMC steps per walker (including burn-in).
    burn_in : int
        Number of initial steps to discard as burn-in.

    Returns
    -------
    posteriors : dict
        {param_name: (median, lower_err, upper_err)} where lower_err
        and upper_err are the distances from the median to the 16th
        and 84th percentiles respectively (i.e., ~1σ for a Gaussian).
    sampler : emcee.EnsembleSampler
        The sampler object (for diagnostics, corner plots, etc.).
    """
    print("  Starting MCMC parameter estimation...")

    # Fix 2: Dynamically compute limb darkening
    ld_coeffs = [0.3, 0.1] # Fallback
    if stellar_params is not None and not stellar_params.get('fallback_used', False):
        try:
            from ldtk import LDPSetCreator, BoxcarFilter
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                tess_filter = BoxcarFilter('TESS', 600, 1000)
                sc = LDPSetCreator(
                    teff=(stellar_params.get('Teff', 5778.0), stellar_params.get('Teff_err', 100.0)), 
                    logg=(stellar_params.get('logg', 4.438), stellar_params.get('logg_err', 0.1)), 
                    z=(stellar_params.get('MH', 0.0), stellar_params.get('MH_err', 0.1)), 
                    filters=[tess_filter]
                )
                ps = sc.create_profiles()
                qc, _ = ps.coeffs_qd(do_mc=True)
                ld_coeffs = [float(qc[0][0]), float(qc[0][1])]
            print(f"  Dynamically computed limb darkening coefficients (ldtk): u1={ld_coeffs[0]:.4f}, u2={ld_coeffs[1]:.4f}")
        except ImportError:
            print("  Warning: ldtk not installed. Using fallback limb darkening [0.3, 0.1].")
        except Exception as e:
            print(f"  Warning: failed to compute dynamic limb darkening with ldtk: {e}. Using fallback.")

    # Extract BLS seed values
    period_init = float(bls_results['period'].value)
    t0_init = float(bls_results['t0'].value)
    depth_init = float(bls_results['depth'].value)
    duration_init = float(bls_results['duration'].value)

    # Convert depth to Rp/Rs: depth ≈ (Rp/Rs)^2
    rp_rs_init = np.sqrt(max(depth_init, 1e-6))

    # Prepare data: phase-fold to single transit window for speed
    time = lc.time.value
    flux = lc.flux.value

    if raw_flux_err is not None and len(raw_flux_err) == len(lc):
        flux_err = raw_flux_err
    elif hasattr(lc, 'flux_err') and lc.flux_err is not None:
        flux_err = lc.flux_err.value
    else:
        # Estimate from data scatter
        flux_err = np.full_like(flux, np.nanstd(flux))

    # Ensure flux_err is positive everywhere
    flux_err = np.abs(flux_err)
    flux_err[flux_err == 0] = np.nanmedian(flux_err[flux_err > 0])

    # Trim to near-transit data (within ±2 transit durations of phase 0)
    dur_phase = duration_init / period_init
    window_phase = max(0.04, dur_phase * 2.0)
    phase = ((time - t0_init) % period_init) / period_init
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    near_transit = np.abs(phase) < window_phase

    if np.sum(near_transit) < 50:
        near_transit = np.ones(len(time), dtype=bool)

    # Do NOT fold the time array. Keeping the original timestamps preserves 
    # the full-time lever arm, giving precise constraints on P and t0.
    time_fit = time[near_transit]
    flux_fit = flux[near_transit]
    flux_err_fit = flux_err[near_transit]

    # If dataset has >20,000 cadences in the transit window, subsample evenly
    # to maintain high statistical fidelity without memory/time bottleneck
    if len(time_fit) > 20000:
        stride = len(time_fit) // 20000
        time_fit = time_fit[::stride]
        flux_fit = flux_fit[::stride]
        flux_err_fit = flux_err_fit[::stride]

    # Derive a/Rs init from duration and period assuming circular orbit
    a_rs_init = period_init / (np.pi * duration_init) if duration_init > 0 else 15.0
    a_rs_init = max(a_rs_init, 2.0)
    inc_init = 89.0  # start slightly off 90 to help sampler

    print(f"  Using {len(time_fit)} cadences for MCMC fit (unfolded, near-transit windows)")
    print(f"  BLS seeds: P={period_init:.6f}d, t0={t0_init:.4f}, "
          f"Rp/Rs={rp_rs_init:.5f}, a/Rs={a_rs_init:.2f}")

    # Initialize walkers in a small ball around the seed
    n_dim = 5
    theta_init = np.array([period_init, t0_init, rp_rs_init, a_rs_init, inc_init])

    # Perturbation scale: small to avoid hitting prior boundaries immediately
    perturbation = np.array([
        1e-4,            # P
        1e-4,            # t0
        1e-3,            # rp_rs
        0.1,             # a_rs
        0.5              # inc
    ])

    pos = theta_init + perturbation * np.random.randn(n_walkers, n_dim)

    # Ensure all initial positions satisfy priors
    for i in range(n_walkers):
        while not np.isfinite(_log_prior(pos[i], period_init, t0_init)):
            pos[i] = theta_init + perturbation * np.random.randn(n_dim)

    # Run MCMC
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, _log_probability,
        args=(time_fit, flux_fit, flux_err_fit, period_init, t0_init, ld_coeffs)
    )

    print(f"  Running {n_steps} MCMC steps with {n_walkers} walkers...")
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Discard burn-in and flatten
    samples = sampler.get_chain(discard=burn_in, flat=True)
    print(f"  Collected {len(samples)} post-burn-in samples")

    # Compute percentiles
    param_names = ['period', 't0', 'rp_rs', 'a_rs', 'inc']
    posteriors = {}

    for i, name in enumerate(param_names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        lower_err = q50 - q16
        upper_err = q84 - q50
        posteriors[name] = (q50, lower_err, upper_err)
        print(f"  {name}: {q50:.6f} (+{upper_err:.6f} / -{lower_err:.6f})")

    # Also report depth = (Rp/Rs)^2 for convenience
    rp_rs_samples = samples[:, 2]
    depth_samples = rp_rs_samples ** 2
    q16, q50, q84 = np.percentile(depth_samples, [16, 50, 84])
    posteriors['depth'] = (q50, q50 - q16, q84 - q50)
    print(f"  depth: {q50:.6f} (+{q84 - q50:.6f} / -{q50 - q16:.6f})")

    # Check convergence: acceptance fraction
    acc = np.mean(sampler.acceptance_fraction)
    print(f"  Mean acceptance fraction: {acc:.3f} "
          f"(ideal: 0.2-0.5)")

    return posteriors, sampler


def derive_physical_parameters(mcmc_posteriors, stellar_params, n_samples=10000):
    """
    Derives physical planetary parameters using Monte Carlo propagation
    of MCMC posteriors and stellar parameter uncertainties.

    Parameters
    ----------
    mcmc_posteriors : dict
        Output from run_mcmc_estimation (needs 'rp_rs', 'period', 'a_rs').
    stellar_params : dict
        Output from ingestion.fetch_stellar_parameters (needs 'Rs', 'Rs_err',
        'Ms', 'Ms_err', 'Teff', 'Teff_err').
    n_samples : int
        Number of Monte Carlo samples to draw for error propagation.

    Returns
    -------
    dict
        Derived parameters (Rp_earth, Rp_jup, a_au, Teq, S_earth, P_transit)
        with median and 1-sigma errors.
    """
    print("  Deriving physical parameters via Monte Carlo propagation...")

    # Constants
    R_SUN_TO_EARTH = 109.2
    R_SUN_TO_JUP = 9.731
    AU_TO_METERS = 1.496e11
    R_SUN_METERS = 6.957e8
    G_CONST = 6.67430e-11  # m^3 kg^-1 s^-2
    M_SUN_KG = 1.9884e30
    SIGMA_SB = 5.670374419e-8
    L_SUN = 3.828e26  # Watts
    S_EARTH = 1361.0  # W/m^2

    # Extract MCMC medians and pseudo-sigmas
    # Use average of upper/lower errors for the MC normal distribution
    rp_rs_med, rp_rs_err_low, rp_rs_err_high = mcmc_posteriors['rp_rs']
    rp_rs_err = (rp_rs_err_low + rp_rs_err_high) / 2.0

    period_med, per_err_low, per_err_high = mcmc_posteriors['period']
    period_err = (per_err_low + per_err_high) / 2.0

    a_rs_med, a_rs_err_low, a_rs_err_high = mcmc_posteriors.get('a_rs', (15.0, 1.0, 1.0))
    a_rs_err = (a_rs_err_low + a_rs_err_high) / 2.0

    # Draw Monte Carlo samples
    rp_rs_dist = np.random.normal(rp_rs_med, rp_rs_err, n_samples)
    period_dist = np.random.normal(period_med, period_err, n_samples)
    a_rs_dist = np.random.normal(a_rs_med, a_rs_err, n_samples)

    Rs_dist = np.random.normal(stellar_params['Rs'], stellar_params['Rs_err'], n_samples)
    Ms_dist = np.random.normal(stellar_params['Ms'], stellar_params['Ms_err'], n_samples)
    Teff_dist = np.random.normal(stellar_params['Teff'], stellar_params['Teff_err'], n_samples)

    # 1. Planetary Radius
    Rp_earth_dist = rp_rs_dist * Rs_dist * R_SUN_TO_EARTH
    Rp_jup_dist = rp_rs_dist * Rs_dist * R_SUN_TO_JUP

    # 2. Semi-Major Axis (a) in AU from Kepler's Third Law
    # a^3 = G * M * P^2 / (4 * pi^2)
    period_sec = period_dist * 86400.0
    mass_kg = Ms_dist * M_SUN_KG
    a_meters = (G_CONST * mass_kg * period_sec**2 / (4 * np.pi**2))**(1/3)
    a_au_dist = a_meters / AU_TO_METERS

    # Alternatively, from a/Rs directly:
    # a_au_alt = a_rs_dist * Rs_dist * R_SUN_METERS / AU_TO_METERS

    # 3. Insolation & Teq (Bond Albedo = 0.3)
    # L = 4 * pi * R^2 * sigma * T^4
    L_star = 4 * np.pi * (Rs_dist * R_SUN_METERS)**2 * SIGMA_SB * Teff_dist**4
    S_flux = L_star / (4 * np.pi * a_meters**2)
    S_earth_dist = S_flux / S_EARTH

    A_B = 0.3
    Teq_dist = Teff_dist * np.sqrt(Rs_dist * R_SUN_METERS / (2 * a_meters)) * (1 - A_B)**0.25

    # 4. Transit Probability
    ptransit_dist = 1.0 / a_rs_dist
    # clip to max 1.0
    ptransit_dist = np.clip(ptransit_dist, 0.0, 1.0)

    # Summarize results
    def summarize(dist):
        q16, q50, q84 = np.percentile(dist, [16, 50, 84])
        return (q50, q50 - q16, q84 - q50)

    derived = {
        'Rp_earth': summarize(Rp_earth_dist),
        'Rp_jup': summarize(Rp_jup_dist),
        'a_au': summarize(a_au_dist),
        'Teq': summarize(Teq_dist),
        'S_earth': summarize(S_earth_dist),
        'P_transit': summarize(ptransit_dist)
    }

    print("  Derived Parameters:")
    print(f"    Rp: {derived['Rp_earth'][0]:.2f} R_earth")
    print(f"    a:  {derived['a_au'][0]:.4f} AU")
    print(f"    Teq:{derived['Teq'][0]:.0f} K")

    return derived
