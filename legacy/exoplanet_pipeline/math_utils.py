import numpy as np

def calculate_odd_even_delta(odd_depth: float, even_depth: float) -> float:
    """
    Calculates the absolute difference between odd and even transit depths.
    """
    return abs(odd_depth - even_depth)

def boxcar_std(flux_array: np.ndarray, window_size: int) -> float:
    """
    Calculates the standard deviation of rolling means over a boxcar window.
    This is used to estimate the empirical standard error in binned light curves.
    """
    flux_array = np.asarray(flux_array)
    flux_array = flux_array[~np.isnan(flux_array)]
    if len(flux_array) < window_size or window_size < 1:
        return np.nan
    
    rolling_means = np.convolve(flux_array, np.ones(window_size)/window_size, mode='valid')
    return np.nanstd(rolling_means)

def keplers_third_law(period_days: float, mass_solar: float) -> float:
    """
    Calculates the semi-major axis (a) in Astronomical Units (AU) 
    given the orbital period in days and stellar mass in solar masses.
    """
    if period_days <= 0 or mass_solar <= 0:
        raise ValueError("Period and mass must be positive.")
        
    G_CONST = 6.67430e-11  # m^3 kg^-1 s^-2
    M_SUN_KG = 1.9884e30
    AU_TO_METERS = 1.496e11
    
    period_sec = period_days * 86400.0
    mass_kg = mass_solar * M_SUN_KG
    a_meters = (G_CONST * mass_kg * period_sec**2 / (4 * np.pi**2))**(1/3)
    
    return a_meters / AU_TO_METERS
