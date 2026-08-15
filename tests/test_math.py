import numpy as np
import pytest
from exoplanet_pipeline.math_utils import calculate_odd_even_delta, boxcar_std, keplers_third_law

def test_calculate_odd_even_delta():
    assert calculate_odd_even_delta(0.01, 0.015) == pytest.approx(0.005)
    assert calculate_odd_even_delta(0.02, 0.02) == 0.0
    assert calculate_odd_even_delta(0.005, 0.001) == pytest.approx(0.004)

def test_boxcar_std():
    np.random.seed(42)
    # create a random array
    flux = np.random.normal(1.0, 0.01, 100)
    std = boxcar_std(flux, window_size=10)
    
    # ensure it returns a valid float
    assert not np.isnan(std)
    assert std > 0
    assert std < 0.01  # the standard deviation of means should be smaller than original std
    
    # edge cases
    assert np.isnan(boxcar_std([1, 2, 3], 5))
    assert np.isnan(boxcar_std([], 5))

def test_keplers_third_law():
    # Earth around the Sun: P=365.25 days, M=1.0 M_sun -> a ~ 1.0 AU
    a_earth = keplers_third_law(365.25, 1.0)
    assert a_earth == pytest.approx(1.0, rel=0.01)
    
    # Jupiter around the Sun: P~4332.6 days, M=1.0 M_sun -> a ~ 5.2 AU
    a_jupiter = keplers_third_law(4332.59, 1.0)
    assert a_jupiter == pytest.approx(5.2, rel=0.01)
    
    with pytest.raises(ValueError):
        keplers_third_law(-10, 1.0)
    
    with pytest.raises(ValueError):
        keplers_third_law(10, -1.0)
