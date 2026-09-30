import numpy as np

from exogargantua import inject as I


def test_limb_darkening_sun_like_and_cool_star():
    u1, u2 = I.limb_darkening(5778, 4.44)
    assert 0.2 < u1 < 0.6 and 0.0 < u2 < 0.4
    c1, c2 = I.limb_darkening(3300, 5.0)  # PHOENIX branch
    assert np.isfinite(c1) and np.isfinite(c2)


def test_planet_depth_matches_request():
    t = np.linspace(-0.2, 0.2, 2001)
    u = I.limb_darkening(5778, 4.44)
    for depth in (2e-4, 1e-3, 1e-2):
        f, _ = I.planet_flux(t, 5.0, 0.0, depth, 0.3, 1.0, 1.0, u)
        assert abs((1 - f.min()) / depth - 1) < 0.02


def test_equal_depth_twin_has_two_equal_eclipses_per_orbit():
    P = 4.0
    t = np.linspace(0, P, 40001)
    u = I.limb_darkening(5778, 4.44)
    f, meta = I.eb_flux(t, P, 0.5, 5e-3, 5e-3, 0.2, 1.0, 1.0, u)
    assert abs(meta["sec_phase"] - 0.5) < 1e-12
    d1 = 1 - f[np.abs(t - 0.5) < 0.01].min()
    d2 = 1 - f[np.abs(t - 2.5) < 0.01].min()
    assert abs(d1 / d2 - 1) < 0.02


def test_eccentric_secondary_is_offset():
    u = I.limb_darkening(5778, 4.44)
    _, meta = I.eb_flux(np.linspace(0, 1, 10), 4.0, 0.5, 5e-3, 3e-3, 0.2, 1.0, 1.0, u, ecc=0.3, omega=0.0)
    assert abs(meta["sec_phase"] - 0.5) > 0.1
