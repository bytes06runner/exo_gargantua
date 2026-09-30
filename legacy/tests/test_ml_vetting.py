import pytest
from exoplanet_pipeline.ml_vetting import MLVetter, generate_training_dataset


def test_ml_vetter_training_and_spatial_veto():
    # 1. Train model on synthetic/simulated multi-modal dataset
    df = generate_training_dataset(n_samples=500)
    vetter = MLVetter()
    vetter.train(df)
    
    # 2. Test a clean true planet
    planet_features = {
        'bls_power': 50.0,
        'snr': 50.0,
        'depth_diff': 0.0001,
        'secondary_eclipse_sigma': 0.2,
        'centroid_shift': 0.05
    }
    pred_planet = vetter.predict(planet_features)
    assert pred_planet['disposition'] == 'CANDIDATE'
    assert pred_planet['planet_probability'] > 0.5
    assert len(pred_planet['flags']) == 0

    # 3. Test a Background Eclipsing Binary with large DIA centroid shift (e.g. 0.5 pix)
    beb_features = {
        'bls_power': 2000.0, # high SNR must not override spatial failure
        'snr': 2000.0,
        'depth_diff': 0.00004,
        'secondary_eclipse_sigma': 0.5,
        'centroid_shift': 0.4965 # Failing DIA threshold (> 0.333)
    }
    pred_beb = vetter.predict(beb_features)
    assert pred_beb['disposition'] == 'FALSE_POSITIVE'
    assert pred_beb['planet_probability'] < 0.05
    assert any('FAILED_CENTROID_DIA' in flag for flag in pred_beb['flags'])

    # 4. Test an Eclipsing Binary with significant secondary eclipse
    eb_features = {
        'bls_power': 50.0,
        'snr': 50.0,
        'depth_diff': 0.0001,
        'secondary_eclipse_sigma': 14.5, # > 10.0 sigma
        'centroid_shift': 0.05
    }
    pred_eb = vetter.predict(eb_features)
    assert pred_eb['disposition'] == 'FALSE_POSITIVE'
    assert any('FAILED_SECONDARY_ECLIPSE' in flag for flag in pred_eb['flags'])

    # 5. Test an Eclipsing Binary with moderate secondary eclipse (Warning only)
    eb_warning_features = {
        'bls_power': 50.0,
        'snr': 50.0,
        'depth_diff': 0.0001,
        'secondary_eclipse_sigma': 4.5, # > 3.0 sigma, < 10.0 sigma
        'centroid_shift': 0.05
    }
    pred_warning_eb = vetter.predict(eb_warning_features)
    assert pred_warning_eb['disposition'] == 'CANDIDATE'
    assert any('WARNING_SECONDARY_ECLIPSE' in flag for flag in pred_warning_eb['flags'])
