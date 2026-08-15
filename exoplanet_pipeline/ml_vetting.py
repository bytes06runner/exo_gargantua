"""
ml_vetting.py — ML Vetting Classifier
=====================================

Replaces heuristic vetting thresholds with a supervised Machine Learning classifier.
Trained on synthetic injection-recovery data, multi-modal astrophysical false positive 
models (BEBs, EBs), and physical Difference Image Analysis (DIA) spatial constraints.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os


class MLVetter:
    def __init__(self, n_estimators=150, max_depth=12, random_state=42):
        """
        Initializes the ML Vetter using a Random Forest classifier.
        Features correspond to:
        - bls_power: SNR/power from BoxLeastSquares
        - depth_diff: Odd/Even transit depth difference
        - secondary_eclipse_sigma: Significance of secondary eclipse
        - centroid_shift: Offset distance from DIA in pixels
        - snr: Transit SNR
        - centroid_fail: Binary flag (1 if centroid_shift >= 0.333, else 0)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            class_weight='balanced'
        )
        self.features = [
            'bls_power', 
            'depth_diff', 
            'secondary_eclipse_sigma', 
            'centroid_shift', 
            'snr',
            'centroid_fail'
        ]
        self.is_trained = False
        self.centroid_threshold = 0.333
        
    def _prepare_features(self, df):
        """Prepares and validates feature matrix with engineered features."""
        df_feat = df.copy()
        if 'centroid_fail' not in df_feat.columns:
            centroid_shifts = df_feat['centroid_shift'].fillna(0.0) if 'centroid_shift' in df_feat.columns else 0.0
            df_feat['centroid_fail'] = (centroid_shifts >= self.centroid_threshold).astype(float)
            
        for f in self.features:
            if f not in df_feat.columns:
                df_feat[f] = 0.0
                
        return df_feat[self.features].fillna(0.0)
        
    def train(self, df):
        """
        Trains the classifier on a DataFrame containing the extracted features.
        The DataFrame must contain a 'label' column (1 = Planet, 0 = False Positive).
        """
        if 'label' not in df.columns:
            raise ValueError("Training data must contain a 'label' column.")
            
        X = self._prepare_features(df)
        y = df['label'].astype(int)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        print("\n--- ML Vetting Model Training Results ---")
        print(classification_report(y_test, preds, zero_division=0))
        
        try:
            auc = roc_auc_score(y_test, probs)
            print(f"ROC AUC: {auc:.3f}")
        except ValueError:
            print("ROC AUC: Not enough classes to compute.")
            
        print("Feature Importances:")
        for name, imp in zip(self.features, self.model.feature_importances_):
            print(f"  {name:25s}: {imp:.4f}")
            
    def predict(self, features_dict):
        """
        Predicts the probability that a candidate is a true planet.
        Enforces physical consistency: if spatial DIA centroid vetting or astrophysical
        eclipsing binary flags fail, the candidate's planet probability is heavily penalized.
        
        Returns
        -------
        dict with keys:
            'planet_probability': float, calibrated probability [0.0, 1.0]
            'disposition': str, 'CANDIDATE' or 'FALSE_POSITIVE'
            'flags': list of str, active vetting failure flags
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
            
        feat_df = pd.DataFrame([features_dict])
        X = self._prepare_features(feat_df)
        rf_prob = float(self.model.predict_proba(X)[0, 1])
        
        flags = []
        centroid_shift = float(features_dict.get('centroid_shift', 0.0))
        sec_sigma = abs(float(features_dict.get('secondary_eclipse_sigma', 0.0)))
        depth_diff = abs(float(features_dict.get('depth_diff', 0.0)))
        
        # Physical Vetting Checks
        if centroid_shift >= self.centroid_threshold:
            flags.append(f"FAILED_CENTROID_DIA (shift={centroid_shift:.3f} >= {self.centroid_threshold:.3f} pix)")
        if sec_sigma > 3.0:
            flags.append(f"FAILED_SECONDARY_ECLIPSE ({sec_sigma:.2f}σ > 3.0σ)")
        if depth_diff > 0.005:
            flags.append(f"FAILED_ODD_EVEN_DEPTH (diff={depth_diff:.5f} > 0.005)")
            
        rp_earth = float(features_dict.get('rp_earth', 0.0))
        depth_val = float(features_dict.get('depth', 0.0))
        if rp_earth > 25.0:
            flags.append(f"FAILED_STELLAR_RADIUS (Rp={rp_earth:.1f} R_earth > 25.0 R_earth / Eclipsing Binary)")
        elif depth_val > 0.03:
            flags.append(f"FAILED_STELLAR_DEPTH (depth={depth_val:.4f} > 3.0% / Eclipsing Binary)")
            
        # Hard Physical Veto / Calibration:
        # Spatial centroid shift is a definitive physical veto indicating a blended false positive
        if centroid_shift >= self.centroid_threshold:
            # Drop planet probability to near-zero (< 0.5%) regardless of 1D light-curve power
            calibrated_prob = min(rf_prob * 0.01, 0.005)
        elif len(flags) > 0:
            calibrated_prob = min(rf_prob * 0.1, 0.05)
        else:
            calibrated_prob = rf_prob
            
        disposition = "CANDIDATE" if (calibrated_prob >= 0.5 and len(flags) == 0) else "FALSE_POSITIVE"
        
        return {
            'planet_probability': calibrated_prob,
            'rf_raw_probability': rf_prob,
            'disposition': disposition,
            'flags': flags
        }
        
    def save(self, filepath='ml_vetter.joblib'):
        """Serializes the trained model weights and configuration to disk."""
        if not self.is_trained:
            print("Warning: Saving an untrained model.")
        payload = {
            'model': self.model,
            'features': self.features,
            'is_trained': self.is_trained,
            'centroid_threshold': self.centroid_threshold
        }
        joblib.dump(payload, filepath)
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath='ml_vetter.joblib'):
        """Loads serialized model weights and configuration from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        payload = joblib.load(filepath)
        instance = cls()
        instance.model = payload['model']
        instance.features = payload['features']
        instance.is_trained = payload.get('is_trained', True)
        instance.centroid_threshold = payload.get('centroid_threshold', 0.333)
        print(f"Model loaded successfully from {filepath}")
        return instance


def generate_training_dataset(n_samples=3000):
    """
    Generates a realistic multi-modal training dataset covering both true exoplanet
    transits and astrophysical false positives (Background Eclipsing Binaries with
    2D centroid shifts, Eclipsing Binaries with secondary eclipses, and low-SNR noise).
    """
    np.random.seed(42)
    data = []
    
    n_planets = int(n_samples * 0.50)
    n_bebs = int(n_samples * 0.25)
    n_ebs = int(n_samples * 0.15)
    n_noise = n_samples - n_planets - n_bebs - n_ebs
    
    # 1. True Planets (Label = 1)
    # Physical traits: High/Moderate SNR, negligible odd/even difference,
    # no secondary eclipse, negligible centroid shift (< 0.20 pixels)
    for _ in range(n_planets):
        snr = np.random.uniform(10.0, 3000.0)
        data.append({
            'bls_power': snr * np.random.uniform(0.9, 1.1),
            'depth_diff': np.random.exponential(scale=0.0005),
            'secondary_eclipse_sigma': np.random.normal(loc=0.0, scale=0.8),
            'centroid_shift': np.random.uniform(0.01, 0.20),
            'snr': snr,
            'label': 1
        })
        
    # 2. Background Eclipsing Binaries (BEB, Label = 0)
    # Physical traits: High SNR, can have clean light curves, but massive centroid shift (> 0.33 pix)
    for _ in range(n_bebs):
        snr = np.random.uniform(10.0, 3000.0)
        data.append({
            'bls_power': snr * np.random.uniform(0.9, 1.1),
            'depth_diff': np.random.exponential(scale=0.001),
            'secondary_eclipse_sigma': np.random.normal(loc=0.0, scale=1.0),
            'centroid_shift': np.random.uniform(0.35, 3.0),  # Failing DIA threshold!
            'snr': snr,
            'label': 0
        })
        
    # 3. Eclipsing Binaries (EB, Label = 0)
    # Physical traits: Large secondary eclipse (> 3.0 sigma) and/or odd-even depth difference
    for _ in range(n_ebs):
        snr = np.random.uniform(15.0, 3000.0)
        data.append({
            'bls_power': snr * np.random.uniform(0.9, 1.1),
            'depth_diff': np.random.uniform(0.006, 0.08),
            'secondary_eclipse_sigma': np.random.uniform(3.5, 20.0),
            'centroid_shift': np.random.uniform(0.01, 0.20),
            'snr': snr,
            'label': 0
        })
        
    # 4. Low-SNR / Instrumental Artifacts (Label = 0)
    for _ in range(n_noise):
        snr = np.random.uniform(3.0, 9.5)
        data.append({
            'bls_power': snr * np.random.uniform(0.8, 1.0),
            'depth_diff': np.random.uniform(0.002, 0.02),
            'secondary_eclipse_sigma': np.random.uniform(1.5, 4.0),
            'centroid_shift': np.random.uniform(0.1, 0.8),
            'snr': snr,
            'label': 0
        })
        
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generate_training_dataset(n_samples=3000)
    vetter = MLVetter()
    vetter.train(df)
    vetter.save('ml_vetter.joblib')
