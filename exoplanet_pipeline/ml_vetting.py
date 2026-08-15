"""
ml_vetting.py — ML Vetting Classifier
=====================================

Replaces heuristic vetting thresholds with a supervised Machine Learning classifier.
Trains on synthetic injection-recovery data (and known false positives).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

class MLVetter:
    def __init__(self):
        """
        Initializes the ML Vetter using a Random Forest classifier.
        Features correspond to:
        - bls_power: SNR from BoxLeastSquares
        - depth_diff: Odd/Even transit depth difference
        - secondary_eclipse_sigma: Significance of secondary eclipse
        - centroid_shift: Offset distance from DIA
        - snr: Transit SNR
        """
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.features = [
            'bls_power', 
            'depth_diff', 
            'secondary_eclipse_sigma', 
            'centroid_shift', 
            'snr'
        ]
        self.is_trained = False
        
    def train(self, df):
        """
        Trains the classifier on a DataFrame containing the extracted features.
        The DataFrame must contain a 'label' column (1 = Planet, 0 = False Positive).
        """
        if 'label' not in df.columns:
            raise ValueError("Training data must contain a 'label' column.")
            
        # Ensure missing feature columns are filled with 0
        for f in self.features:
            if f not in df.columns:
                df[f] = 0.0
                
        X = df[self.features].fillna(0)
        y = df['label']
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
            print(f"  {name}: {imp:.3f}")
            
    def predict(self, features_dict):
        """
        Predicts the probability that a candidate is a true planet.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
            
        # Convert dictionary to DataFrame using predefined feature order
        X = pd.DataFrame([features_dict], columns=self.features).fillna(0)
        prob = self.model.predict_proba(X)[0, 1]
        return prob
        
    def save(self, filepath='ml_vetter.joblib'):
        """Serializes the trained model to disk."""
        if not self.is_trained:
            print("Warning: Saving an untrained model.")
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath='ml_vetter.joblib'):
        """Loads a serialized model from disk."""
        instance = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return instance

def generate_dummy_dataset(n_samples=500):
    """
    Generates a dummy dataset for demonstration purposes.
    In production, this is populated from the output of validate_injection.py
    and real false positive target runs.
    """
    np.random.seed(42)
    data = []
    
    # Generate True Planets (label=1)
    for _ in range(n_samples // 2):
        data.append({
            'bls_power': np.random.uniform(10, 50),
            'depth_diff': np.random.uniform(0.0, 0.003), # Small difference
            'secondary_eclipse_sigma': np.random.uniform(0, 2.5), # Not significant
            'centroid_shift': np.random.uniform(0, 0.1), # Small shift
            'snr': np.random.uniform(10, 100),
            'label': 1
        })
        
    # Generate False Positives (label=0)
    for _ in range(n_samples // 2):
        # Determine FP type: 0=EB (high depth diff/eclipse), 1=BEB (high centroid shift)
        fp_type = np.random.choice([0, 1])
        if fp_type == 0:
            data.append({
                'bls_power': np.random.uniform(10, 50),
                'depth_diff': np.random.uniform(0.005, 0.05), # High difference
                'secondary_eclipse_sigma': np.random.uniform(3.0, 10.0), # High significance
                'centroid_shift': np.random.uniform(0, 0.1),
                'snr': np.random.uniform(10, 100),
                'label': 0
            })
        else:
            data.append({
                'bls_power': np.random.uniform(10, 50),
                'depth_diff': np.random.uniform(0.0, 0.003),
                'secondary_eclipse_sigma': np.random.uniform(0, 2.5),
                'centroid_shift': np.random.uniform(0.4, 2.0), # High shift
                'snr': np.random.uniform(10, 100),
                'label': 0
            })
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dummy_dataset()
    vetter = MLVetter()
    vetter.train(df)
    vetter.save('ml_vetter.joblib')
