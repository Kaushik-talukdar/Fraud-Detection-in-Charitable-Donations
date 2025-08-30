import pandas as pd
import numpy as np
import joblib
import yaml
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.preprocess import DataPreprocessor

# Load configuration
config_path = os.path.join(project_root, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class FraudDetector:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
    def load_model(self):
        try:
            model_path = os.path.join(project_root, config['paths']['model'])
            preprocessor_path = os.path.join(project_root, config['paths']['preprocessor'])
            
            self.model = joblib.load(model_path)
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessor(preprocessor_path)
            self.feature_names = self.preprocessor.feature_names
            print("Model and preprocessor loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, donation_data):
        if self.model is None or self.preprocessor is None:
            if not self.load_model():
                raise Exception("Model could not be loaded")
        
        if isinstance(donation_data, dict):
            donation_df = pd.DataFrame([donation_data])
        else:
            donation_df = donation_data.copy()
        
        X, _, df_processed = self.preprocessor.preprocess_data(donation_df, fit=False)
        
        if config['model']['algorithm'] == 'isolation_forest':
            prediction = self.model.predict(X)
            fraud_prob = self.model.decision_function(X)
            fraud_score = -fraud_prob
        else:
            prediction = self.model.fit_predict(X)
            fraud_score = -self.model.negative_outlier_factor_
        
        if config['model']['algorithm'] == 'lof' and len(X) == 1:
            prediction = np.array([1 if fraud_score[0] > 0.5 else -1])
        
        is_fraud = (prediction == -1)
        
        return {
            'is_fraud': bool(is_fraud[0]),
            'fraud_score': float(fraud_score[0]),
            'processed_data': df_processed.iloc[0].to_dict()
        }

detector = FraudDetector()

def predict_fraud(donation_data):
    return detector.predict(donation_data)