import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import yaml
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Load configuration
config_path = os.path.join(project_root, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class DataPreprocessor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.preprocessor = None
        self.feature_names = None
        
    def extract_sentiment(self, text):
        if not text or pd.isna(text):
            return 0
        return self.sia.polarity_scores(text)['compound']
    
    def preprocess_data(self, df, fit=False):
        df_processed = df.copy()
        
        # Extract sentiment from comments
        df_processed['sentiment_score'] = df_processed['donor_comment'].apply(self.extract_sentiment)
        
        # Prepare features for modeling
        X = df_processed[config['features']['numerical'] + config['features']['categorical']]
        y = df_processed['label'] if 'label' in df_processed.columns else None
        
        # Create preprocessing pipeline
        numerical_features = config['features']['numerical']
        categorical_features = config['features']['categorical']
        
        if fit or self.preprocessor is None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])
            
            if fit:
                X_transformed = preprocessor.fit_transform(X)
                self.preprocessor = preprocessor
                # Get feature names
                feature_names = numerical_features.copy()
                ohe = preprocessor.named_transformers_['cat']
                cat_feature_names = ohe.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
                self.feature_names = feature_names
            else:
                X_transformed = preprocessor.transform(X)
        else:
            X_transformed = self.preprocessor.transform(X)
            
        return X_transformed, y, df_processed
    
    def save_preprocessor(self, filepath):
        if self.preprocessor:
            full_path = os.path.join(project_root, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            joblib.dump(self.preprocessor, full_path)
            print(f"Preprocessor saved to {full_path}")
    
    def load_preprocessor(self, filepath):
        full_path = os.path.join(project_root, filepath)
        self.preprocessor = joblib.load(full_path)
        print(f"Preprocessor loaded from {full_path}")
        
        # Reconstruct feature names
        numerical_features = config['features']['numerical']
        categorical_features = config['features']['categorical']
        
        feature_names = numerical_features.copy()
        ohe = self.preprocessor.named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
        self.feature_names = feature_names