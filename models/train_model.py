import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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

def train_model():
    print("Loading and preprocessing data...")
    
    # Load data
    data_path = os.path.join(project_root, config['data']['output_file'])
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    
    preprocessor = DataPreprocessor()
    X, y, df_processed = preprocessor.preprocess_data(df, fit=True)
    
    # Save the preprocessor
    preprocessor_path = os.path.join(project_root, config['paths']['preprocessor'])
    preprocessor.save_preprocessor(preprocessor_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['model']['test_size'], 
        random_state=config['model']['random_state'], stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train the model
    if config['model']['algorithm'] == 'isolation_forest':
        model = IsolationForest(
            n_estimators=100,
            contamination=config['data']['fraud_ratio'],
            random_state=config['model']['random_state'],
            n_jobs=-1
        )
    elif config['model']['algorithm'] == 'lof':
        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=config['data']['fraud_ratio'],
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown algorithm: {config['model']['algorithm']}")
    
    print("Training model...")
    
    if config['model']['algorithm'] == 'isolation_forest':
        model.fit(X_train)
        y_pred = model.predict(X_test)
        y_pred_binary = np.where(y_pred == -1, 1, 0)
    else:
        y_pred = model.fit_predict(X_test)
        y_pred_binary = np.where(y_pred == -1, 1, 0)
    
    # Evaluate
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred_binary))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_binary)
        print(f"ROC AUC Score: {roc_auc:.4f}")
    except:
        print("Could not calculate ROC AUC")
    
    # Save the model
    if config['model']['algorithm'] == 'isolation_forest':
        model_path = os.path.join(project_root, config['paths']['model'])
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print("Model saved successfully")
    
    return model, preprocessor

if __name__ == "__main__":
    train_model()