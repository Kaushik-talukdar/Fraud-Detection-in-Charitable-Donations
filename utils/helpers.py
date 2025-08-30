import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_explanation(prediction_result: Dict[str, Any]) -> str:
    data = prediction_result['processed_data']
    is_fraud = prediction_result['is_fraud']
    score = prediction_result['fraud_score']
    
    reasons = []
    
    if is_fraud:
        if data['donation_frequency_from_ip'] > 5:
            reasons.append(f"High donation frequency from this IP ({data['donation_frequency_from_ip']} donations)")
        
        if data['geo_distance_from_campaign'] > 2000:
            reasons.append(f"Large geographic distance from campaign ({data['geo_distance_from_campaign']:.0f} km)")
        
        if data['is_donor_anonymous'] and data['amount'] > 500:
            reasons.append("Large anonymous donation")
        
        if not data['donor_comment'] or data['donor_comment'].strip() == "":
            reasons.append("Missing donor comment")
        
        if 'sentiment_score' in data and data['sentiment_score'] < -0.5:
            reasons.append("Negative sentiment in comment")
        
        if len(reasons) == 0:
            reasons.append("Combination of multiple suspicious factors")
        
        explanation = f"🚨 Potential Fraud Detected (Score: {score:.3f})\nReasons: " + "; ".join(reasons)
    else:
        explanation = f"✅ Legitimate Donation (Score: {score:.3f})"
    
    return explanation

def validate_donation_data(data: Dict[str, Any]) -> tuple:
    required_fields = [
        'amount', 'donation_time', 'donor_comment',
        'donation_frequency_from_ip', 'device_type', 
        'geo_distance_from_campaign', 'is_donor_anonymous', 
        'campaign_age'
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing fields: {', '.join(missing_fields)}"
    
    try:
        float(data['amount'])
        int(data['donation_frequency_from_ip'])
        float(data['geo_distance_from_campaign'])
        int(data['campaign_age'])
        bool(data['is_donor_anonymous'])
    except ValueError as e:
        return False, f"Invalid data types: {str(e)}"
    
    return True, "Valid data"