from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import yaml
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.predict import predict_fraud, detector
from utils.helpers import generate_explanation, validate_donation_data

# Load configuration
config_path = os.path.join(project_root, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class DonationData(BaseModel):
    amount: float
    donation_time: str
    donor_comment: str
    donation_frequency_from_ip: int
    device_type: str
    geo_distance_from_campaign: float
    is_donor_anonymous: bool
    campaign_age: int
    donation_id: Optional[str] = None

class PredictionResponse(BaseModel):
    donation_id: Optional[str]
    is_fraud: bool
    fraud_score: float
    explanation: str

app = FastAPI(title="Charity Fraud Detection API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    detector.load_model()

@app.get("/")
async def root():
    return {"message": "Charity Fraud Detection API", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(donation: DonationData):
    try:
        donation_data = donation.dict()
        
        is_valid, message = validate_donation_data(donation_data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        prediction_result = predict_fraud(donation_data)
        explanation = generate_explanation(prediction_result)
        
        response = PredictionResponse(
            donation_id=donation_data.get('donation_id'),
            is_fraud=prediction_result['is_fraud'],
            fraud_score=prediction_result['fraud_score'],
            explanation=explanation
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": detector.model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host=config['app']['host'], port=config['app']['port'])