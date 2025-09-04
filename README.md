# 🤝 Fraud Detection in Charitable Donations
A machine learning system that detects fraudulent donations made with stolen credit cards to legitimate charitable campaigns. This system helps NGOs prevent financial losses from chargebacks and protect their reputation.

## 🚀 Features
Synthetic Data Generation: Creates realistic donation data with 5% fraud rate

Real-time Fraud Detection: Analyzes donations in real-time using ML models

Explainable AI: Provides clear reasons for fraud flags

Web Interface: User-friendly dashboard for NGO staff

RESTful API: Programmatic access for integration

Anomaly Detection: Uses Isolation Forest for imbalanced data

## 📊 Dataset Overview
Total Samples: 25,000 synthetic donations

Fraud Rate: 5% (1,250 fraudulent donations)

Features: Amount, geographic distance, donation frequency, device type, sentiment analysis, and more

Fraud Patterns Detected:
High frequency donations from same IP

Geographic anomalies (long distances)

Large anonymous donations

Missing or negative comments

Unusual donation amounts

## 🛠️ Tech Stack
### Backend:
Python 3.8+

FastAPI - High-performance API framework

Scikit-learn - Machine learning algorithms

Isolation Forest - Anomaly detection model

### Frontend:
Streamlit - Web application interface

Plotly - Data visualization

### Data Processing:
Pandas - Data manipulation

Numpy - Numerical computing

NLTK - Sentiment analysis

### Utilities:
Faker - Synthetic data generation

SHAP/LIME - Model interpretability

Joblib - Model persistence

## 📁 Project Structure
text
fraud-detection-charity/
├── data/
│   ├── synthetic_data_generator.py  # Creates synthetic donation data
│   ├── raw/                         # Raw CSV data storage
│   └── processed/                   # Processed data storage
├── models/
│   ├── train_model.py               # ML model training
│   ├── predict.py                   # Fraud prediction logic
│   └── saved_models/                # Trained model storage
├── app/
│   ├── main.py                      # FastAPI backend server
│   └── frontend.py                  # Streamlit web interface
├── utils/
│   ├── preprocess.py                # Data preprocessing
│   └── helpers.py                   # Utility functions
├── config.yaml                      # Configuration settings
├── requirements.txt                 # Python dependencies
├── run.py                           # Main execution script
└── README.md                        # This file
## ⚙️ Installation
Prerequisites:
Python 3.8 or higher

pip package manager

Step 1: Clone and Setup
bash


# Install required packages
pip install -r requirements.txt

# Download NLTK data (for sentiment analysis)
python -c "import nltk; nltk.download('vader_lexicon')"
🚀 Quick Start
Method 1: Using the Menu System (Recommended)
bash
python run.py
Then follow the menu prompts to:

Generate synthetic data

Train the model

Start API server

Launch web interface

Method 2: Manual Execution
bash
# 1. Generate synthetic data (25,000 samples)
python data/synthetic_data_generator.py

# 2. Train the machine learning model
python models/train_model.py

# 3. Start the API server (Terminal 1)
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 4. Start the web interface (Terminal 2)
streamlit run app/frontend.py
🌐 Access Points
Web Application: http://localhost:8501

API Documentation: http://localhost:8000/docs

API Health Check: http://localhost:8000/health

#### 🧪 Testing the System
Example Legitimate Donation:
json
{
  "amount": 50.00,
  "donation_time": "2024-01-15 14:30:00",
  "donor_comment": "Happy to support this cause!",
  "donation_frequency_from_ip": 1,
  "device_type": "desktop",
  "geo_distance_from_campaign": 100.5,
  "is_donor_anonymous": false,
  "campaign_age": 30
}
Example Fraudulent Donation:
json
{
  "amount": 500.00,
  "donation_time": "2024-01-15 15:05:00",
  "donor_comment": "",
  "donation_frequency_from_ip": 15,
  "device_type": "desktop",
  "geo_distance_from_campaign": 5000.8,
  "is_donor_anonymous": true,
  "campaign_age": 25
}
## 📊 Model Performance
The Isolation Forest model typically achieves:

Precision: 85-90% for fraud detection

Recall: 70-75% for fraud detection

ROC AUC: 0.85-0.90

F1-Score: 0.75-0.80 for fraud class

#### ⚙️ Configuration
Edit config.yaml to customize:

yaml
data:
  synthetic_samples: 25000    # Number of samples to generate
  fraud_ratio: 0.05           # Fraud percentage (5%)
  
model:
  algorithm: "isolation_forest"  # Options: isolation_forest, lof
  test_size: 0.2                 # Validation split size

app:
  host: "127.0.0.1"           # API host address
  port: 8000                   # API port
  frontend_port: 8501          # Web interface port
#### 🎯 Usage Examples
Single Donation Check:
Open web interface at http://localhost:8501

Select "Single Donation"

Fill in donation details

Get instant fraud assessment with explanations

Batch Processing:
Upload CSV file with multiple donations

Process all donations in batch

Download results with fraud scores

API Integration:
python
import requests

api_url = "http://localhost:8000/predict"
donation_data = {
    "amount": 100.00,
    "donation_time": "2024-01-15 10:00:00",
    # ... other fields
}

response = requests.post(api_url, json=donation_data)
result = response.json()
print(f"Fraud: {result['is_fraud']}, Score: {result['fraud_score']}")
#### 🔍 How It Works
Data Generation: Synthetic data mimics real donation patterns with embedded fraud signals

Preprocessing:

Sentiment analysis on donor comments

Feature scaling and encoding

Anomaly detection preparation

Model Training: Isolation Forest learns normal donation patterns

Prediction: Real-time scoring of new donations

Explanation: SHAP-based reasoning for fraud flags

#### 🚨 Fraud Detection Patterns
The system flags donations based on:

⚡ Frequency Patterns: Multiple donations from same IP in short time

🌍 Geographic Anomalies: Donations from unusual locations

🕵️ Anonymous Large Gifts: Big amounts from unknown donors

💬 Comment Analysis: Missing or negative sentiments

💰 Amount Suspiciousness: Unusually large donations

#### 📈 Performance Considerations
Data Generation: 2-3 minutes for 25,000 samples

Model Training: 3-5 minutes on standard hardware

Prediction Speed: <100ms per donation

Memory Usage: ~50-100MB for full operation

#### 🤝 Contributing
To extend this project:

Add new fraud detection features

Improve model interpretability

Enhance web interface

Add database integration

Implement real-time monitoring



## 🎯 Future Enhancements
Real payment processor integration

Database persistence

Advanced visualization dashboards

Email alerts for fraud detection

Multi-language support

Mobile application

### ⭐ Star this project if you found it helpful!

### Helping NGOs protect their missions from fraudulent activities 🤝
