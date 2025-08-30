import streamlit as st
import pandas as pd
import requests
import json
import yaml
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load configuration
config_path = os.path.join(project_root, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="Charity Fraud Detection", page_icon="🤝", layout="wide")

st.title("🤝 Charity Fraud Detection System")
st.markdown("Detect fraudulent donations made with stolen credit cards to legitimate campaigns.")

if 'predictions' not in st.session_state:
    st.session_state.predictions = []

API_URL = f"http://{config['app']['host']}:{config['app']['port']}/predict"

st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose input method:", ("Single Donation", "Batch Processing (CSV)"))

if option == "Single Donation":
    st.header("Check a Single Donation")
    
    with st.form("donation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Donation Amount ($)", min_value=0.0, value=50.0, step=1.0)
            donation_time = st.text_input("Donation Time", value="2023-08-15 14:30:00")
            donor_comment = st.text_area("Donor Comment", value="Happy to support this cause!")
            donation_frequency = st.number_input("Donations from this IP (last hour)", min_value=0, value=1, step=1)
        
        with col2:
            device_type = st.selectbox("Device Type", ["desktop", "mobile", "tablet"])
            geo_distance = st.number_input("Distance from Campaign (km)", min_value=0.0, value=100.0, step=10.0)
            is_anonymous = st.checkbox("Anonymous Donor")
            campaign_age = st.number_input("Campaign Age (days)", min_value=0, value=30, step=1)
        
        submitted = st.form_submit_button("Check for Fraud")
        
        if submitted:
            donation_data = {
                "amount": amount, "donation_time": donation_time, "donor_comment": donor_comment,
                "donation_frequency_from_ip": donation_frequency, "device_type": device_type,
                "geo_distance_from_campaign": geo_distance, "is_donor_anonymous": is_anonymous,
                "campaign_age": campaign_age
            }
            
            try:
                response = requests.post(API_URL, json=donation_data)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.predictions.append(result)
                    
                    st.subheader("Result")
                    if result['is_fraud']:
                        st.error(f"🚨 Potential Fraud Detected (Score: {result['fraud_score']:.3f})")
                    else:
                        st.success(f"✅ Legitimate Donation (Score: {result['fraud_score']:.3f})")
                    
                    st.info(result['explanation'])
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

else:
    st.header("Batch Process Donations from CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Process All Donations"):
                results = []
                progress_bar = st.progress(0)
                
                for i, row in df.iterrows():
                    donation_data = row.to_dict()
                    try:
                        response = requests.post(API_URL, json=donation_data)
                        if response.status_code == 200:
                            result = response.json()
                            result['donation_id'] = donation_data.get('donation_id', f"row_{i}")
                            results.append(result)
                    except Exception as e:
                        st.error(f"Error processing row {i}: {str(e)}")
                    progress_bar.progress((i + 1) / len(df))
                
                if results:
                    st.subheader("Batch Processing Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    fraud_count = results_df['is_fraud'].sum()
                    st.metric("Fraudulent Donations Detected", fraud_count)
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

if st.session_state.predictions:
    st.sidebar.header("Prediction History")
    for i, pred in enumerate(st.session_state.predictions[-10:]):
        status = "🚨 Fraud" if pred['is_fraud'] else "✅ Legit"
        st.sidebar.text(f"{i+1}. {status} (Score: {pred['fraud_score']:.3f})")
    
    if st.sidebar.button("Clear History"):
        st.session_state.predictions = []
        st.experimental_rerun()

st.markdown("---")
st.markdown("**Note**: This is a demonstration system using synthetic data.")