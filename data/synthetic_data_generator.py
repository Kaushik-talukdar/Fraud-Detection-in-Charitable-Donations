import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
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

def generate_synthetic_data(num_samples=25000, fraud_ratio=0.05):
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create directories if they don't exist
    output_dir = os.path.dirname(config['data']['output_file'])
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    fraud_count = 0
    
    # Generate campaign start dates
    campaign_start_dates = [fake.date_between(start_date='-365d', end_date='-30d') for _ in range(50)]
    
    print(f"Generating {num_samples} donations with {fraud_ratio*100}% fraud rate...")
    
    for i in range(num_samples):
        if (i + 1) % 5000 == 0:
            print(f"Generated {i + 1} samples so far...")
            
        donation_id = f"don_{i:06d}"
        amount = round(np.random.lognormal(mean=3.5, sigma=1.2), 2)
        
        donation_time = fake.date_time_between(start_date='-30d', end_date='now')
        
        if random.random() < 0.8:
            comments = [
                "Great cause! Happy to help.", "Hope this makes a difference.",
                "Keep up the good work!", "For a better tomorrow."
            ]
            donor_comment = random.choice(comments)
        else:
            donor_comment = ""
        
        donation_frequency_from_ip = np.random.poisson(lam=1.5)
        device_type = random.choice(["desktop", "mobile", "tablet"])
        geo_distance_from_campaign = np.random.exponential(scale=500)
        is_donor_anonymous = random.random() < 0.3
        
        campaign_start = random.choice(campaign_start_dates)
        campaign_age = (donation_time.date() - campaign_start).days
        
        # Determine if fraud
        is_fraud = 0
        fraud_probability = 0.0
        
        if donation_frequency_from_ip > 8:
            fraud_probability += 0.6
        elif donation_frequency_from_ip > 5:
            fraud_probability += 0.4
            
        if geo_distance_from_campaign > 5000:
            fraud_probability += 0.7
        elif geo_distance_from_campaign > 2000:
            fraud_probability += 0.5
            
        if is_donor_anonymous and amount > 1000:
            fraud_probability += 0.8
        elif is_donor_anonymous and amount > 500:
            fraud_probability += 0.6
            
        if not donor_comment and (donation_frequency_from_ip > 3 or geo_distance_from_campaign > 1000):
            fraud_probability += 0.4
            
        if amount > 2000:
            fraud_probability += 0.5
            
        fraud_probability += random.uniform(-0.1, 0.1)
        fraud_probability = max(0, min(1, fraud_probability))
        
        if random.random() < fraud_probability:
            is_fraud = 1
            fraud_count += 1
        
        data.append([
            donation_id, amount, donation_time, donor_comment, 
            donation_frequency_from_ip, device_type, geo_distance_from_campaign,
            is_donor_anonymous, campaign_age, is_fraud
        ])
    
    columns = [
        "donation_id", "amount", "donation_time", "donor_comment",
        "donation_frequency_from_ip", "device_type", "geo_distance_from_campaign",
        "is_donor_anonymous", "campaign_age", "label"
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    print(f"\nGenerated {len(df)} samples with {fraud_count} fraudulent donations ({fraud_count/len(df)*100:.2f}%)")
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(
        num_samples=config['data']['synthetic_samples'],
        fraud_ratio=config['data']['fraud_ratio']
    )
    output_path = os.path.join(project_root, config['data']['output_file'])
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")