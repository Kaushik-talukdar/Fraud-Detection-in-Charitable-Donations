import os
import sys
import subprocess

def main():
    print("Fraud Detection System - Main Menu")
    print("1. Generate synthetic data (25,000 samples)")
    print("2. Train machine learning model")
    print("3. Start API server")
    print("4. Start web interface")
    print("5. Run all steps")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == "1":
        print("Generating synthetic data...")
        subprocess.run([sys.executable, "data/synthetic_data_generator.py"])
        
    elif choice == "2":
        print("Training model...")
        subprocess.run([sys.executable, "models/train_model.py"])
        
    elif choice == "3":
        print("Starting API server...")
        subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"])
        
    elif choice == "4":
        print("Starting web interface...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/frontend.py"])
        
    elif choice == "5":
        print("Running all steps...")
        subprocess.run([sys.executable, "data/synthetic_data_generator.py"])
        subprocess.run([sys.executable, "models/train_model.py"])
        print("Please start API server and web interface separately")
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()