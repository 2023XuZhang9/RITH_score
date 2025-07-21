import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

def preprocess_train(input_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_excel(input_path)
    
    # Assume the first column is Patient ID, the rest are features
    features = df.columns[1:]
    X = df[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Combine ID with scaled features
    scaled_df = pd.DataFrame(X_scaled, columns=features)
    scaled_df.insert(0, df.columns[0], df.iloc[:, 0])  # Insert Patient ID as first column
    
    # Save scaled data and scaler
    scaled_df.to_excel(f"{output_dir}/train_scaled.xlsx", index=False)
    dump(scaler, f"{output_dir}/scaler.joblib")
    
    print(f"Preprocessing completed. Results saved to {output_dir}")

if __name__ == "__main__":
    preprocess_train("xxx/your_file.xlsx", "xxx/output_dir")
