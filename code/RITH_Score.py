import pandas as pd
from RSES import rsea
import numpy as np

def calculate_train_scores(scaled_path, feature_dir, output_path):
    """
    Calculate RITH-related scores for the training set.
    
    Parameters:
        scaled_path (str): Path to the standardized training data.
        feature_dir (str): Directory containing selected feature lists.
        output_path (str): Path to save the output Excel file.
    """
    # Load standardized training data
    df = pd.read_excel(scaled_path)
    features = df.columns[2:].tolist()  # Skip PatientID and Cluster columns
    X = df[features].values
    
    # Load high and low feature lists
    high_features = pd.read_excel(f"{feature_dir}/high_features.xlsx")["High_Features"].tolist()
    low_features = pd.read_excel(f"{feature_dir}/low_features.xlsx")["Low_Features"].tolist()
    
    # Compute scores
    scores_df = rsea(X, np.array(features), high_features, low_features)
    
    # Combine with original metadata
    result_df = pd.concat([df[["PatientID", "Cluster"]], scores_df], axis=1)
    result_df.to_excel(output_path, index=False)
    print(f"Training scores saved to {output_path}")

if __name__ == "__main__":
    calculate_train_scores(
        "xxx/train_scaled.xlsx",
        "xxx/feature_lists",
        "xxx/results/train_scores.xlsx"
    )
