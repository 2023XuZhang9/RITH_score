import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

def group_specific_pca(X, labels, group, n_components=15, top_n=100):
    """
    Perform group-specific PCA and return top contributing features.
    
    Parameters:
        X (ndarray): Feature matrix.
        labels (ndarray): Cluster labels (0 or 1).
        group (int): Group label to extract (0 or 1).
        n_components (int): Number of PCA components.
        top_n (int): Number of top features to return.
    
    Returns:
        indices (ndarray): Indices of selected features.
        pca (PCA object): Fitted PCA object.
    """
    mask = (labels == group)
    X_group = X[mask]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_group)

    loadings = np.abs(pca.components_).sum(axis=0)
    return np.argsort(loadings)[-top_n:], pca

def save_feature_lists(features, high_idx, low_idx, output_dir):
    """
    Save selected feature names for each group to Excel.
    
    Parameters:
        features (Index): List of feature names.
        high_idx (ndarray): Indices of features for high group.
        low_idx (ndarray): Indices of features for low group.
        output_dir (str): Path to save output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    high_features = [features[i] for i in high_idx]
    low_features = [features[i] for i in low_idx]

    pd.DataFrame({"High_Features": high_features}).to_excel(
        f"{output_dir}/high_features.xlsx", index=False)
    pd.DataFrame({"Low_Features": low_features}).to_excel(
        f"{output_dir}/low_features.xlsx", index=False)

def train_feature_selection(input_path, output_dir):
    """
    Main function to perform PCA-based feature selection by group.
    
    Parameters:
        input_path (str): Path to input Excel file.
        output_dir (str): Directory to save selected features.
    """
    df = pd.read_excel(input_path)
    features = df.columns[2:]  # Skip ID and Cluster columns
    X = df[features].values
    labels = df["Cluster"].values

    high_idx, high_pca = group_specific_pca(X, labels, 1)
    low_idx, low_pca = group_specific_pca(X, labels, 0)

    save_feature_lists(features, high_idx, low_idx, output_dir)

    print(f"Feature selection completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    input_path = "xxx/train_scaled.xlsx"
    output_dir = "xxx/feature_lists"
    train_feature_selection(input_path, output_dir)
