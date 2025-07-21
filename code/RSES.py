import numpy as np
import pandas as pd

def rsea(X, feature_names, high_features, low_features):
    """
    Core scoring function using ssgsea-like method.
    
    Parameters:
        X: Standardized feature matrix of shape (n_samples, n_features)
        feature_names: List or array of feature names
        high_features: List of high group feature names
        low_features: List of low group feature names
    
    Returns:
        result_df: DataFrame containing normalized scores for ITH_High, ITH_Low, and RITH_score
    """
    # Convert feature names to index positions
    high_idx = [np.where(feature_names == f)[0][0] for f in high_features]
    low_idx = [np.where(feature_names == f)[0][0] for f in low_features]
    
    # Scoring function for a single sample
    def _calculate(sample, features):
        sorted_indices = np.argsort(sample)[::-1]  # Rank features by descending value
        rank_weights = np.arange(len(sample), 0, -1)  # Higher rank gets higher weight
        hit = np.isin(sorted_indices, features).astype(float)  # 1 if feature is in selected set
        phit = np.cumsum(hit * rank_weights)  # Running sum of hits
        pmiss = np.cumsum((1 - hit) / (len(sample) - len(features)))  # Running sum of misses
        es = phit - pmiss  # Enrichment score
        return es.max() if es.max() > 0 else es.min()  # Use max or min deviation from zero

    # Apply to all samples
    high_scores = np.array([_calculate(s, high_idx) for s in X])
    low_scores = np.array([_calculate(s, low_idx) for s in X])
    
    # Normalize scores by maximum absolute value across all
    max_abs = np.maximum(np.abs(high_scores).max(), np.abs(low_scores).max())
    high_scores /= max_abs
    low_scores /= max_abs
    rith_scores = high_scores - low_scores
    
    return pd.DataFrame({
        "ITH_High": high_scores,
        "ITH_Low": low_scores,
        "RITH_score": rith_scores
    })
