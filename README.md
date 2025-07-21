# RITH_score

📘 Radiomics-based RITH Score Analysis Pipeline
This repository provides a pipeline for calculating the Radiomics Intratumoral Heterogeneity (RITH) score.

├── Consensus_Clustering.py # Perform consensus clustering to define RITH phenotypes
├── Standard_Preprocess.py # Standardize radiomics features
├── Feature_Selection.py # Select group-specific features using PCA
├── RSES.py 
├── RITH_Score.py # Calculation of RITH score
├── data/
│ ├── train_scaled.xlsx
│ ├── feature_lists/
│ └── results/
