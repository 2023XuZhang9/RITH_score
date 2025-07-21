import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker

# === Clustering hyperparameters (customize as needed) ===
xxx = 2024     # Controls random seed for reproducibility
yyy = 1e-5     # Prevents singular covariance in Gaussian Mixture Models
zzz = 0.6      # eps parameter for DBSCAN (defines neighborhood radius)
www = 6        # min_samples parameter for DBSCAN
nnn = 100      # Number of repeats per clustering method (affects consensus robustness)
ppp = 50       # Number of repeats per cluster count (affects stability across k)

# Fix random seed for reproducibility
np.random.seed(xxx)  # xxx = seed value

# Load data
file_path = r"xxx/feature_data.xlsx"
data = pd.read_excel(file_path)

# Extract feature columns (assume the first column is Patient ID)
features = data.iloc[:, 1:].values
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Define multiple clustering methods with placeholder parameters
cluster_methods = {
    'KMeans': lambda n_clusters: KMeans(n_clusters=n_clusters, n_init='auto', random_state=xxx),
    'Agglomerative': lambda n_clusters: AgglomerativeClustering(n_clusters=n_clusters),
    'Spectral': lambda n_clusters: SpectralClustering(n_clusters=n_clusters, random_state=xxx, affinity='nearest_neighbors'),
    'GaussianMixture': lambda n_clusters: GaussianMixture(n_components=n_clusters, random_state=xxx, reg_covar=yyy),
    'DBSCAN': lambda _: DBSCAN(eps=zzz, min_samples=www),  # zzz = eps value, www = min_samples value
}

# Function to generate consensus matrix
def generate_consensus_matrix(features, n_clusters, n_repeats=nnn):  # nnn = number of repeats
    n_samples = features.shape[0]
    consensus_matrix = np.zeros((n_samples, n_samples))

    for method_name, method_func in cluster_methods.items():
        for _ in range(n_repeats):
            sampled_features = features[:, np.random.choice(features.shape[1], int(0.8 * features.shape[1]), replace=False)]

            # Perform clustering
            if method_name == 'DBSCAN':
                model = method_func(None)
                labels = model.fit_predict(sampled_features)
            else:
                model = method_func(n_clusters)
                labels = model.fit_predict(sampled_features)

            valid_indices = labels != -1
            labels = labels[valid_indices]

            # Update consensus matrix
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if labels[i] == labels[j]:
                        consensus_matrix[i, j] += 1

    consensus_matrix /= (len(cluster_methods) * n_repeats)
    return consensus_matrix

# Set clustering range and repeat times
n_clusters_list = range(2, 6) 
consensus_matrices = {}
average_consensus_scores = []

# Function to compute consensus score
def calculate_consensus_score(matrix):
    return np.mean(matrix[np.triu_indices_from(matrix, k=1)])

# Run clustering for each cluster number
for n_clusters in n_clusters_list:
    consensus_matrix = generate_consensus_matrix(features, n_clusters, n_repeats=ppp)  # ppp = number of repetitions
    consensus_matrices[n_clusters] = consensus_matrix
    avg_score = calculate_consensus_score(consensus_matrix)
    average_consensus_scores.append(avg_score)

# Plot elbow curve
plt.figure(figsize=(8, 6))
plt.plot(n_clusters_list, average_consensus_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Consensus Score')
plt.title('Delta area')
plt.show()

# Determine best number of clusters
best_n_clusters = n_clusters_list[np.argmax(average_consensus_scores)]
print(f"Best number of clusters: {best_n_clusters}")

# Plot CDF curve
plt.figure(figsize=(8, 6))
for n_clusters in n_clusters_list:
    consensus_matrix = consensus_matrices[n_clusters]
    upper_triangle_values = consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)]
    sorted_values = np.sort(upper_triangle_values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    plt.plot(sorted_values, cdf, label=f'{n_clusters}')

plt.xlabel('Consensus Index')
plt.ylabel('CDF')
plt.title('Consensus CDF Curve')
plt.legend()
plt.show()

# Confirm best cluster number again
best_consensus_matrix = consensus_matrices[best_n_clusters]

# Save best consensus matrix
best_consensus_matrix_file = r"xxx/best_consensus_matrix.csv"
np.savetxt(best_consensus_matrix_file, best_consensus_matrix, delimiter=",")
print(f"Best consensus matrix saved to {best_consensus_matrix_file}")

# Visualize best consensus matrix
sns.heatmap(best_consensus_matrix, cmap="Blues")
plt.title(f"Consensus Matrix for {best_n_clusters} Clusters")
plt.show()

# Final clustering
final_model = KMeans(n_clusters=best_n_clusters, random_state=xxx)
group_labels = final_model.fit_predict(features)
data['Cluster'] = group_labels

# Save processed features and clusters
output_data_path = r"xxx/umap_processed_data.xlsx"
processed_data = {
    'Patient_ID': data.iloc[:, 0],
    'Features': [list(row) for row in features],
    'Cluster_Labels': group_labels
}
pd.DataFrame(processed_data).to_excel(output_data_path, index=False)
print(f"Processed data saved to {output_data_path}")

# Save final result
output_file = r"xxx/umap_consensus_clustering_results.xlsx"
data.to_excel(output_file, index=False)
print(f"Clustering results saved to {output_file}")
