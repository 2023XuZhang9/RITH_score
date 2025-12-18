# -*- coding: utf-8 -*-
"""
KMeans-only Consensus Clustering (outputs only, publication-friendly)

Stage 1: Scan K with fewer repeats to select best_k (PAC + CDF area)
Stage 2: Rebuild consensus matrix for best_k with higher repeats and export final labels

Outputs (in out_dir):
  - kmeans_consensus_metrics_scan.csv
  - kmeans_consensus_metrics_final_bestk.csv
  - Best_Consensus_Matrix_KMeans_k{best_k}.csv
  - KMeans_Consensus_Clustering_Results_k{best_k}.xlsx
  - KMeans_Labels_Only_k{best_k}.xlsx
  - Borderline_Samples_k{best_k}.xlsx
  - (optional) Cluster_vs_Meta_Crosstabs.xlsx
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency


# =========================
# Configuration
# =========================
warnings.filterwarnings("ignore")

SEED = 42
rng = np.random.default_rng(SEED)

file_path = r"F:\ITH\1_FIGURE_CODE\Radiomics_features_bincount_762.xlsx"
out_dir = Path(r"F:\ITH\1_FIGURE_CODE")
out_dir.mkdir(parents=True, exist_ok=True)

# Recommended to scan at least up to 10
n_clusters_list = list(range(2, 11))  # 2..10

# Subsampling ratios (samples + features)
sample_subsample_ratio = 0.8
feature_subsample_ratio = 0.8

# PAC thresholds (main + sensitivity)
PAC_LO_MAIN, PAC_HI_MAIN = 0.1, 0.9
PAC_LO_SENS, PAC_HI_SENS = 0.2, 0.8

# Two-stage repeats
n_repeats_scan = 1000    # scanning K
n_repeats_final = 1000   # final best_k

# CDF grid bins
CDF_BINS = 100

# Linkage method for deriving labels from consensus matrix
HCLUST_METHOD = "average"

# Assume the first column is the ID column
ID_COL_INDEX = 0

# If any of these columns exist, crosstabs will be exported
META_CANDIDATES = ["Protocol", "Center", "Scanner", "Site", "Hospital"]


# =========================
# Helper functions
# =========================
def consensus_kmeans(
    X: np.ndarray,
    k: int,
    n_repeats: int,
    sample_ratio: float,
    feature_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Standard consensus matrix definition:
      C_ij = (# times i,j are co-sampled AND assigned to same cluster) /
             (# times i,j are co-sampled)

    Key points:
      - Subsample both samples and features
      - Use co-occurrence counts for denominator normalization
      - Update numerator by cluster membership (avoid building full sub x sub matrices)
    """
    n, p = X.shape
    n_sub = max(k + 1, int(np.floor(sample_ratio * n)))
    p_sub = max(1, int(np.floor(feature_ratio * p)))

    # uint16 is sufficient for repeats <= ~1000 and saves memory
    same_sum = np.zeros((n, n), dtype=np.uint16)
    pair_cnt = np.zeros((n, n), dtype=np.uint16)

    for _ in range(n_repeats):
        idx = rng.choice(n, size=n_sub, replace=False)
        fidx = rng.choice(p, size=p_sub, replace=False)
        X_sub = X[np.ix_(idx, fidx)]

        rs = int(rng.integers(1, 2_000_000_000))
        km = KMeans(n_clusters=k, n_init="auto", random_state=rs)
        labels = km.fit_predict(X_sub)

        # Denominator: co-sampled pairs +1
        pair_cnt[np.ix_(idx, idx)] += 1

        # Numerator: same-cluster pairs +1 (by cluster)
        for g in range(k):
            sub = idx[labels == g]
            if sub.size <= 1:
                continue
            same_sum[np.ix_(sub, sub)] += 1

    C = np.zeros((n, n), dtype=np.float32)
    mask = pair_cnt > 0
    C[mask] = same_sum[mask].astype(np.float32) / pair_cnt[mask].astype(np.float32)
    np.fill_diagonal(C, 1.0)
    return C


def cdf_and_area(C: np.ndarray, n_bins: int = 100):
    """
    Estimate CDF on a fixed grid (0..1) and compute A(k) = ∫ CDF(x) dx.
    This approximates ConsensusClusterPlus-style CDF/area using a grid estimator.
    """
    vals = C[np.triu_indices_from(C, k=1)]
    vals = np.sort(vals)

    grid = np.linspace(0.0, 1.0, n_bins + 1)
    cdf = np.searchsorted(vals, grid, side="right") / vals.size
    area = float(np.trapz(cdf, grid))
    return grid, cdf, area


def pac(C: np.ndarray, lo: float, hi: float) -> float:
    """PAC: proportion of consensus values in (lo, hi). Lower indicates higher stability."""
    vals = C[np.triu_indices_from(C, k=1)]
    return float(np.mean((vals > lo) & (vals < hi)))


def labels_from_consensus(C: np.ndarray, k: int, method: str = "average"):
    """
    Derive final labels from consensus matrix:
      - Build D = 1 - C
      - Hierarchical clustering on D
      - Cut into k clusters
    """
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method=method)
    lab = fcluster(Z, t=k, criterion="maxclust") - 1  # 0..k-1
    return lab, Z


def item_consensus(C: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """For each sample, average consensus to other members in the same cluster (lower indicates borderline)."""
    labels = np.asarray(labels)
    n = len(labels)
    ic = np.zeros(n, dtype=float)
    for i in range(n):
        same = np.where(labels == labels[i])[0]
        same = same[same != i]
        ic[i] = float(np.mean(C[i, same])) if len(same) > 0 else 1.0
    return ic


def cluster_consensus(C: np.ndarray, labels: np.ndarray) -> dict:
    """Within-cluster average consensus for each cluster."""
    labels = np.asarray(labels)
    out = {}
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        if len(idx) <= 1:
            out[int(g)] = 1.0
            continue
        vals = C[np.ix_(idx, idx)]
        out[int(g)] = float(np.mean(vals[np.triu_indices_from(vals, k=1)]))
    return out


def cramers_v_from_table(ct: pd.DataFrame) -> float:
    """Compute Cramer's V from a contingency table."""
    chi2, p, dof, expected = chi2_contingency(ct.values)
    n = ct.values.sum()
    if n == 0:
        return np.nan
    r, c = ct.shape
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / denom))


# =========================
# Load data and build feature matrix
# =========================
data = pd.read_excel(file_path)

id_col = data.columns[ID_COL_INDEX]
patient_id = data[id_col].astype(str).values

meta_cols_found = [c for c in META_CANDIDATES if c in data.columns]
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c != id_col and c not in meta_cols_found]

if len(feature_cols) == 0:
    raise ValueError("No numeric feature columns found. Please check the Excel file.")

X = data[feature_cols].values.astype(np.float32)
X = StandardScaler().fit_transform(X).astype(np.float32)

n_samples, n_features = X.shape
print(f"[INFO] n_samples={n_samples}, n_features={n_features}, n_feature_cols={len(feature_cols)}")


# =========================
# Stage 1: Scan K (fewer repeats)
# =========================
print("[INFO] Stage 1: scanning K to select best_k ...")
consensus_mats_scan = {}
rows_scan = []

for k in n_clusters_list:
    print(f"[INFO] Building consensus matrix (scan) for k={k} ...")
    Ck = consensus_kmeans(
        X=X,
        k=k,
        n_repeats=n_repeats_scan,
        sample_ratio=sample_subsample_ratio,
        feature_ratio=feature_subsample_ratio,
        rng=rng,
    )
    consensus_mats_scan[k] = Ck

    _, _, area = cdf_and_area(Ck, n_bins=CDF_BINS)
    pac_main = pac(Ck, PAC_LO_MAIN, PAC_HI_MAIN)
    pac_sens = pac(Ck, PAC_LO_SENS, PAC_HI_SENS)
    mean_cons = float(np.mean(Ck[np.triu_indices_from(Ck, k=1)]))

    rows_scan.append({
        "k": k,
        "cdf_area": area,
        f"pac_{PAC_LO_MAIN}_{PAC_HI_MAIN}": pac_main,
        f"pac_{PAC_LO_SENS}_{PAC_HI_SENS}": pac_sens,
        "mean_consensus": mean_cons
    })

metrics_scan = pd.DataFrame(rows_scan).sort_values("k").reset_index(drop=True)

# Delta area: standard difference (k=2 will be NaN for delta_area)
metrics_scan["delta_area"] = metrics_scan["cdf_area"].diff()

# ConsensusClusterPlus-style "relative change":
# For k=2 use A(2), for k>2 use A(k)-A(k-1)
metrics_scan["delta_area_relative"] = metrics_scan["cdf_area"].diff()
metrics_scan.loc[metrics_scan.index[0], "delta_area_relative"] = metrics_scan.loc[metrics_scan.index[0], "cdf_area"]

metrics_scan_csv = out_dir / "kmeans_consensus_metrics_scan.csv"
metrics_scan.to_csv(metrics_scan_csv, index=False, encoding="utf-8-sig")
print(f"[INFO] Scan metrics saved: {metrics_scan_csv}")

pac_col_main = f"pac_{PAC_LO_MAIN}_{PAC_HI_MAIN}"
best_k = int(metrics_scan.sort_values([pac_col_main, "cdf_area"], ascending=[True, False]).iloc[0]["k"])
print(f"[INFO] Best k selected from scan: {best_k}")


# =========================
# Stage 2: Rebuild consensus matrix for best_k (more repeats)
# =========================
print("[INFO] Stage 2: rebuilding consensus matrix for best_k with higher repeats ...")

# Reset RNG for reproducibility of the final matrix independent of scan consumption
rng_final = np.random.default_rng(SEED)

C_best = consensus_kmeans(
    X=X,
    k=best_k,
    n_repeats=n_repeats_final,
    sample_ratio=sample_subsample_ratio,
    feature_ratio=feature_subsample_ratio,
    rng=rng_final,
)

# Save best consensus matrix
cm_csv = out_dir / f"Best_Consensus_Matrix_KMeans_k{best_k}.csv"
np.savetxt(cm_csv, C_best, delimiter=",")
print(f"[INFO] Best consensus matrix saved: {cm_csv}")

# Final metrics for best_k
_, _, area_best = cdf_and_area(C_best, n_bins=CDF_BINS)
pac_best_main = pac(C_best, PAC_LO_MAIN, PAC_HI_MAIN)
pac_best_sens = pac(C_best, PAC_LO_SENS, PAC_HI_SENS)
mean_cons_best = float(np.mean(C_best[np.triu_indices_from(C_best, k=1)]))

metrics_best = pd.DataFrame([{
    "k": best_k,
    "cdf_area": area_best,
    f"pac_{PAC_LO_MAIN}_{PAC_HI_MAIN}": pac_best_main,
    f"pac_{PAC_LO_SENS}_{PAC_HI_SENS}": pac_best_sens,
    "mean_consensus": mean_cons_best,
    "n_repeats_final": n_repeats_final,
    "sample_ratio": sample_subsample_ratio,
    "feature_ratio": feature_subsample_ratio,
    "cdf_bins": CDF_BINS,
    "hclust_method": HCLUST_METHOD
}])

metrics_best_csv = out_dir / "kmeans_consensus_metrics_final_bestk.csv"
metrics_best.to_csv(metrics_best_csv, index=False, encoding="utf-8-sig")
print(f"[INFO] Final best_k metrics saved: {metrics_best_csv}")


# =========================
# Final labels: consensus-derived (recommended) + direct KMeans on full features (reference)
# =========================
lab_consensus, _ = labels_from_consensus(C_best, best_k, method=HCLUST_METHOD)

km_full = KMeans(n_clusters=best_k, n_init="auto", random_state=SEED)
lab_km_full = km_full.fit_predict(X)

ari = adjusted_rand_score(lab_consensus, lab_km_full)
print(f"[INFO] ARI(consensus-derived vs direct full-feature KMeans) = {ari:.4f}")

ic = item_consensus(C_best, lab_consensus)
cc = cluster_consensus(C_best, lab_consensus)
for g, v in cc.items():
    print(f"[INFO] Cluster {g} cluster-consensus = {v:.4f}")


# =========================
# Export result tables
# =========================
out_df = data.copy()
out_df["Cluster_consensus"] = lab_consensus
out_df["Cluster_kmeans_full"] = lab_km_full
out_df["ItemConsensus"] = ic

cc_df = pd.DataFrame({"Cluster": list(cc.keys()), "ClusterConsensus": list(cc.values())})

out_xlsx = out_dir / f"KMeans_Consensus_Clustering_Results_k{best_k}.xlsx"
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
    out_df.to_excel(w, index=False, sheet_name="AllData_withClusters")
    cc_df.to_excel(w, index=False, sheet_name="ClusterConsensus")
    metrics_scan.to_excel(w, index=False, sheet_name="ScanMetrics")
    metrics_best.to_excel(w, index=False, sheet_name="FinalBestKMetrics")
print(f"[INFO] Results workbook saved: {out_xlsx}")

label_df = pd.DataFrame({
    "PatientID": patient_id,
    "Cluster_consensus": lab_consensus,
    "Cluster_kmeans_full": lab_km_full,
    "ItemConsensus": ic
})
label_xlsx = out_dir / f"KMeans_Labels_Only_k{best_k}.xlsx"
label_df.to_excel(label_xlsx, index=False)
print(f"[INFO] Labels saved: {label_xlsx}")

border_df = label_df.sort_values("ItemConsensus", ascending=True)
border_xlsx = out_dir / f"Borderline_Samples_k{best_k}.xlsx"
border_df.to_excel(border_xlsx, index=False)
print(f"[INFO] Borderline samples saved: {border_xlsx}")


# =========================
# Optional: association between cluster and metadata (Protocol/Center/etc.)
# =========================
meta_cols_found = [c for c in META_CANDIDATES if c in data.columns]
if len(meta_cols_found) > 0:
    print(f"[INFO] Meta columns found: {meta_cols_found}")
    crosstab_path = out_dir / "Cluster_vs_Meta_Crosstabs.xlsx"

    with pd.ExcelWriter(crosstab_path, engine="openpyxl") as w:
        for mc in meta_cols_found:
            tmp = out_df[[mc, "Cluster_consensus"]].dropna()
            ct = pd.crosstab(tmp[mc].astype(str), tmp["Cluster_consensus"])
            ct.to_excel(w, sheet_name=f"{mc}_crosstab")

            chi2, p, dof, _ = chi2_contingency(ct.values)
            v = cramers_v_from_table(ct)

            stat = pd.DataFrame([{
                "MetaColumn": mc,
                "Chi2": float(chi2),
                "dof": int(dof),
                "p_value": float(p),
                "CramersV": float(v),
                "n_used": int(ct.values.sum())
            }])
            stat.to_excel(w, sheet_name=f"{mc}_stats", index=False)

    print(f"[INFO] Cluster vs meta crosstabs saved: {crosstab_path}")
else:
    print("[INFO] No metadata columns found. Crosstab export skipped.")

print("[DONE] All outputs are saved in:", out_dir)
