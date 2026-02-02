import numpy as np

print("PCA LAB EXPERIMENT (COMBINED PROGRAM)")
print("=" * 50)

# -------------------------------
# STEP 1: DATASET
# -------------------------------
data = np.array([
    [160, 55, 22],
    [165, 60, 24],
    [170, 65, 26],
    [175, 70, 28],
    [180, 75, 30]
])

print("\nSTEP 1: DATASET")
print(data)

# -------------------------------
# STEP 2: STANDARDIZATION
# -------------------------------
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized_data = (data - mean) / std

print("\nSTEP 2: STANDARDIZED DATA")
print(standardized_data)

# -------------------------------
# STEP 3: COVARIANCE MATRIX
# -------------------------------
cov_matrix = np.cov(standardized_data.T)

print("\nSTEP 3: COVARIANCE MATRIX")
print(cov_matrix)

# -------------------------------
# STEP 4: EIGEN VALUES & VECTORS
# -------------------------------
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("\nSTEP 4: EIGEN VALUES")
print(eigen_values)

print("\nSTEP 4: EIGEN VECTORS")
print(eigen_vectors)

# -------------------------------
# STEP 5: PCA (2 COMPONENTS)
# -------------------------------
idx = np.argsort(eigen_values)[::-1]
top_2_vectors = eigen_vectors[:, idx[:2]]

pca_result = standardized_data.dot(top_2_vectors)

print("\nSTEP 5: PCA RESULT (2 COMPONENTS)")
print(pca_result)

print("\nPCA EXPERIMENT COMPLETED")
