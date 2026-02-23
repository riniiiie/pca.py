# =====================================
# PCA — DIMENSIONALITY REDUCTION
# =====================================

# -------------------------------
# Step 1: Upload Dataset
# -------------------------------
from google.colab import files
uploaded = files.upload()

# -------------------------------
# Step 2: Import Libraries
# -------------------------------
import pandas as pd
import numpy as np

# -------------------------------
# Step 3: Load Dataset
# -------------------------------
df = pd.read_csv(list(uploaded.keys())[0])

# --- Start of fix ---
# Convert categorical columns to numeric using one-hot encoding
df_numeric = pd.get_dummies(df)

# Ensure all columns are numerical before converting to numpy array
# Convert boolean columns to int (0 or 1) and other types to float
data = df_numeric.astype(float).values
# --- End of fix ---

print("\nSTEP 1: DATASET")
print(data)

# -------------------------------
# Step 4: Standardization
# -------------------------------
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

# --- Start of fix ---
# Handle potential division by zero for columns with zero standard deviation (e.g., constant columns)
std[std == 0] = 1 # Replace 0 with 1 to avoid ZeroDivisionError, effectively making mean=0 for such columns.
# --- End of fix ---

standardized_data = (data - mean) / std

print("\nSTEP 2: STANDARDIZED DATA")
print(standardized_data)

# -------------------------------
# Step 5: Covariance Matrix
# -------------------------------
cov_matrix = np.cov(standardized_data.T)

print("\nSTEP 3: COVARIANCE MATRIX")
print(cov_matrix)

# -------------------------------
# Step 6: Eigen Values & Vectors
# -------------------------------
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("\nSTEP 4: EIGEN VALUES")
print(eigen_values)

print("\nSTEP 4: EIGEN VECTORS")
print(eigen_vectors)

# -------------------------------
# Step 7: PCA (Top 2 Components)
# -------------------------------
idx = np.argsort(eigen_values)[::-1]
top_2_vectors = eigen_vectors[:, idx[:2]]

pca_result = standardized_data.dot(top_2_vectors)

print("\nSTEP 5: PCA RESULT (2 COMPONENTS)")
print(pca_result)

print("\nPCA EXPERIMENT COMPLETED")

Kaggle link-https://www.kaggle.com/datasets/sehaj1104/student-productivity-and-digital-distraction-dataset
