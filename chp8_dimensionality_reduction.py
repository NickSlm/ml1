
import numpy as np
from sklearn.decomposition import PCA

# ===========================================================================
# Building 3D dataset
# ===========================================================================

np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))

X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) + 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:,0] * w1 + X[:,1] * w2 + noise * np.random.randn(m)

# ===========================================================================
# PCA 
# Steps:
# - Center the dataset
# - Calculate covariance matrix using x_variance, y_variance and covariance values
# - Calculate eigenvectors and eigenvalues using the covariance
# - Get rid of the smallest  eigenvalues that you get, and keep whatever you need.
# - Plot the eigenvector with eigenvalues on our dataset and project the data onto the new plane. 
# ===========================================================================

# Manual
# X_centered = X - X.mean(axis=0)
# U, s, Vt = np.linalg.svd(X_centered)

# W2 = Vt[:,:2]
# X2D = X_centered.dot(W2)
# print(X2D)

# Using Scikit
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)
