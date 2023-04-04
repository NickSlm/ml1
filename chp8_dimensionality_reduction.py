
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.datasets import fetch_openml, make_swiss_roll
from sklearn.model_selection import train_test_split

# ===========================================================================
# Loading MNIST dataset
# ===========================================================================

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)


# ===========================================================================
# Building a random 3D dataset
# ===========================================================================

# np.random.seed(4)
# m = 60
# w1, w2 = 0.1, 0.3
# noise = 0.1

# angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
# X = np.empty((m, 3))

# X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
# X[:, 1] = np.sin(angles) + 0.7 + noise * np.random.randn(m) / 2
# X[:, 2] = X[:,0] * w1 + X[:,1] * w2 + noise * np.random.randn(m)

# ===========================================================================
# PCA 
# Steps:
# - Center the dataset
# - Calculate covariance matrix using x_variance, y_variance and covariance values
# - Calculate eigenvectors and eigenvalues using the covariance
# - Get rid of the smallest  eigenvalues that you get, and keep whatever you need.
# - Plot the eigenvector with eigenvalues on our dataset and project the data onto the new plane. 
# ===========================================================================

# ===========================================================================
# Manual
# ===========================================================================

# X_centered = X - X.mean(axis=0)
# U, s, Vt = np.linalg.svd(X_centered)
# W2 = Vt[:,:2]
# X2D = X_centered.dot(W2)

# ===========================================================================
# Using Scikit
# ===========================================================================

# pca = PCA(n_components=2)
# X2D = pca.fit_transform(X)

# ===========================================================================
# Computing the minimum number of dimensions required to preserver 95% variance
# ===========================================================================

pca = PCA()
pca.fit(X_train)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(d)

# option 2
# pca = PCA(n_components=0.95)
# X_reduced = pca.fit_transform(X)

# ===========================================================================
# Incremental PCA
# ===========================================================================

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(batch)
    
X_reduced = inc_pca.transform(X_train)

# ===========================================================================
# Kernel PCA
# ===========================================================================

X, y = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lin_pca = KernelPCA(n_components=2,kernel="linear")
rbf_pca = KernelPCA(n_components=2,kernel="rbf")
sig_pca = KernelPCA(n_components=2, kernel="sigmoid")

