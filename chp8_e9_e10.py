import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

np.random.seed(42)

mnist = fetch_openml(name="mnist_784", version=1, as_frame=False, data_home="D:\ml1\datasets")

# ==========================================================================
# PCA speed test
# ==========================================================================

# X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=10000)

# rf_clf = RandomForestClassifier(n_estimators=100,random_state=42)

# t0 = time.time()
# rf_clf.fit(X_train, y_train)
# t1 = time.time()

# print("Time it took to fit:", t1 - t0)

# y_pred = rf_clf.predict(X_test)
# score = accuracy_score(y_test, y_pred)
# print(score)

# pca = PCA(n_components=0.95, random_state=42)
# X_train_reduced = pca.fit_transform(X_train)

# rf_clf = RandomForestClassifier(n_estimators=100,random_state=42)
# rf_clf.fit(X_train_reduced, y_train)

# X_test_reduced = pca.transform(X_test)
# y_pred = rf_clf.predict(X_test_reduced)
# print(accuracy_score(y_test, y_pred))


# ==========================================================================
# e10 
# ==========================================================================

indx = np.random.permutation(60000)[:10000]
X, y = mnist["data"][indx], mnist["target"][indx]
y = y.astype(np.uint8)

tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

plt.figure(figsize=(12,10))
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()