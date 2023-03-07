import numpy as np


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

# =========================================================================
# Load dataset
# =========================================================================

mnist = fetch_openml("mnist_784", version=1, as_frame=False, data_home="D:\ml1\datasets")

X = mnist["data"]
y = mnist["target"]

# =========================================================================
# Split dataset into 50000/10000/10000 (X_train, X_test, X_val)
# =========================================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000,random_state=42)

# =========================================================================
# Train classifiers: SVM, RandomForest, ExtraTrees
# =========================================================================

svm_clf = LinearSVC(max_iter=100,tol=20, random_state=42)
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)

estimators = [svm_clf, random_forest_clf, extra_trees_clf]

for estimator in estimators:
    estimator.fit(X_train, y_train)
    
named_estimators = [("svm", svm_clf),
                    ("random_forest", random_forest_clf),
                    ("extra_tree", extra_trees_clf)]

voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)

print([estimator for estimator in voting_clf.estimators_])
