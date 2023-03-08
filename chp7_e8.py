import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

# =========================================================================
# Load dataset
# =========================================================================
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

X = mnist["data"]
y = mnist["target"]

# =========================================================================
# Split dataset into 50000/10000/10000 (X_train, X_test, X_val)
# =========================================================================
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000,random_state=42)

# =========================================================================
# Train classifiers: SVM, RandomForest, ExtraTrees AND Ensemble clf
# =========================================================================
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf]
for estimator in estimators:
    estimator.fit(X_train, y_train)
    
named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf)
]
voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)

voting_clf.set_params(svm_clf=None)
del voting_clf.estimators_[2]
voting_clf.voting = "soft"

new_training_set = np.empty((len(X_val), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    y_pred = estimator.predict(X_val)
    new_training_set[:,index] = y_pred

rnd_forest_blender = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=42)
rnd_forest_blender.fit(new_training_set, y_val)
print(rnd_forest_blender.oob_score_)