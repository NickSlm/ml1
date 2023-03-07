from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

dataset = load_iris()

X = dataset.data
y = dataset.target

# ==========================================================================
# Feature importances
# ==========================================================================

rnd_forest_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, oob_score=True)
rnd_forest_clf.fit(X, y)
print(rnd_forest_clf.oob_score_)
# for name, score in zip(dataset.feature_names, rnd_forest_clf.feature_importances_):
#     print(name, score)

