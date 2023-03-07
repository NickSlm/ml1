from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ==========================================================================
# Voting Classifier
# ==========================================================================

# log_clf = LogisticRegression(solver="lbfgs", random_state=42)
# rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# svm_clf = SVC(gamma="scale", random_state=42, probability=True)

# voting_clf = VotingClassifier(estimators=[("lr", log_clf), ("rf", rnd_clf), ("svc", svm_clf)],
#                               voting="soft")
# voting_clf.fit(X_train, y_train)

# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# ==========================================================================
# Bagging and Pasting
# ==========================================================================

# bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
#                             max_samples=100, bootstrap=True,
#                             n_jobs=-1, random_state=42)
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
# print("Bagging: ", accuracy_score(y_test, y_pred))

# decision_tree_clf = DecisionTreeClassifier(random_state=42)
# decision_tree_clf.fit(X_train, y_train)
# y_pred = decision_tree_clf.predict(X_test)
# print("Tree: ", accuracy_score(y_test, y_pred))

# ==========================================================================
# OOB evaluation
# ==========================================================================

# bag_clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,
#                             bootstrap=True, n_jobs=-1, oob_score=True)
# bag_clf.fit(X_train, y_train)
# print(bag_clf.oob_decision_function_)