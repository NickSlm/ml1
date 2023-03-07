import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

# ==========================================================================
# AdaBoost
# ==========================================================================

# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200,
#                              algorithm="SAMME.R", learning_rate=0.5)
# ada_clf.fit(X, y)

# ==========================================================================
# Gradient Boost
# ==========================================================================

# grbt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=3, max_depth=2)
# grbt.fit(X, y)

# xgboost -> Gradient boosting library (extremely fast, scalable and portable
