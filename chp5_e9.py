import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

df = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)


X = df["data"]
y = df["target"].astype(np.int8)


x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
x_test_scaled = scaler.fit_transform(x_test.astype(np.float32))


# lin_svc = LinearSVC(random_state=42, verbose=3)
# lin_svc.fit(x_train_scaled, y_train)
svc_clf = SVC(kernel="rbf", gamma="scale")
svc_clf.fit(x_train_scaled, y_train)

param_dist = {"gamma":reciprocal(0.001, 0.1), "C":uniform(1, 10)}
rnd_search = RandomizedSearchCV(svc_clf, param_dist, n_iter=10, verbose=3, cv=3)

rnd_search.fit(x_train_scaled, y_train)

y_pred = rnd_search.best_estimator_.predict(x_test_scaled)
print(accuracy_score(y_test, y_pred))