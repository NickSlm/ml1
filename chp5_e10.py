import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR, SVR
from scipy.stats import reciprocal, uniform


housing = fetch_california_housing()

X = housing["data"]
y = housing["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline([("scaler", StandardScaler())])

X_train_scale = pipeline.fit_transform(X_train)
X_test_scale = pipeline.transform(X_test)

param_dist = {"gamma": reciprocal(0.001, 0.1), "C":uniform(1,10)}
rnd_search_cv = RandomizedSearchCV(SVR(),param_distributions=param_dist,n_iter=10, verbose=2, cv=3, random_state=42)
rnd_search_cv.fit(X_train_scale,y_train)

y_pred = rnd_search_cv.best_estimator_.predict(X_train_scale)
mse = mean_squared_error(y_train, y_pred)
print("train_set: ",np.sqrt(mse))

y_pred_test = rnd_search_cv.best_estimator_.predict(X_test_scale)
mse = mean_squared_error(y_test, y_pred_test)
print("test_set: ", np.sqrt(mse))


