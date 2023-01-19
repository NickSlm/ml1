import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x = 3 * np.random.rand(20, 1)
y = 0.5 * x + 1 + np.random.randn(20, 1) / 1.5

# Increasing the bias to lower the variance

# Ridge Regression
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(x, y)
# print("Ridge: {0}".format(ridge_reg.predict([[1.5]])))

sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(x, y.ravel())
y_pred = sgd_reg.predict([[1.5]])
# print(f"SGD Ridge: {y_pred}")

# Lasso Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x, y)
# print(f"Lasso:", lasso_reg.predict([[1.5]]))

sgd_reg_p1 = SGDRegressor(penalty="l1")
sgd_reg_p1.fit(x,y.ravel())
# print("SGD Lasso:", sgd_reg_p1.predict([[1.5]]))

# Elastic Net
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(x,y)
# print(f"Elastic net: {elastic_net.predict([[1.5]])}")


# Early Stopping

x = 6 * np.random.rand(100, 1) - 3
y = 2 + x + 0.5 * x**2 + np.random.randn(100, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.5)
poly_scaler = Pipeline([("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
                        ("std_scaler", StandardScaler())])


x_train_poly = poly_scaler.fit_transform(x_train)
x_test_poly= poly_scaler.transform(x_test)

sgd_reg = SGDRegressor(penalty=None, max_iter=1, tol=np.infty, warm_start=True, learning_rate="constant", eta0=0.0005)

minimum_val_error = float("inf")
best_epoch = None
best_model = None

for epoch in range(1000):
    sgd_reg.fit(x_train_poly,y_train)
    y_train_predict = sgd_reg.predict(x_train_poly)
    val_error = mean_squared_error(y_train, y_train_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)



