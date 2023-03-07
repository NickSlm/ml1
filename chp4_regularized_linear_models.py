import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# x = 3 * np.random.rand(20, 1)
# y = 0.5 * x + 1 + np.random.randn(20, 1) / 1.5

# =========================================================================
# Increasing the bias to lower the variance
# =========================================================================
# =========================================================================
# Ridge Regression
# =========================================================================

# ridge_reg = Ridge(alpha=1, solver="cholesky")
# ridge_reg.fit(x, y)
# print("Ridge: {0}".format(ridge_reg.predict([[1.5]])))

# sgd_reg = SGDRegressor(penalty="l2")
# sgd_reg.fit(x, y.ravel())
# y_pred = sgd_reg.predict([[1.5]])
# print(f"SGD Ridge: {y_pred}")

# =========================================================================
# Lasso Regression
# =========================================================================

# lasso_reg = Lasso(alpha=0.1)
# lasso_reg.fit(x, y)
# print(f"Lasso:", lasso_reg.predict([[1.5]]))

# sgd_reg_p1 = SGDRegressor(penalty="l1")
# sgd_reg_p1.fit(x,y.ravel())
# print("SGD Lasso:", sgd_reg_p1.predict([[1.5]]))

# =========================================================================
# Elastic Net
# =========================================================================

# elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
# elastic_net.fit(x,y)
# print(f"Elastic net: {elastic_net.predict([[1.5]])}")

# =========================================================================
# Early Stopping
# =========================================================================

np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(100, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

poly_scaler = Pipeline([("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
                        ("std_scaler",StandardScaler())])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

train_errors, val_errors = [], []
for epoch in range(500):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))

best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])

print(best_epoch)

plt.annotate('Best model',
             xy=(best_epoch, best_val_rmse),
             xytext=(best_epoch, best_val_rmse + 1),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=16,
            )

best_val_rmse -= 0.03
plt.plot([0, 500], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.show()