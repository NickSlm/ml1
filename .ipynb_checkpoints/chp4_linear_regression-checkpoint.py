import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================================================================
# linear equation
# =========================================================================

x  = 2 * np.random.rand(5, 1)
y =  4 + 3 * x + np.random.randn(5, 1)
# plt.plot(x,y,"y.")
# plt.xlabel("$x_1$",fontsize=18)
# plt.ylabel("$y$",rotation=0,fontsize=18)
# plt.axis([0,2,0,15])
# plt.show()


x_b = np.c_[np.ones((5,1)),x]
# =========================================================================
# normal equation - linear regression => (xT*x)^-1*xT*y
# =========================================================================

# theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
# print(theta_best)

# Linear Regression SVD O(n^2)
# lin_reg = LinearRegression()
# lin_reg.fit(x, y)
# print(lin_reg.intercept_, lin_reg.coef_)

# linalg.lstsq
# theta_best_svd, residuals, rank, s = np.linalg.lstsq(x_b, y, rcond=1e-6)
# print(theta_best_svd)

# =========================================================================
# Batch Gradient Descent
# =========================================================================

learning_rate = 0.01
n_iterations = 1000
m = 100

theta = np.random.randn(2,1)  

print(x_b)
print(theta)

for iteration  in range(n_iterations):
    gradient = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - learning_rate * gradient
print(theta)    
