import numpy as np
import numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

x = 6 * rnd.rand(100, 1) - 3
y = 0.5 * x**2 + x + 2 + rnd.randn(100, 1 )

poly_features = PolynomialFeatures(degree=10, include_bias=False)
x_poly = poly_features.fit_transform(x)


def plot_learning_curve(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    train_errors, test_errors = [], []
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_predict = model.predict(x_train[:m])
        y_test_predict = model.predict(x_test)
        train_errors.append(mean_squared_error(y_train[:m],y_predict))
        test_errors.append(mean_squared_error(y_test, y_test_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(test_errors), "b-",  linewidth=3, label="val")
        

lin_reg = LinearRegression()
plot_learning_curve(lin_reg, x ,y)
plt.axis([0, 80, 0, 3])

plt.show()