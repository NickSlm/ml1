import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()
# x = iris["data"][:,3:]
# y = (iris["target"] == 2).astype(int)

# print(iris["data"][:,2:])

# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(x, y)

# x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# y_predict = log_reg.predict_proba(x_new)

# plt.plot(x_new, y_predict[:,1:],"b-",label="Iris virginica")
# plt.plot(x_new, y_predict[:,0:],"r--",label="Not Iris virginica")
# plt.xlabel("Petal width (cm)")
# plt.ylabel("Probability")
# plt.show()

# =========================================================================
# softmax
# =========================================================================

x = iris["data"][:,(2,3)]
y = iris["target"]

softmax_reg = LogisticRegression(C=10, solver="lbfgs", multi_class="multinomial")
softmax_reg.fit(x, y)

print(softmax_reg.predict_proba([[2,3]]))