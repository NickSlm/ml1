import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC,SVC

# iris = load_iris()
# x = iris["data"][:,(2,3)]
# y = (iris["target"] == 2).astype(np.float64)
# pipeline = Pipeline([("scaler", StandardScaler())])
# x_scaled = pipeline.fit_transform(x)
# svm_clf = LinearSVC(C=1, loss="hinge")
# svm_clf.fit(x_scaled, y)
# print(svm_clf.predict([[5.5, 1.7]]))

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
print(X)
def plot_dataset(X,y, axes):
    plt.plot(X[:,0][y==0], X[:,1][y==0],"bs")
    plt.plot(X[:,0][y==1], X[:,1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True,which="both")
    plt.xlabel(r"$X_1$", fontsize=20)
    plt.ylabel(r"$X_2$", fontsize=20, rotation=0)
    
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0,x1 = np.meshgrid(x0s,x1s)
    X = np.c_[x0.ravel(),x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    print(clf.decision_function(X))
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
    
polynomial_svm_clf = Pipeline([("poly", PolynomialFeatures(degree=3)),
                     ("scaler", StandardScaler()),
                     ("SVC", LinearSVC(C=10, loss="hinge",random_state=42))])

polynomial_svm_clf.fit(X, y)
plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])   
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()


