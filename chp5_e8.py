import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = load_iris()


x = df["data"][:, (2,3)]
y = df["target"]

seto_versi = (y==0) | (y==1)
y = df["target"][seto_versi]
x = x[seto_versi]
# plt.scatter(x[:,0][y==0],x[:,1][y==0])
# plt.scatter(x[:,0][y==1],x[:,1][y==1])
# plt.legend()
# plt.show()

lin_svc = LinearSVC(C=5, loss="hinge", random_state=42)
svc = SVC(C=5,kernel="linear")
sgd_clf = SGDClassifier(loss="hinge", learning_rate="constant", eta0=1e-03, alpha=1/(5*len(x)), tol=1e-3, max_iter=1000, random_state=42)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

lin_svc.fit(x_scaled, y)
svc.fit(x_scaled,y)
sgd_clf.fit(x_scaled,y)

print(f"linearSVC -> Coeffiecnt:{lin_svc.coef_}, intercept: {lin_svc.intercept_}")
print(f"SVC(linear kernel) -> Coeffiecnt:{svc.coef_}, intercept: {svc.intercept_}")
print(f"SGDClassifier -> Coeffiecnt:{sgd_clf.coef_}, intercept: {sgd_clf.intercept_}")

w1 = -lin_svc.coef_[0,0]/lin_svc.coef_[0,1]
b1 = -lin_svc.intercept_[0]/lin_svc.coef_[0,1]
w2 = -svc.coef_[0,0]/svc.coef_[0,1]
b2 = -svc.intercept_[0]/svc.coef_[0,1]
w3 = -sgd_clf.coef_[0,0]/sgd_clf.coef_[0,1]
b3 = -sgd_clf.intercept_[0]/sgd_clf.coef_[0,1]
# =========================================================================
# Transform the decision boundary lines back to the original scale
# =========================================================================
line1 = scaler.inverse_transform([[-10, -10 * w1 + b1], [10, 10 * w1 + b1]])
line2 = scaler.inverse_transform([[-10, -10 * w2 + b2], [10, 10 * w2 + b2]])
line3 = scaler.inverse_transform([[-10, -10 * w3 + b3], [10, 10 * w3 + b3]])

print(line1)

# =========================================================================
# Plot all three decision boundaries
# =========================================================================
plt.figure(figsize=(11, 4))
plt.plot(line1[:, 0], line1[:, 1], "k:", label="LinearSVC")
plt.plot(line2[:, 0], line2[:, 1], "b--", linewidth=2, label="SVC")
plt.plot(line3[:, 0], line3[:, 1], "r-", label="SGDClassifier")
plt.plot(x[:, 0][y==1], x[:, 1][y==1], "bs") # label="Iris versicolor"
plt.plot(x[:, 0][y==0], x[:, 1][y==0], "yo") # label="Iris setosa"
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper center", fontsize=14)
plt.axis([0, 5.5, 0, 2])

plt.show()