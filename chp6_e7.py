import numpy as np

from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

moons_df = make_moons(n_samples=10000, noise=0.4, random_state=42)

X = moons_df[0]
y = moons_df[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier(random_state=42)

param_grid = {"max_leaf_nodes":list(range(2, 100)), "min_samples_split":[2,3,4], "criterion":["gini", "entropy"]}

grid_search = GridSearchCV(tree_clf, param_grid,verbose=3, cv=3)

grid_search.fit(X_train, y_train)

n_dtree = 1000
n_instances =  100

subsets = []

rs = StratifiedShuffleSplit(n_splits=n_dtree, test_size=len(X_train) - n_instances, random_state=42)

for mini_train_index, mini_test_index in rs.split(X_train, y_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    subsets.append((X_mini_train, y_mini_train))

forest = [clone(grid_search.best_estimator_) for _ in range(n_dtree)]

accuracy_scores = []

for tree, (x_train_sub, y_train_sub) in zip(forest, subsets):
    tree.fit(x_train_sub, y_train_sub)
    y_predict = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_predict))

Y_pred = np.empty((n_dtree, len(X_test)), dtype=np.uint8)    

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

y_pred_majority_votes, n_votes = mode(Y_pred,axis=0)
print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))
 
 
    