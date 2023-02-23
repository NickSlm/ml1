import sys
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source



image_path = os.path.join(".", "images", "decision_tree")
os.makedirs(image_path, exist_ok=True)

df = load_iris()
X = df["data"][:,2:]
y = df["target"]

dtc = DecisionTreeClassifier(max_depth=2, criterion="gini")
dtc.fit(X, y)

print(dtc.predict_proba([[5, 1.5]]))

# export_graphviz(dtc,
#                 out_file=os.path.join(image_path, "iris_tree.dot"),
#                 feature_names=df.feature_names[2:],
#                 class_names=df.target_names,2
#                 rounded=True,
#                 filled=True)

# Source.from_file(os.path.join(image_path, "iris_tree.dot"))