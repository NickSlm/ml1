import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict,cross_val_score
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


mnist = fetch_openml("mnist_784", version=1, as_frame=False, data_home="D:\ml1\datasets")

x, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
x_train, x_test, y_train, y_test = x[:60000], x[60000:],y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
# sgd_clf.fit(x_train, y_train_5)

some_digit = x_train[0]
some_digit.reshape(28,28)

# custom cross_validation implementation
# kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# for train_index, test_index in kfolds.split(x_train, y_train_5):
#     clone_clf = clone(sgd_clf)
    
#     x_train_fold = x_train[train_index]
#     y_train_fold = y_train_5[train_index]
    
#     x_test_fold = x_train[test_index]
#     y_test_fold = y_train_5[test_index]
    
#     clone_clf.fit(x_train_fold, y_train_fold)
#     y_predict = clone_clf.predict(x_test_fold)
    
#     num_true = sum(y_predict == y_test_fold)
    
#     print(num_true / len(y_predict))
# 


# y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
# print(confusion_matrix(y_train_5, y_train_pred))
# print("precision:\t",precision_score(y_train_5, y_train_pred))
# print("recall:\t\t",recall_score(y_train_5, y_train_pred))
# print("f1 score:\t",f1_score(y_train_5, y_train_pred))

# y_scores = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3, method="decision_function")

# precision vs recall Graph
# precision, recall, thresholds = precision_recall_curve(y_train_5, y_scores)
# def plot_precision_recall_vs_threshold(precision, recall, thresholds):
#     plt.plot(thresholds, precision[:-1], "b--", label="precision")
#     plt.plot(thresholds, recall[:-1], "g-", label="recall")
#     plt.legend(loc="center right", fontsize=16) # Not shown in the book
#     plt.xlabel("Threshold", fontsize=16)        # Not shown
#     plt.grid(True)                              # Not shown
#     plt.axis([-50000, 50000, 0, 1])             # Not shown
    
# recall_90_precision = recall[np.argmax(precision >= 0.90)]
# threshold_90_precision = thresholds[np.argmax(precision >= 0.90)]


# plt.figure(figsize=(8, 4))                                                                  # Not shown
# plot_precision_recall_vs_threshold(precision, recall, thresholds)
# plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 # Not shown
# plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                # Not shown
# plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")# Not shown
# plt.plot([threshold_90_precision], [0.9], "ro")                                             # Not shown
# plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             # Not shown
# plt.show()

# roc_curve Graph
# forest_clf = RandomForestClassifier(random_state=42)
# y_probas_02 = cross_val_predict(forest_clf, x_train ,y_train_5, cv=3, method="predict_proba")
# y_scores_02 = y_probas_02[:,1]

# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores_02)
# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0,1],[0,1], 'k--')
#     plt.axis([0,1,0,1])
#     plt.ylabel("recall - False Positive Rate")
#     plt.xlabel("Fall out - True Positive Rate")
    

# plot_roc_curve(fpr,tpr)
# plt.show()



sdg_clf = SGDClassifier()
sdg_clf.fit(x_train, y_train)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
y_train_pred = cross_val_predict(sdg_clf, x_train_scaled, y_train, cv=3)
conf_mtx = confusion_matrix(y_train, y_train_pred)

row_sum = conf_mtx.sum(axis=1, keepdims=True)
norm_conf_mtx = conf_mtx / row_sum
np.fill_diagonal(norm_conf_mtx, 0)
plt.matshow(norm_conf_mtx, cmap=plt.cm.gray)
plt.show()







