import numpy as np
import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict,cross_val_score,GridSearchCV
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve,roc_auc_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(images_per_row, len(instances))
    
    images = [instance.reshape(size,size) for instance in instances]
    
    
    n_rows = math.ceil(len(instances) / images_per_row)
    
    row_images = []
    
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    
    for row in range(n_rows):
        rimages = images[row * images_per_row: images_per_row * (row + 1)]
        row_images.append(np.concatenate(rimages,axis=1))
    
    image = np.concatenate(row_images, axis=0)
    # And finally we can just `imshow()` this big image, forwarding any options we were given:
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    # And drop the horizontal and vertical axes (and their labels):
    plt.axis("off")

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



# sdg_clf = SGDClassifier()
# sdg_clf.fit(x_train, y_train)

# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
# y_train_pred = cross_val_predict(sdg_clf, x_train_scaled, y_train, cv=3)
# conf_mtx = confusion_matrix(y_train, y_train_pred)

# row_sum = conf_mtx.sum(axis=1, keepdims=True)
# norm_conf_mtx = conf_mtx / row_sum
# np.fill_diagonal(norm_conf_mtx, 0)
# plt.matshow(norm_conf_mtx, cmap=plt.cm.gray)
# plt.show()

# cl_a, cl_b = 3, 5
# x_aa = x_train[(y_train == cl_a) & (y_train_pred == cl_a)]
# x_ab = x_train[(y_train == cl_a) & (y_train_pred == cl_b)]
# x_ba = x_train[(y_train == cl_b) & (y_train_pred == cl_a)]
# x_bb = x_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# plt.subplot(221); plot_digits(x_aa[:25], 5)
# plt.subplot(222); plot_digits(x_ab[:25], 5)
# plt.subplot(223); plot_digits(x_ba[:25], 5)
# plt.subplot(224); plot_digits(x_bb[:25], 5)
# plt.show()



# Multilabel Classification
# y_train_large = (y_train >= 7)
# y_train_odd = (y_train % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]
# kneighb_clf = KNeighborsClassifier()
# kneighb_clf.fit(x_train, y_multilabel)
# kneighb_clf.predict([some_digit])

# Multioutput Classification

# 1
kneighb_clf = KNeighborsClassifier()
kneighb_clf.fit(x_train, y_train)
param_grid = {"weights":["uniform","distance"],"n_neighbors":[3,4,5]}
grid_search = GridSearchCV(kneighb_clf, param_grid,cv=3, verbose=3)
grid_search.fit(x_train, y_train)


# 2
def shift_image(image,dx, dy):
    image = image.reshape((28,28))
    shifted_image = shift(image,[dy, dx],mode="constant", cval=0)
    return shifted_image.reshape([-1])


x_train_augmented = [image for image in x_train]
y_train_augmented = [label for label in y_train]

for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
    for image, label in zip(x_train, y_train):        
        x_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)
    
x_train_augmented = np.array(x_train_augmented)
y_train_augmented = np.array(y_train_augmented)
    
kneigh_clf = KNeighborsClassifier(**grid_search.best_params_)
kneigh_clf.fit(x_train_augmented, y_train_augmented)
y_test_pred = kneigh_clf.predict(x_test)


