import pandas as pd
import numpy as np
import os
import urllib
from urllib import request
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV


TITANIC_PATH = os.path.join("datasets", "titanic")
DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/"

def fetch_titanic_data(url=DOWNLOAD_URL, path=TITANIC_PATH):
    if not os.path.isdir(path):
        os.makedirs(TITANIC_PATH, exist_ok=True)
    for file_name in ["test.csv", "train.csv"]:
        file_path = os.path.join(TITANIC_PATH, file_name)
        urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path )
    
def load_data(file_name, path=TITANIC_PATH):
    file_path = os.path.join(path, file_name)
    return pd.read_csv(file_path)

fetch_titanic_data()

df_train = load_data("train.csv")
df_train["RelativesOnboard"] = df_train["SibSp"] + df_train["Parch"]
df_train.drop(["SibSp","Parch"], axis=1)

df_test = load_data("test.csv")
df_test["RelativesOnboard"] = df_test["SibSp"] + df_test["Parch"]
df_test.drop(["SibSp","Parch"], axis=1)


df_train = df_train.set_index("PassengerId")
df_test = df_test.set_index("PassengerId")

# fill up missing values; drop not important attribs like cabin
# scale the data if needed using stdScaler

num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])

cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

num_attribs = ["Age", "RelativesOnboard", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocessing_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),
                                   ("cat",cat_pipeline, cat_attribs)])

# train dataset
x_train = preprocessing_pipeline.fit_transform(df_train[num_attribs + cat_attribs])


# test dataset
x_test = preprocessing_pipeline.fit_transform(df_test[num_attribs + cat_attribs])

# train dataset labels
y_train = df_train["Survived"]


# knb_clf = KNeighborsClassifier()
# knb_clf.fit(x_train,y_train)
# scores_knb = cross_val_score(knb_clf, x_train, y_train, cv=10)

rndforest_clf = RandomForestClassifier()
rndforest_clf.fit(x_train, y_train)
scores_forest = cross_val_score(rndforest_clf, x_train, y_train, cv=10)
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
grid_search = GridSearchCV(rndforest_clf, param_grid, cv=5, verbose=3)
grid_search.fit(x_train, y_train)

svm_clf = SVC(gamma="auto")
svm_clf.fit(x_train, y_train)
scores_svm = cross_val_score(svm_clf, x_train, y_train, cv=10)



plt.figure(figsize=(8,4))
plt.plot([1] * 10, scores_svm, ".")
plt.plot([2] * 10, scores_forest, ".")
plt.boxplot([scores_svm, scores_forest],labels=["SMV","RandomForest"])
plt.ylabel("Accuracy")
plt.show()