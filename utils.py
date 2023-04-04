import os
import tarfile
import urllib
import joblib
import math
from urllib import request
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from zlib import crc32
import numpy as np


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_path=HOUSING_PATH, housing_url=HOUSING_URL):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

def check_if_test(id, ratio):
    return crc32(np.int64(id)) < ratio * 2 ** 32

def split_test_train_by_id(data, ratio, index):
    data_ids = data[index]
    test_set = data_ids.apply(lambda id: check_if_test(id, ratio))
    return data.loc[~test_set], data.loc[test_set]

def display_score(score):
    print("Scores:", score)
    print("Mean:", np.mean(score))
    print("Standard deviation:", np.std(score))
    
def save_model_to_disk(model_name, file_name):
    """
    Save model to disk
    Args:
        model_name (object): _description_
        file_name (str): filename
    """
    joblib.dump(model_name, file_name)

def load_model_from_disk(file_name):
    return joblib.load(file_name)

class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    rooms_ix, households_ix, population_ix, bedrooms_ix = 3, 4, 5, 6
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def top_k_indices(features, k):
    return np.sort(np.argpartition(np.array(features), -k)[-k:])

class TopFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, array, k):
        self.array = array
        self.k = k
    
    def fit(self, X,y=None):
        self.feature_importances_ = top_k_indices(self.array, self.k)
        return self
    
    def transform(self, X):
        return X[:,self.feature_importances_]
    