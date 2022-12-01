import os
import tarfile
import urllib
from urllib import request
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

    
    
