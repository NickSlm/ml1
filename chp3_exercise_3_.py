import pandas as pd
import numpy as np
import os
import urllib
from urllib import request

TITANIC_PATH = os.path.join("datasets", "titanic")
DOWNLOAD_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/"

def fetch_titanic_data(url=DOWNLOAD_URL, path=TITANIC_PATH):
    os.makedirs(TITANIC_PATH, exist_ok=True)
    for file_name in ["test.csv", "train.csv"]:
        file_path = os.path.join(TITANIC_PATH, file_name)
        urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path )
    
fetch_titanic_data()