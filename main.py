from utils import fetch_housing_data,load_housing_data, split_test_train_by_id
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from zlib import crc32
import numpy as np

def main():

    fetch_housing_data()
    data_frame = load_housing_data()
    data_frame['income_cat'] = pd.cut(data_frame['median_income'],bins=[0., 1.5, 3.0, 4.5, 6, np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data_frame, data_frame['income_cat']):
        strat_train_set = data_frame.loc[train_index]
        strat_test_set = data_frame.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                         s=strat_train_set["population"] / 100, label="population", figsize=(10,7),
                         c="median_house_value",colormap="jet", colorbar=True)

    # FIRST WAY TO CHECK CORRELATION    
    # corr_matrix = strat_train_set.corr()
    # print(corr_matrix['median_house_value'].sort_values(ascending=False))
    
    # SECOND WAY TO CHECK CORRELATION BETWEEN ALL THE ATTRIBUTES LISTED
    attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
    scatter_matrix(strat_train_set[attributes], figsize=(12, 8))
    strat_train_set["rooms_per_household"] = strat_train_set["total_rooms"] / strat_train_set["households"]
    print(strat_train_set.info())



if __name__ == "__main__":
    main()
