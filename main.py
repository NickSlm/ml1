from utils import fetch_housing_data,load_housing_data, split_test_train_by_id, CombinedAttributeAdder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
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
    
    # strat_train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    #                      s=strat_train_set["population"] / 100, label="population", figsize=(10,7),
    #                      c="median_house_value",colormap="jet", colorbar=True)

    # FIRST WAY TO CHECK CORRELATION    
    # corr_matrix = strat_train_set.corr()
    # print(corr_matrix['median_house_value'].sort_values(ascending=False))
    
    # SECOND WAY TO CHECK CORRELATION BETWEEN ALL THE ATTRIBUTES LISTED
    # attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
    # scatter_matrix(strat_train_set[attributes], figsize=(12, 8))
    
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    
    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)
    
    # Text and Categorical attributes changes (Changing non interger attributes to integer)
    housing_cat = housing[["ocean_proximity"]]
    cat_encoder = OneHotEncoder(sparse=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    
    attr_adder = CombinedAttributeAdder(add_bedrooms_per_room=False)
    # adding new attributes to our dataset = rooms_per_household, population_per_household
    # the new dataset wont have any column names so we will have to add it
    housing_extra_attribs = attr_adder.transform(housing.values)
    # adding column names into our new dataset with new attributes
    housing_extra_attribs = pd.DataFrame(housing_extra_attribs,
                                         columns=list(housing.columns) + ["rooms_per_household", "population_per_household"],
                                         index=housing.index)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attr_adder", CombinedAttributeAdder()),
        ("std_scaler", StandardScaler())
    ])
    
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
    housing_prepared = full_pipeline.fit_transform(housing)
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared,housing_labels)
    
    some_data = housing.iloc[:5]
    some_lables = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    
    print(lin_reg.predict(some_data_prepared))
    print(list(some_lables))

if __name__ == "__main__":
    main()
