from utils import fetch_housing_data, load_housing_data, CombinedAttributeAdder, display_score, save_model_to_disk, load_model_from_disk, top_k_indices, TopFeatures

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from zlib import crc32
from scipy.stats import randint, reciprocal, expon

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def main():
    k = 5
    

# =========================================================================
# Fetch data
# ========================================================================= 
    data_frame = load_housing_data()
    data_frame['income_cat'] = pd.cut(data_frame['median_income'],bins=[0., 1.5, 3.0, 4.5, 6, np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data_frame, data_frame['income_cat']):
        strat_train_set = data_frame.loc[train_index]
        strat_test_set = data_frame.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    

    housing_num = housing.drop("ocean_proximity", axis=1)
 
# =========================================================================
# Text and Categorical attributes changes (Changing non interger attributes to integer)
# =========================================================================
    housing_cat = housing[["ocean_proximity"]]
    cat_encoder = OneHotEncoder(sparse=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    
    attr_adder = CombinedAttributeAdder(add_bedrooms_per_room=False)
    
# =========================================================================
# adding new attributes to our dataset = rooms_per_household, population_per_household
# the new dataset wont have any column names so we will have to add it
# =========================================================================
    housing_extra_attribs = attr_adder.transform(housing.values)
    housing_extra_attribs = pd.DataFrame(housing_extra_attribs,
                                         columns=list(housing.columns) + ["rooms_per_household", "population_per_household"],
                                         index=housing.index)
# =========================================================================
# Building a pipeline to clean our data
# =========================================================================
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
    
# =========================================================================
# Run gridsearch on random_forest to find the best hyperparameters,features
# =========================================================================
    param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(housing_prepared, housing_labels)
    
    feature_importances = grid_search.best_estimator_.feature_importances_
    
    svr = SVR()
    param_distribs = {
        'kernel': ['rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }  
    rnd_search = RandomizedSearchCV(svr, param_distributions=param_distribs,
                                    verbose=2, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)
    
    prepare_select_and_predict_pipeline = Pipeline([('preparation', full_pipeline),
                                                    ('feature_selection', TopFeatures(feature_importances, k)),
                                                    ('prediction', SVR(**rnd_search.best_params_))])
    
    print(prepare_select_and_predict_pipeline.fit(housing, housing_labels))
    
    
main()