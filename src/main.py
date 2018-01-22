# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:02:38 2017

@author: Alexandre
"""

#%% 0.1 Import libraries
from framework_extensions.NYC_taxis_preprocessor import NYCTaxisPreprocessor
from framework_extensions.NYC_taxis_feature_selector import NYCTaxisFeatureSelector

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
#%% 0.2 Import datasets
train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='id', nrows=10000)
test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='id', nrows=10000)

#%% 1.0 Preprocess the data
data_preprocessor = NYCTaxisPreprocessor('trip_duration')

data_preprocessor.prepare(train_dataframe)
train_eng = data_preprocessor.cook(train_dataframe)
test_eng = data_preprocessor.cook(test_dataframe)

feature_selector = NYCTaxisFeatureSelector(['trip_duration', 'log_duration'])
X_train, y_train = feature_selector.prepare(train_eng).cook_and_split(train_eng)
X_test, _ = feature_selector.cook_and_split(test_eng)

#%% Data pre-analysis (jupyter notebook)


#%% KNN

corr_factor = [1,1,1,1,1e-6,1e-6]

neigh = KNeighborsRegressor(n_neighbors=5, weights = 'distance',n_jobs=2)
neigh.fit(X_train * corr_factor, y_train)
pred = np.exp(neigh.predict(X_test))


#%% Submit
y_pred = pd.DataFrame(pred, index = X_test.index, columns = ['trip_duration'])
y_pred.to_csv('..\\output\\Knn.csv')
