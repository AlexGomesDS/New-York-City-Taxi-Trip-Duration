# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:10:13 2017

@author: Alexandre


This class implements the following preprocessing methods:
    - apply standardscaling to every numeric feature
    - select only some of the features
"""


#%% Import onyl the necessary libraries
import pandas as pd
import numpy as np
from features.preprocessor.abstract_preprocessor import AbstractPreprocessor 
from sklearn.preprocessing import StandardScaler

#%% Implementing class with our one version of the preprocessing methods

class NYCTaxisFeatureSelector(AbstractPreprocessor):    
    # apply standardscaling to every numeric feature
    def _set_num_scaler(self, dataframe):
        self.num_scalers = StandardScaler().fit(dataframe[self.numerical_features])
    
    def get_y(self, df):
        return df[ [self._cols_to_predict[-1]] ]
    
    def _feat_eng(self, dataframe):
        columns_to_keep = [
        'pickup_hour', 
        'pickup_weekday',
        'pickup_longitude', 
        'pickup_latitude', 
        'dropoff_longitude', 
        'dropoff_latitude',
        'log_duration']
        
        columns_to_drop = [col for col in dataframe.columns if col not in columns_to_keep]
        dataframe.drop(columns_to_drop, axis = 1,inplace=True)
        
#%% Testing

if __name__ == '__main__':    
    data_preprocessor = NYCTaxisFeatureSelector(["trip_duration", "log_duration"])


    train_dataframe = pd.read_csv('..\\input\\train.csv', index_col='id',nrows=1000)
    test_dataframe = pd.read_csv("..\\input\\test.csv", index_col='id',nrows=1000)
    train_dataframe["log_duration"] = np.log(train_dataframe.trip_duration)
    
    data_preprocessor.prepare(train_dataframe)
    X_train, y_train = data_preprocessor.cook_and_split(train_dataframe)
    X_test, _ = data_preprocessor.cook_and_split(test_dataframe)
    
    assert X_train.shape == (1000,4), "X_train.shape should be {}, and is {}.".format((1000,4), X_train.shape)
    assert X_test.shape == (1000,4), "X_test.shape should be {}, and is {}.".format((1000,4), X_test.shape)
    assert y_train.shape == (1000,1), "y_train.shape should be {}, and is {}.".format((1000,1), y_train.shape)
