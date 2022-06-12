import os
import time
import json
import datetime
import numpy as np
import pandas as pd
import sys
# from pycaret.regression import *
from xgboost import XGBRegressor
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pycaret.regression import *
import tensorflow as tf
# from sklearn.pipeline import make_pipeline, Pipeline

# tf.config.list_logical_devices()

# 1. 데이터 (raw_data_㈜코오롱워터앤에너지 경산사업소)
data = pd.read_csv('./data/ai_contest/raw_data_test.csv')
print(data.shape)   # (735699, 9)
# print(data.head())
data = data.drop(['Unnamed: 0', 'local'], axis=1)
print(data.head())


# def split_xy(dataset, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_column
        
#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i:x_end_number, :]
#         tmp_y = dataset[x_end_number:y_end_number, :]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)
        
# # x, y = split_xy(data, 2000, 699)
# # print(x, "\n", y)
# # print(x.shape)
# # print(y.shape)
# data_v = data[['pH', 'COD', 'SS', 'N', 'P', 'T']].values
# print(data_v.shape)  # (735699, 6)

# x, y = split_xy(data_v, 10, 10)
# print(x.shape, y.shape)   # (735610, 80, 6) (735610, 10, 6)
# # x = x.reshape(1000, 1, 6)
# # x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])


# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
# # print(x_train.shape, y_train.shape)  # (588488, 80, 6) (588488, 10, 6)

# # x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
# # print(x_train.shape)  # (588488, 60)
# # x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
# # print(x_test.shape)
# # y_train = y_train.reshape(y_train.shape[0],y_train.shape[1]*y_train.shape[2])
# # print(y_train.shape)  # (588488, 60)
# # y_test = y_test.reshape(y_test.shape[0],y_test.shape[1]*y_test.shape[2])
# # print(y_test.shape)   # (147122, 60)

# model = setup(data_v, target = y, ignore_features=['PassengerId'])
              
# start = time.time()
# model.fit(x_train, y_train, verbose=1,
#           eval_set = [(x_train, y_train), (x_test, y_test)],
#           eval_metric='mae' 
#           )
# end = time.time()

# results = model.score(x_test, y_test)
# print("results: ", round(results, 4))

# # y_predict = model.predict(x_test)

# # r2 = r2_score(y_test, y_predict)
# # print('r2스코어 : ', r2)