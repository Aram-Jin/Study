import numpy as np
import pandas as pd
import time
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = fetch_covtype()
x = datasets.data 
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

x_train = np.load('../save/_save/smote_x.npy')
y_train = np.load('../save/_save/smote_y.npy')

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.9,
#               enable_categorical=False, eval_metric='merror', gamma=0,
#               gpu_id=-1, importance_type=None, interaction_constraints='',
#               learning_rate=0.3, max_delta_step=0, max_depth=5,
#               min_child_weight=1, monotone_constraints='()',
#               n_estimators=1000, n_jobs=8, num_parallel_tree=1,
#               objective='multi:softprob', predictor= 'gpu_predictor', random_state=0,
#               reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='gpu_hist', validate_parameters=1, verbosity=None 
# )  

# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)

# print("model.score : ", round(score, 4))

# y_predict = model.predict(x_test)
# print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
# print("f1_score : ", round(f1_score(y_test, y_predict, average='macro'),4))

# path = '../save/_save/'

# import joblib
# joblib.dump(model, path +"weight_save.dat")

'''
model.score :  0.9387
accuracy score:  0.9387
f1_score :  0.9354
'''

import joblib
model = joblib.load('D:\Study\_Report\\weight_save.dat')

score = model.score(x_test, y_test)

print("model.score : ", round(score, 4))

y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
print("f1_score : ", round(f1_score(y_test, y_predict, average='macro'),4))