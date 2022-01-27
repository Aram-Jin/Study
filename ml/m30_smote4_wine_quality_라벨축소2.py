from random import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time

#1. 데이터
path = "../_data/"    

data = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';', dtype=float)
# print(data.shape)  # (4898, 12)
# print(data.head())
# print(data.describe())   # pandas에서 볼수있는기능 (수치데이터에서 용이함)
# print(data.info())
# print(data.columns)
# Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'quality'],
#       dtype='object')
# print(type(data))   # <class 'pandas.core.frame.DataFrame'>

x = data.drop(['quality'], axis=1)  
# print(x.shape)  # (4898, 11)
y = data['quality']
# print(y.shape)  # (4898,)
# print(np.unique(y, return_counts=True))   # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# print(y)
# 0       6.0
# 1       6.0
# 2       6.0
# 3       6.0
# 4       6.0
#        ...
# 4893    6.0
# 4894    5.0
# 4895    6.0
# 4896    7.0
# 4897    6.0
# Name: quality, Length: 4898, dtype: float64

for index, value in enumerate(y):
    if value == 9 :    # == : 비교하는것
       y[index] = 7
    elif value == 8 :   
       y[index] = 7
    elif value == 7 :   
        y[index] = 7
    elif value == 6 :   
        y[index] = 6
    elif value == 5 :   
        y[index] = 5
    elif value == 4 :   
        y[index] = 4
    elif value == 3 :   
        y[index] = 4
    else :
        y[index] = 0  
             
print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0    1060
# 4.0     183
# Name: quality, dtype: int64

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=66, stratify=y)

print(pd.Series(y_train).value_counts())
# 6.0    1648
# 5.0    1093
# 7.0     795
# 4.0     137
# Name: quality, dtype: int64

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ", round(score, 4))

y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))

print("=========================================== SMOTE 적용 ===============================================")
start = time.time()

smote = SMOTE(random_state=66, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)

end = time.time()

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print("smote걸린시간: ", end - start)
print("model.score : ", round(score, 4))

y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
print("f1_score : ", round(f1_score(y_test, y_predict, average='macro')),4)


'''
model.score :  0.6767
accuracy score:  0.6767
=========================================== SMOTE 적용 ===============================================
smote걸린시간:  0.01296544075012207
model.score :  0.6531
accuracy score:  0.6531
f1_score :  1 4
'''
          