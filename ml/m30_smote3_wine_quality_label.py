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
print(np.unique(y, return_counts=True))   # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
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

x_new = x[:-1000]
y_new = y[:-1000]
print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    18
# dtype: int64
print(y_new)
# 0       6.0
# 1       6.0
# 2       6.0
# 3       6.0
# 4       6.0
#        ...
# 3893    5.0
# 3894    6.0
# 3895    6.0
# 3896    6.0
# 3897    6.0
# Name: quality, Length: 3898, dtype: float64



x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, train_size=0.75, shuffle=True, random_state=66, stratify=y_new)

print(pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2    14
# dtype: int64

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ", round(score, 4))

y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))

print("=========================================== SMOTE 적용 ===============================================")

start = time.time()

smote = SMOTE(random_state=66,k_neighbors=3 )
x_train, y_train = smote.fit_resample(x_train, y_train)

end = time.time()
print("smote걸린시간: ", end - start)

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ", round(score, 4))

y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
print("f1_score : ", round(f1_score(y_test, y_predict, average='macro')),4)


'''
model.score :  0.6379
accuracy score:  0.6379
=========================================== SMOTE 적용 ===============================================
smote걸린시간:  0.019946575164794922
model.score :  0.6256
accuracy score:  0.6256
f1_score :  0 4
'''