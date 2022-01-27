import time
import numpy as np
import pandas as pd
from sklearn import datasets
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

#1. 데이터
path = "../_data/"    

data = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';', dtype=float)
# print(data.shape)  # (4898, 12)
print(data.head())
print(data.describe())   # pandas에서 볼수있는기능 (수치데이터에서 용이함)
print(data.info())
print(data.columns)
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

# newlist = []
# for i in y:
#     # print(i)
#     if i<=4 :
#         newlist +=[0]
#     elif i<=7:
#         newlist +=[1]
#     else:
#         newlist +=[2]
            
# y = np.array(newlist)
# print(y)        
# print(np.unique(y, return_counts=True))   

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66, stratify=y)

print(pd.Series(y_train).value_counts())
# 6.0    1758
# 5.0    1166
# 7.0     704
# 8.0     140
# 4.0     130
# 3.0      16
# 9.0       4
# Name: quality, dtype: int64

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ", round(score, 4))

y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))

print("=========================================== SMOTE 적용 ===============================================")

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

smote = SMOTE(random_state=66, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = model.score(x_test, y_test)

print("model.score : ", round(score, 4))
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
print("f1_score : ", round(f1_score(y_test, y_predict, average='macro')),4)





'''
model.score :  0.6582
accuracy score:  0.6582
=========================================== SMOTE 적용 ===============================================
model.score :  0.6418
accuracy score:  0.6418

model.score :  0.6439
accuracy score:  0.6439

'''
