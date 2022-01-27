# 1  357
# 0  212

# 라벨 0을 112개 삭제하여 재구성
# smote 사용하기 , 전 후 비교
import numpy as np
import pandas as pd
import time
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = load_breast_cancer()   #(506, 13) (506,)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)    #(506, 13) (506,)

print(np.unique(y, return_counts=True))   # (array([0, 1]), array([212, 357], dtype=int64))

for index, value in enumerate(y):
    if y[index] == 0:
        
        
      
       

      
print(np.unique(y, return_counts=True))   
            
print(pd.Series(y).value_counts())

# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# model = XGBClassifier(n_jobs=4)
# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)
# print("model.score : ", round(score, 4))
# y_predict = model.predict(x_test)
# print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
# print("f1_score : ", round(f1_score(y_test, y_predict, average='macro')),4)

# print("=========================================== SMOTE 적용 ===============================================")
# start = time.time()

# smote = SMOTE(random_state=66)
# x_train, y_train = smote.fit_resample(x_train, y_train)

# end = time.time()

# model = XGBClassifier(n_jobs=4)
# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)

# print("smote걸린시간: ", end - start)
# print("model.score : ", round(score, 4))

# y_predict = model.predict(x_test)
# print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
# print("f1_score : ", round(f1_score(y_test, y_predict, average='macro')),4)