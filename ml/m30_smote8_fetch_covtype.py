# 증폭한 후 저장
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

print(x.shape, y.shape)  # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))   # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
print(pd.Series(y).value_counts())
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# dtype: int64

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

start = time.time()

smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)

end = time.time()

np.save('../save/_save/smote_x.npy',arr=x_train)
np.save('../save/_save/smote_y.npy', arr=y_train)

print("smote걸린시간: ", end - start)

"""
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)


print("model.score : ", round(score, 4))

y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
print("f1_score : ", round(f1_score(y_test, y_predict, average='macro'),4))
"""







'''
model.score :  0.7898
accuracy score:  0.7898
f1_score :  1 4
=========================================== SMOTE 적용 ===============================================
smote걸린시간:  404.43740797042847
model.score :  0.836
accuracy score:  0.836
f1_score :  1 4
'''
