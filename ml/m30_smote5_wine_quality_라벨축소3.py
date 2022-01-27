# 3,4 -> 0
# 5,6,7, -> 1
# 8,9 -> 2

# 3,4,5 -> 0
# 6 -> 1
# 7,8,9 -> 2

# 성능비교!!  // acc, f1
# smote 전후 / 라벨축소 전후 -> 총 8가지 결과

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time

#1. 데이터
path = "../_data/"    

data = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';', dtype=float)
# print(data.shape)  # (4898, 12)

x = data.drop(['quality'], axis=1)  
# print(x.shape)  # (4898, 11)
y = data['quality']
# print(y.shape)  # (4898,)
print(np.unique(y, return_counts=True))   # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# print(y)

# for index, value in enumerate(y):
#     if value == 9 :    # == : 비교하는것
#        y[index] = 2
#     elif value == 8 :   
#        y[index] = 2
#     elif value == 7 :   
#         y[index] = 2
#     elif value == 6 :   
#         y[index] = 1
#     else :
#         y[index] = 0  
             
print(pd.Series(y).value_counts())
# 1.0    4535
# 0.0     183
# 2.0     180
# Name: quality, dtype: int64

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=66, stratify=y)

model = XGBClassifier(n_jobs=4)
model.fit(x_train, y_train)


score = model.score(x_test, y_test)
print("model.score : ", round(score, 4))
y_predict = model.predict(x_test)
print("accuracy score: ", round(accuracy_score(y_test, y_predict),4))
print("f1_score : ", round(f1_score(y_test, y_predict, average='macro')),4)

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
# 라벨축소 전
model.score :  0.6433
accuracy score:  0.6433
f1_score :  0 4
=========================================== SMOTE 적용 ===============================================
smote걸린시간:  0.022939682006835938
model.score :  0.6433
accuracy score:  0.6433
f1_score :  0 4

# 1
model.score :  0.9412
accuracy score:  0.9412
f1_score :  1 4
=========================================== SMOTE 적용 ===============================================
smote걸린시간:  0.007010221481323242
model.score :  0.9265
accuracy score:  0.9265
f1_score :  1 4

#2
model.score :  0.7004
accuracy score:  0.7004
f1_score :  1 4
=========================================== SMOTE 적용 ===============================================
smote걸린시간:  0.01395106315612793
model.score :  0.7037
accuracy score:  0.7037
f1_score :  1 4
'''