import time
import numpy as np
import pandas as pd
from sklearn import datasets
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder, OneHotEncoder

#1. 데이터
path = "../_data/"    

data = pd.read_csv(path + 'winequality-white.csv', index_col=None, header=0, sep=';', dtype=float)
# print(data.shape)  # (4898, 12)
print(data.head())
print(data.describe())   # pandas에서 볼수있는기능 (수치데이터에서 용이함)
print(data.info())
# print(data.columns)
# print(type(data))   # <class 'pandas.core.frame.DataFrame'>

data = data.values   # numpy형태로 변경하는 방법
print(type(data))   # <class 'numpy.ndarray'>
print(data.shape)  # (4898, 12)

# numpy형태에서 x,y나누기 
x = data[:, :11]
y = data[:, 11]

# x = data.drop(['quality'], axis=1)  
# # print(x.shape)  # (4898, 11)
# y = data['quality']
# # print(y.shape)  # (4898,)

print(np.unique(y, return_counts=True))  # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# le = LabelEncoder()
# le.fit(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(
                     n_estimators=10000,   # n_estimators는 epoch와 같은 의미
                     n_jobs=-1,           
                     learning_rate=0.005, 
                     max_depth=13,         
                     min_child_weight=1,
                     subsample=1,
                     colsample_bytree=1,
                     reg_alpha=0,         # 규제  L1  
                     reg_lambda=1         # 규제  L2
                     )     

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set = [(x_test, y_test)],
          eval_metric= 'merror',        #rmse, mae, logloss, error
          early_stopping_rounds=900 
          )  # mlogloss
end = time.time()

results = model.score(x_test, y_test)
print("results: ", round(results, 4))

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", round(acc, 4))
print("f1_score : ", f1_score(y_test, y_predict, average='macro'))
print("f1_score : ", f1_score(y_test, y_predict, average='micro'))

print("걸린시간: ", end - start)


print("=====================================")
'''
results:  0.7102
accuracy_score :  0.7102
걸린시간:  38.32083296775818

results:  0.6745
accuracy_score :  0.6745
f1_score :  0.4523239916472014
걸린시간:  24.807340145111084
'''
