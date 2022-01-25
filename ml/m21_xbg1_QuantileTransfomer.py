from sklearn import datasets
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time

#1. 데이터
# datasets = fetch_california_housing()
datasets = load_boston()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)   #(20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = XGBRegressor()
model = XGBRegressor(n_estimators=20000,   # 0.8444120980222539   # n_estimators는 epoch와 같은 의미
                     n_jobs=-1,           # 0.8532492079366552
                     learning_rate=0.04825)     

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1)
end = time.time()

results = model.score(x_test, y_test)
print("results: ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)
print("걸린시간: ", end - start)
'''
results:  0.843390038548427
r2 :  0.843390038548427
'''
print("=====================================")
# hist = model.evals_result()
# print(hist)

'''
results:  0.8566291699938181
r2 :  0.8566291699938181
걸린시간:  8.984177827835083
=====================================

results:  0.8563880295311915
r2 :  0.8563880295311915
걸린시간:  50.08405351638794
=====================================
results:  0.9313449710981906
r2 :  0.9313449710981906
걸린시간:  0.5760297775268555
=====================================
results:  0.9313449710981906
r2 :  0.9313449710981906
걸린시간:  27.75403666496277
=====================================
results:  0.9328812557278715
r2 :  0.9328812557278715
걸린시간:  29.966202974319458
=====================================
results:  0.9340318377491721
r2 :  0.9340318377491721
걸린시간:  3.3955209255218506
=====================================





'''
'''
i) QuantileTransformer
기본적으로 1000개 분위를 사용하여 데이터를 '균등분포' 시킵니다.
Robust처럼 이상치에 민감X, 0~1사이로 압축합니다.

ii) PowerTransformer
데이터의 특성별로 정규분포형태에 가깝도록 변환
히스토그램을 그려서 확인 꼭 해보는게 좋다네요 책에서
'''
