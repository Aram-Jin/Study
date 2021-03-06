import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.decomposition import PCA    # PCA -> 고차원의 데이터를 저차원의 데이터로 축소시키는 차원 축소 방법

#2.모델
# model = SVC()
# model = make_pipeline(MinMaxScaler(), StandardScaler(), SVC())   # 스케일러 두번 적용 가능 ; 성능은 잘 모름
model = make_pipeline(MinMaxScaler(), PCA(), SVC())    # 스케일러 두번 적용 가능 ; 성능은 잘 모름 (fit, transform 두개를 이용하여 변환시키는 애들 다 넣을 수 있음)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("model.score : ", result)

'''
model.score :  0.9666666666666667

<make_pipeline>
model.score :  1.0

<스케일러 두번 적용 가능>
model.score :  0.9333333333333333
'''