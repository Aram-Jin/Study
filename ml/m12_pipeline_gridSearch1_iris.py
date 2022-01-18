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

from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# parameters = [
#     {'randomforestclassifier__max_depth' : [6, 8, 10]},
#     {'randomforestclassifier__min_samples_leaf' :[3, 5, 7], 'randomforestclassifier__min_samples_split' : [3, 5, 10]}
# ]   # (1) make_pipeline사용시, 원래 파라미터는 pipe(RandomForestClassifier)의 파라미터 이므로 파라미터에 명시를 하고 GridSearchCV의 파라미터 형태로 바꿔줌(래핑)

parameters = [
    {'rF__max_depth' : [6, 8, 10]},
    {'rF__min_samples_leaf' :[3, 5, 7], 'rF__min_samples_split' : [3, 5, 10]}
]   # (2) Pipeline사용시, RandomForestClassifier를 "rF"로 명시해주었으므로 rF로 래핑하여 파라미터를 바꿔줌


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline

#2.모델
# model = SVC()
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())   # 스케일러 두번 적용 가능 ; 성능은 잘 모름
pipe = Pipeline([("mm", MinMaxScaler()), ("rF", RandomForestClassifier())])   # -> make_pipeline(MinMaxScaler(), RandomForestClassifier()) 와 동일함

model = GridSearchCV(pipe, parameters, cv=5, verbose=1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("model.score : ", result)

'''
model.score :  0.9333333333333333
model.score :  0.9333333333333333
'''