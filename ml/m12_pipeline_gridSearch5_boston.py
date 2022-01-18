import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_boston()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

parameters = [
    {'randomforestregressor__max_depth' : [6, 8, 10], 'randomforestregressor__min_samples_leaf' :[3, 5, 7]},
    {'randomforestregressor__min_samples_leaf' :[3, 5, 7], 'randomforestregressor__min_samples_split' : [3, 5, 10]}
]   # (1) make_pipeline사용시, 원래 파라미터는 pipe(RandomForestRegressor)의 파라미터 이므로 파라미터에 명시를 하고 GridSearchCV의 파라미터 형태로 바꿔줌(래핑)

# parameters = [
#     {'rF__max_depth' : [6, 8, 10], 'rF__min_samples_leaf' :[3, 5, 7]},
#     {'rF__min_samples_leaf' :[3, 5, 7], 'rF__min_samples_split' : [3, 5, 10]}
# ]   # (2) Pipeline사용시, RandomForestRegressor를 "rF"로 명시해주었으므로 rF로 래핑하여 파라미터를 바꿔줌


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.pipeline import make_pipeline, Pipeline

#2.모델
# model = SVC()
pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())   # 스케일러 두번 적용 가능 ; 성능은 잘 모름
# pipe = Pipeline([("mm", MinMaxScaler()), ("rF", RandomForestRegressor())])   # -> make_pipeline(MinMaxScaler(), RandomForestRegressor()) 와 동일함

# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
# model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)
model = HalvingRandomSearchCV(pipe, parameters, cv=5, verbose=1)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("걸린시간 : ", end - start)
print("model.score : ", result)
print("r2_score : ", r2)

'''
GridSearchCV
Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린시간 :  11.021263360977173
model.score :  0.9209196495681634
r2_score :  0.9209196495681634

RandomizedSearchCV
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린시간 :  6.195126533508301
model.score :  0.9245713226021559
r2_score :  0.9245713226021559

HalvingGridSearchCV
Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린시간 :  10.992571353912354
model.score :  0.9215401350414401
r2_score :  0.9215401350414401

HalvingRandomSearchCV
Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린시간 :  9.751389026641846
model.score :  0.906278084475882
r2_score :  0.906278084475882


'''