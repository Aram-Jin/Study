import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

# model : RandomForestClassifier
# 파라미터 조합으로 2개이상 엮을 것

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' :[3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100], 'max_depth' : [6, 12], 'min_samples_leaf' :[7, 10], 'min_samples_split' : [2, 3]},
    {'n_estimators' : [200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 5], 'min_samples_split' : [5, 10]},
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8], 'min_samples_leaf' :[3, 10], 'min_samples_split' : [2, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True)   # -> 너무 중요해~~

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)   # 최적의 매개변수 :  SVC(C=1, kernel='linear')
print("최적의 파라미터 : ", model.best_params_)      # 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}

print("best_score_ : ", model.best_score_)            # best_score_ :  0.9916666666666668  -> train부분에서 훈련시킨 값 중 최고 값(acc)
print("model.score : ", model.score(x_test, y_test))  # model.score :  0.9666666666666667  -> test까지 넣어서 나온 값 중 최고값(val_acc).  iris는 분류모델이므로 accuracy 값

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))   # accuracy_score :  0.9666666666666667  -> test까지 넣어서 나온 값 중 최고값(val_acc).  iris는 분류모델이므로 accuracy 값


y_pred_best = model.best_estimator_.predict(x_test)    # gridsearch 사용할떄 model.predict보다는 model.best_estimator_.predict 사용하길 권장함
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))
