from inspect import Parameter
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3,4,5]},   # 12
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},     # 6
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                     
     "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}                   # 24
]                                                                 # 총 42개

#2. 모델구성
model = GridSearchCV(SVC(), parameters, cv=kfold)
# model = SVC(C=1, kernel='linear', degree=3)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
aaa = model.score(x_test, y_test)
print(aaa)

'''
0.9666666666666667   -> iris는 분류모델이므로 accuracy 값
'''

