import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

#2. 모델구성
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True)   # -> 너무 중요해~~
model = SVC(C=1, kernel='linear', degree=3)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("model.score : ", model.score(x_test, y_test))  # model.score :  0.9666666666666667  -> test까지 넣어서 나온 값 중 최고값(val_acc).  iris는 분류모델이므로 accuracy 값

y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))   # accuracy_score :  0.9666666666666667  -> test까지 넣어서 나온 값 중 최고값(val_acc).  iris는 분류모델이므로 accuracy 값

'''
model.score :  0.9666666666666667
accuracy_score :  0.9666666666666667
'''