from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier   # Classifier :분류모델
from sklearn.linear_model import LogisticRegression  # LogisticRegression :분류모델**  / 보통 Regression은 회귀모델이지만..
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8, shuffle=True, random_state=66)

#2.모델
# model = Perceptron()
# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)    # score는 자동으로 맞춰서 반환해줌; 여기서 반환해주는건 'accuracy' (분류모델이기 때문에)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

# print("Perceptron : ", result)
# print("LinearSVC : ", result)
# print("SVC : ", result)
print("KNeighborsClassifier : ", result)
# print("LogisticRegression : ", result)
# print("DecisionTreeClassifier : ", result)
# print("RandomForestClassifier : ", result)

'''
Perceptron :  0.49991824651687133

'''